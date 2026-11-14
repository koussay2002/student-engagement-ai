"""Capture -> detect -> track -> face analysis -> score -> annotated stream."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from .capture import VideoStream
from .config import Settings
from .scoring import EngagementModel, Features, summarize
from .tracker import IoUTracker

log = logging.getLogger("engagement")

# BGR colours, matched to the dashboard palette.
COLORS = {"Focused": (74, 122, 31), "Distracted": (43, 57, 192),
          "Drowsy": (31, 121, 183), "Confused": (181, 91, 107),
          "Unknown": (172, 162, 152)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Pipeline:
    def __init__(self, settings: Settings, detector=None, faces=None, stream=None):
        self.s = settings
        self.stream = stream or VideoStream(settings.source, settings.max_width)
        if detector is None:
            from .detector import Detector
            detector = Detector(settings.model, settings.imgsz, settings.conf, settings.device)
        if faces is None:
            from .face import FaceAnalyzer
            faces = FaceAnalyzer()
        self.detector, self.faces = detector, faces
        self.tracker = IoUTracker()
        self.model = EngagementModel()
        self.pool = ThreadPoolExecutor(max_workers=settings.face_workers)
        self.results: dict = {}

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._vis = None
        self._counter = 0
        self._snapshot = {"fps": 0.0, "students": [], "summary": summarize([])}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.stream.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.stream.stop()
        self.pool.shutdown(wait=False)

    # ---- public accessors ------------------------------------------------
    def latest(self):
        with self._lock:
            return self._counter, self._jpeg

    def latest_frame(self):
        with self._lock:
            return self._vis

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _phone_near(box, phones) -> bool:
        x1, y1, x2, y2 = box
        mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        for p in phones:
            cx, cy = (p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0
            if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
                return True
        return False

    def _draw(self, frame, tracks, phones):
        vis = frame.copy()
        h, w = vis.shape[:2]
        k = max(0.45, w / 1800.0)
        th = max(1, round(2 * k))
        fs, lh, pad = 0.55 * k, int(26 * k), 4

        for p in phones:
            cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 140, 255), th)

        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.box)
            r = self.results.get(t.sid)
            color = COLORS.get(r.label if r else "Unknown", COLORS["Unknown"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, th)
            lines = [t.sid]
            if r:
                lines += [r.label, f"CONF {int(r.confidence * 100)}%",
                          f"PRED {int(round(r.predicted))}/100"]
            else:
                lines.append("Analyzing")
            bw = max(cv2.getTextSize(s, FONT, fs, th)[0][0] for s in lines) + 2 * pad
            bh = lh * len(lines) + pad
            ty = y1 - bh if y1 - bh >= 0 else y1
            tx = min(x1, max(0, w - bw))
            cv2.rectangle(vis, (tx, ty), (tx + bw, ty + bh), color, -1)
            for i, s in enumerate(lines):
                cv2.putText(vis, s, (tx + pad, ty + lh * (i + 1) - pad), FONT, fs,
                            (255, 255, 255), th, cv2.LINE_AA)
        return vis

    def _loop(self) -> None:
        s = self.s
        last_fid, i = 0, 0
        tracks, phones = [], []
        frames, t_fps, fps = 0, time.time(), 0.0
        while not self._stop.is_set():
            fid, frame = self.stream.read()
            if frame is None or fid == last_fid:
                time.sleep(0.05 if self.stream.ended else 0.003)
                continue
            last_fid, i = fid, i + 1
            try:
                if (i - 1) % s.detect_every == 0:
                    persons, phones = self.detector(frame)
                    tracks = self.tracker.update([p[:4] for p in persons])
                    self.model.prune({t.sid for t in self.tracker.tracks})

                now = time.time()
                if tracks and i % s.face_every == 0:
                    jobs = [(t, self.pool.submit(self.faces.analyze, frame, t.box)) for t in tracks]
                    for t, fut in jobs:
                        try:
                            f = fut.result()
                        except Exception:
                            log.exception("face analysis failed")
                            f = Features(face=False)
                        f.phone = self._phone_near(t.box, phones)
                        self.results[t.sid] = self.model.update(t.sid, f, now)

                vis = self._draw(frame, tracks, phones)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])

                frames += 1
                if now - t_fps >= 1.0:
                    fps, frames, t_fps = frames / (now - t_fps), 0, now
                fh, fw = frame.shape[:2]
                live = [(t, self.results[t.sid]) for t in tracks if t.sid in self.results]
                students = [{"id": t.sid, "label": r.label, "score": round(r.score),
                             "confidence": round(r.confidence * 100),
                             "predicted": round(r.predicted), "trend": round(r.trend, 1),
                             "x": round((t.box[0] + t.box[2]) / 2 / fw, 3),
                             "y": round((t.box[1] + t.box[3]) / 2 / fh, 3)}
                            for t, r in live]
                snap = {"fps": round(fps, 1), "students": students,
                        "summary": summarize(r for _, r in live)}
                with self._lock:
                    self._vis = vis
                    if ok:
                        self._jpeg = buf.tobytes()
                        self._counter += 1
                    self._snapshot = snap
            except Exception:
                log.exception("pipeline iteration failed")
                time.sleep(0.05)
