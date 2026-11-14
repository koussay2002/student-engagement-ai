"""Threaded capture: always serves the newest frame, never a stale backlog."""
from __future__ import annotations

import threading
import time

import cv2


class VideoStream:
    def __init__(self, source: str, max_width: int = 1280, loop: bool = True):
        s = str(source)
        live = s.isdigit() or s.startswith(("rtsp://", "http://", "https://"))
        self.is_file = not live
        src = int(s) if s.isdigit() else s
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.max_width = max_width
        self.loop = loop
        self.ended = False
        self._frame = None
        self._fid = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "VideoStream":
        self._thread.start()
        return self

    def _run(self) -> None:
        delay = 1.0 / self.fps if self.is_file else 0.0
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop and self._fid > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                break
            h, w = frame.shape[:2]
            if self.max_width and w > self.max_width:
                k = self.max_width / w
                frame = cv2.resize(frame, (self.max_width, int(h * k)), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._frame = frame
                self._fid += 1
            if delay:
                time.sleep(max(0.0, delay - (time.time() - t0)))

    def read(self):
        """Return (frame_id, frame). Frames are replaced, never mutated."""
        with self._lock:
            return (0, None) if self._frame is None else (self._fid, self._frame)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
