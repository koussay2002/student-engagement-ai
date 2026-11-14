"""Lightweight IoU tracker that gives each student a stable anonymous ID."""
from __future__ import annotations

from dataclasses import dataclass


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple
    missed: int = 0
    hits: int = 1

    @property
    def sid(self) -> str:
        return f"STD-{self.id:04d}"


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 min_hits: int = 2, smooth: float = 0.6):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.smooth = smooth
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, boxes) -> list[Track]:
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, b in enumerate(boxes):
                s = iou(t.box, b)
                if s >= self.iou_threshold:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            k = self.smooth
            t.box = tuple(k * n + (1 - k) * o for n, o in zip(boxes[di][:4], t.box))
            t.missed = 0
            t.hits += 1

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, b in enumerate(boxes):
            if di not in used_d:
                self.tracks.append(Track(self._next, tuple(float(v) for v in b[:4])))
                self._next += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.missed == 0 and t.hits >= self.min_hits]
