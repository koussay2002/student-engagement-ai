"""Head pose (yaw/pitch/roll) and eye aspect ratio from MediaPipe Face Mesh.

Runs on a crop of each student's head, so cost scales with people, not frame size.
Each worker thread owns its own FaceMesh instance (MediaPipe objects are not thread-safe).
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from .scoring import Features

# Generic 3D face model (mm) and matching Face Mesh landmark ids:
# nose tip, chin, outer eye corners (image-left, image-right), mouth corners.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)
_POSE_IDS = [1, 152, 33, 263, 61, 291]
_EYE_A = [33, 160, 158, 133, 153, 144]
_EYE_B = [362, 385, 387, 263, 373, 380]
_MIN_SIDE = 192


def _wrap(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def _ear(pts: np.ndarray, idx) -> float:
    p = pts[idx]
    d = np.linalg.norm
    return float((d(p[1] - p[5]) + d(p[2] - p[4])) / (2.0 * d(p[0] - p[3]) + 1e-6))


def head_crop(frame, box):
    """Upper part of a person box, where the head is (seated students)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - 0.1 * bw, x2 + 0.1 * bw
    y2 = y1 + 0.5 * bh
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = frame[y1:y2, x1:x2]
    short = min(crop.shape[:2])
    if short < _MIN_SIDE:
        k = _MIN_SIDE / short
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return crop


class FaceAnalyzer:
    def __init__(self):
        self._local = threading.local()

    def _mesh(self):
        m = getattr(self._local, "mesh", None)
        if m is None:
            import mediapipe as mp
            m = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.4)
            self._local.mesh = m
        return m

    def analyze(self, frame, box) -> Features:
        crop = head_crop(frame, box)
        if crop is None:
            return Features(face=False)
        res = self._mesh().process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return Features(face=False)

        h, w = crop.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

        cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts[_POSE_IDS], cam, np.zeros(4),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        pitch = yaw = roll = 0.0
        if ok:
            angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
            pitch, yaw, roll = _wrap(angles[0]), _wrap(angles[1]), _wrap(angles[2])

        ear = (_ear(pts, _EYE_A) + _ear(pts, _EYE_B)) / 2.0
        return Features(face=True, yaw=yaw, pitch=pitch, roll=roll, ear=ear)
