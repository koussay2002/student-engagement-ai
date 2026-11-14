from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    source: str = "0"              # webcam index, video file, or rtsp:// URL
    model: str = "yolov8n.pt"      # .pt, .onnx or .engine (TensorRT)
    imgsz: int = 960               # larger = better for small phones / far rows
    conf: float = 0.35
    device: str | None = None      # None = auto (cuda > mps > cpu)
    detect_every: int = 2          # run the detector every N frames
    face_every: int = 3            # run face analysis every N frames
    face_workers: int = 4
    jpeg_quality: int = 75
    max_width: int = 1280
