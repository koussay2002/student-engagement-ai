"""YOLO person + cell-phone detector (COCO classes 0 and 67)."""
from __future__ import annotations

import numpy as np

PERSON, PHONE = 0, 67


def _auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, model: str = "yolov8n.pt", imgsz: int = 960,
                 conf: float = 0.35, device: str | None = None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.imgsz, self.conf = imgsz, conf
        self.device = device or _auto_device()
        self.half = self.device.startswith("cuda")  # FP16 only on NVIDIA GPUs
        self(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))  # warm-up

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               classes=[PERSON, PHONE], device=self.device,
                               half=self.half, verbose=False)[0]
        persons, phones = [], []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for b, c, p in zip(xyxy, cls, cf):
                item = (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(p))
                (persons if c == PERSON else phones).append(item)
        return persons, phones
