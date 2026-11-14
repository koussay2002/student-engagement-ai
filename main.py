"""Entry point: `python main.py --source 0` (webcam) or `--source class.mp4`."""
from __future__ import annotations

import argparse
import logging

from engagement.config import Settings


def parse_args():
    d = Settings()
    p = argparse.ArgumentParser(description="Real-time classroom engagement analysis")
    p.add_argument("--source", default=d.source, help="webcam index, video file or rtsp:// URL")
    p.add_argument("--model", default=d.model, help="YOLO weights (.pt/.onnx/.engine)")
    p.add_argument("--imgsz", type=int, default=d.imgsz)
    p.add_argument("--conf", type=float, default=d.conf)
    p.add_argument("--device", default=d.device, help="cuda:0, mps or cpu (default: auto)")
    p.add_argument("--detect-every", type=int, default=d.detect_every)
    p.add_argument("--face-every", type=int, default=d.face_every)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--headless", action="store_true", help="OpenCV window instead of the web dashboard")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.INFO)
    s = Settings(source=a.source, model=a.model, imgsz=a.imgsz, conf=a.conf,
                 device=a.device, detect_every=a.detect_every, face_every=a.face_every)
    from engagement.pipeline import Pipeline
    pipe = Pipeline(s)

    if a.headless:
        import cv2
        pipe.start()
        try:
            while True:
                frame = pipe.latest_frame()
                if frame is not None:
                    cv2.imshow("engagement", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
        finally:
            pipe.stop()
            cv2.destroyAllWindows()
        return

    import uvicorn
    from engagement.server import create_app
    print(f"Dashboard: http://{a.host}:{a.port}")
    uvicorn.run(create_app(pipe), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
