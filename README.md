# Classroom Engagement AI

Real-time engagement analysis for a classroom camera. It detects each student, tracks them with an
anonymous ID, reads head pose, eye closure and phone use, and turns that into a live engagement
score with a 5-minute forecast, shown on a web dashboard.

```
camera / video / RTSP
        |
  threaded capture ---- always the newest frame, no backlog
        |
  YOLO (person + phone) every N frames ---- FP16 on GPU, ONNX/TensorRT ready
        |
  IoU tracker ---- stable anonymous IDs (STD-0001 ...)
        |
  Face Mesh on head crops, parallel workers ---- yaw / pitch / roll, eye aspect ratio
        |
  Engagement model ---- smoothed score, majority-vote label, confidence, trend forecast
        |
  FastAPI ---- MJPEG video + WebSocket feed ---- dashboard
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --source 0                  # webcam
python main.py --source class.mp4          # video file
python main.py --source rtsp://user:pass@ip/stream
python main.py --source 0 --headless       # OpenCV window, press q to quit
```

Open <http://127.0.0.1:8000>. YOLO weights download automatically on first run.

## Performance

| Lever | How |
| --- | --- |
| GPU | Auto-selects CUDA, then Apple MPS, then CPU. FP16 is enabled on CUDA. |
| Frame skipping | `--detect-every 2` and `--face-every 3` reuse tracks between runs. |
| Cheaper face analysis | Face Mesh runs on head crops only, in a thread pool. |
| No stale frames | Capture runs in its own thread and keeps only the latest frame. |
| Faster model | Export once, then pass the file: `yolo export model=yolov8n.pt format=onnx` (or `engine` for TensorRT), `python main.py --model yolov8n.onnx`. |
| Far rows and small phones | Raise `--imgsz` to 1280, or use `yolov8s.pt`. |

## How the score works

The model in `engagement/scoring.py` is rule-based and readable:

- head turned away (yaw) and head down (pitch) lower the score,
- eyes closed (eye aspect ratio) marks the student as drowsy,
- a phone near the student marks them as distracted,
- head tilt is treated as a confusion cue,
- the score is smoothed (exponential average), the label is a majority vote over recent frames,
  and the forecast extends the recent trend over five minutes.

Replace `instant()` with a classifier trained on labelled data (for example the DAiSEE
dataset) and the tracker, pipeline and dashboard keep working unchanged.

## Limits you should know about

- The score measures visible attention cues. It is not a validated predictor of grades.
  Looking down can mean taking notes, and tilted heads can mean many things.
- Thresholds depend on camera angle and distance. Tune them in `scoring.py` for your room.
- Accuracy drops for faces that are small, occluded or lit from behind.

## Privacy

Faces are analysed in memory. No images, names or biometric templates are stored, and students
are shown only as anonymous IDs. Before using this in a real classroom, get consent from students
(and guardians for minors), tell them what is measured, and check local rules such as GDPR.
Use it to understand the class as a group, not to grade or discipline individuals.

## Project layout

```
main.py                   entry point
engagement/capture.py     threaded video capture
engagement/detector.py    YOLO person and phone detection
engagement/tracker.py     IoU tracker, anonymous IDs
engagement/face.py        head pose and eye aspect ratio
engagement/scoring.py     score, label, confidence, forecast
engagement/pipeline.py    the processing loop and annotated stream
engagement/server.py      FastAPI app
engagement/static/        dashboard
tests/                    pytest suite (runs without a GPU or camera)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Roadmap

- Train a classifier on DAiSEE and compare it with the rule-based model
- Per-session CSV export of anonymous class-level statistics
- Docker image with CUDA
- Calibration helper for camera angle

## License

MIT
