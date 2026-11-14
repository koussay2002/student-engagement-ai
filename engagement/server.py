"""FastAPI app: MJPEG video, JSON snapshot and a WebSocket feed for the dashboard."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

STATIC = Path(__file__).parent / "static"


def create_app(pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.start()
        yield
        pipeline.stop()

    app = FastAPI(title="Classroom engagement", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/students")
    async def students():
        return pipeline.snapshot()

    @app.get("/video")
    async def video():
        async def frames():
            last = -1
            while True:
                counter, jpeg = pipeline.latest()
                if jpeg is None or counter == last:
                    await asyncio.sleep(0.01)
                    continue
                last = counter
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(pipeline.snapshot())
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
