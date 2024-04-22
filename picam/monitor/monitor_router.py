from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from picamera2 import Picamera2

from picam.monitor.camera_agent import CameraAgent


monitor_router = APIRouter(tags=['monitor'], prefix='/monitor')
camera_agent = CameraAgent()


def generate_image():
    while True:
        img_bytes = camera_agent.capture()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n'
        )


@monitor_router.get('', include_in_schema=False)
def respond_root(request: Request):
    return StreamingResponse(
        generate_image(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
