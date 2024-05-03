from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from picamera2 import Picamera2

from picam.detection.object_detector import ObjectDetector
from picam.monitor.camera_agent import CameraAgent


detection_router = APIRouter(tags=['detection'], prefix='/detection')
camera_agent = CameraAgent()
object_detector = ObjectDetector()


def generate_image():
    while True:
        pil_img = camera_agent.capture(is_bytearray=False)
        img_bytes = object_detector.detect(pil_img)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n'
        )


@detection_router.get('', include_in_schema=False)
def respond_root(request: Request):
    return StreamingResponse(
        generate_image(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
