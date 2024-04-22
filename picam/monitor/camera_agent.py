import io
import time

from picamera2 import Picamera2

from summer_toolkit.utility.singleton import Singleton


class CameraAgent(metaclass=Singleton):
    def __init__(self):
        self.camera = Picamera2()
        self.camera_still_config = self.camera.create_still_configuration(
            main={
                'size': (1024, 768)
            }
        )
        self.camera.configure(self.camera_still_config)
        self.camera.start()
        time.sleep(2)

    def capture(self, is_bytearray=True):
        captured = self.camera.capture_image('main')

        if is_bytearray:
            result = io.BytesIO()
            captured.save(result, format='jpeg')

            return result.getvalue()

        return captured
