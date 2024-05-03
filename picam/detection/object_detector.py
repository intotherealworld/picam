import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from summer_toolkit.utility.singleton import Singleton

object_class_map = {
    "0": "background",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "12",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "26",
    "27": "backpack",
    "28": "umbrella",
    "29": "29",
    "30": "30",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "45",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "66",
    "67": "dining table",
    "68": "68",
    "69": "69",
    "70": "toilet",
    "71": "71",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "83",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush",
}


class ObjectDetector(metaclass=Singleton):
    def __init__(self):
        self.current_dir = os.getcwd()
        model_path = os.path.join(self.current_dir, 'ssd_mobilenet_v1_coco_2017_11_17')

        self.cv_net = cv2.dnn.readNetFromTensorflow(
            os.path.join(model_path, 'frozen_inference_graph.pb'),
            os.path.join(model_path, 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt'),
            )

    def detect(self, pil_image):
        img = np.array(pil_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rows = img.shape[0]
        cols = img.shape[1]
        self.cv_net.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        cv_out = self.cv_net.forward()

        for detection in cv_out[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                left_top = (int(left), int(top))
                text_org = (left_top[0] + 10, left_top[1] + 20)
                right_bottom = (int(right), int(bottom))
                class_name = object_class_map[str(int(detection[1]))]
                text = f'{class_name}({int(score * 100)}%)'
                cv2.rectangle(img, left_top, right_bottom, (23, 230, 210), thickness=2)
                cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 230, 210), 2)

        return cv2.imencode(".jpg", img)[1].tobytes()
