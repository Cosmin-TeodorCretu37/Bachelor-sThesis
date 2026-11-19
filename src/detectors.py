import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import FACE_TASK, HAND_TASK

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]


class _VideoDetector:
    def __init__(self, detector):
        self.detector = detector
        self.last_timestamp = -1

    def _run(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.last_timestamp = max(int(time.monotonic() * 1000), self.last_timestamp + 1)
        return self.detector.detect_for_video(mp_image, self.last_timestamp)

    def close(self):
        self.detector.close()


class HandDetector(_VideoDetector):
    def __init__(self, confidence=0.6, task_path=HAND_TASK):
        super().__init__(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )))

    def detect(self, frame):
        result = self._run(frame)
        return result.hand_landmarks[0] if result.hand_landmarks else None


class FaceDetector(_VideoDetector):
    CROP_EXPANSION = 1.22
    VERTICAL_OFFSET = -0.07

    def __init__(self, confidence=0.5, task_path=FACE_TASK):
        super().__init__(vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=confidence
        )))

    def detect(self, frame):
        result = self._run(frame)
        if not result.detections:
            return None

        b = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        img_h, img_w = frame.shape[:2]
        side = min(int(max(b.width, b.height) * self.CROP_EXPANSION), img_w, img_h)
        cx = b.origin_x + b.width / 2
        cy = b.origin_y + b.height / 2 + self.VERTICAL_OFFSET * side
        x = int(min(max(cx - side / 2, 0), img_w - side))
        y = int(min(max(cy - side / 2, 0), img_h - side))
        return x, y, side, side


def normalize_hand(hand_landmarks):
    wrist = hand_landmarks[0]
    relative_coords = []
    for lm in hand_landmarks:
        relative_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    max_value = max(map(abs, relative_coords)) or 1.0
    return [v / max_value for v in relative_coords]


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 215, 255), 2, cv2.LINE_AA)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
