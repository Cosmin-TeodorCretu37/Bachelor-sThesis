import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import DIR_MODELE

CONEXIUNI_PALMA = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]


class DetectorMana:
    def __init__(self, incredere=0.6, calea_task=DIR_MODELE / "hand_landmarker.task"):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(calea_task)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=incredere,
            min_hand_presence_confidence=incredere,
            min_tracking_confidence=incredere
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.ultimul_timestamp = -1

    def detecteaza(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.ultimul_timestamp = max(int(time.monotonic() * 1000), self.ultimul_timestamp + 1)
        rezultat = self.detector.detect_for_video(mp_image, self.ultimul_timestamp)
        return rezultat.hand_landmarks[0] if rezultat.hand_landmarks else None

    def inchide(self):
        self.detector.close()


def normalizeaza(hand_landmarks):
    baza = hand_landmarks[0]
    coordonate_relative = []
    for lm in hand_landmarks:
        coordonate_relative.extend([lm.x - baza.x, lm.y - baza.y, lm.z - baza.z])

    val_max = max(map(abs, coordonate_relative)) or 1.0
    return [v / val_max for v in coordonate_relative]


def deseneaza(frame, hand_landmarks):
    h, w = frame.shape[:2]
    puncte = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in CONEXIUNI_PALMA:
        cv2.line(frame, puncte[start], puncte[end], (0, 215, 255), 2, cv2.LINE_AA)
    for (x, y) in puncte:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
