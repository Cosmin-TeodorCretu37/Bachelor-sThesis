import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import TASK_FATA, TASK_MANA

CONEXIUNI_PALMA = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]


class _DetectorVideo:
    def __init__(self, detector):
        self.detector = detector
        self.ultimul_timestamp = -1

    def _ruleaza(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.ultimul_timestamp = max(int(time.monotonic() * 1000), self.ultimul_timestamp + 1)
        return self.detector.detect_for_video(mp_image, self.ultimul_timestamp)

    def inchide(self):
        self.detector.close()


class DetectorMana(_DetectorVideo):
    def __init__(self, incredere=0.6, calea_task=TASK_MANA):
        super().__init__(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(calea_task)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=incredere,
            min_hand_presence_confidence=incredere,
            min_tracking_confidence=incredere
        )))

    def detecteaza(self, frame):
        rezultat = self._ruleaza(frame)
        return rezultat.hand_landmarks[0] if rezultat.hand_landmarks else None


class DetectorFata(_DetectorVideo):
    EXTINDERE_DECUPAJ = 1.22
    DECALAJ_VERTICAL = -0.07

    def __init__(self, incredere=0.5, calea_task=TASK_FATA):
        super().__init__(vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(calea_task)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=incredere
        )))

    def detecteaza(self, frame):
        rezultat = self._ruleaza(frame)
        if not rezultat.detections:
            return None

        b = max(rezultat.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        h_img, w_img = frame.shape[:2]
        latura = min(int(max(b.width, b.height) * self.EXTINDERE_DECUPAJ), w_img, h_img)
        cx = b.origin_x + b.width / 2
        cy = b.origin_y + b.height / 2 + self.DECALAJ_VERTICAL * latura
        x = int(min(max(cx - latura / 2, 0), w_img - latura))
        y = int(min(max(cy - latura / 2, 0), h_img - latura))
        return x, y, latura, latura


def normalizeaza_mana(hand_landmarks):
    baza = hand_landmarks[0]
    coordonate_relative = []
    for lm in hand_landmarks:
        coordonate_relative.extend([lm.x - baza.x, lm.y - baza.y, lm.z - baza.z])

    val_max = max(map(abs, coordonate_relative)) or 1.0
    return [v / val_max for v in coordonate_relative]


def deseneaza_mana(frame, hand_landmarks):
    h, w = frame.shape[:2]
    puncte = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in CONEXIUNI_PALMA:
        cv2.line(frame, puncte[start], puncte[end], (0, 215, 255), 2, cv2.LINE_AA)
    for (x, y) in puncte:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
