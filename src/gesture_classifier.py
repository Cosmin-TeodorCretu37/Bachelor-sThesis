import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from src.config import GESTURE_LABELS, GESTURE_MODEL
from src.detectors import HandDetector, normalize_hand


@dataclass
class GestureResult:
    landmarks: Any
    gesture: str
    confidence: float


class GestureClassifier:
    EMERGENCY_GESTURE = 'fist'
    CANCEL_GESTURE = 'open_palm'

    MEDICAL_COMMANDS = {
        'fist': 'MEDICAL EMERGENCY!',
        'ok': 'I need water.',
        'one': 'I am in pain. \nNeed painkillers.',
        'two': 'Please call a nurse.',
        'three': 'Please adjust the bed.',
        'open_palm': 'Cancel command.'
    }

    def __init__(self, model_path=GESTURE_MODEL, labels_path=GESTURE_LABELS):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path) as f:
            self.index_to_gesture = {int(v): k for k, v in json.load(f).items()}

        self.detector = HandDetector(confidence=0.6)

    def predict(self, hand_landmarks):
        model_input = np.array([normalize_hand(hand_landmarks)], dtype=np.float32)
        probabilities = self.model(model_input, training=False).numpy()[0]
        best = int(np.argmax(probabilities))
        return self.index_to_gesture[best], float(probabilities[best])

    def process_frame(self, frame):
        landmarks = self.detector.detect(frame)
        if landmarks is None:
            return None
        gesture, confidence = self.predict(landmarks)
        return GestureResult(landmarks, gesture, confidence)

    def close(self):
        self.detector.close()
