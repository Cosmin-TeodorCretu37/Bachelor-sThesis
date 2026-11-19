import json
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

from src.config import EMOTION_LABELS, EMOTION_MODEL
from src.detectors import FaceDetector


@dataclass
class EmotionResult:
    box: tuple
    emotion_name: str
    state: str
    confidence: float


class EmotionClassifier:

    EMOTION_NAMES = {
        'angry': 'Anger', 'disgust': 'Disgust', 'fear': 'Fear',
        'happy': 'Happiness', 'neutral': 'Neutral', 'sad': 'Sadness',
        'surprise': 'Surprise',
    }

    CLINICAL_STATES = {
        'neutral': 'Stable',
        'surprise': 'Stable',

        'happy': 'Comfort',

        'sad': 'Discomfort/Pain',
        'angry': 'Discomfort/Pain',
        'fear': 'Discomfort',
        'disgust': 'Discomfort',
    }

    DISCOMFORT_STATES = ('Discomfort/Pain', 'Discomfort')

    STATE_COLORS = {
        'Stable': ((255, 200, 0), '#58a6ff'),
        'Comfort': ((0, 255, 0), '#3fb950'),
        'Discomfort/Pain': ((0, 0, 255), '#f85149'),
        'Discomfort': ((0, 0, 255), '#f85149'),
    }

    def __init__(self, model_path=EMOTION_MODEL, labels_path=EMOTION_LABELS):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path) as f:
            self.index_to_emotion = {int(v): k for k, v in json.load(f).items()}

        self.detector = FaceDetector()

    def predict(self, face_gray):
        face_resized = cv2.resize(face_gray, (48, 48))
        model_input = (face_resized / 255.0).astype(np.float32).reshape(1, 48, 48, 1)

        probabilities = self.model(model_input, training=False).numpy()[0]
        best = int(np.argmax(probabilities))
        return self.index_to_emotion[best], float(probabilities[best])

    def process_frame(self, frame):
        box = self.detector.detect(frame)
        if box is None:
            return None

        x, y, w, h = box
        face_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        emotion, confidence = self.predict(face_gray)
        return EmotionResult(box, self.EMOTION_NAMES[emotion], self.CLINICAL_STATES[emotion], confidence)

    def close(self):
        self.detector.close()
