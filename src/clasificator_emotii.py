import json
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

from src.config import MAPARE_EMOTII, MODEL_EMOTII
from src.detectoare import DetectorFata


@dataclass
class RezultatEmotie:
    box: tuple
    emotie_ro: str
    stare: str
    incredere: float


class ClasificatorEmotii:

    EMOTII_RO = {
        'angry': 'Furie', 'disgust': 'Dezgust', 'fear': 'Frica',
        'happy': 'Bucurie', 'neutral': 'Neutru', 'sad': 'Tristete',
        'surprise': 'Surpriza',
    }

    STARI_CLINICE = {
        'neutral': 'Stable',
        'surprise': 'Stable',

        'happy': 'Comfort',

        'sad': 'Discomfort/Pain',
        'angry': 'Discomfort/Pain',
        'fear': 'Discomfort',
        'disgust': 'Discomfort',
    }

    STARI_DISCONFORT = ('Discomfort/Pain', 'Discomfort')

    CULORI_STARI = {
        'Stable': ((255, 200, 0), '#58a6ff'),
        'Comfort': ((0, 255, 0), '#3fb950'),
        'Discomfort/Pain': ((0, 0, 255), '#f85149'),
        'Discomfort': ((0, 0, 255), '#f85149'),
    }

    def __init__(self, calea_model=MODEL_EMOTII, calea_mapare=MAPARE_EMOTII):
        self.model = tf.keras.models.load_model(calea_model, compile=False)
        with open(calea_mapare) as f:
            self.idx_la_emotie = {int(v): k for k, v in json.load(f).items()}

        self.detector = DetectorFata()

    def predictie(self, fata_gri):
        fata_redimensionata = cv2.resize(fata_gri, (48, 48))
        fata_input = (fata_redimensionata / 255.0).astype(np.float32).reshape(1, 48, 48, 1)

        procente = self.model(fata_input, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))
        return self.idx_la_emotie[max_idx], float(procente[max_idx])

    def proceseaza_cadru(self, frame):
        box = self.detector.detecteaza(frame)
        if box is None:
            return None

        x, y, w, h = box
        fata_gri = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        emotie, incredere = self.predictie(fata_gri)
        return RezultatEmotie(box, self.EMOTII_RO[emotie], self.STARI_CLINICE[emotie], incredere)

    def inchide(self):
        self.detector.inchide()
