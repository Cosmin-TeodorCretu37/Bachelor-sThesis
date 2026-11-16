import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

DIR_MODELE = Path(__file__).resolve().parent.parent / "modele"


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

    CULORI_STARI = {
        'Stable': ((255, 200, 0), '#58a6ff'),
        'Comfort': ((0, 255, 0), '#3fb950'),
        'Discomfort/Pain': ((0, 0, 255), '#f85149'),
        'Discomfort': ((0, 0, 255), '#f85149'),
    }

    PRAG_SIGURANTA_FURIE = 0.65
    SCALA_DETECTIE = 0.5

    def __init__(self, calea_model=DIR_MODELE / "model_emotii_best.h5",
                 calea_mapare=DIR_MODELE / "mapare_emotii.json"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.mapare = json.load(f)

        self.idx_la_emotie = {int(v): k for k, v in self.mapare.items()}

        self.detectie_fata = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predictie(self, fata_gri):
        fata_redimensionata = cv2.resize(fata_gri, (48, 48))
        fata_input = (fata_redimensionata / 255.0).astype(np.float32).reshape(1, 48, 48, 1)

        procente = self.model(fata_input, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))
        emotie_raw = self.idx_la_emotie[max_idx]
        incredere = float(procente[max_idx])

        if emotie_raw == 'angry' and incredere < self.PRAG_SIGURANTA_FURIE:
            emotie_raw = 'neutral'
            incredere = max(float(procente[self.mapare['neutral']]), 0.50)

        return emotie_raw, incredere

    def detecteaza_fata(self, gri):
        gri_mic = cv2.resize(gri, None, fx=self.SCALA_DETECTIE, fy=self.SCALA_DETECTIE)
        latura_minima = int(60 * self.SCALA_DETECTIE)
        fete = self.detectie_fata.detectMultiScale(gri_mic, scaleFactor=1.1, minNeighbors=5,
                                                   minSize=(latura_minima, latura_minima))
        if len(fete) == 0:
            return None

        x, y, w, h = max(fete, key=lambda f: f[2] * f[3])
        return tuple(int(v / self.SCALA_DETECTIE) for v in (x, y, w, h))

    def proceseaza_cadru(self, frame):
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = self.detecteaza_fata(gri)
        if box is None:
            return None

        x, y, w, h = box
        emotie_raw, incredere = self.predictie(gri[y:y + h, x:x + w])
        return {
            'box': box,
            'emotie_ro': self.EMOTII_RO[emotie_raw],
            'stare': self.STARI_CLINICE[emotie_raw],
            'incredere': incredere,
        }
