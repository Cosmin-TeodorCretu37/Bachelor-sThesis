import json

import cv2
import numpy as np
import tensorflow as tf

from src.config import DIR_MODELE


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

    SCALA_DETECTIE = 0.5

    def __init__(self, calea_model=DIR_MODELE / "model_emotii_best.h5",
                 calea_mapare=DIR_MODELE / "mapare_emotii.json"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.idx_la_emotie = {int(v): k for k, v in json.load(f).items()}

        self.detectie_fata = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predictie(self, fata_gri):
        fata_redimensionata = cv2.resize(fata_gri, (48, 48))
        fata_input = (fata_redimensionata / 255.0).astype(np.float32).reshape(1, 48, 48, 1)

        procente = self.model(fata_input, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))
        return self.idx_la_emotie[max_idx], float(procente[max_idx])

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
        emotie, incredere = self.predictie(gri[y:y + h, x:x + w])
        return {
            'box': box,
            'emotie_ro': self.EMOTII_RO[emotie],
            'stare': self.STARI_CLINICE[emotie],
            'incredere': incredere,
        }
