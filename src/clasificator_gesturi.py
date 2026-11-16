import json

import numpy as np
import tensorflow as tf

from src import repere_mana
from src.config import DIR_MODELE


class ClasificatorGesturi:
    GEST_URGENTA = 'pumn_inchis'
    GEST_ANULARE = 'palma_deschisa'

    COMENZI_MEDICALE = {
        'pumn_inchis': 'MEDICAL EMERGENCY!',
        'ok': 'I need water.',
        'unu': 'I am in pain. \nNeed painkillers.',
        'doi': 'Please call a nurse.',
        'trei': 'Please adjust the bed.',
        'palma_deschisa': 'Cancel command.'
    }

    def __init__(self, calea_model=DIR_MODELE / "model_gesturi.h5",
                 calea_mapare=DIR_MODELE / "mapare_gesturi.json"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.idx_la_gest = {int(v): k for k, v in json.load(f).items()}

        self.detector = repere_mana.DetectorMana(incredere=0.6)

    def predictie(self, hand_landmarks):
        input_model = np.array([repere_mana.normalizeaza(hand_landmarks)], dtype=np.float32)
        procente = self.model(input_model, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))
        return self.idx_la_gest[max_idx], float(procente[max_idx])

    def proceseaza_cadru(self, frame):
        landmarks = self.detector.detecteaza(frame)
        if landmarks is None:
            return None, None, 0.0
        gest, incredere = self.predictie(landmarks)
        return landmarks, gest, incredere

    def inchide(self):
        self.detector.inchide()
