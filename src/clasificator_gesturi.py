import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from src.config import MAPARE_GESTURI, MODEL_GESTURI
from src.detectoare import DetectorMana, normalizeaza_mana


@dataclass
class RezultatGest:
    landmarks: Any
    gest: str
    incredere: float


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

    def __init__(self, calea_model=MODEL_GESTURI, calea_mapare=MAPARE_GESTURI):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.idx_la_gest = {int(v): k for k, v in json.load(f).items()}

        self.detector = DetectorMana(incredere=0.6)

    def predictie(self, hand_landmarks):
        input_model = np.array([normalizeaza_mana(hand_landmarks)], dtype=np.float32)
        procente = self.model(input_model, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))
        return self.idx_la_gest[max_idx], float(procente[max_idx])

    def proceseaza_cadru(self, frame):
        landmarks = self.detector.detecteaza(frame)
        if landmarks is None:
            return None
        gest, incredere = self.predictie(landmarks)
        return RezultatGest(landmarks, gest, incredere)

    def inchide(self):
        self.detector.inchide()
