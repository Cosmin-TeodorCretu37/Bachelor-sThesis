import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src import repere_mana

DIR_MODELE = Path(__file__).resolve().parent.parent / "modele"


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
                 calea_mapare=DIR_MODELE / "mapare_gesturi.json",
                 calea_task=DIR_MODELE / "hand_landmarker.task"):
        self.model = tf.keras.models.load_model(calea_model)
        with open(calea_mapare) as f:
            self.mapare = json.load(f)

        self.idx_la_gest = {int(v): k for k, v in self.mapare.items()}

        base_options = python.BaseOptions(model_asset_path=str(calea_task))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.ultimul_timestamp = -1

    def predictie(self, hand_landmarks):
        input_model = np.array([repere_mana.normalizeaza(hand_landmarks)], dtype=np.float32)
        procente = self.model(input_model, training=False).numpy()[0]
        max_idx = int(np.argmax(procente))

        gest_raw = self.idx_la_gest[max_idx]
        comanda_en = self.COMENZI_MEDICALE[gest_raw]

        return gest_raw, comanda_en, float(procente[max_idx])

    def proceseaza_cadru(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp = max(int(time.monotonic() * 1000), self.ultimul_timestamp + 1)
        self.ultimul_timestamp = timestamp
        rezultat_detectie = self.detector.detect_for_video(mp_image, timestamp)

        if rezultat_detectie.hand_landmarks:
            landmarks = rezultat_detectie.hand_landmarks[0]
            gest_raw, comanda_en, incredere = self.predictie(landmarks)
            return landmarks, gest_raw, comanda_en, incredere
        return None, None, None, 0.0

    def inchide(self):
        self.detector.close()
