import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DIR_PROIECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIR_PROIECT))

from src import repere_mana

#config
GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]
EXEMPLE_PER_GEST = 300
DIR_DATE = DIR_PROIECT / "antrenare" / "date_gesturi"
CALE_TASK = DIR_PROIECT / "modele" / "hand_landmarker.task"

DIR_DATE.mkdir(parents=True, exist_ok=True)

#init
base_options = python.BaseOptions(model_asset_path=str(CALE_TASK))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

#bucla principala colectare date
cap = cv2.VideoCapture(0)

for index_gest, gest in enumerate(GESTURI):
    fisier_csv = DIR_DATE / f"{gest}.csv"
    exemple_colectate = 0

    print(f"\nPregateste-te pentru gestul: {gest.upper()}")
    print("Apasa SPATIU cand esti gata sa incepi colectarea")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        #instructiuni
        cv2.putText(frame, f"Gest: {gest.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Apasa SPATIU pentru a incepe", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gest {index_gest + 1} din {len(GESTURI)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Colectare date", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    #write CSV
    with open(fisier_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        while exemple_colectate < EXEMPLE_PER_GEST:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            if results.hand_landmarks:
                writer.writerow(repere_mana.normalizeaza(results.hand_landmarks[0]))
                exemple_colectate += 1

            #progress bar
            progress = int((exemple_colectate / EXEMPLE_PER_GEST) * 400)
            cv2.rectangle(frame, (10, 450), (410, 475), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progress, 475), (0, 255, 100), -1)

            cv2.putText(frame, f"{gest.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
            cv2.putText(frame, f"Colectat: {exemple_colectate}/{EXEMPLE_PER_GEST}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Tine gestul constant!", (10, 425),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            if results.hand_landmarks:
                repere_mana.deseneaza(frame, results.hand_landmarks[0])

            cv2.imshow("Colectare date", frame)
            cv2.waitKey(1)

        print(f"{gest}: {exemple_colectate} exemple salvate in {fisier_csv}")

cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"\nColectare finalizata! Fisierele CSV sunt in folderul '{DIR_DATE}'")
