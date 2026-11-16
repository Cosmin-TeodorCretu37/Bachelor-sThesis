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

GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]
EXEMPLE_PER_GEST = 500
DIR_DATE = DIR_PROIECT / "antrenare" / "date_gesturi"
CALE_TASK = DIR_PROIECT / "modele" / "hand_landmarker.task"
NUME_DEGETE = ["Aratator", "Mijlociu", "Inelar", "Mic"]

DIR_DATE.mkdir(parents=True, exist_ok=True)

base_options = python.BaseOptions(model_asset_path=str(CALE_TASK))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)


def degete_ridicate(hand_landmarks):
    #ret degete sunt ridicate
    #feedback vizual
    varfuri = [8, 12, 16, 20]
    baze = [6, 10, 14, 18]
    return [hand_landmarks[varf].y < hand_landmarks[baza].y for varf, baza in zip(varfuri, baze)]


def gest_valid(gest, degete):
    if gest == "pumn_inchis":
        return sum(degete) == 0
    if gest == "palma_deschisa":
        return sum(degete) == 4
    if gest == "unu":
        return degete == [True, False, False, False]
    if gest == "doi":
        return degete == [True, True, False, False]
    if gest == "trei":
        return degete == [True, True, True, False]
    return True


def afiseaza_degete(frame, degete):
    for i, (ridicat, nume) in enumerate(zip(degete, NUME_DEGETE)):
        culoare = (0, 255, 0) if ridicat else (0, 0, 255)
        cv2.putText(frame, f"{nume}: {'SUS' if ridicat else 'JOS'}",
                    (10, 160 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, culoare, 1)


#instructiuni gesturi
instructiuni = {
    "pumn_inchis": "Strange TOATE degetele in pumn",
    "palma_deschisa": "Deschide TOATE cele 5 degete larg",
    "unu": "Ridica DOAR aratator, rest stranse",
    "doi": "Ridica DOAR aratator+mijlociu",
    "trei": "Ridica DOAR aratator+mijlociu+inelar",
    "ok": "Formeaza cerc cu deget mare+aratator"
}

cap = cv2.VideoCapture(0)

for index_gest, gest in enumerate(GESTURI):
    fisier_csv = DIR_DATE / f"{gest}.csv"
    exemple_colectate = 0

    print(f"\n---{gest.upper()}---")
    print(f"Instructiune: {instructiuni[gest]}")
    print("Apasa SPATIU cand esti gata")

    #asteptam SPATIU
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"GEST {index_gest + 1}/{len(GESTURI)}: {gest.upper()}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, instructiuni[gest],
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Apasa SPATIU pentru a incepe",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        if results.hand_landmarks:
            repere_mana.deseneaza(frame, results.hand_landmarks[0])
            afiseaza_degete(frame, degete_ridicate(results.hand_landmarks[0]))

        cv2.imshow("Colectare date", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    #colectare cu feedback vizual
    with open(fisier_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        while exemple_colectate < EXEMPLE_PER_GEST:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                #verificare corectitudine gest
                degete = degete_ridicate(hand_landmarks)
                valid = gest_valid(gest, degete)

                if valid:
                    writer.writerow(repere_mana.normalizeaza(hand_landmarks))
                    exemple_colectate += 1

                repere_mana.deseneaza(frame, hand_landmarks)

                #feedback culoare: verde=valid, rosu=invalid
                culoare_status = (0, 255, 0) if valid else (0, 0, 255)
                status_text = "VALID - salvez!" if valid else "INVALID - corecteaza gestul!"
                cv2.putText(frame, status_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, culoare_status, 2)

                afiseaza_degete(frame, degete)

            #progress bar
            progress = int((exemple_colectate / EXEMPLE_PER_GEST) * 400)
            cv2.rectangle(frame, (10, 450), (410, 472), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progress, 472), (0, 255, 100), -1)
            cv2.putText(frame, f"{gest.upper()}: {exemple_colectate}/{EXEMPLE_PER_GEST}", (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Colectare date", frame)
            cv2.waitKey(1)

        print(f"{gest}: {exemple_colectate} exemple salvate")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\nColectare finalizata!")
