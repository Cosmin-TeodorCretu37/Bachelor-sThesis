import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import repere_mana
from src.config import DIR_ANTRENARE

#config
GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]
EXEMPLE_PER_GEST = 500
DIR_DATE = DIR_ANTRENARE / "date_gesturi"
NUME_DEGETE = ["Aratator", "Mijlociu", "Inelar", "Mic"]

#instructiuni gesturi
INSTRUCTIUNI = {
    "pumn_inchis": "Strange TOATE degetele in pumn",
    "palma_deschisa": "Deschide TOATE cele 5 degete larg",
    "unu": "Ridica DOAR aratator, rest stranse",
    "doi": "Ridica DOAR aratator+mijlociu",
    "trei": "Ridica DOAR aratator+mijlociu+inelar",
    "ok": "Formeaza cerc cu deget mare+aratator"
}


def degete_ridicate(hand_landmarks):
    #ret degete sunt ridicate
    #feedback vizual
    varfuri = [8, 12, 16, 20]
    baze = [6, 10, 14, 18]
    return [hand_landmarks[varf].y < hand_landmarks[baza].y for varf, baza in zip(varfuri, baze)]


def gest_valid(gest, degete):
    asteptat = {
        "pumn_inchis": [False, False, False, False],
        "palma_deschisa": [True, True, True, True],
        "unu": [True, False, False, False],
        "doi": [True, True, False, False],
        "trei": [True, True, True, False],
    }
    return gest not in asteptat or degete == asteptat[gest]


def afiseaza_degete(frame, degete):
    for i, (ridicat, nume) in enumerate(zip(degete, NUME_DEGETE)):
        culoare = (0, 255, 0) if ridicat else (0, 0, 255)
        cv2.putText(frame, f"{nume}: {'SUS' if ridicat else 'JOS'}", (10, 160 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, culoare, 1)


def citeste_cadru(cap):
    success, frame = cap.read()
    return cv2.flip(frame, 1) if success else None


def asteapta_start(cap, detector, index_gest, gest):
    while True:
        frame = citeste_cadru(cap)
        if frame is None:
            return False
        landmarks = detector.detecteaza(frame)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"GEST {index_gest + 1}/{len(GESTURI)}: {gest.upper()}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, INSTRUCTIUNI[gest],
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "SPATIU = start, Q = iesire",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        if landmarks is not None:
            repere_mana.deseneaza(frame, landmarks)
            afiseaza_degete(frame, degete_ridicate(landmarks))

        cv2.imshow("Colectare date", frame)
        tasta = cv2.waitKey(1) & 0xFF
        if tasta == ord(' '):
            return True
        if tasta == ord('q'):
            return False


def colecteaza(cap, detector, gest):
    exemple_colectate = 0
    with open(DIR_DATE / f"{gest}.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        while exemple_colectate < EXEMPLE_PER_GEST:
            frame = citeste_cadru(cap)
            if frame is None:
                break
            landmarks = detector.detecteaza(frame)

            if landmarks is not None:
                #verificare corectitudine gest
                degete = degete_ridicate(landmarks)
                valid = gest_valid(gest, degete)
                if valid:
                    writer.writerow(repere_mana.normalizeaza(landmarks))
                    exemple_colectate += 1

                repere_mana.deseneaza(frame, landmarks)

                #feedback culoare: verde=valid, rosu=invalid
                culoare_status = (0, 255, 0) if valid else (0, 0, 255)
                status_text = "VALID - salvez!" if valid else "INVALID - corecteaza gestul!"
                cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, culoare_status, 2)
                afiseaza_degete(frame, degete)

            #progress bar
            progres = int((exemple_colectate / EXEMPLE_PER_GEST) * 400)
            cv2.rectangle(frame, (10, 450), (410, 472), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progres, 472), (0, 255, 100), -1)
            cv2.putText(frame, f"{gest.upper()}: {exemple_colectate}/{EXEMPLE_PER_GEST}", (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Colectare date", frame)
            cv2.waitKey(1)

    print(f"{gest}: {exemple_colectate} exemple salvate")


def main():
    DIR_DATE.mkdir(parents=True, exist_ok=True)
    detector = repere_mana.DetectorMana(incredere=0.7)
    cap = cv2.VideoCapture(0)

    try:
        for index_gest, gest in enumerate(GESTURI):
            print(f"\n---{gest.upper()}---\nInstructiune: {INSTRUCTIUNI[gest]}")
            if not asteapta_start(cap, detector, index_gest, gest):
                print("Colectare oprita.")
                return
            colecteaza(cap, detector, gest)
        print(f"\nColectare finalizata! Fisierele CSV sunt in '{DIR_DATE}'")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.inchide()


if __name__ == "__main__":
    main()
