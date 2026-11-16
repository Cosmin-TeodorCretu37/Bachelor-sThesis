import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DIR_ANTRENARE

FISIER_PIXELI = DIR_ANTRENARE / "fer2013.csv"
FISIER_ETICHETE = DIR_ANTRENARE / "ferplus" / "fer2013new.csv"
DIR_IESIRE = DIR_ANTRENARE / "ferplus"

EMOTII_FERPLUS = {
    "neutral": "neutral", "happiness": "happy", "surprise": "surprise", "sadness": "sad",
    "anger": "angry", "disgust": "disgust", "fear": "fear",
}
COLOANE_VOTURI = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear",
                  "contempt", "unknown", "NF"]
SUBSETURI = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def eticheta_majoritara(voturi):
    voturi = np.where(voturi > 1, voturi, 0)
    total = voturi.sum()
    if total == 0 or voturi.max() <= 0.5 * total:
        return None
    eticheta = COLOANE_VOTURI[int(np.argmax(voturi))]
    return EMOTII_FERPLUS.get(eticheta)


def main():
    pixeli = pd.read_csv(FISIER_PIXELI)
    etichete = pd.read_csv(FISIER_ETICHETE)
    if len(pixeli) != len(etichete) or not (pixeli.Usage.values == etichete.Usage.values).all():
        sys.exit("Fisierele fer2013.csv si fer2013new.csv nu corespund.")

    statistici = {}
    for i, (rand_pixeli, rand_etichete) in enumerate(zip(pixeli.itertuples(), etichete.itertuples(), strict=True)):
        emotie = eticheta_majoritara(np.array([getattr(rand_etichete, c) for c in COLOANE_VOTURI], dtype=float))
        subset = SUBSETURI[rand_pixeli.Usage]
        if emotie is None:
            statistici[(subset, "eliminate")] = statistici.get((subset, "eliminate"), 0) + 1
            continue

        imagine = np.array(rand_pixeli.pixels.split(), dtype=np.uint8).reshape(48, 48)
        director = DIR_IESIRE / subset / emotie
        director.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(director / f"{i:05d}.png"), imagine)
        statistici[(subset, emotie)] = statistici.get((subset, emotie), 0) + 1

    for subset in SUBSETURI.values():
        valori = {e: n for (s, e), n in sorted(statistici.items()) if s == subset}
        print(f"{subset}: {valori}")


if __name__ == "__main__":
    main()
