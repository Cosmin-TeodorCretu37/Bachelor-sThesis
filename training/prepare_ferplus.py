import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAINING_DIR

PIXELS_FILE = TRAINING_DIR / "fer2013.csv"
LABELS_FILE = TRAINING_DIR / "ferplus" / "fer2013new.csv"
OUTPUT_DIR = TRAINING_DIR / "ferplus"

FERPLUS_EMOTIONS = {
    "neutral": "neutral", "happiness": "happy", "surprise": "surprise", "sadness": "sad",
    "anger": "angry", "disgust": "disgust", "fear": "fear",
}
VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear",
                "contempt", "unknown", "NF"]
SUBSETS = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def majority_label(votes):
    votes = np.where(votes > 1, votes, 0)
    total = votes.sum()
    if total == 0 or votes.max() <= 0.5 * total:
        return None
    label = VOTE_COLUMNS[int(np.argmax(votes))]
    return FERPLUS_EMOTIONS.get(label)


def main():
    pixels = pd.read_csv(PIXELS_FILE)
    labels = pd.read_csv(LABELS_FILE)
    if len(pixels) != len(labels) or not (pixels.Usage.values == labels.Usage.values).all():
        sys.exit("The files fer2013.csv and fer2013new.csv do not match.")

    stats = {}
    for i, (pixel_row, label_row) in enumerate(zip(pixels.itertuples(), labels.itertuples(), strict=True)):
        emotion = majority_label(np.array([getattr(label_row, c) for c in VOTE_COLUMNS], dtype=float))
        subset = SUBSETS[pixel_row.Usage]
        if emotion is None:
            stats[(subset, "removed")] = stats.get((subset, "removed"), 0) + 1
            continue

        image = np.array(pixel_row.pixels.split(), dtype=np.uint8).reshape(48, 48)
        directory = OUTPUT_DIR / subset / emotion
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / f"{i:05d}.png"), image)
        stats[(subset, emotion)] = stats.get((subset, emotion), 0) + 1

    for subset in SUBSETS.values():
        counts = {e: n for (s, e), n in sorted(stats.items()) if s == subset}
        print(f"{subset}: {counts}")


if __name__ == "__main__":
    main()
