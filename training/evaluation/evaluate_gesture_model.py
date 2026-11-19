import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import GESTURE_LABELS, GESTURE_MODEL, TRAINING_DIR

TRAINING_ORDER = ["fist", "open_palm", "one", "two", "three", "ok"]

print("Loading the model and the data...")

#model + gesture labels loading
model = tf.keras.models.load_model(GESTURE_MODEL, compile=False)
with open(GESTURE_LABELS) as f:
    label_map = json.load(f)

#gesture sorting
index_to_gesture = {v: k for k, v in label_map.items()}
GESTURES = [index_to_gesture[i] for i in range(len(label_map))]

#data loading
data, labels = [], []
for gesture in TRAINING_ORDER:
    df = pd.read_csv(TRAINING_DIR / "gesture_data" / f"{gesture}.csv", header=None)
    data.extend(df.values)
    labels.extend([label_map[gesture]] * len(df))

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

_, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
X_mirrored = X_test.copy()
X_mirrored[:, 0::3] = -X_mirrored[:, 0::3]
data = np.concatenate([X_test, X_mirrored])
true_labels = np.concatenate([y_test, y_test])
print(f"Test samples: {len(true_labels)}")

print("Generating predictions...")
predictions = np.argmax(model.predict(data, verbose=0), axis=1)

#confusion matrix generation + saving
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[g.replace('_', ' ').upper() for g in GESTURES],
            yticklabels=[g.replace('_', ' ').upper() for g in GESTURES],
            annot_kws={"size": 14}) # bigger numbers inside the cells

plt.title('Confusion Matrix - Gesture Detection (test set)', fontsize=18, pad=20)
plt.xlabel('Model Prediction', fontsize=14, labelpad=15)
plt.ylabel('True Gesture', fontsize=14, labelpad=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

#plot saving
image_path = TRAINING_DIR / "gesture_confusion_matrix.png"
plt.savefig(image_path, dpi=300)
print(f"\n=> SUCCESS: the image was saved in '{image_path}'")

#report generation
print('GESTURE MODEL PERFORMANCE REPORT')
report = classification_report(true_labels, predictions, target_names=[g.upper() for g in GESTURES])
print(report)
