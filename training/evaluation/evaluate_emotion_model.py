import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import EMOTION_LABELS, EMOTION_MODEL, TRAINING_DIR

print("Loading the model and the data...")

#model and labels loading
model = tf.keras.models.load_model(EMOTION_MODEL, compile=False)
with open(EMOTION_LABELS) as f:
    label_map = json.load(f)

index_to_emotion = {v: k for k, v in label_map.items()}
EMOTIONS = [index_to_emotion[i] for i in range(len(label_map))]

#FER+ test data loading
IMG_SIZE = 48

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TRAINING_DIR / "ferplus" / "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print("Generating predictions...")
probabilities = model.predict(test_generator, verbose=1)
predictions = np.argmax(probabilities, axis=1)
true_labels = test_generator.classes

#confusion matrix
cm = confusion_matrix(true_labels, predictions)
display_labels = [e.upper() for e in EMOTIONS]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=display_labels,
            yticklabels=display_labels, annot_kws={"size": 12})
plt.title('Confusion Matrix - Emotion Detection', fontsize=16, pad=15)
plt.xlabel('Model Prediction', fontsize=13, labelpad=12)
plt.ylabel('True Emotion', fontsize=13, labelpad=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
image_path = TRAINING_DIR / "emotion_confusion_matrix.png"
plt.savefig(image_path, dpi=300)
plt.show()
print(f"Saved: {image_path}")

#detailed per-class report
print('\nEMOTION MODEL PERFORMANCE REPORT:')
print(classification_report(true_labels, predictions, target_names=display_labels))
