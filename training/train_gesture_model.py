import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GESTURE_LABELS, GESTURE_MODEL, MODELS_DIR, TRAINING_DIR

DATA_DIR = TRAINING_DIR / "gesture_data"

#data loading
GESTURES = ["fist", "open_palm", "one", "two", "three", "ok"]

data = []
labels = []

for gesture in GESTURES:
    df = pd.read_csv(DATA_DIR / f"{gesture}.csv", header=None)
    data.extend(df.values)
    labels.extend([gesture] * len(df))

data = np.array(data, dtype=np.float32)
labels = np.array(labels)


def add_mirrored(x, y):
    x_mirrored = x.copy()
    x_mirrored[:, 0::3] = -x_mirrored[:, 0::3]
    return np.concatenate([x, x_mirrored]), np.concatenate([y, y])


#label encoding (text -> numbers)
encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(labels)

print("\nGesture -> number mapping:")
for i, gesture in enumerate(encoder.classes_):
    print(f"{i} = {gesture}")

#data split train80/test20
X_train, X_test, y_train, y_test = train_test_split(
    data, numeric_labels,
    test_size=0.2,
    random_state=42,
    stratify=numeric_labels
)

X_train, y_train = shuffle(*add_mirrored(X_train, y_train), random_state=42)
X_test, y_test = add_mirrored(X_test, y_test)

y_train = keras.utils.to_categorical(y_train, num_classes=len(GESTURES))
y_test = keras.utils.to_categorical(y_test, num_classes=len(GESTURES))

print(f"\nTraining data: {X_train.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")

#network architecture - MLP
model = keras.Sequential([
    #input layer
    keras.layers.Input(shape=(63,)),

    #layer1: 128 neurons
    #relu = activation function (adds non-linearity)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),

    #layer2: 64 neurons
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),

    #layer3: 32 neurons
    keras.layers.Dense(32, activation='relu'),

    #output layer: 6 neurons
    #softmax turns the values into probabilities (sum = 1)
    keras.layers.Dense(len(GESTURES), activation='softmax')
])

model.summary()

model.compile(
    #adam = optimization algorithm (adjusts the weights)
    optimizer='adam',
    #categorical crossentropy = loss function for multi-class classification
    loss='categorical_crossentropy',
    #accuracy = the metric we track
    metrics=['accuracy']
)

print("\nTraining started...")

history = model.fit(
    X_train, y_train,
    epochs=50,           #number of passes through the whole dataset
    batch_size=32,       #how many samples are processed at once
    validation_split=0.1, #10% of the training data for validation
    verbose=1
)

#evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy * 100:.1f}%")

#save model + labels
MODELS_DIR.mkdir(exist_ok=True)
model.save(GESTURE_MODEL)

label_map = {gesture: int(i) for i, gesture in enumerate(encoder.classes_)}
with open(GESTURE_LABELS, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\nModel and labels saved in '{MODELS_DIR}'")

#accuracy plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(MODELS_DIR / "gesture_training_history.png")
plt.show()
print(f"Plot saved in '{MODELS_DIR / 'gesture_training_history.png'}'")
