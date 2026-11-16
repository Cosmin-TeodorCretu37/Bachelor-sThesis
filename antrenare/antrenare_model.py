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

from src.config import DIR_ANTRENARE, DIR_MODELE, MAPARE_GESTURI, MODEL_GESTURI

DIR_DATE = DIR_ANTRENARE / "date_gesturi"

#incarcare date
GESTURI = ["pumn_inchis", "palma_deschisa", "unu", "doi", "trei", "ok"]

date = []
etichete = []

for gest in GESTURI:
    df = pd.read_csv(DIR_DATE / f"{gest}.csv", header=None)
    date.extend(df.values)
    etichete.extend([gest] * len(df))

date = np.array(date, dtype=np.float32)
etichete = np.array(etichete)


def adauga_oglindite(x, y):
    x_oglindit = x.copy()
    x_oglindit[:, 0::3] = -x_oglindit[:, 0::3]
    return np.concatenate([x, x_oglindit]), np.concatenate([y, y])


#codificare etichete(text -> numere)
encoder = LabelEncoder()
etichete_numerice = encoder.fit_transform(etichete)

print("\nMapare gesturi -> numere:")
for i, gest in enumerate(encoder.classes_):
    print(f"{i} = {gest}")

#impartire date  antrenare80/test20
X_train, X_test, y_train, y_test = train_test_split(
    date, etichete_numerice,
    test_size=0.2,
    random_state=42,
    stratify=etichete_numerice
)

X_train, y_train = shuffle(*adauga_oglindite(X_train, y_train), random_state=42)
X_test, y_test = adauga_oglindite(X_test, y_test)

y_train = keras.utils.to_categorical(y_train, num_classes=len(GESTURI))
y_test = keras.utils.to_categorical(y_test, num_classes=len(GESTURI))

print(f"\nDate antrenare: {X_train.shape[0]} exemple")
print(f"Date testare: {X_test.shape[0]} exemple")

#arhitectura retelei - MLP
model = keras.Sequential([
    #strat intrare
    keras.layers.Input(shape=(63,)),

    #strat1: 128 neuroni
    #relu = functia de activare (introduce neliniaritate)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),

    #strat2: 64 neuroni
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),

    #strat3: 32 neuroni
    keras.layers.Dense(32, activation='relu'),

    #strat iesire: 6 neuroni
    #softmax transforma valorile in probabilitati (suma = 1)
    keras.layers.Dense(len(GESTURI), activation='softmax')
])

model.summary()

model.compile(
    #adam = algoritmul de optimizare (ajusteaza ponderile)
    optimizer='adam',
    #categorical crossentropy = functia de pierdere pentru clasificare multipla
    loss='categorical_crossentropy',
    #accuracy = metrica pe care o urmarim
    metrics=['accuracy']
)

print("\nIncepe antrenarea...")

history = model.fit(
    X_train, y_train,
    epochs=50,           #numarul de treceri prin tot datasetul
    batch_size=32,       #cate exemple proceseaza odata
    validation_split=0.1, #10% din datele de antrenare pentru validare
    verbose=1
)

#evaluare
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcuratete pe datele de test: {accuracy * 100:.1f}%")

#save model + mapare
DIR_MODELE.mkdir(exist_ok=True)
model.save(MODEL_GESTURI)

mapare = {gest: int(i) for i, gest in enumerate(encoder.classes_)}
with open(MAPARE_GESTURI, "w") as f:
    json.dump(mapare, f, indent=2)

print(f"\nModel si mapare salvate in '{DIR_MODELE}'")

#grafic acuratete
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Antrenare')
plt.plot(history.history['val_accuracy'], label='Validare')
plt.title('Acuratete')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Antrenare')
plt.plot(history.history['val_loss'], label='Validare')
plt.title('Pierdere (Loss)')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(DIR_MODELE / "grafic_antrenare.png")
plt.show()
print(f"Grafic salvat in '{DIR_MODELE / 'grafic_antrenare.png'}'")
