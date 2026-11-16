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

from src.config import DIR_ANTRENARE, MAPARE_EMOTII, MODEL_EMOTII

print("Se incarca modelul si datele...")

#incarcare model si mapare
model = tf.keras.models.load_model(MODEL_EMOTII, compile=False)
with open(MAPARE_EMOTII) as f:
    mapare = json.load(f)

idx_to_emotie = {v: k for k, v in mapare.items()}
EMOTII = [idx_to_emotie[i] for i in range(len(mapare))]

#incarcare date test din FER2013
IMG_SIZE = 48

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    DIR_ANTRENARE / "fer2013" / "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print("Se genereaza predictiile...")
probabilitati = model.predict(test_generator, verbose=1)
predictii = np.argmax(probabilitati, axis=1)
etichete_reale = test_generator.classes

#matrice de confuzie
cm = confusion_matrix(etichete_reale, predictii)
etichete_display = [e.upper() for e in EMOTII]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=etichete_display,
            yticklabels=etichete_display, annot_kws={"size": 12})
plt.title('Matrice de Confuzie - Detectie Emotii', fontsize=16, pad=15)
plt.xlabel('Predictia Modelului', fontsize=13, labelpad=12)
plt.ylabel('Emotia Reala', fontsize=13, labelpad=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
cale_imagine = DIR_ANTRENARE / "matrice_confuzie_emotii.png"
plt.savefig(cale_imagine, dpi=300)
plt.show()
print(f"Salvat: {cale_imagine}")

#raport detaliat per clasa
print('\nRAPORT PERFORMANTA EMOTII:')
print(classification_report(etichete_reale, predictii, target_names=etichete_display))
