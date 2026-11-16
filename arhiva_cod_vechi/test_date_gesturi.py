import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

DIR_PROIECT = Path(__file__).resolve().parent.parent
DIR_MODELE = DIR_PROIECT / "modele"
DIR_ANTRENARE = DIR_PROIECT / "antrenare"

print("Se incarca modelul si datele...")

#incarcare+mapare gesturi
model = tf.keras.models.load_model(DIR_MODELE / "model_gesturi.h5")
with open(DIR_MODELE / "mapare_gesturi.json") as f:
    mapare = json.load(f)

#sortare gesturi
idx_to_gest = {v: k for k, v in mapare.items()}
GESTURI = [idx_to_gest[i] for i in range(len(mapare))]

#incarcare date
date, etichete = [], []
for gest in GESTURI:
    try:
        df = pd.read_csv(DIR_ANTRENARE / "date_gesturi" / f"{gest}.csv", header=None)
        date.extend(df.values)
        etichete.extend([mapare[gest]] * len(df))
    except FileNotFoundError:
        print(f"Atentie: Fisierul pentru {gest} nu a fost gasit.")

date = np.array(date, dtype=np.float32)
etichete_adevarate = np.array(etichete)

print("Se genereaza predictiile...")
predictii = np.argmax(model.predict(date, verbose=0), axis=1)

#generare+salvare matrice de confuzie
cm = confusion_matrix(etichete_adevarate, predictii)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[g.replace('_', ' ').upper() for g in GESTURI],
            yticklabels=[g.replace('_', ' ').upper() for g in GESTURI],
            annot_kws={"size": 14}) # Cifre mai mari in interiorul casutelor

plt.title('Matrice de Confuzie - Detectie Gesturi', fontsize=18, pad=20)
plt.xlabel('Predictia Modelului ', fontsize=14, labelpad=15)
plt.ylabel('Gestul Real (Ce era de fapt)', fontsize=14, labelpad=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

#salvare grafic
cale_imagine = DIR_ANTRENARE / "matrice_confuzie_gesturi.png"
plt.savefig(cale_imagine, dpi=300)
print(f"\n=> SUCCES: Imaginea a fost salvata in '{cale_imagine}'")

#generare raport (de copiat in teza)
print('RAPORT DE PERFORMANTA PENTRU CAPITOLUL 4 (LICENTA)')
raport = classification_report(etichete_adevarate, predictii, target_names=[g.upper() for g in GESTURI])
print(raport)
