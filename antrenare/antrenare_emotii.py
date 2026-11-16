import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras

DIR_PROIECT = Path(__file__).resolve().parent.parent
DIR_DATE = DIR_PROIECT / "antrenare" / "fer2013"
DIR_MODELE = DIR_PROIECT / "modele"

#parametri generali
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 50
PROCENT_VALIDARE = 0.1

#augmentare date antrenare
#generam variante ale imaginilor (rotite, oglindite, zoom)
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=PROCENT_VALIDARE
)

#datele de validare si test doar se normalizeaza
val_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=PROCENT_VALIDARE
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

#incarcare imagini
parametri_comuni = dict(
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  #fer2013 e alb-negru
    class_mode='categorical'
)

train_generator = train_datagen.flow_from_directory(
    DIR_DATE / "train", subset='training', **parametri_comuni
)
val_generator = val_datagen.flow_from_directory(
    DIR_DATE / "train", subset='validation', shuffle=False, **parametri_comuni
)
test_generator = test_datagen.flow_from_directory(
    DIR_DATE / "test", shuffle=False, **parametri_comuni
)

print(f"\nClase detectate: {train_generator.class_indices}")
print(f"Imagini antrenare: {train_generator.samples}")
print(f"Imagini validare: {val_generator.samples}")
print(f"Imagini test: {test_generator.samples}")

#arhitectura CNN
#spre deosebire de gesturi (unde aveam coordonate numerice),
#aici lucram cu imagini, deci folosim straturi Conv2D
model = keras.Sequential([
    keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    #bloc1: trasaturi simple (muchii, linii)
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(), #stabilizeaza antrenarea
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),  #reduce dimensiunea la jumatate
    keras.layers.Dropout(0.25),

    #bloc2:trasaturi complexe (ochi, gura, sprancene)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    #bloc3: trasaturi abstracte (combinatii de expresii)
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    #trecem de la 2D la 1D pentru clasificare
    keras.layers.Flatten(),

    #straturi dense finale
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    #strat iesire: 7 emotii
    keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#callbacks (mecanisme care controleaza antrenarea automat)
DIR_MODELE.mkdir(exist_ok=True)

#opreste antrenarea daca nu se imbunateste in 10 epoci
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

#injumatateste learning rate-ul daca modelul stagneaza
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

#salveaza automat cel mai bun model
checkpoint = keras.callbacks.ModelCheckpoint(
    str(DIR_MODELE / "model_emotii_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nIncepe antrenarea... (poate dura 10-20 minute)")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

#evaluare
loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nAcuratete finala pe test: {accuracy * 100:.1f}%")

#salvare mapare emotii
with open(DIR_MODELE / "mapare_emotii.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)

print(f"Model si mapare salvate in '{DIR_MODELE}'")

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
plt.savefig(DIR_MODELE / "grafic_antrenare_emotii.png")
plt.show()
print(f"Grafic salvat in '{DIR_MODELE / 'grafic_antrenare_emotii.png'}'")
