import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMOTION_LABELS, EMOTION_MODEL, MODELS_DIR, TRAINING_DIR

DATA_DIR = TRAINING_DIR / "ferplus"

#general parameters
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 50

#training data augmentation
#we generate variants of the images (rotated, mirrored, zoomed)
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

#validation and test data are only normalized
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

#image loading
common_params = dict(
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  #fer2013 is grayscale
    class_mode='categorical'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR / "train", **common_params
)
val_generator = test_datagen.flow_from_directory(
    DATA_DIR / "val", shuffle=False, **common_params
)
test_generator = test_datagen.flow_from_directory(
    DATA_DIR / "test", shuffle=False, **common_params
)

print(f"\nDetected classes: {train_generator.class_indices}")
print(f"Training images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Test images: {test_generator.samples}")

#CNN architecture
#unlike gestures (where we had numeric coordinates),
#here we work with images, so we use Conv2D layers
model = keras.Sequential([
    keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    #block1: simple features (edges, lines)
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(), #stabilizes training
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),  #halves the size
    keras.layers.Dropout(0.25),

    #block2: complex features (eyes, mouth, eyebrows)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    #block3: abstract features (expression combinations)
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    #from 2D to 1D for classification
    keras.layers.Flatten(),

    #final dense layers
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    #output layer: 7 emotions
    keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#callbacks (mechanisms that control the training automatically)
MODELS_DIR.mkdir(exist_ok=True)

#stops training if there is no improvement in 10 epochs
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

#halves the learning rate if the model stagnates
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

#automatically saves the best model
checkpoint = keras.callbacks.ModelCheckpoint(
    str(EMOTION_MODEL),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nTraining started... (it can take up to an hour on CPU)")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

#evaluation
loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nFinal test accuracy: {accuracy * 100:.1f}%")

#save emotion labels
with open(EMOTION_LABELS, "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)

print(f"Model and labels saved in '{MODELS_DIR}'")

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
plt.savefig(MODELS_DIR / "emotion_training_history.png")
plt.show()
print(f"Plot saved in '{MODELS_DIR / 'emotion_training_history.png'}'")
