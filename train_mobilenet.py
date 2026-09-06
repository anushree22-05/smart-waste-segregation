import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

MODEL_PATH = "waste_mobilenetv2_best.keras"

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n======================================")
print("LOADING WASTE DATASET")
print("======================================\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i} = {name}")

print(f"\nNumber of classes: {num_classes}")

# ============================================================
# 3. PERFORMANCE SETTINGS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================================
# 5. MOBILENETV2 BASE MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# First phase: freeze MobileNetV2
base_model.trainable = False

# ============================================================
# 6. BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# IMPORTANT:
# MobileNetV2 preprocessing
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 8. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# 9. INITIAL TRAINING
# ============================================================

print("\n======================================")
print("INITIAL MOBILE NET V2 TRAINING")
print("======================================\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# ============================================================
# 10. FINE-TUNING
# ============================================================

print("\n======================================")
print("STARTING FINE-TUNING")
print("======================================\n")

base_model.trainable = True

# Freeze most layers and fine-tune the final part
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history1.history["accuracy"]),
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n======================================")
print("MODEL TRAINING COMPLETED")
print("======================================")

print(f"\nModel saved as:")
print(MODEL_PATH)

# ============================================================
# 12. COMBINE TRAINING HISTORY
# ============================================================

accuracy = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    + history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

epochs_range = range(1, len(accuracy) + 1)

# ============================================================
# 13. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    "mobilenet_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 14. LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("CREATING CONFUSION MATRIX")
print("======================================\n")

y_true = []
y_pred = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(9, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="viridis",
    xticks_rotation=45
)

plt.title("MobileNetV2 Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ============================================================
# 17. FINAL ACCURACY
# ============================================================

print("\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    f"Best Training Accuracy: "
    f"{max(accuracy) * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{max(val_accuracy) * 100:.2f}%"
)

print("\nClasses used:")

for i, name in enumerate(class_names):
    print(f"{i} = {name}")

print("\n======================================")
print("DONE")
print("======================================")