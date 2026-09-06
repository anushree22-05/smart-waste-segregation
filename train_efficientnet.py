import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

MODEL_PATH = "waste_efficientnet_best.keras"


# ============================================================
# 2. CLASS NAMES
# ============================================================

class_names = [
    "battery",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

NUM_CLASSES = len(class_names)

print("\n==========================================")
print("   AI SMART WASTE SEGREGATION")
print("   EfficientNetB0 Training")
print("==========================================\n")

print("Classes:")

for i, name in enumerate(class_names):
    print(f"{i} = {name}")


# ============================================================
# 3. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

for class_name in class_names:

    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

print("Dataset folders found successfully.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=True
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

print("\nDataset loaded successfully.")

print("\nTraining batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(validation_ds).numpy())


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")


# ============================================================
# 6. PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)


# ============================================================
# 7. BUILD EFFICIENTNETB0
# ============================================================

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# EfficientNetB0 includes its own input preprocessing
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 10. INITIAL TRAINING
# ============================================================

print("\n==========================================")
print("INITIAL TRAINING")
print("==========================================\n")

history1 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 11. FINE-TUNING
# ============================================================

print("\n==========================================")
print("FINE-TUNING EFFICIENTNETB0")
print("==========================================\n")

base_model.trainable = True


# Freeze the first part of EfficientNet
fine_tune_from = 150

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


# Use a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 12. COMBINE TRAINING HISTORY
# ============================================================

train_accuracy = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

train_loss = (
    history1.history["loss"]
    + history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

epochs_range = range(1, len(train_accuracy) + 1)


# ============================================================
# 13. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("EfficientNetB0 Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "efficientnet_accuracy.png",
    dpi=300
)

plt.show()

print("\nAccuracy graph saved as:")
print("efficientnet_accuracy.png")


# ============================================================
# 14. LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.title("EfficientNetB0 Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "efficientnet_loss.png",
    dpi=300
)

plt.show()

print("\nLoss graph saved as:")
print("efficientnet_loss.png")


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully.")


# ============================================================
# 16. VALIDATION ACCURACY
# ============================================================

print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

loss, accuracy = best_model.evaluate(
    validation_ds,
    verbose=1
)

print("\nFinal Validation Accuracy:")
print(f"{accuracy * 100:.2f}%")

print(f"\nFinal Validation Loss:")
print(f"{loss:.4f}")


# ============================================================
# 17. PREDICTIONS FOR CONFUSION MATRIX
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in validation_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

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
    cmap="Blues",
    values_format="d",
    ax=plt.gca(),
    xticks_rotation=45
)

plt.title(
    "EfficientNetB0 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "efficientnet_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as:")
print("efficientnet_confusion_matrix.png")


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = "waste_efficientnet_final.keras"

best_model.save(
    FINAL_MODEL_PATH
)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("\nBest model:")
print(MODEL_PATH)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nGenerated files:")
print("1. efficientnet_accuracy.png")
print("2. efficientnet_loss.png")
print("3. efficientnet_confusion_matrix.png")
print("4. waste_efficientnet_best.keras")
print("5. waste_efficientnet_final.keras")

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i} = {name}")

print("\n==========================================")
print("DONE")
print("==========================================")