import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


class_names = train_dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i} = {name}")


# ============================================================
# 3. IMPROVE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    ),

    layers.RandomContrast(
        0.1
    )

])


# ============================================================
# 5. LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers initially

base_model.trainable = False


# ============================================================
# 6. BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.4
)(x)


outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


print("\nMobileNetV2 model created.")

model.summary()


# ============================================================
# 8. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001
    )

]


# ============================================================
# 9. INITIAL TRAINING
# ============================================================

print("\n==========================================")
print("STARTING INITIAL TRAINING")
print("==========================================")

history1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks

)


# ============================================================
# 10. FINE-TUNING
# ============================================================

print("\n==========================================")
print("STARTING FINE-TUNING")
print("==========================================")


# Unfreeze MobileNetV2

base_model.trainable = True


# Freeze most of the early layers
# and train only the last 30 layers

for layer in base_model.layers[:-30]:

    layer.trainable = False


# Recompile with smaller learning rate

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


history2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks

)


# ============================================================
# 11. COMBINE TRAINING HISTORY
# ============================================================

accuracy = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model.save(
    "waste_classifier.keras"
)

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "waste_classifier.keras"
)


# ============================================================
# 13. ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.title(
    "MobileNetV2 Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid()

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title(
    "MobileNetV2 Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

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

print("\nCreating confusion matrix...")

true_labels = []

predicted_labels = []


for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# ============================================================
# 16. DISPLAY CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. FINAL EVALUATION
# ============================================================

print("\nEvaluating model...")

final_loss, final_accuracy = model.evaluate(
    validation_dataset
)


print("\n==========================================")
print("FINAL MOBILE NET V2 RESULTS")
print("==========================================")

print(
    f"Validation Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: "
    f"{final_loss:.4f}"
)

print("==========================================")