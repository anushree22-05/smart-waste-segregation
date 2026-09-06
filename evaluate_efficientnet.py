import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"
MODEL_PATH = "waste_efficientnet_best.keras"

CLASS_NAMES = [
    "battery",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 20% images from EVERY class
VALIDATION_PERCENT = 0.20

# Fixed seed so the same images are selected every time
RANDOM_SEED = 42

# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking dataset...")

for class_name in CLASS_NAMES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Class folder not found: {folder}"
        )

    print(f"{class_name}: folder found")

# ============================================================
# COLLECT IMAGES FROM EACH CLASS
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

random.seed(RANDOM_SEED)

all_image_paths = []
all_labels = []

print("\nSelecting 20% images from EACH class...\n")

for label, class_name in enumerate(CLASS_NAMES):

    folder = os.path.join(DATASET_DIR, class_name)

    images = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(image_extensions):
            images.append(
                os.path.join(folder, filename)
            )

    if len(images) == 0:
        raise ValueError(
            f"No images found in {folder}"
        )

    # Shuffle images inside this class
    random.shuffle(images)

    # Select 20% from this class
    num_validation = max(
        1,
        int(len(images) * VALIDATION_PERCENT)
    )

    selected_images = images[:num_validation]

    print(
        f"{class_name:10s} -> "
        f"{len(images)} total, "
        f"{len(selected_images)} selected"
    )

    all_image_paths.extend(selected_images)

    all_labels.extend(
        [label] * len(selected_images)
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD IMAGES
# ============================================================

def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image).astype("float32")

    return image


print("\nLoading validation images...")

X = np.array([
    load_image(path)
    for path in all_image_paths
])

y_true = np.array(all_labels)

print("Images loaded:", len(X))
print("Image shape:", X.shape)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

predictions = model.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

print("\n======================================")
print("6 x 6 CONFUSION MATRIX")
print("======================================\n")

print(cm)

# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("EfficientNetB0 - 6 Class Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "efficientnet_confusion_matrix_6x6.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved as:"
    "\nefficientnet_confusion_matrix_6x6.png"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================\n")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

with open(
    "efficientnet_classification_report.txt",
    "w"
) as f:

    f.write(report)

print(
    "Classification report saved as:"
    "\nefficientnet_classification_report.txt"
)

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n======================================")
print("PER-CLASS ACCURACY")
print("======================================\n")

for i, class_name in enumerate(CLASS_NAMES):

    total = cm[i].sum()

    if total > 0:
        accuracy = cm[i, i] / total * 100
    else:
        accuracy = 0

    print(
        f"{class_name:10s}: "
        f"{accuracy:.2f}% "
        f"({cm[i, i]}/{total})"
    )

print("\n======================================")
print("EVALUATION COMPLETE")
print("======================================")