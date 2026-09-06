import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model("waste_mobilenetv2_best.keras")


# ==========================================
# 2. CLASS NAMES
# ==========================================

class_names = [
    "battery",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]


# ==========================================
# 3. OPEN FILE SELECTION WINDOW
# ==========================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a waste image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()


# ==========================================
# 4. CHECK IF IMAGE WAS SELECTED
# ==========================================

if not image_path:
    print("No image selected.")
    exit()


print("\nSelected image:")
print(image_path)


# ==========================================
# 5. LOAD IMAGE
# ==========================================

image = Image.open(image_path).convert("RGB")

# Resize to the same size used during training
image = image.resize((224, 224))


# ==========================================
# 6. CONVERT IMAGE TO NUMPY ARRAY
# ==========================================

image = image.resize((224, 224))

image_array = np.array(image).astype("float32")

image_array = np.expand_dims(image_array, axis=0)

image_array = preprocess_input(image_array)

prediction = model.predict(image_array, verbose=0)



# ==========================================
# 7. MAKE PREDICTION
# ==========================================

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("\n======================================")
print("       WASTE CLASSIFICATION")
print("======================================")

print("Predicted waste :", predicted_class)
print(f"Confidence      : {confidence:.2f}%")

print("======================================")


# ==========================================
# 9. SHOW ALL CLASS PROBABILITIES
# ==========================================

print("\nPrediction probabilities:")

for i, class_name in enumerate(class_names):
    probability = predictions[0][i] * 100

    print(
        f"{class_name:10s} : {probability:.2f}%"
    )