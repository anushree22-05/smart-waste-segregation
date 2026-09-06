import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "waste_classifier.keras"

class_names = [
    "battery",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

except Exception as e:
    model = None
    print("Model loading error:", e)


# ============================================================
# GLOBAL CAMERA VARIABLES
# ============================================================

camera = None
camera_running = False


# ============================================================
# WINDOW
# ============================================================

root = tk.Tk()

root.title("AI Smart Waste Segregation")
root.geometry("1100x750")
root.configure(bg="#eef2f3")

root.resizable(False, False)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg="#173f5f",
    height=100
)

header.pack(fill="x")
header.pack_propagate(False)

title = tk.Label(
    header,
    text="♻ AI SMART WASTE SEGREGATION",
    font=("Arial", 26, "bold"),
    bg="#173f5f",
    fg="white"
)

title.pack(pady=(20, 2))

subtitle = tk.Label(
    header,
    text="MobileNetV2 Based Waste Classification System",
    font=("Arial", 12),
    bg="#173f5f",
    fg="white"
)

subtitle.pack()


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg="#eef2f3"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=20
)


# ============================================================
# LEFT FRAME
# ============================================================

left_frame = tk.Frame(
    main_frame,
    bg="white",
    width=520,
    height=520,
    relief="solid",
    borderwidth=1
)

left_frame.pack(
    side="left",
    fill="y",
    padx=(0, 15)
)

left_frame.pack_propagate(False)


image_title = tk.Label(
    left_frame,
    text="Waste Image / Camera",
    font=("Arial", 18, "bold"),
    bg="white"
)

image_title.pack(pady=15)


# ============================================================
# IMAGE AREA
# ============================================================

image_box = tk.Frame(
    left_frame,
    bg="#f7f7f7",
    width=460,
    height=350,
    relief="solid",
    borderwidth=1
)

image_box.pack(pady=5)

image_box.pack_propagate(False)


image_label = tk.Label(
    image_box,
    text="No image selected\n\nUpload an image or use the camera",
    font=("Arial", 15),
    bg="#f7f7f7",
    fg="#777777",
    justify="center"
)

image_label.pack(
    expand=True
)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    left_frame,
    bg="white"
)

button_frame.pack(
    pady=15
)


# ============================================================
# UPLOAD BUTTON
# ============================================================

upload_button = tk.Button(
    button_frame,
    text="📤 Upload Image",
    font=("Arial", 12, "bold"),
    bg="#2a9d8f",
    fg="white",
    padx=15,
    pady=9,
    relief="flat",
    cursor="hand2",
    command=lambda: upload_image()
)

upload_button.grid(
    row=0,
    column=0,
    padx=5
)


# ============================================================
# CAMERA BUTTON
# ============================================================

camera_button = tk.Button(
    button_frame,
    text="📷 Camera",
    font=("Arial", 12, "bold"),
    bg="#457b9d",
    fg="white",
    padx=15,
    pady=9,
    relief="flat",
    cursor="hand2",
    command=lambda: start_camera()
)

camera_button.grid(
    row=0,
    column=1,
    padx=5
)


# ============================================================
# CAPTURE BUTTON
# ============================================================

capture_button = tk.Button(
    button_frame,
    text="📸 Capture",
    font=("Arial", 12, "bold"),
    bg="#e76f51",
    fg="white",
    padx=15,
    pady=9,
    relief="flat",
    cursor="hand2",
    command=lambda: capture_image()
)

capture_button.grid(
    row=0,
    column=2,
    padx=5
)


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_button = tk.Button(
    button_frame,
    text="🔄 Clear",
    font=("Arial", 12, "bold"),
    bg="#777777",
    fg="white",
    padx=15,
    pady=9,
    relief="flat",
    cursor="hand2",
    command=lambda: clear_image()
)

clear_button.grid(
    row=0,
    column=3,
    padx=5
)


# ============================================================
# CAMERA STATUS
# ============================================================

camera_status = tk.Label(
    left_frame,
    text="Camera: OFF",
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#777777"
)

camera_status.pack(pady=2)


# ============================================================
# RIGHT FRAME
# ============================================================

right_frame = tk.Frame(
    main_frame,
    bg="white",
    width=500,
    height=520,
    relief="solid",
    borderwidth=1
)

right_frame.pack(
    side="right",
    fill="y"
)

right_frame.pack_propagate(False)


result_title = tk.Label(
    right_frame,
    text="Classification Result",
    font=("Arial", 18, "bold"),
    bg="white"
)

result_title.pack(
    pady=(15, 10)
)


# ============================================================
# PREDICTION
# ============================================================

prediction_label = tk.Label(
    right_frame,
    text="Prediction: ---",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#173f5f"
)

prediction_label.pack(pady=5)


confidence_label = tk.Label(
    right_frame,
    text="Confidence: ---",
    font=("Arial", 17, "bold"),
    bg="white"
)

confidence_label.pack(pady=5)


status_label = tk.Label(
    right_frame,
    text="",
    font=("Arial", 13, "bold"),
    bg="white"
)

status_label.pack(pady=5)


# ============================================================
# SEPARATOR
# ============================================================

separator = tk.Frame(
    right_frame,
    bg="#dddddd",
    height=2
)

separator.pack(
    fill="x",
    padx=25,
    pady=10
)


# ============================================================
# PROBABILITY TITLE
# ============================================================

prob_title = tk.Label(
    right_frame,
    text="Prediction Probabilities",
    font=("Arial", 15, "bold"),
    bg="white"
)

prob_title.pack(pady=5)


# ============================================================
# PROBABILITY BARS
# ============================================================

bar_frame = tk.Frame(
    right_frame,
    bg="white"
)

bar_frame.pack(
    padx=25,
    pady=5,
    fill="x"
)

progress_bars = {}
percentage_labels = {}


for waste_class in class_names:

    row = tk.Frame(
        bar_frame,
        bg="white"
    )

    row.pack(
        fill="x",
        pady=4
    )

    name_label = tk.Label(
        row,
        text=waste_class.capitalize(),
        font=("Arial", 11),
        bg="white",
        width=10,
        anchor="w"
    )

    name_label.pack(side="left")

    progress = tk.Canvas(
        row,
        width=250,
        height=18,
        bg="#e6e6e6",
        highlightthickness=0
    )

    progress.pack(
        side="left",
        padx=5
    )

    percentage = tk.Label(
        row,
        text="0.00%",
        font=("Arial", 10, "bold"),
        bg="white",
        width=8
    )

    percentage.pack(side="left")

    progress_bars[waste_class] = progress
    percentage_labels[waste_class] = percentage


# ============================================================
# CATEGORY INFORMATION
# ============================================================

info_label = tk.Label(
    right_frame,
    text="",
    font=("Arial", 11),
    bg="#f5f5f5",
    fg="#333333",
    wraplength=430,
    justify="center",
    padx=10,
    pady=8
)

info_label.pack(
    fill="x",
    padx=25,
    pady=10
)


category_info = {

    "battery":
        "Battery waste should be collected separately and handled safely.",

    "glass":
        "Glass waste can be collected for reuse or recycling.",

    "metal":
        "Metal waste is recyclable and should be separated from other waste.",

    "organic":
        "Organic waste can be composted or processed into useful materials.",

    "paper":
        "Paper waste can generally be recycled.",

    "plastic":
        "Plastic waste should be separated and sent for appropriate recycling."
}


# ============================================================
# UPDATE PROGRESS BAR
# ============================================================

def update_bar(canvas, value):

    canvas.delete("all")

    width = 250
    height = 18

    filled_width = int(
        width * value / 100
    )

    canvas.create_rectangle(
        0,
        0,
        filled_width,
        height,
        fill="#2a9d8f",
        outline=""
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(predictions):

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )


    prediction_label.config(
        text=f"Prediction: {predicted_class.upper()}"
    )

    confidence_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )


    # --------------------------------------------------------
    # CONFIDENCE STATUS
    # --------------------------------------------------------

    if confidence >= 80:

        status_label.config(
            text="✓ HIGH CONFIDENCE",
            fg="#218838"
        )

    elif confidence >= 60:

        status_label.config(
            text="⚠ MEDIUM CONFIDENCE",
            fg="#e09f00"
        )

    else:

        status_label.config(
            text="⚠ LOW CONFIDENCE - TRY A CLEARER IMAGE",
            fg="#d62828"
        )


    info_label.config(
        text=category_info[
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    for i, waste_class in enumerate(class_names):

        probability = (
            predictions[i] * 100
        )

        percentage_labels[
            waste_class
        ].config(
            text=f"{probability:.2f}%"
        )

        update_bar(
            progress_bars[waste_class],
            probability
        )


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image):

    if model is None:

        messagebox.showerror(
            "Model Error",
            "waste_classifier.keras could not be loaded."
        )

        return


    try:

        # Convert OpenCV image to RGB
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image_rgb
        )


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        display_image = pil_image.copy()

        display_image.thumbnail(
            (440, 330)
        )

        photo = ImageTk.PhotoImage(
            display_image
        )

        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        model_image = pil_image.resize(
            (224, 224)
        )


        image_array = np.array(
            model_image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # MOBILENETV2 PREPROCESSING
        # ----------------------------------------------------

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        display_prediction(
            predictions
        )


    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    stop_camera()

    file_path = filedialog.askopenfilename(

        title="Select a Waste Image",

        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )

    if file_path:

        try:

            image = cv2.imread(
                file_path
            )

            if image is None:

                messagebox.showerror(
                    "Error",
                    "Could not open the image."
                )

                return

            predict_image(
                image
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global camera
    global camera_running


    if camera_running:

        return


    camera = cv2.VideoCapture(0)


    if not camera.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Could not open the webcam.\n\n"
            "Check that your camera is connected "
            "and not being used by another application."
        )

        camera = None

        return


    camera_running = True

    camera_status.config(
        text="Camera: ON",
        fg="#218838"
    )


    update_camera()


# ============================================================
# UPDATE CAMERA
# ============================================================

def update_camera():

    global camera_running


    if not camera_running:

        return


    ret, frame = camera.read()


    if ret:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        camera_image = Image.fromarray(
            frame_rgb
        )

        camera_image.thumbnail(
            (440, 330)
        )

        photo = ImageTk.PhotoImage(
            camera_image
        )

        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo


    root.after(
        30,
        update_camera
    )


# ============================================================
# CAPTURE CAMERA IMAGE
# ============================================================

def capture_image():

    global camera


    if not camera_running or camera is None:

        messagebox.showwarning(
            "Camera",
            "Please click the Camera button first."
        )

        return


    ret, frame = camera.read()


    if not ret:

        messagebox.showerror(
            "Camera Error",
            "Could not capture image."
        )

        return


    # Stop live camera
    stop_camera()


    # Predict captured frame
    predict_image(
        frame
    )


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global camera
    global camera_running


    camera_running = False


    if camera is not None:

        camera.release()

        camera = None


    camera_status.config(
        text="Camera: OFF",
        fg="#777777"
    )


# ============================================================
# CLEAR
# ============================================================

def clear_image():

    stop_camera()


    image_label.config(
        image="",
        text="No image selected\n\n"
             "Upload an image or use the camera"
    )

    image_label.image = None


    prediction_label.config(
        text="Prediction: ---"
    )

    confidence_label.config(
        text="Confidence: ---"
    )

    status_label.config(
        text=""
    )

    info_label.config(
        text=""
    )


    for waste_class in class_names:

        percentage_labels[
            waste_class
        ].config(
            text="0.00%"
        )

        update_bar(
            progress_bars[
                waste_class
            ],
            0
        )


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="Powered by MobileNetV2 • AI-Based Waste Classification",
    font=("Arial", 10),
    bg="#eef2f3",
    fg="#666666"
)

footer.pack(
    pady=(0, 10)
)


# ============================================================
# CLOSE CAMERA WHEN WINDOW CLOSES
# ============================================================

def on_closing():

    stop_camera()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    on_closing
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()