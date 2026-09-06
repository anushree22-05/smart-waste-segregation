import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Smart Waste Segregation",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "waste_mobilenetv2_best.keras"

class_names = [
    "battery",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model = None
    model_loaded = False
    st.error("Model could not be loaded.")
    st.code(str(e))


# ============================================================
# CATEGORY INFORMATION
# ============================================================

category_info = {

    "battery":
        "🔋 Battery waste should be collected separately and handled safely.",

    "glass":
        "🍾 Glass waste can be collected for reuse or recycling.",

    "metal":
        "🥫 Metal waste is recyclable and should be separated from other waste.",

    "organic":
        "🍃 Organic waste can be composted or processed into useful materials.",

    "paper":
        "📄 Paper waste can generally be recycled.",

    "plastic":
        "🧴 Plastic waste should be separated and sent for appropriate recycling."
}


# ============================================================
# HEADER
# ============================================================

st.title("♻️ AI SMART WASTE SEGREGATION")

st.markdown(
    "### MobileNetV2 Based Waste Classification System"
)

st.markdown("---")


# ============================================================
# IMAGE INPUT
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("📤 Upload Waste Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp"]
    )


with col2:

    st.subheader("📷 Use Camera")

    camera_file = st.camera_input(
        "Take a picture of waste"
    )


# ============================================================
# SELECT IMAGE
# ============================================================

image_file = uploaded_file

if camera_file is not None:
    image_file = camera_file


# ============================================================
# PREDICTION
# ============================================================

if image_file is not None and model_loaded:

    try:

        image = Image.open(image_file).convert("RGB")

        st.markdown("---")

        left, right = st.columns(2)


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        with left:

            st.subheader("🖼️ Waste Image")

            st.image(
                image,
                caption="Selected Waste Image",
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        model_image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            model_image,
            dtype=np.float32
        )


        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with right:

            st.subheader("🔍 Classification Result")

            st.success(
                f"Prediction: {predicted_class.upper()}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # Confidence status

            if confidence >= 80:

                st.success(
                    "✓ HIGH CONFIDENCE"
                )

            elif confidence >= 60:

                st.warning(
                    "⚠ MEDIUM CONFIDENCE"
                )

            else:

                st.error(
                    "⚠ LOW CONFIDENCE - TRY A CLEARER IMAGE"
                )


            # Category information

            st.info(
                category_info[predicted_class]
            )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Prediction Probabilities")


        for i, waste_class in enumerate(class_names):

            probability = float(
                predictions[i] * 100
            )

            st.write(
                f"**{waste_class.capitalize()}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                min(probability / 100, 1.0)
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader("♻️ Waste Management Recommendation")

        st.write(
            category_info[predicted_class]
        )


    except Exception as e:

        st.error(
            "Prediction error occurred."
        )

        st.code(str(e))


# ============================================================
# NO IMAGE
# ============================================================

elif image_file is None:

    st.info(
        "👆 Upload a waste image or use the camera to start classification."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by MobileNetV2 • AI-Based Waste Classification"
)
