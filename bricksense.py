import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image, ImageOps
import numpy as np
import cv2
from keras.models import Model
import math
import matplotlib.cm as cm
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="Brick Analyzer",
    page_icon="static/brickicon8.png",
    layout="centered",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'Developed by BrickSense Team | © 2024'}
)

imagelogo = Image.open("static/BSbasicboxhightran1.png")
st.image(imagelogo, use_container_width=True, width=150)

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("static/BScirclehightran1.png", width=200)

st.sidebar.header("About This App")
st.sidebar.write("""This app uses AI models to:
Predict flexural strength of individual bricks

**Developed by:**  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com""")

st.header("🤪 Predict Flexural Strength of Brick")

@st.cache_resource
def load_strength_model():
    return tf.keras.models.load_model('brick_FlexureStrength_Reg_model_epoch50.keras')

@st.cache_resource
def load_class_model():
    return tf.keras.models.load_model('brick_classification_model.keras')

strength_model = load_strength_model()
class_model = load_class_model()

file = st.file_uploader("Upload an image of the individual brick", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))
dry_weight = st.number_input("Enter standardized dry weight of brick (0-1):", min_value=0.0, max_value=1.0, step=0.01)

# Define denormalization range
MIN_KN = 2.0  # Example min strength
MAX_KN = 14.0  # Example max strength

if file:
    try:
        image = Image.open(file).convert('RGB')
        resized_image = image.resize((224, 224))
        img_array = np.array(resized_image) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        class_pred = class_model.predict(img_tensor)
        class_label = np.argmax(class_pred[0])
        label_1 = 1 if class_label == 0 else 0
        label_2 = 1 if class_label == 1 else 0
        label_3 = 1 if class_label == 2 else 0

        tabular_input = np.array([[label_1, label_2, label_3, dry_weight]])
        prediction = strength_model.predict([img_tensor, tabular_input])
        strength_norm = float(prediction[0][0])
        strength_denorm = strength_norm * (MAX_KN - MIN_KN) + MIN_KN

        st.image(image, caption=f"Uploaded Brick (Predicted Class: {['1st', '2nd', '3rd'][class_label]})", use_container_width=True)

        st.success(f"🧪 Normalized Flexural Strength: **{strength_norm:.3f}** (0–1 scale)")
        st.success(f"🧪 Estimated Real Flexural Strength: **{strength_denorm:.2f} kN**")

        st.subheader("📊 Classification Probabilities")
        st.write("""
        - **1st Class Brick:** {:.2f}%
        - **2nd Class Brick:** {:.2f}%
        - **3rd Class Brick:** {:.2f}%
        """.format(class_pred[0][0] * 100, class_pred[0][1] * 100, class_pred[0][2] * 100))

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
    Developed with Streamlit & TensorFlow | © 2024 BrickSense
</div>
""", unsafe_allow_html=True)
