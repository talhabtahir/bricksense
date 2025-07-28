# Brick Property Prediction App (App 1)
import streamlit as st
import tensorflow as tf
from PIL import Image, ExifTags
import numpy as np
import cv2

st.title("🔍 Brick Property Predictor")

# Load models
@st.cache_resource
def load_models():
    strength_model = tf.lite.Interpreter(model_path='brick_FlexureStrength_Reg_model_epoch50.tflite')
    strength_model.allocate_tensors()

    class_model = tf.lite.Interpreter(model_path='brick_classification_model trial 2.tflite')
    class_model.allocate_tensors()

    absorption_model = tf.lite.Interpreter(model_path='brick_absorption_Model.tflite')
    absorption_model.allocate_tensors()

    return strength_model, class_model, absorption_model

def run_tflite_inference(interpreter, inputs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_tensor in enumerate(inputs):
        interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))

    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except:
        pass
    return image

strength_model, class_model, absorption_model = load_models()

file = st.file_uploader("Upload an image of the individual brick", type=["jpg", "jpeg", "png"])
dry_weight_grams = st.number_input("Enter dry weight of brick (in grams):", min_value=1500.0, max_value=3500.0)

# Normalize dry weight for model
min_val, max_val = 2610, 3144
norm_dry_weight = (dry_weight_grams - min_val) / (max_val - min_val)
MIN_KN, MAX_KN = 2.13, 12.48

if file:
    image = Image.open(file).convert('RGB')
    image = correct_orientation(image)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(image_bgr, (224, 224)) / 255.0
    img_tensor = np.expand_dims(resized_image.astype(np.float32), axis=0)

    class_pred = run_tflite_inference(class_model, [img_tensor])
    class_label = np.argmax(class_pred[0])

    if class_label == 3:
        st.image(image, caption="Uploaded Image")
        st.error("🚫 This is not a brick. Please upload a valid brick image.")
    else:
        label_1 = 1 if class_label == 0 else 0
        label_2 = 1 if class_label == 1 else 0
        label_3 = 1 if class_label == 2 else 0

        tabular_input = np.array([[label_1, label_2, label_3, norm_dry_weight]], dtype=np.float32)
        strength_pred = run_tflite_inference(strength_model, [img_tensor, tabular_input])
        strength_val = float(strength_pred[0][0]) * (MAX_KN - MIN_KN) + MIN_KN

        absorption_input = np.array([[dry_weight_grams, class_label + 1]], dtype=np.float32)
        absorption_pred = run_tflite_inference(absorption_model, [img_tensor, absorption_input])
        absorption_val = float(absorption_pred[0][0])

        st.image(image, caption=f"Brick Image (Predicted Class: {['1st', '2nd', '3rd'][class_label]})", use_container_width=True)
        st.success(f"🧪 Estimated Flexural Strength: **{strength_val:.2f} kN**")
        st.success(f"💧 Estimated Absorption: **{absorption_val:.2f}%**")

        st.subheader("📊 Classification Probabilities")
        st.write("""
        - **1st Class Brick:** {:.2f}%  
        - **2nd Class Brick:** {:.2f}%  
        - **3rd Class Brick:** {:.2f}%  
        - **Not a Brick:** {:.2f}%
        """.format(*[x * 100 for x in class_pred[0]]))
