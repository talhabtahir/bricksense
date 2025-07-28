import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
import math
import matplotlib.cm as cm
from streamlit_image_comparison import image_comparison


# Helper to load and run TFLite model
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, inputs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_tensor in enumerate(inputs):
        interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Load models
@st.cache_resource
def load_models():
    strength_interpreter = load_tflite_model('brick_FlexureStrength_Reg_model_epoch50.tflite')
    # class_interpreter = load_tflite_model('brick_classification_model.tflite')
    class_interpreter = load_tflite_model('brick_classification_model trial 2.tflite')
    absorption_interpreter = load_tflite_model('brick_absorption_Model.tflite')
    return strength_interpreter, class_interpreter, absorption_interpreter
def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image
strength_model, class_model, absorption_model = load_models()

file = st.file_uploader("Upload an image of the individual brick", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))
dry_weight_grams = st.number_input("Enter dry weight of brick (in grams):", min_value=1500.0, max_value=3500.0, step=1.0)

# Normalize dry weight
min_val= 2610 # typical min dry weight in grams
max_val= 3144 # typical max dry weight in grams
dry_weight = (dry_weight_grams-min_val) / (max_val-min_val)

# Define denormalization range
MIN_KN = 2.13
MAX_KN = 12.48

if file:
    try:
          # Read and convert image to BGR (to match training setup)
        image = Image.open(file).convert('RGB')  # PIL loads as RGB
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
        resized_image = cv2.resize(image_bgr, (224, 224)) / 255.0
        img_tensor = np.expand_dims(resized_image.astype(np.float32), axis=0)

        class_pred = run_tflite_inference(class_model, [img_tensor])
        class_label = np.argmax(class_pred[0])

        # st.subheader("🧪 Debug Info")
        # st.write("🔢 Raw classification prediction vector:", class_pred)
        # st.write("🔍 Predicted class index (argmax):", class_label)
        image = correct_orientation(file) # correction of orientation for images in taken from mobile
        if class_label == 3:
            st.image(image, caption="Uploaded Image")
            st.error("🚫 This is not a brick. Please upload a valid brick image.")
        else:
            label_1 = 1 if class_label == 0 else 0
            label_2 = 1 if class_label == 1 else 0
            label_3 = 1 if class_label == 2 else 0

            tabular_input = np.array([[label_1, label_2, label_3, dry_weight]], dtype=np.float32)

            strength_pred = run_tflite_inference(strength_model, [img_tensor, tabular_input])
            strength_norm = float(strength_pred[0][0])
            strength_denorm = strength_norm * (MAX_KN - MIN_KN) + MIN_KN
            # --- Absorption Prediction (Real Output) ---
            try:
               # Create single combined tabular input: [[dry_weight, class_label]]
                tabular_input = np.array([[dry_weight_grams, class_label + 1]], dtype=np.float32)  # shape (1, 2)
            
                            
                # Run inference
                absorption_pred = run_tflite_inference(absorption_model, [img_tensor, tabular_input])
                absorption_real = float(absorption_pred[0][0])  # already in %
                        
            except Exception as e:
                st.error(f"Error in absorption prediction: {e}")

            st.image(image, caption=f"Uploaded Brick (Predicted Class: {['1st', '2nd', '3rd'][class_label]})", use_container_width=True)

            # st.success(f"🧪 Normalized Flexural Strength: **{strength_norm:.3f}** (0–1 scale)")
            st.success(f"🧪 Estimated Real Flexural Strength: **{strength_denorm:.2f} kN**")
            st.success(f"💧 Estimated Absorption: **{absorption_real :.2f}%**")
        st.subheader("📊 Classification Probabilities")
        st.write("""
        - **1st Class Brick:** {:.2f}%
        - **2nd Class Brick:** {:.2f}%
        - **3rd Class Brick:** {:.2f}%
        - **Not a Brick:** {:.2f}%
        """.format(class_pred[0][0] * 100, class_pred[0][1] * 100, class_pred[0][2] * 100, class_pred[0][3] * 100))

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
    Developed with Streamlit & TensorFlow Lite | © 2024 BrickSense
</div>
""", unsafe_allow_html=True)
