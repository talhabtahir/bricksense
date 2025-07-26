import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
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
Group 24 Batch 213 and Talha Bin Tahir
**Email:** talhabtahir@gmail.com""")

# st.header("🤪 Predict Flexural Strength of Brick")

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
    return strength_interpreter, class_interpreter
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
strength_model, class_model = load_models()

file = st.file_uploader("Upload an image of the individual brick", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))
dry_weight_grams = st.number_input("Enter dry weight of brick (in grams):", min_value=100.0, max_value=5000.0, step=1.0)

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
        image = correct_orientation(file) # correction of orientation for images in taken from mobile
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

            st.image(image, caption=f"Uploaded Brick (Predicted Class: {['1st', '2nd', '3rd'][class_label]})", use_container_width=True)

            # st.success(f"🧪 Normalized Flexural Strength: **{strength_norm:.3f}** (0–1 scale)")
            st.success(f"🧪 Estimated Real Flexural Strength: **{strength_denorm:.2f} kN**")

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
#________________________________________________________________________________________________
#__________________________________________________________________________________________________
# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
# import math
# import matplotlib.cm as cm
# from streamlit_image_comparison import image_comparison

# st.set_page_config(
#     page_title="Brick Analyzer",
#     page_icon="static/brickicon8.png",
#     layout="centered",
#     menu_items={
#         'Get Help': 'https://example.com/help',
#         'Report a bug': 'https://example.com/bug',
#         'About': 'Developed by BrickSense Team | © 2024'}
# )

# imagelogo = Image.open("static/BSbasicboxhightran1.png")
# st.image(imagelogo, use_container_width=True, width=150)

# with st.sidebar:
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.image("static/BScirclehightran1.png", width=200)

# st.sidebar.header("About This App")
# st.sidebar.write("""This app uses AI models to:
# Predict flexural strength of individual bricks

# **Developed by:**  
# Talha Bin Tahir  
# **Email:** talhabtahir@gmail.com""")

# # st.header("🤪 Predict Flexural Strength of Brick")

# # Helper to load and run TFLite model
# def load_tflite_model(path):
#     interpreter = tf.lite.Interpreter(model_path=path)
#     interpreter.allocate_tensors()
#     return interpreter

# def run_tflite_inference(interpreter, inputs):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     for i, input_tensor in enumerate(inputs):
#         interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))

#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     return output_data

# # Load models
# @st.cache_resource
# def load_models():
#     strength_interpreter = load_tflite_model('brick_FlexureStrength_Reg_model_epoch50.tflite')
#     class_interpreter = load_tflite_model('brick_classification_model.tflite')
#     return strength_interpreter, class_interpreter

# strength_model, class_model = load_models()

# file = st.file_uploader("Upload an image of the individual brick", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))
# dry_weight_grams = st.number_input("Enter dry weight of brick (in grams):", min_value=100.0, max_value=5000.0, step=1.0)

# # Normalize dry weight
# NORMALIZATION_BASE = 3144-2610  # typical average dry weight in grams
# dry_weight = dry_weight_grams / NORMALIZATION_BASE

# # Define denormalization range
# MIN_KN = 2.138
# MAX_KN = 12.48

# if file:
#     try:
#         image = Image.open(file).convert('RGB')
#         resized_image = image.resize((224, 224))
#         img_array = np.array(resized_image) / 255.0
#         img_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

#         class_pred = run_tflite_inference(class_model, [img_tensor])
#         class_label = np.argmax(class_pred[0])

#         label_1 = 1 if class_label == 0 else 0
#         label_2 = 1 if class_label == 1 else 0
#         label_3 = 1 if class_label == 2 else 0

#         tabular_input = np.array([[label_1, label_2, label_3, dry_weight]], dtype=np.float32)

#         strength_pred = run_tflite_inference(strength_model, [img_tensor, tabular_input])
#         strength_norm = float(strength_pred[0][0])
#         strength_denorm = strength_norm * (MAX_KN - MIN_KN) + MIN_KN

#         st.image(image, caption=f"Uploaded Brick (Predicted Class: {['1st', '2nd', '3rd'][class_label]})", use_container_width=True)

#         st.success(f"🧪 Normalized Flexural Strength: **{strength_norm:.3f}** (0–1 scale)")
#         st.success(f"🧪 Estimated Real Flexural Strength: **{strength_denorm:.2f} kN**")

#         st.subheader("📊 Classification Probabilities")
#         st.write("""
#         - **1st Class Brick:** {:.2f}%
#         - **2nd Class Brick:** {:.2f}%
#         - **3rd Class Brick:** {:.2f}%
#         """.format(class_pred[0][0] * 100, class_pred[0][1] * 100, class_pred[0][2] * 100))

#     except Exception as e:
#         st.error(f"Error processing image: {e}")

# st.markdown("""
# <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
#     Developed with Streamlit & TensorFlow Lite | © 2024 BrickSense
# </div>
# """, unsafe_allow_html=True)


