import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
import math
import matplotlib.cm as cm
from keras.models import Model
from streamlit_image_comparison import image_comparison

# ----------------- SETUP -------------------
st.set_page_config(
    page_title="BrickSense Multi-App",
    page_icon="static/brickicon8.png",
    layout="centered",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'Developed by BrickSense Team | © 2024'}
)

# ----------------- LOGO -------------------
imagelogo = Image.open("static/BSbasicboxhightran1.png")
st.image(imagelogo, use_container_width=True, width=150)

# ----------------- SIDEBAR -------------------
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("static/BScirclehightran1.png", width=200)

# ----------------- APP SELECTOR -------------------
app_choice = st.radio("Choose the App to Run:", ("Brick Property Prediction (App 1)", "Wall Crack Detection (App 2)"), horizontal=True)

# ----------------- COMMON FILE UPLOAD -------------------
file = st.file_uploader("Upload your image", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# ----------------- COMMON FUNCTIONS -------------------
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

# ----------------- APP 1: Brick Properties -------------------
def run_app1(file):
    st.sidebar.header("App 1: Brick Strength and Absorption")
    st.sidebar.write("""Uses AI models to predict flexural load and absorption of individual bricks.
    
    **Developed by:**  
    Group 24 (Batch 213)  
    Group 25 (Batch 203)  
    Talha Bin Tahir  
    **Email:** talhabtahir@gmail.com""")

    @st.cache_resource
    def load_models():
        strength_model = tf.lite.Interpreter(model_path='brick_FlexureStrength_Reg_model_epoch50.tflite')
        class_model = tf.lite.Interpreter(model_path='brick_classification_model trial 2.tflite')
        absorption_model = tf.lite.Interpreter(model_path='brick_absorption_Model.tflite')
        strength_model.allocate_tensors()
        class_model.allocate_tensors()
        absorption_model.allocate_tensors()
        return strength_model, class_model, absorption_model

    def run_tflite_inference(interpreter, inputs):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        for i, input_tensor in enumerate(inputs):
            interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    dry_weight_grams = st.number_input("Enter dry weight of brick (in grams):", min_value=1500.0, max_value=3500.0, step=1.0)
    dry_weight = (dry_weight_grams - 2610) / (3144 - 2610)

    strength_model, class_model, absorption_model = load_models()

    try:
        image = Image.open(file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        resized = cv2.resize(image_bgr, (224, 224)) / 255.0
        img_tensor = np.expand_dims(resized.astype(np.float32), axis=0)
        class_pred = run_tflite_inference(class_model, [img_tensor])
        class_label = np.argmax(class_pred[0])
        image = correct_orientation(image)
        if class_label == 3:
            st.image(image, caption="Uploaded Image")
            st.error("🚫 This is not a brick.")
        else:
            labels = [1 if class_label == i else 0 for i in range(3)]
            tabular_input = np.array([[*labels, dry_weight]], dtype=np.float32)
            strength_norm = run_tflite_inference(strength_model, [img_tensor, tabular_input])[0][0]
            strength_real = strength_norm * (12.48 - 2.13) + 2.13

            tabular_input2 = np.array([[dry_weight_grams, class_label + 1]], dtype=np.float32)
            absorption = run_tflite_inference(absorption_model, [img_tensor, tabular_input2])[0][0]

            st.image(image, caption=f"Predicted Class: {['1st', '2nd', '3rd'][class_label]}")
            st.success(f"🧪 Estimated Flexural Strength: **{strength_real:.2f} kN**")
            st.success(f"💧 Estimated Absorption: **{absorption:.2f}%**")

            st.subheader("📊 Class Probabilities")
            st.write("""
            - **1st Class:** {:.2f}%
            - **2nd Class:** {:.2f}%
            - **3rd Class:** {:.2f}%
            - **Not a Brick:** {:.2f}%
            """.format(*(class_pred[0] * 100)))
    except Exception as e:
        st.error(f"Error: {e}")

# ----------------- APP 2: Crack Detection -------------------
def run_app2(file):
    st.sidebar.header("App 2: Wall Crack Detection")
    st.sidebar.write("""CNN-based detection of cracked or normal brick walls.

    **Developed by:**  
    Talha Bin Tahir  
    **Email:** talhabtahir@gmail.com""")

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')

    model = load_model()

    def add_white_border(image, border_size):
        return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))

    def add_canvas(image, fill_color=(255, 255, 255)):
        w, h = image.size
        canvas = Image.new("RGB", (w + int(0.015 * w), h + int(0.07 * h)), fill_color)
        canvas.paste(image, ((canvas.width - w) // 2, (canvas.height - h) // 7))
        return canvas

    def predict(image, sensitivity=9):
        original_img = np.array(image)
        if original_img.shape[-1] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        img_resized = cv2.resize(original_img, (224, 224))
        tensor = np.expand_dims(img_resized, axis=0) / 255.0
        custom_model = Model(inputs=model.inputs, outputs=[model.layers[sensitivity].output, model.layers[-1].output])
        feature_maps, preds = custom_model.predict(tensor)
        heat_map = np.mean(feature_maps.squeeze(), axis=-1)
        heat_map = np.maximum(heat_map, 0) / np.max(heat_map)
        heat_map_img = cv2.resize(np.uint8(255 * cm.jet(heat_map)[:, :, :3]), original_img.shape[1::-1])
        heat_map_pil = Image.fromarray(heat_map_img)
        pred_class = np.argmax(preds)
        return preds[0], heat_map_pil, pred_class

    try:
        image = Image.open(file).convert("RGB")
        image = correct_orientation(image)
        preds, heatmap_img, pred_class = predict(image)

        labels = ["Normal", "Cracked", "Not a Wall"]
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if pred_class == 0:
            st.success("✅ Normal brick wall.")
        elif pred_class == 1:
            st.error("❌ Cracked brick wall.")
        else:
            st.warning("⚠️ Not a wall.")

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
            <div style="text-align: center; flex: 1;">🟢 Normal: {preds[0]*100:.2f}%</div>
            <div style="text-align: center; flex: 1;">🔴 Cracked: {preds[1]*100:.2f}%</div>
            <div style="text-align: center; flex: 1;">🟠 Not a Wall: {preds[2]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        sensitivity = st.slider("Adjust Detection Sensitivity", 0, 12, 9, 1)
        _, heatmap_img, _ = predict(image, sensitivity)

        st.image(heatmap_img, caption="Heatmap Overlay", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------- EXECUTION -------------------
if file:
    if app_choice.startswith("Brick Property"):
        run_app1(file)
    else:
        run_app2(file)
else:
    st.info("Please upload an image to get started.")

# ----------------- FOOTER -------------------
st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
    Developed with Streamlit & TensorFlow | © 2024 BrickSense
</div>
""", unsafe_allow_html=True)
