# Brick Wall Crack Detection App (App 2)
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
import cv2
import math
from keras.models import Model
from streamlit_image_comparison import image_comparison
import matplotlib.cm as cm

st.title("🧱 Brick Wall Crack Detection")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

file = st.file_uploader("Upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
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

def add_white_border(image, border_size):
    return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))

def add_canvas(image, fill_color=(255, 255, 255)):
    w, h = image.size
    cw = w + math.ceil(0.015 * w)
    ch = h + math.ceil(0.07 * h)
    canvas = Image.new("RGB", (cw, ch), fill_color)
    pos = ((cw - w) // 2, (ch - h) // 7)
    canvas.paste(image, pos)
    return canvas

def import_and_predict(image_data, sensitivity=9):
    original_img = np.array(image_data)
    if original_img.shape[-1] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

    h, w, _ = original_img.shape
    thickness = max(2, int(max(w, h) / 200))
    img_resized = cv2.resize(original_img, (224, 224)) / 255.0
    img_tensor = np.expand_dims(img_resized, axis=0)

    custom_model = Model(inputs=model.inputs,
                         outputs=(model.layers[sensitivity].output, model.layers[-1].output))
    conv_output, pred_vec = custom_model.predict(img_tensor)
    conv_output = np.squeeze(conv_output)

    pred = np.argmax(pred_vec)
    heat_map_resized = cv2.resize(conv_output, (w, h))
    heat_map = np.mean(heat_map_resized, axis=-1)
    heat_map = np.maximum(heat_map, 0)
    heat_map = heat_map / heat_map.max()

    _, thresh_map = cv2.threshold(np.uint8(255 * heat_map), int(255 * 0.5), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heatmap_rgb = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
    heatmap_image = Image.fromarray(heatmap_rgb)
    contoured_img = original_img.copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), thickness)
    contoured_image = Image.fromarray(contoured_img)

    overlay = Image.blend(Image.fromarray(original_img).convert("RGBA"),
                          heatmap_image.convert("RGBA"), alpha=0.5)
    overlay_np = np.array(overlay.convert("RGB"))
    cv2.drawContours(overlay_np, contours, -1, (0, 0, 0), thickness)
    overlay_img = Image.fromarray(overlay_np)

    class_labels = ["Normal", "Cracked", "Not a Wall"]
    image_with_border = add_white_border(image_data, 10)
    contours_with_border = add_white_border(overlay_img, 10)
    return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img

if file:
    image = Image.open(file)
    image = correct_orientation(image)

    predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img = import_and_predict(image)

    predicted_class = np.argmax(predictions)
    prediction_percentages = predictions[0] * 100

    if predicted_class == 0:
        st.success("✅ This is a normal brick wall.")
    elif predicted_class == 1:
        st.error("❌ This wall is a cracked brick wall.")
    else:
        st.warning("⚠️ This is not a brick wall.")

    st.markdown(f"""
    <div style='background-color: #808080; color: white; padding: 6px; border-radius: 9px;'>
    🟢 Normal Wall: {prediction_percentages[0]:.2f}% &nbsp;&nbsp;
    🔴 Cracked Wall: {prediction_percentages[1]:.2f}% &nbsp;&nbsp;
    🟠 Not a Wall: {prediction_percentages[2]:.2f}%
    </div>
    """, unsafe_allow_html=True)

    sensitivity = st.slider("🔍 Detection Sensitivity", 0, 12, 9)
    predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img = import_and_predict(image, sensitivity=sensitivity)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(contoured_image if predicted_class == 1 else image, caption="Crack(s) Location", use_container_width=True)
    with col3:
        st.image(heatmap_image if predicted_class == 1 else image, caption="Crack(s) Heatmap", use_container_width=True)
    with col4:
        st.image(overlay_img if predicted_class == 1 else image, caption="Localization Overlay", use_container_width=True)

    image_with_border = add_canvas(image_with_border)
    contours_with_border = add_canvas(contours_with_border)

    if st.checkbox("🖼️ Original vs Crack Slider"):
        st.markdown("""<div style='display: flex; justify-content: center;'>""", unsafe_allow_html=True)
        image_comparison(
            img1=image_with_border,
            img2=contours_with_border,
            label1="Original",
            label2="Localized Cracks",
            show_labels=False,
        )
        st.markdown("""</div>""", unsafe_allow_html=True)
