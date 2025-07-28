# Full-featured Brick Wall Crack Detection Module
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
import cv2
from keras.models import Model
from streamlit_image_comparison import image_comparison
import math
import matplotlib.cm as cm

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Utilities
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

def add_canvas(image, fill_color=(255, 255, 255)):
    w, h = image.size
    canvas_w = w + math.ceil(0.015 * w)
    canvas_h = h + math.ceil(0.07 * h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), fill_color)
    pos = ((canvas_w - w) // 2, (canvas_h - h) // 7)
    canvas.paste(image, pos)
    return canvas

def add_white_border(image, border_size):
    return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))

def import_and_predict(image_data, sensitivity=9):
    try:
        original_img = np.array(image_data)
        if original_img.shape[-1] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        h, w, _ = original_img.shape
        contour_thickness = max(2, int(max(w, h) / 200))

        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

        custom_model = Model(inputs=model.inputs,
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))

        conv_output, pred_vec = custom_model.predict(img_tensor)
        conv_output = np.squeeze(conv_output)
        pred = np.argmax(pred_vec)

        heat_map_resized = cv2.resize(conv_output, (w, h), interpolation=cv2.INTER_LINEAR)
        heat_map = np.mean(heat_map_resized, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        heat_map = heat_map / heat_map.max()

        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * 0.5), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        heatmap_image = Image.fromarray(heatmap_colored)

        contoured_img = original_img.copy()
        cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
        contoured_image = Image.fromarray(contoured_img)

        heatmap_overlay = Image.blend(Image.fromarray(original_img).convert("RGBA"),
                                      heatmap_image.convert("RGBA"), alpha=0.5)
        overlay_img_np = np.array(heatmap_overlay.convert("RGB"))
        cv2.drawContours(overlay_img_np, contours, -1, (0, 0, 0), contour_thickness)
        overlay_img = Image.fromarray(overlay_img_np)

        class_labels = ["Normal", "Cracked", "Not a Wall"]
        predicted_class = class_labels[pred]

        bordered_original = add_white_border(image_data, 10)
        bordered_overlay = add_white_border(overlay_img, 10)

        return pred_vec, bordered_original, bordered_overlay, heatmap_image, contoured_image, overlay_img, pred

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None, None, None, None, None

# Crack Detection Execution
if file:
    image = Image.open(file)
    image = correct_orientation(image)

    with st.expander("🔍 Sensitivity Settings"):
        sensitivity = st.slider("Adjust Detection Sensitivity", 0, 12, 9, 1)

    with st.spinner("Analyzing image..."):
        predictions, img_with_border, contours_img, heatmap_img, contoured_img, overlay_img, predicted_class_idx = import_and_predict(image, sensitivity)

    if predictions is not None:
        class_labels = ["Normal Wall", "Cracked Wall", "Not a Wall"]
        prediction_percentages = predictions[0] * 100
        st.success(f"✅ Prediction: {class_labels[predicted_class_idx]}")

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
            <div style="text-align: center; flex: 1;">🟢 <strong>Normal:</strong> {prediction_percentages[0]:.2f}%</div>
            <div style="text-align: center; flex: 1;">🔴 <strong>Cracked:</strong> {prediction_percentages[1]:.2f}%</div>
            <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(contoured_img if predicted_class_idx == 1 else image, caption="Crack(s) Location", use_container_width=True)
        with col3:
            st.image(heatmap_img if predicted_class_idx == 1 else image, caption="Heatmap", use_container_width=True)
        with col4:
            st.image(overlay_img if predicted_class_idx == 1 else image, caption="Localization Overlay", use_container_width=True)

        if st.checkbox("Original vs Crack Slider"):
            center_style = """
            <style>
            .centered-image-container {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """
            st.markdown(center_style, unsafe_allow_html=True)
            st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
            image_comparison(
                img1=img_with_border,
                img2=contours_img,
                label1="Uploaded Image",
                label2="Cracks Localization",
                show_labels=False,
            )
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload an image to start crack detection.")
