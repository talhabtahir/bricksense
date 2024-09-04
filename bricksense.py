import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Detection",
    page_icon="static/brickicon8.png",  # Path to your favicon file
    layout="centered"
)

# Custom CSS for additional styling
st.markdown(
    """
    <link rel="icon" href="static/brickicon8.png" type="image/x-icon">
    <style>
        .reportview-container {
            background-color: #f7f9fc;
            padding-top: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f7f9fc;
        }
        .main-header {
            color: #ff6347;
            text-align: center;
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: small;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with an icon
st.markdown("<h1 class='main-header'>🧱 Brick Detection 🧱</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('170kmodelv1_version_cam_1.keras')

model = load_model()

# Sidebar for app information
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) model to detect brick walls and classify them as either normal, cracked, or not a wall. 
You can upload an image, and the app will analyze it to provide predictions.
""")
st.sidebar.write("""
**Developed by:**  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com
""")

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# Placeholder for image and prediction
image_placeholder = st.empty()
prediction_placeholder = st.empty()

# Function to correct image orientation based on EXIF data
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

# Function to make predictions using the TensorFlow model
def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

if file:
    try:
        # Display the uploaded image
        image = Image.open(file)
        
        # Correct the orientation if necessary
        image = correct_orientation(image)
        
        # Update the image placeholder
        image_placeholder.image(image, caption=f"Uploaded Image: {file.name}", use_column_width=True)
        
        # Perform prediction
        predictions = import_and_predict(image, model)
        if predictions is not None:
            predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
            prediction_percentages = predictions[0] * 100  # Convert to percentages
            
            # Update the prediction placeholder
            prediction_placeholder.write(f"**Prediction Percentages for {file.name}:**")
            prediction_placeholder.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
            prediction_placeholder.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
            prediction_placeholder.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
            
            # Display the predicted class
            if predicted_class == 0:
                prediction_placeholder.success(f"✅ This is a normal wall.")
            elif predicted_class == 1:
                prediction_placeholder.error(f"⚠️ This wall is cracked.")
            elif predicted_class == 2:
                prediction_placeholder.warning(f"⚠️ This is not a wall.")
            else:
                prediction_placeholder.error(f"❓ Unknown prediction result: {predicted_class}")
    
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
