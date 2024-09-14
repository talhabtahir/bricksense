import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Detection",
    page_icon="static/brickicon8.png",
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

# Display logo instead of header
imagelogo = Image.open("static/sidelogo.png")
st.image(imagelogo, use_column_width=True, width=150)  # Update the path to your logo file

# Sidebar information
st.sidebar.image("static/sidelogo.png", width=200, use_column_width=True)
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses a CNN model to detect brick walls and classify them as either normal, cracked, or not a wall. 
You can upload an image, and the app will analyze it to provide a prediction.
""")
st.sidebar.write("""
**Developed by:**  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com
""")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv3_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

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

# Function to generate heatmap and contours based on the predictions
def generate_heatmap_and_contours(img_array, model):
    try:
        img_tensor = np.expand_dims(img_array, axis=0) / 255.0
        preprocessed_img = img_tensor
        
        # Define a new model that outputs the conv2d_3 feature maps and the prediction
        custom_model = tf.keras.Model(inputs=model.inputs, outputs=(model.layers[10].output, model.layers[-1].output))
        
        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)  # 28x28x32 feature maps
        
        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Average all the filters to get a single activation map (heatmap)
        heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1
        
        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5
        heat_map_thresh = np.uint8(255 * heat_map)  # Convert heatmap to 8-bit image
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image (without the heatmap overlay)
        contoured_img_only = img_array.copy()
        cv2.drawContours(contoured_img_only, contours, -1, (0, 255, 0), 2)  # Draw green contours (lines)
        
        return contoured_img_only, heat_map, pred
    except Exception as e:
        st.error(f"Error generating heatmap and contours: {e}")
        return None, None, None

# Function to make predictions and generate the heatmap and contours
def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = np.asarray(image).astype(np.float32)

        contoured_img, heatmap, predicted_class = generate_heatmap_and_contours(img_array, model)
        
        return contoured_img, predicted_class
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            # Display the uploaded image
            image = Image.open(file)
            
            # Correct the orientation if necessary
            image = correct_orientation(image)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Perform prediction
            contoured_img, predicted_class = import_and_predict(image, model)
            
            if contoured_img is not None:
                st.image(contoured_img, caption="Detected Contours", use_column_width=True)
                
                # Display the predicted class
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall.")
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")
        
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
