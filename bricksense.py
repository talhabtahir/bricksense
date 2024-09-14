import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

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

# Display logo instead of header
imagelogo = Image.open("static/sidelogo.png")
st.image(imagelogo, use_column_width=True, width=150)  # Update the path to your logo file

# Add space below the logo
st.write("")  # Creates a blank line
st.write(" ")  # Creates an extra line for more space
st.write(" ")  # Adjust the number of empty lines for desired spacing

# Sidebar navigation with icons
st.sidebar.image("static/sidelogo.png", width=200, use_column_width=True)
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv3_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Sidebar for app information
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) model to detect brick walls and classify them as either normal, cracked, or not a wall. 
You can upload an image, and the app will analyze it to provide a prediction.
""")
st.sidebar.write("""
**Developed by:**  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com
""")

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

# Function to make predictions and generate heatmaps/contours
def import_and_predict(image_data, model):
    try:
        # Preprocess the image
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]  # Add batch dimension

        # Get model predictions and feature maps
        custom_model = tf.keras.Model(inputs=model.inputs, outputs=(model.layers[10].output, model.layers[-1].output))  # Layer selection
        conv2d_3_output, predictions = custom_model.predict(img_reshape)
        conv2d_3_output = np.squeeze(conv2d_3_output)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        prediction_percentages = predictions[0] * 100  # Convert to percentages

        # Generate heatmap
        upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()

        # Threshold the heatmap and find contours
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * 0.5), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

        return predictions, img_with_contours, heat_map, predicted_class
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            # Display the uploaded image
            image = Image.open(file)
            image = correct_orientation(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform prediction and display heatmap/contours
            predictions, img_with_contours, heat_map, predicted_class = import_and_predict(image, model)
            if predictions is not None:
                st.image(img_with_contours, caption="Image with Contours", use_column_width=True)

                # Display heatmap
                fig, ax = plt.subplots()
                ax.imshow(heat_map, cmap='viridis')
                ax.set_title("Heatmap")
                st.pyplot(fig)

                # Display prediction percentages
                st.write(f"**Prediction Percentages:**")
                st.write(f"Normal Wall: {predictions[0][0] * 100:.2f}%")
                st.write(f"Cracked Wall: {predictions[0][1] * 100:.2f}%")
                st.write(f"Not a Wall: {predictions[0][2] * 100:.2f}%")

                # Display predicted class
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
