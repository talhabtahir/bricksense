import streamlit as st
from PIL import Image, ImageOps, ExifTags
import numpy as np
import tensorflow as tf
import cv2
from keras.models import Model
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv3_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

def correct_orientation(image):
    try:
        if hasattr(image, '_getexif'):
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
    except Exception as e:
        st.error(f"Error correcting orientation: {e}")
    return image

def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)

        custom_model = Model(inputs=model.inputs, 
        outputs=(model.layers[8].output, model.layers[-1].output))  # `conv2d_3` and predictions

        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(img_reshape/255.0)
        conv2d_3_output = np.squeeze(conv2d_3_output)  # 28x28x32 feature maps
        
        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)  # (224, 224, 32)
        
        # Average all the filters from conv2d_3 to get a single activation map
        heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)  # Take the mean of the 32 filters, resulting in (224, 224)
        
        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1
        
        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5  # You can adjust this threshold
        heat_map_thresh = np.uint8(255 * heat_map)  # Convert heatmap to 8-bit image
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours along the heatmap regions
        contoured_img = img.copy()
        cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)  # Draw green contours (lines)
        
        # Plot the original image with heatmap and contours overlaid
        fig, ax = plt.subplots()
        ax.imshow(contoured_img)  # Image with contours (lines)
        ax.imshow(heat_map, cmap='jet', alpha=0.4)  # Overlay heatmap with transparency
        
        return prediction, fig  # Return the figure instead of `plt.show()`
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Debugging wrapper to display file details
def display_file_details(uploaded_file):
    st.write(f"Uploaded file type: {type(uploaded_file)}")
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
    st.write(f"File type: {uploaded_file.type}")

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=["jpg", "png", "jpeg", "bmp", "tiff", "webp"])

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            # Display file details for debugging
            display_file_details(file)

            # Try to open the uploaded image using PIL
            image = Image.open(file)
            if image is None:
                raise ValueError("Uploaded file is not a valid image.")

            # Correct the orientation if necessary
            image = correct_orientation(image)

            # Ensure the image format is valid
            if image.format not in ["JPEG", "PNG", "BMP", "TIFF", "WEBP"]:
                raise ValueError("Unsupported image format. Please upload JPG, PNG, BMP, TIFF, or WEBP files.")

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform prediction
            predictions, heatmap_fig = import_and_predict(image, model)
            if predictions is not None:
                predicted_class = np.argmax(predictions[0])
                prediction_percentages = predictions[0] * 100

                st.write(f"**Prediction Percentages:**")
                st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
                st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
                st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
                
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall.")
                    # Display the heatmap and contours figure
                    st.pyplot(heatmap_fig)
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
