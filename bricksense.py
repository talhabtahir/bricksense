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
        # Get original image size
        original_size = image_data.size  # (width, height)
        size = (224, 224)

        # Resize the image for model prediction
        image_resized = image_data.convert("RGB")
        image_resized = ImageOps.fit(image_resized, size, Image.LANCZOS)
        img = np.asarray(image_resized).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]

        # Get predictions from the model
        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[10].output, model.layers[-1].output))  # `conv2d_3` and predictions
        conv2d_3_output, pred_vec = custom_model.predict(img_reshape)
        
        # Get the predicted class and confidence
        pred = np.argmax(pred_vec)

        # Extract the feature map output
        conv2d_3_output = np.squeeze(conv2d_3_output)  # Shape (28, 28, 32)
        
        # Average across the depth dimension (32 filters) to generate the heatmap
        heat_map = np.mean(conv2d_3_output, axis=-1)  # Shape (28, 28)

        # Normalize the heatmap between 0 and 1 for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map /= np.max(heat_map)  # Normalize to 0-1

        # Resize heatmap to the size of the resized image (224, 224)
        heatmap_resized = cv2.resize(heat_map, size, interpolation=cv2.INTER_LINEAR)

        # Threshold the heatmap to get regions of interest
        _, thresh_map = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert original image to numpy array (for contour drawing)
        original_img_np = np.array(image_data)

        # Ensure the original image is in 3 channels (RGB) for blending
        if len(original_img_np.shape) == 2:  # If grayscale, convert to RGB
            original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)

        # Resize the original image to match the heatmap size (224, 224)
        original_resized = cv2.resize(original_img_np, size, interpolation=cv2.INTER_LINEAR)

        # Draw contours on the resized image
        cv2.drawContours(original_resized, contours, -1, (255, 0, 0), 2)  # Green contours

        # Resize the image with contours back to its original size
        contours_img_original_size = cv2.resize(original_resized, original_size, interpolation=cv2.INTER_LINEAR)

        # Convert back to RGB for display in Streamlit
        contours_img_rgb = cv2.cvtColor(contours_img_original_size, cv2.COLOR_BGR2RGB)
        
        # Convert to a PIL Image for display in Streamlit
        contours_pil = Image.fromarray(contours_img_rgb)

        # Create a figure to display the results
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size for better clarity
        ax.imshow(contours_pil)
        ax.axis('off')  # Hide the axes for a cleaner visualization
        
        return pred_vec, fig
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=["jpg", "png", "jpeg", "bmp", "tiff", "webp"])

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
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
            predictions, contours_fig = import_and_predict(image, model)
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                st.write(f"**Prediction Percentages:**")
                st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
                st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
                st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
                
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall.")
                    # Display the contours figure
                    st.pyplot(contours_fig)
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
