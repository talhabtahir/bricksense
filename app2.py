import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
from keras.models import Model
from streamlit_image_comparison import image_comparison
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt

@st.cache_resource
#__________________________________________________________________________________________________________________________________________________________________________________
#For single model selection
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
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
    
# Adding Canvas Background
def add_canvas(image, fill_color=(255, 255, 255)):
    """Automatically adjusts canvas size according to image size, with added padding and centers the image on the canvas."""
    # Get the original image size
    image_width, image_height = image.size
    
    # Calculate new canvas size with padding
    canvas_width = image_width + math.ceil(0.015 * image_width)
    canvas_height = image_height + math.ceil(0.07 * image_height)
    
    # Create a new image (canvas) with the calculated size
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    
    # Calculate the position to paste the image at the center of the canvas
    paste_position = (
        (canvas_width - image_width) // 2,
        (canvas_height - image_height) // 7
    )
    
    # Paste the original image onto the canvas
    canvas.paste(image, paste_position)
    
    return canvas


# Function to localize the crack and to make predictions using the TensorFlow model
def import_and_predict(image_data, sensitivity=9):
    try:
        # Convert image to numpy array
        original_img = np.array(image_data)

        # Convert image to RGB if it has an alpha channel
        if original_img.shape[-1] == 4:  # Check if the image has 4 channels
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        
        # Save original dimensions
        orig_height, orig_width, _ = original_img.shape
        
        # Calculate the maximum dimension of the original image
        max_dimension = max(orig_width, orig_height)
        
        # Set the scaling factor for contour line thickness based on the max dimension
        contour_thickness = max(2, int(max_dimension / 200))  # Adjust the divisor to control scaling

        # Preprocess the image for the model
        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
        preprocessed_img = img_tensor
        
        # Define a new model that outputs the conv2d_3 feature maps and the prediction
        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))  # `conv2d_3` and predictions

        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)  # (28, 28, 32) feature maps

        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        # Average all the filters from conv2d_3 to get a single activation map
        heat_map = np.mean(heat_map_resized, axis=-1)  # (orig_height, orig_width)

        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1

        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5  # Adjust this threshold if needed
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert heatmap to RGB for display
        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        
        # Convert heatmap to PIL format
        heatmap_image = Image.fromarray(heatmap_colored)
        
        # Create contoured image
        contoured_img = original_img.copy()  # Copy original image
        cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)  # Draw blue contours
        
        # Convert contoured image to PIL format
        contoured_image = Image.fromarray(contoured_img)

        # Overlay heatmap on original image
        heatmap_image_rgba = heatmap_image.convert("RGBA")
        original_img_pil = Image.fromarray(original_img).convert("RGBA")
        heatmap_overlay = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

        # Draw contours on the heatmap-overlayed image
        # Convert heatmap-overlayed image to RGB for contour drawing
        heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
        heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
        # heatmap_overlay_np = np.array(heatmap_overlay)
        cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)  # Draw blue contours

        # Convert overlay image to PIL format
        overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

        # Get the predicted class name
        class_labels = ["Normal", "Cracked", "Not a Wall"]
        predicted_class = class_labels[pred]

        # Add white borders
        border_size = 10  # Set the border size
        image_with_border = add_white_border(image_data, border_size)
        contours_with_border = add_white_border(overlay_img, border_size)

        return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img 
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None, None, None
        
# Adds border to the image

def add_white_border(image, border_size):
    """Add a white border to the image."""
    return ImageOps.expand(image, border= border_size, fill=(255, 255, 255))

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

            
            
            # Perform prediction
            predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image)
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                 # Display prediction result
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall. ")
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")

                st.write(f"**Prediction Percentages:**")
                # Display predictions in one line
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
                        <div style="text-align: center; flex: 1;">🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")  # Creates a blank line

                # st.write("")  # Creates a blank line

                # Create an expander for sensitivity adjustment
                with st.expander("🔍 Sensitivity Settings"):
                    # Add a slider for selecting the sensitivity dynamically
                    sensitivity = st.slider(
                        "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                        min_value=0,   # Minimum value for sensitivity
                        max_value=12,   # Maximum value for sensitivity
                        value=9,       # Default value for sensitivity
                        step=1,        # Step for incremental changes
                        format="%.1f"    # Format to display sensitivity with one decimal
                                            )


                # Perform prediction again
                predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image, sensitivity=sensitivity)

                #in one row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    if predicted_class == 1:
                        st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
                        
                with col3:
                    if predicted_class == 1:
                        st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
                
                with col4:
                    if predicted_class == 1:
                        st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
         
                image_with_border = add_canvas(image_with_border)
                contours_with_border = add_canvas(contours_with_border)               
                st.write("")  # Creates a blank line
                if st.checkbox("Original vs Cracked Slider"):
                    # HTML/CSS for centering the image comparison component
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
                    
                    # Opening div tag to center the image comparison component
                    st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
                    
                    # Conditionally display image comparison
                    if predicted_class == 1:
                        image_comparison(
                            img1=image_with_border, 
                            img2=contours_with_border,
                            label1="Uploaded Image",
                            label2="Cracks Localization",
                            show_labels=False,
                        )
                    else:
                        image_comparison(
                            img1=image_with_border, 
                            img2=image_with_border,
                            label1="Uploaded Image",
                            label2="Cracks Localization",
                            show_labels=False,
                        )
                
                    # Closing div tag
                    st.markdown('</div>', unsafe_allow_html=True)
                            
        
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
