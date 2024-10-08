import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np

def run():
    # Custom CSS for additional styling
    st.markdown(
        """
        <link rel="icon" href="static/brickicon3.png" type="image/x-icon">
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
            .stButton > button {
                background-color: #ff6347;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }
            .stButton > button:hover {
                background-color: #ff4500;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display logo instead of header
    imagelogo = Image.open("static/head2.png")
    st.image(imagelogo, use_column_width=True)  # Update the path to your logo file

    # Add space below the logo
    st.write("")  # Creates a blank line
    st.write(" ")  # Adjust the number of empty lines for desired spacing
    
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('170kmodelv3_version_cam_1.keras')
    
    model = load_model()
    
    # Sidebar for app information
    with st.sidebar.expander("About the Version"):
        st.write("""
        This version of Brick Sense is an AI-powered solution designed to identify cracks and classify brick walls using advanced computer vision techniques. The app leverages a Convolutional Neural Network (CNN) pre-trained model that has been fine-tuned using a large dataset of approximately 200,000 images. This extensive dataset ensures high accuracy and robust performance in real-world scenarios, enabling the app to accurately classify both single and multiple images simultaneously.
        """)
        st.write("""
        **Developed by:**  
        Talha Bin Tahir  
        **Email:** talhabtahir@gmail.com
        """)
    
    # Upload options
    upload_option = st.radio("Choose your upload option:", ("Single Image", "Multiple Images"), index=0)
    
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
    
    if upload_option == "Single Image":
        file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))
        # Check if a file was uploaded
        if not file:
            st.info("Please upload an image file to start the detection.")
        elif file:
            try:
                # Display the uploaded image
                image = Image.open(file)
                
                # Correct the orientation if necessary
                image = correct_orientation(image)
                
                st.image(image, caption="Uploaded Image (Corrected Orientation)", use_column_width=True)
                
                # Perform prediction
                predictions = import_and_predict(image, model)
                if predictions is not None:
                    predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
                    prediction_percentages = predictions[0] * 100  # Convert to percentages
                    
                    # Display prediction percentages for each class
                    st.write(f"**Prediction Percentages:**")
                    st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
                    st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
                    st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
                    
                    # Display the predicted class
                    if predicted_class == 0:
                        st.success(f"✅ This is a normal wall.")
                    elif predicted_class == 1:
                        st.error(f"❌ This wall is cracked.")
                    elif predicted_class == 2:
                        st.warning(f"⚠️ This is not a wall.")
                    else:
                        st.error(f"❓ Unknown prediction result: {predicted_class}")
            
            except Exception as e:
                st.error(f"Error processing the uploaded image: {e}")
    
    elif upload_option == "Multiple Images":
        files = st.file_uploader("Please upload images of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"), accept_multiple_files=True)
        # Check if a file was uploaded
        if not files:
            st.info("Please upload an image file to start the detection.")
        elif files:
            for file in files:
                try:
                    # Display the uploaded image
                    image = Image.open(file)
                    
                    # Correct the orientation if necessary
                    image = correct_orientation(image)
                    
                    st.image(image, caption=f"Uploaded Image: {file.name}", use_column_width=True)
                    
                    # Perform prediction
                    predictions = import_and_predict(image, model)
                    if predictions is not None:
                        predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
                        prediction_percentages = predictions[0] * 100  # Convert to percentages
                        
                        # Display prediction percentages for each class
                        st.write(f"**Prediction Percentages for {file.name}:**")
                        st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
                        st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
                        st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
                        
                        # Display the predicted class
                        if predicted_class == 0:
                            st.success(f"✅ This is a normal wall.")
                        elif predicted_class == 1:
                            st.error(f"❌ This wall is cracked.")
                        elif predicted_class == 2:
                            st.warning(f"⚠️ This is not a wall.")
                        else:
                            st.error(f"❓ Unknown prediction result: {predicted_class}")
                
                except Exception as e:
                    st.error(f"Error processing the uploaded image {file.name}: {e}")
    
    # Footer
    st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
