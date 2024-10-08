import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
from auth import check_authentication

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
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # # Header with an icon
    # st.markdown("<h1 class='main-header'>🧱 Brick Cracks Detection 🧱</h1>", unsafe_allow_html=True)
  
    # Header with an image (centered)
    # Display logo instead of header
    imagelogo = Image.open("static/head1.png")
    st.image(imagelogo, use_column_width=True, width=150)  # Update the path to your logo file

    # Add space below the logo
    st.write("")  # Creates a blank line
    st.write(" ")  # Creates an extra line for more space
    st.write(" ")  # Adjust the number of empty lines for desired spacing
    
    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model('170kmodelv1_version_cam_1.keras')
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    model = load_model()
    
    # Sidebar for app information
    with st.sidebar.expander("About the Version"):
        st.write("""
        This version of BrickSense App is a powerful tool designed to automatically detect cracks 
        in brick walls, leveraging cutting-edge deep learning technique. Built using a Convolutional 
        Neural Network (CNN) model pre-trained on a dataset of more than 150,000 images, the app 
        specializes in identifying structural defects in brick walls with high accuracy. The app 
        can analyze and classify a single image at a time, under three categoies namely noraml, cracked 
        and not a wall, making it easy for users to quickly check the condition of brick structures in real-time.
        """)
        st.write("""
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
