import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, resnet50
from PIL import Image, ImageOps, ExifTags
import numpy as np
import torch
import cv2

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Crack Detection",
    page_icon="static/brickicon4.png",  # Path to your favicon file
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
st.markdown("<h1 class='main-header'>🧱 Brick Crack Detection 🧱</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('170kmodelv1_version_cam_1')

@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@st.cache_resource
def load_imagenet_model():
    # Load the pre-trained ResNet50 model with ImageNet weights
    model = ResNet50(weights='imagenet')
    return model

model = load_model()
yolo_model = load_yolo_model()
imagenet_model = load_imagenet_model()

# Sidebar for app information
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) model to detect cracks in brick walls. 
You can upload an image of a brick wall, and the app will analyze the image to determine if the wall is cracked or normal. 
The output will provide a predicted class and the probability of the wall being cracked or normal.
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

# Function to analyze image with YOLOv5
def analyze_with_yolo(image_path):
    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            st.error(f"Error: Could not open or find the image at {image_path}")
            return None
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        results = yolo_model(img_rgb)
        results_df = results.pandas().xyxy[0]
        return results_df
    except Exception as e:
        st.error(f"An error occurred during YOLO analysis: {e}")
        return None

# Function to make predictions using the ImageNet model
def import_and_predict_imagenet(image_data, model):
    try:
        size = (224, 224)
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32)
        img_reshape = np.expand_dims(img, axis=0)
        img_preprocessed = resnet50.preprocess_input(img_reshape)
        prediction = model.predict(img_preprocessed)
        decoded_predictions = resnet50.decode_predictions(prediction, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"An error occurred during ImageNet prediction: {e}")
        return None

if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    try:
        # Display the uploaded image
        image = Image.open(file)
        
        # Correct the orientation if necessary
        image = correct_orientation(image)
        
        st.image(image, caption="Uploaded Image (Corrected Orientation)", use_column_width=True)
        
        # Convert the image to RGB and save temporarily for YOLO processing
        image = image.convert("RGB")  # Ensure image is in RGB mode
        image_path = '/tmp/uploaded_image.jpg'
        image.save(image_path, format='JPEG')  # Save as JPEG to avoid format issues
        
        # Initialize flags for detection
        yolo_detected = False
        resnet50_detected = False

        # Collect detection results
        confres = []

        # Step 1: Analyze with YOLOv5
        yolo_detected_classes = []  # Initialize to avoid NameError

        yolo_results = analyze_with_yolo(image_path)
        if yolo_results is not None and not yolo_results.empty:
            # Check if any detected class name contains "wall"
            if any("wall" in class_name.lower() for class_name in yolo_results['name']):
                yolo_detected = True
            else:
                # Filter results with confidence >= 0.6
                high_confidence_results = yolo_results[yolo_results['confidence'] >= 0.6]
                
                # Check if there are high confidence results
                if not high_confidence_results.empty:
                    yolo_detected_classes = high_confidence_results['name'].unique().tolist()
                    confres.extend([class_name.capitalize() for class_name in yolo_detected_classes])
                    st.write("#### YOLO Classification Results:")
                    # Capitalize each class name individually
                    st.write(f"YOLOv5 detected the following classes with high confidence: {', '.join([name.capitalize() for name in yolo_detected_classes])}")
        
        # Handle case where yolo_results is None or empty
        if not yolo_detected_classes:
            # st.write("No classes detected with high confidence using YOLOv5.")
            pass
        
        # Step 2: ImageNet classification
        imagenet_predictions = import_and_predict_imagenet(image, imagenet_model)
        if imagenet_predictions:
            if any("wall" in name.lower() for _, name, score in imagenet_predictions): # Detection of any wall class in imagenet predictions and getting out of loop
                resnet50_detected = True
            else:
                high_confidence_imagenet = [(name, score) for _, name, score in imagenet_predictions if score >= 0.6]
                if high_confidence_imagenet:
                    st.write("#### ImageNet Classification Results:")
                    confres.extend([class_name.capitalize() for class_name, score in high_confidence_imagenet])
                    for class_name, score in high_confidence_imagenet:
                        st.write(f"Class: {class_name}, Score: {score:.4f}")
                
        # Decision to proceed with TensorFlow model
        if yolo_detected or resnet50_detected:
            # st.info("Proceeding with TensorFlow model prediction based on detection.")
            predictions = import_and_predict(image, model)
            if predictions is not None:
                probability = predictions[0][0]
                if probability > 0.5:
                    predicted_class = "cracked"
                    st.error(f"⚠️ This brick wall is {predicted_class}.")
                    st.write(f"**Predicted Probability:** {probability * 100:.2f}% cracked.")
                else:
                    predicted_class = "normal"
                    st.success(f"✅ This brick wall is {predicted_class}.")
                    st.write(f"**Predicted Probability:** {(1 - probability) * 100:.2f}% normal.")
        else:
            # Decision based on detection results
            if confres:
                st.info(f"Following objects/subjects were detected: {', '.join(confres)}. Please upload an image of brick wall")
            else:
                # Step 3: TensorFlow model prediction
                # st.info("Neither YOLOv5 nor ImageNet detected relevant classes with high confidence. Proceeding with TensorFlow model prediction.")
                predictions = import_and_predict(image, model)
                if predictions is not None:
                    probability = predictions[0][0]
                    if probability > 0.5:
                        predicted_class = "cracked"
                        st.error(f"⚠️ This brick wall is {predicted_class}.")
                        st.write(f"**Predicted Probability:** {probability * 100:.2f}% cracked.")
                    else:
                        predicted_class = "normal"
                        st.success(f"✅ This brick wall is {predicted_class}.")
                        st.write(f"**Predicted Probability:** {(1 - probability) * 100:.2f}% normal.")
    
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
