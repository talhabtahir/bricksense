import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model

# Streamlit app title
st.title("Brick Wall Crack Detection")

# Upload CSV file
uploaded_csv = st.file_uploader("Upload the CSV file", type=["csv"])

# Upload model file
uploaded_model = st.file_uploader("Upload the model (.keras file)", type=["keras"])

# Class dictionary built-in
class_dict = {
    0: 'Normal',
    1: 'Cracked',
    2: 'Not a Wall'
}

if uploaded_csv and uploaded_model:
    # Load the CSV data
    df_test = pd.read_csv(uploaded_csv)

    # Filter the DataFrame
    df_test = df_test[df_test['label'] == 1]

    # Upload an image file
    file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])

    if file is None:
        st.warning("Please upload a file.")
    else:
        # Read the uploaded image file
        img = Image.open(file)
        img = img.resize((224, 224))
        img = np.array(img)

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        img_tensor = np.expand_dims(img, axis=0) / 255.0
        preprocessed_img = img_tensor

        # Load the custom model
        model = tf.keras.models.load_model(uploaded_model)

        # Define a new model that outputs feature maps and prediction
        custom_model = Model(inputs=model.inputs, outputs=(model.layers[10].output, model.layers[-1].output))

        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)

        # Prediction for the image
        pred = np.argmax(pred_vec)

        # Resize the conv2d_3 output
        upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Generate heatmap
        heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        heat_map = heat_map / heat_map.max()

        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contoured_img_only = img.copy()
        cv2.drawContours(contoured_img_only, contours, -1, (0, 255, 0), 2)

        # Fetch the class name for the prediction
        predicted_class = class_dict[pred]

        # Display the image with contours and predicted class
        st.image(contoured_img_only, caption=f"Predicted Class: {predicted_class}", use_column_width=True)

        # Optionally, you can add heatmap visualization
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(heat_map, cmap='jet', alpha=0.4)
        ax.set_title("Heatmap")
        st.pyplot(fig)
