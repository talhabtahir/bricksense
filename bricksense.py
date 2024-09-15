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
                             outputs=(model.layers[8].output, model.layers[-1].output))  # `conv2d_3` and predictions
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

        # Resize heatmap to the size of the original image
        heatmap_resized = cv2.resize(heat_map, original_size, interpolation=cv2.INTER_LINEAR)

        # Apply colormap to the heatmap for better visualization
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Convert original image to numpy array (for blending)
        original_img_np = np.array(image_data)

        # Resize the original image to match the input image shape (224x224) for blending
        original_resized = cv2.resize(original_img_np, size, interpolation=cv2.INTER_LINEAR)

        # Overlay the heatmap onto the resized image
        overlay_img_resized = cv2.addWeighted(cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)

        # Now, resize the overlaid image back to its original size
        overlay_img_original_size = cv2.resize(overlay_img_resized, original_size, interpolation=cv2.INTER_LINEAR)

        # Convert back to RGB for display in Streamlit
        overlay_img_rgb = cv2.cvtColor(overlay_img_original_size, cv2.COLOR_BGR2RGB)
        
        # Convert to a PIL Image for display in Streamlit
        overlay_pil = Image.fromarray(overlay_img_rgb)

        # Threshold the heatmap to get regions of interest
        _, thresh_map = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the resized image
        cv2.drawContours(overlay_img_rgb, contours, -1, (0, 255, 0), 2)  # Green contours

        # Create a figure to display the results
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size for better clarity
        ax.imshow(overlay_img_rgb)
        ax.axis('off')  # Hide the axes for a cleaner visualization
        
        return pred_vec, fig
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None
