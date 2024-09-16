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
            predictions, image_with_border, contours_with_border = import_and_predict(image, model)
            
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                # Display prediction result
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall.")
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
                
                # Conditionally display image comparison
                if predicted_class == 1:
                    image_comparison(
                        img1=image_with_border, 
                        img2=contours_with_border,
                        label1="Uploaded Image",
                        label2="Cracks Localization",
                        show_labels=True
                    )
                else:
                    st.image(image_with_border, caption="Uploaded Image with Border")

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
