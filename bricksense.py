# Combined Streamlit App with Brick Property Prediction and Crack Detection
import streamlit as st

# Import components from both apps
from PIL import Image

# Set up page
st.set_page_config(
    page_title="Brick Sense",
    page_icon="static/brickicon8.png",
    layout="centered",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'Developed by BrickSense Team | © 2024'}
)

# Header Logo
imagelogo = Image.open("static/BSbasicboxhightran1.png")
st.image(imagelogo, use_container_width=True, width=150)

# Sidebar
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("static/BScirclehightran1.png", width=200)

st.sidebar.header("About This App")
st.sidebar.write("""This app uses AI models for two purposes:

1. **Predict Flexural Strength and Absorption** of individual bricks
2. **Detect Cracks in Brick Walls** with visual explanations

**Developed by:**  
Group 24 (Batch 213) & Group 25 (Batch 203)  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com""")

# App Selector
app_mode = st.radio("Select Task", ["Predict Brick Properties", "Detect Brick Wall Cracks"])

if app_mode == "Predict Brick Properties":
    # Insert App 1 code here (everything from App 1 except `st.set_page_config` and sidebar/header)
    import bricksense_strength_absorption  # Replace with actual inline code if preferred

elif app_mode == "Detect Brick Wall Cracks":
    # Insert full-featured crack detection code from new version (the upgraded App 2)
    import bricksense_OG  # Replace with actual inline code if preferred

# Footer
st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
    Developed with Streamlit & TensorFlow | © 2024 BrickSense
</div>
""", unsafe_allow_html=True)
