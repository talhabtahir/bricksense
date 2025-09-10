import streamlit as st
from PIL import Image

# Shared Sidebar and Branding
st.set_page_config(
    page_title="Brick Sense",
    page_icon="static/brickicon8.png",
    layout="centered",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'Developed by BrickSense Team | © 2024'}
)

imagelogo = Image.open("static/BSbasicboxhightran1.png")
st.image(imagelogo, use_container_width=True, width=150)
#_____________________
#Clickabale Logo
#_____________________
import base64

# Load and encode the image
with open("static/BSbasicboxhightran1.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

# Create clickable image HTML
link_url = "https://your-target-site.com"  # Replace with your actual URL
html = f"""
<a href="{link_url}" target="_blank">
    <img src="data:image/png;base64,{encoded}" width="150" style="display:block; margin:auto;" />
</a>
"""

# Display clickable logo
st.markdown(html, unsafe_allow_html=True)

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("static/BScirclehightran1.png", width=200)

st.sidebar.header("About This App")
st.sidebar.write("""This app uses AI models for two purposes:\n
1. **Predict Flexural Load and Water Absorption** of individual bricks
2. **Detect Cracks in Brick Walls** with visual explanations\n\n

**Developed by:**\n
Group 24 (Batch 213)\n
Group 25 (Batch 203)\n
Talha Bin Tahir\n
**Email:** talhabtahir@gmail.com
""")

# App Selection using dropdown
app_options = ["Select an App", "Predict Brick Properties", "Detect Brick Wall Cracks"]
app_mode = st.selectbox("Choose Application", options=app_options, index=0)

# App Execution Based on Selection
if app_mode == "Predict Brick Properties":
    exec(open("app1.py").read())

elif app_mode == "Detect Brick Wall Cracks":
    exec(open("app2.py").read())

# Common Footer
st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
    Developed with Streamlit & TensorFlow | © 2025 BrickSense
</div>
""", unsafe_allow_html=True)

#_______________________________________________________________________________________________________
# import streamlit as st

# # Shared Sidebar and Branding
# st.set_page_config(
#     page_title="Brick Sense",
#     page_icon="static/brickicon8.png",
#     layout="centered",
#     menu_items={
#         'Get Help': 'https://example.com/help',
#         'Report a bug': 'https://example.com/bug',
#         'About': 'Developed by BrickSense Team | © 2024'}
# )

# from PIL import Image

# imagelogo = Image.open("static/BSbasicboxhightran1.png")
# st.image(imagelogo, use_container_width=True, width=150)

# with st.sidebar:
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.image("static/BScirclehightran1.png", width=200)

# st.sidebar.header("About This App")
# st.sidebar.write("""This app uses AI models to predict brick strength & absorption or detect cracks in brick walls.

# **Developed by:**\n
# Group 24 (batch 213)\n
# Group 25 (Batch 203)\n
# Talha Bin Tahir\n
# **Email:** talhabtahir@gmail.com
# """)

# # App Selection
# app_mode = st.radio("Select Function", ["Predict Brick Properties", "Detect Brick Wall Cracks"])

# # App 1: Brick Property Prediction
# if app_mode == "Predict Brick Properties":
#     # Paste full App 1 code below from your existing app1 file
#     # Example:
#     # import bricksense_strength_absorption
#     # OR paste its full code inline here
#     # For now, placeholder:
#     exec(open("app1.py").read())

# # App 2: Brick Wall Crack Detection
# elif app_mode == "Detect Brick Wall Cracks":
#     # Paste full App 2 code below (with sensitivity, contour, overlay, slider etc.)
#     # For now, placeholder:
#     exec(open("app2.py").read())

# # Common Footer
# st.markdown("""
# <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: gray; text-align: center; font-size: small; padding: 10px;">
#     Developed with Streamlit & TensorFlow | © 2024 BrickSense
# </div>
# """, unsafe_allow_html=True)
