import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# Load the model
model = tf.keras.models.load_model('NASNetMobile.keras')

# App Title
st.title('Lychee Image Classifier')

# Custom Styling with CSS and Markdown for Modern, Clean UI
st.markdown("""
    <style>
        /* Global Styles */
        body {
            background-color: #f4f7fc;
            font-family: 'Arial', sans-serif;
        }

        /* Header Styling */
        .header {
            text-align: center;
            font-size: 36px;
            color: #2d3e50;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Subtitle Styling */
        .subheader {
            text-align: center;
            font-size: 18px;
            color: #4CAF50;
            margin-bottom: 40px;
        }

        /* Image Container Styling */
        .image-container {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Results Container Styling */
        .result-container {
            padding: 15px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            border: 2px solid #e0e0e0;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Slider Styling */
        .stSlider>div>label {
            font-size: 16px;
            color: #333;
        }

        .stSlider>div>div {
            margin-top: 10px;
        }
        
        /* Layout: Center all the contents in columns */
        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-top: 50px;
        }
    </style>
    """, unsafe_allow_html=True)

# Layout: Using columns for better organization
col1, col2 = st.columns([2, 1])

with col1:
    # Upload Images (multiple files allowed)
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

with col2:
    st.markdown("<div class='header'>Lychee Image Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Upload images to classify them as different lychee varieties.</div>", unsafe_allow_html=True)

# Slider for image resizing
image_size = st.slider("Resize Image to (px)", min_value=100, max_value=500, value=224, step=10)

# If images are uploaded, process and predict
if uploaded_images:
    for uploaded_image in uploaded_images:
        # Open the image file
        img = Image.open(uploaded_image)
        
        with col1:
            st.markdown(f"### Image: {uploaded_image.name}")
            st.image(img, caption='Uploaded Image', use_column_width=True)

        # Resize image as per user-selected size
        img = img.resize((image_size, image_size))  # Resize to match the model's input
        img_array = np.array(img) / 255.0  # Normalize the image

        # Expand dimensions to match the model's input shape
        img_array = np.expand_dims(img_array, axis=0)

        # Display progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)  # Simulate a slow prediction
            progress_bar.progress(i + 1)

        # Make the prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)

        # Confidence score (probability)
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Class labels (adjust according to your dataset)
        labels = ['Algal_Spot_Indirect', 'Anthracnose_Cloudy', 'Dry_Leaves', 'Entomosporium_Spot', 'Leaf_Mites_Direct', 'Mayetiola_PostRain']
        predicted_class = labels[class_idx[0]]

        # Display the results in a cleaner format
        with col2:
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.write("Please upload images to make predictions!")
