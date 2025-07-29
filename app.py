import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

# Load the model
model = tf.keras.models.load_model('NASNetMobile.keras')

st.title('Lychee Image Classifier')

# Upload Image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the image file
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((224, 224))  # Resize the image to match the input shape of your model
    img_array = np.array(img) / 255.0  # Normalize the image

    # If your model takes 4D input (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)

    # You should have a dictionary for your class labels
    labels = ['Anthracnose_Cloudy', 'Anthracnose_Cloudy', 'Dry_Leaves', 'Entomosporium_Spot', 'Leaf_Mites_Direct', 'Mayetiola_PostRain']  # Update as per your model
    predicted_class = labels[class_idx[0]]

    st.write(f"Prediction: {predicted_class}")
