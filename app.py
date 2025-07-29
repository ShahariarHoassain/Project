import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
from io import BytesIO

# Streamlit File Uploader for the model
st.title('Lychee Image Classifier')

# Upload model
uploaded_model = st.file_uploader("Upload the Model", type=["h5", "keras"])

if uploaded_model is not None:
    model = tf.keras.models.load_model(BytesIO(uploaded_model.read()))

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
