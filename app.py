import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
import requests
import geopy.distance

# Load trained model
model_path = "final_cancer_stage_model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

# Preprocessing function
def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Hospital data (mockup, replace with real database or API)
hospitals = [
    {"name": "Oral Cancer Center A", "location": (19.0760, 72.8777)},  # Example coordinates (Mumbai)
    {"name": "Oral Cancer Center B", "location": (28.7041, 77.1025)},  # Example coordinates (Delhi)
]

# Function to find the nearest hospital
def find_nearest_hospital(user_location):
    nearest_hospital = min(hospitals, key=lambda h: geopy.distance.distance(user_location, h["location"]).km)
    return nearest_hospital

# Streamlit UI
st.title("Oral Cancer Stage Prediction")

uploaded_file = st.file_uploader("Upload an image of the oral cavity", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict cancer stage
    input_tensor = preprocess_image(image)
    predictions = model.predict(input_tensor)
    predicted_stage = class_labels[np.argmax(predictions)]

    st.write(f"**Predicted Cancer Stage**: {predicted_stage}")

    # Geolocation for nearby hospitals
    user_lat = st.number_input("Enter your latitude", format="%.6f")
    user_lon = st.number_input("Enter your longitude", format="%.6f")

    if st.button("Find Nearest Hospital"):
        if user_lat and user_lon:
            nearest_hospital = find_nearest_hospital((user_lat, user_lon))
            st.write(f"Nearest Hospital: **{nearest_hospital['name']}**")
            st.write(f"Location: {nearest_hospital['location']}")
        else:
            st.write("Please provide valid latitude and longitude.")
