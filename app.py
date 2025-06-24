import streamlit as st
from PIL import Image
import os
from predict import predict_image

st.title("Face Race Prediction app")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_filename = "temp_uploaded_image.jpg"

    # FIX: Use PIL to open and save clean JPG — avoids dlib error!!
    image = Image.open(uploaded_file).convert("RGB")
    image.save(temp_filename, format='JPEG')

    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Predicting race...")

    race = predict_image(temp_filename)

    st.success(f"Predicted Race: {race}")

    os.remove(temp_filename)
