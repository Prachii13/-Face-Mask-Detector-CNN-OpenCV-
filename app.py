import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector.model")

st.title("😷 Face Mask Detector")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0][0]
    st.subheader("Prediction: " + ("Mask 😷" if pred < 0.5 else "No Mask 😐"))
