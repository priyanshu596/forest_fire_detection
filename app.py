import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt

# Load your trained RandomForest model
clf = joblib.load('random_forest_model.pkl')

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False

class_names = ['Fire', 'No Fire']

def predict_fire(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    prediction = clf.predict(features)
    return class_names[int(prediction[0])]

st.title("Wildfire Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = predict_fire("temp.jpg")
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{result}**")
