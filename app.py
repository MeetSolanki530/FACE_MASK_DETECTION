from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array,load_img
import numpy as np
import streamlit as st
import tensorflow as tf

model = load_model('best_model.h5')

height,width = 150,150

def preprocess_image(image):
    image = load_img(image,target_size=(height,width))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image,axis=0)
    return image


# Streamlit app
st.title("Face Mask Detection")
st.write("Upload an image to check if a person is wearing a mask or not.")


image = st.file_uploader('Enter Image:- ',type=['jpg','jpeg','png'])


if st.button('Predict'):
    if image is not None:
        st.image(image,caption='Uploaded Image',use_column_width=True)
        image = preprocess_image(image)
        
        # Predict using the model
        prediction = model.predict(image)
        
        # Determine the class label
        if prediction[0] > 0.5:
            label = "Without Mask"
            confidence = prediction[0][0] * 100
        else:
            label = "With Mask"
            confidence = (1 - prediction[0][0]) * 100
        
        # Display the prediction result
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

