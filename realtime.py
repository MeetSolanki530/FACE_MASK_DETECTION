from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array,load_img
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon = ":brain:",
    initial_sidebar_state = 'auto'
)

with st.sidebar:
        st.subheader("Below are sample images With Detection result")
        st.title("With FaceMask")
        st.image(r'C:\Users\solan\Downloads\face-mask-detection\Sample\test.jpeg')
        st.title("Without FaceMask")
        st.image(r'C:\Users\solan\Downloads\face-mask-detection\Sample\download.jpeg')
        st.subheader("Below are sample video With Detection result")
        st.title("With FaceMask")
        st.video(r'C:\Users\solan\Downloads\face-mask-detection\Sample\test.mp4')
        

model = load_model('best_model.h5')

height,width = 150,150

def preprocess_image(image):
    image = load_img(image,target_size=(height,width))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image,axis=0)
    return image





def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    label = ["With Mask","Without Mask"]
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    predictions = model.predict(frames)

    # Aggregate predictions over frames (e.g., averaging or voting)
    aggregated_prediction = np.mean(predictions, axis=0)  # You can use other aggregation techniques based on your requirements
    predicted_label = np.argmax(aggregated_prediction)

    return label[predicted_label]



def main():
    # Streamlit app
    st.title("Face Mask Detection")
    st.write("Upload an image to check if a person is wearing a mask or not.")
    upload_option = st.radio("Choose upload option:", ("Image", "Video"))
    if upload_option == "Image":
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
    
    elif upload_option == "Video":
        video = st.file_uploader("Upload a video...", type=["mp4"])
        if st.button('Predict'):
            if video is not None:
                video_path = 'temp_video.mp4'
                with open(video_path, "wb") as f:
                    f.write(video.getbuffer())

                st.video(video)
                prediction = predict_video(video_path)
                st.write("Predicted:", prediction)
                

if __name__ == "__main__":
    main()


    
