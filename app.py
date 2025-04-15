import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile
import shap
import matplotlib.pyplot as plt

# Suppress TensorFlow oneDNN warnings
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the model
MODEL_PATH = "model_weights.h5.keras"  # Adjust path if needed
model = load_model(MODEL_PATH)

# Define emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_RESPONSES = {
    "Angry": "It's okay to feel angry sometimes. Take a deep breath. 🧘‍♂️",
    "Disgust": "You seem displeased. Hope things get better! 🤗",
    "Fear": "Feeling afraid? You're stronger than you think. 💪",
    "Happy": "Happiness looks great on you! 😃",
    "Neutral": "Staying neutral is fine. Stay balanced! ⚖️",
    "Sad": "Feeling down? Remember, you're not alone. ❤️",
    "Surprise": "Surprised? Hope it's a good one! 🎉"
}

# Streamlit UI Setup
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")
st.title("🧠 Emotion Detection from Image or Video")
st.markdown("## Upload an Image or Video and Discover the Emotion")
st.write("This AI-powered tool helps recognize emotions from images and videos with empathy.")

# Upload file
uploaded_file = st.file_uploader("📂 Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        # Load and preprocess the image
        img_size = (56, 56)
        img = load_img(uploaded_file, target_size=img_size, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Display the uploaded image
        st.image(img, caption='🎨 Uploaded Image', use_column_width=True)
        st.write("⏳ Analyzing Emotion...")

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_emotion = EMOTION_LABELS[predicted_class]

        # Display result
        st.success(f"**Predicted Emotion: {predicted_emotion}**")
        st.write(EMOTION_RESPONSES[predicted_emotion])

        # Explainability using SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(img_array)
        plt.figure()
        shap.image_plot(shap_values)
        plt.savefig("shap_plot.png")
        st.image("shap_plot.png", caption="Feature Importance", use_column_width=True)

    elif file_type == 'video':
        # Save and open the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.write("📽️ Processing Video Frames...")

        frame_count = 0
        skip_frames = 5  # Process every 5th frame for better performance

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue  # Skip frames for faster processing

            # Preprocess the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray, (56, 56))
            img_array = img_to_array(resized_frame) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_emotion = EMOTION_LABELS[predicted_class]

            # Overlay emotion on frame
            cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")

        cap.release()
        st.success(f"Final Predicted Emotion: {predicted_emotion}")
        st.write(EMOTION_RESPONSES[predicted_emotion])

st.markdown("---")
st.write("💡 *Emotions are complex, and this AI is here to help you understand them better.* 😊")