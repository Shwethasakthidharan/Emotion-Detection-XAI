import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile
import shap
import matplotlib.pyplot as plt
import json
import os

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the model
MODEL_PATH = "model_weights.h5.keras"  # Adjust path if needed
model = load_model(MODEL_PATH)

# Define emotion labels and responses
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_RESPONSES = {
    "Angry": "It's okay to feel angry sometimes. Take a deep breath! 🧘‍♂️",
    "Disgust": "Something bothering you? Let’s turn it around! 🤗",
    "Fear": "Feeling scared? You’re tougher than you know! 💪",
    "Happy": "You’re glowing with happiness! Keep it up! 😃",
    "Neutral": "Cool and calm—perfectly balanced! ⚖️",
    "Sad": "Feeling low? You’re not alone—we’re here! ❤️",
    "Surprise": "Wow, what a surprise! Hope it’s a good one! 🎉"
}

# Theme-specific configurations
THEME_CONFIG = {
    "Classroom": {
        "roles": ["Student", "Teacher"],
        "welcome_msg": "Welcome to Classroom Emotion Insights!",
        "login_file": "classroom_users.json",
        "icon": "https://cdn-icons-png.flaticon.com/512/167/167756.png"
    },
    "Office": {
        "roles": ["Employee", "Manager"],
        "welcome_msg": "Welcome to Office Emotion Analytics!",
        "login_file": "office_users.json",
        "icon": "https://cdn-icons-png.flaticon.com/512/1251/1251266.png"
    },
    "Retail": {
        "roles": ["Staff", "Customer"],
        "welcome_msg": "Welcome to Retail Emotion Experience!",
        "login_file": "retail_users.json",
        "icon": "https://cdn-icons-png.flaticon.com/512/4290/4290854.png"
    }
}

# Initialize user data
def initialize_user_data(theme):
    sample_data = {
        "Classroom": {"Student": {"student1": "pass123"}, "Teacher": {"teacher1": "teach456"}},
        "Office": {"Employee": {"emp1": "work789"}, "Manager": {"mgr1": "lead101"}},
        "Retail": {"Staff": {"staff1": "shop202"}, "Customer": {"cust1": "buy303"}}
    }
    filename = THEME_CONFIG[theme]["login_file"]
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(sample_data[theme], f)

# Login and Signup functions
def check_login(theme, role, username, password):
    filename = THEME_CONFIG[theme]["login_file"]
    try:
        with open(filename, "r") as f:
            users = json.load(f)
        return username in users[role] and users[role][username] == password
    except:
        return False

def signup_user(theme, role, username, password):
    filename = THEME_CONFIG[theme]["login_file"]
    try:
        with open(filename, "r") as f:
            users = json.load(f)
    except:
        users = {r: {} for r in THEME_CONFIG[theme]["roles"]}
    
    if username in users[role]:
        return False
    users[role][username] = password
    with open(filename, "w") as f:
        json.dump(users, f)
    return True

# Function to process frame and detect emotion
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (56, 56))
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_emotion = EMOTION_LABELS[predicted_class]

    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, predicted_emotion

# Streamlit UI Setup
st.set_page_config(page_title="Emotion Insights", page_icon="😊", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>div>input {border-radius: 5px;}
    .stSelectbox {background-color: #ffffff; border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201614.png", width=100)
    st.title("Emotion Insights")
    theme = st.selectbox("Choose Your Environment", list(THEME_CONFIG.keys()))
    initialize_user_data(theme)
    st.image(THEME_CONFIG[theme]["icon"], width=50)
    role = st.selectbox("Select Your Role", THEME_CONFIG[theme]["roles"])

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "recording" not in st.session_state:
    st.session_state.recording = False

# Main Content
st.title(THEME_CONFIG[theme]["welcome_msg"])
st.markdown("Discover emotions with AI-powered insights—upload files or use your camera!")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if check_login(theme, role, login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(f"Welcome back, {login_username} ({role})!")
            else:
                st.error("Invalid credentials. Try again or sign up!")

    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("New Username", key="signup_user")
        signup_password = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up", key="signup_btn"):
            if signup_user(theme, role, signup_username, signup_password):
                st.success(f"Account created for {signup_username} ({role})! Please log in.")
            else:
                st.error("Username already exists. Try a different one!")
else:
    st.markdown(f"### Hello, {st.session_state.username} ({role})!")
    input_method = st.radio("Choose Input Method", ["Upload File", "Live Camera"])

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("📸 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], key="uploader")
        if uploaded_file is not None:
            file_type = uploaded_file.type.split('/')[0]

            if file_type == 'image':
                img_size = (56, 56)
                img = load_img(uploaded_file, target_size=img_size, color_mode="grayscale")
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption='Uploaded Image', use_column_width=True)
                with col2:
                    st.write("⏳ Analyzing Emotion...")
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions)
                    predicted_emotion = EMOTION_LABELS[predicted_class]
                    st.success(f"**Emotion Detected: {predicted_emotion}**")
                    st.write(EMOTION_RESPONSES[predicted_emotion])

                with st.expander("See What the AI Sees"):
                    # Fix: Provide a masker for SHAP
                    masker = shap.maskers.Image("inpaint_telea", img_array[0].shape)
                    explainer = shap.Explainer(model, masker=masker)
                    shap_values = explainer(img_array)
                    plt.figure()
                    shap.image_plot(shap_values)
                    plt.savefig("shap_plot.png")
                    st.image("shap_plot.png", caption="Feature Importance", use_column_width=True)

            elif file_type == 'video':
                st.write("📽️ Streaming Uploaded Video with Emotion Detection...")
                stframe = st.empty()
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)

                frame_count = 0
                skip_frames = 5
                last_emotion = "Neutral"

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue

                    frame, predicted_emotion = process_frame(frame)
                    last_emotion = predicted_emotion
                    stframe.image(frame, channels="BGR", use_column_width=True)

                cap.release()
                os.unlink(tfile.name)
                st.success(f"Final Emotion: {last_emotion}")
                st.write(EMOTION_RESPONSES[last_emotion])

    elif input_method == "Live Camera":
        st.write("📷 Live Emotion Detection via Webcam")
        stframe = st.empty()
        start_stop = st.button("Start Recording" if not st.session_state.recording else "Stop Recording")

        if start_stop:
            st.session_state.recording = not st.session_state.recording

        if st.session_state.recording:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access the webcam. Please ensure it’s connected and accessible.")
                st.session_state.recording = False
            else:
                last_emotion = "Neutral"
                while st.session_state.recording:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, predicted_emotion = process_frame(frame)
                    last_emotion = predicted_emotion
                    stframe.image(frame, channels="BGR", use_column_width=True)

                    st.session_state.last_emotion = last_emotion
                    st.success(f"**Current Emotion: {last_emotion}**")
                    st.write(EMOTION_RESPONSES[last_emotion])

                    if "recording" in st.session_state and not st.session_state.recording:
                        break

                cap.release()
                st.success(f"Recording Stopped. Last Emotion: {last_emotion}")
                st.write(EMOTION_RESPONSES[last_emotion])
        else:
            if "last_emotion" in st.session_state:
                st.success(f"**Last Emotion: {st.session_state.last_emotion}**")
                st.write(EMOTION_RESPONSES[st.session_state.last_emotion])

    st.markdown("---")
    if st.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        # Reset the app state by clearing the session state
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

st.markdown(f"<p style='text-align: center;'>💡 <i>Empowering {theme} with Emotional Intelligence</i> 😊</p>", unsafe_allow_html=True)