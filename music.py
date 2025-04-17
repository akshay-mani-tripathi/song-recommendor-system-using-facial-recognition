import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import webbrowser

st.title("🎵 Emotion-based Music Recommender")

# Load the model
if os.path.exists("model.h5"):
    try:
        model = load_model("model.h5", compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.error("model.h5 not found.")
    st.stop()

# Load labels
if os.path.exists("labels.npy"):
    labels = np.load("labels.npy")
else:
    st.error("labels.npy not found.")
    st.stop()

# Mediapipe setup
holistic = mp.solutions.holistic.Holistic(static_image_mode=True)
drawing = mp.solutions.drawing_utils

# Inputs
lang = st.text_input("Enter Language (e.g. English, Hindi)")
singer = st.text_input("Enter Singer (optional)")
uploaded_image = st.file_uploader("Upload a clear selfie image", type=["jpg", "jpeg", "png"])

emotion = ""

# Emotion detection
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.flip(img, 1)
    res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    features = []

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            features.append(i.x - res.face_landmarks.landmark[1].x)
            features.append(i.y - res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                features.append(i.x - res.left_hand_landmarks.landmark[8].x)
                features.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            features.extend([0.0]*42)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                features.append(i.x - res.right_hand_landmarks.landmark[8].x)
                features.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            features.extend([0.0]*42)

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        emotion = labels[np.argmax(prediction)]
        st.success(f"Detected Emotion: **{emotion}**")

        # Optionally show image with landmarks
        drawing.draw_landmarks(img, res.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(img, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(img, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        st.image(img, channels="BGR", caption="Detected Landmarks")
    else:
        st.warning("Face not detected properly. Please upload a clear selfie.")

# Recommend
if st.button("🎧 Recommend me a song"):
    if emotion:
        query = f"{lang} {emotion} song {singer}"
        st.markdown(f"🔗 [Search on YouTube](https://www.youtube.com/results?search_query={query})")
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    else:
        st.warning("Please upload a valid image first.")
