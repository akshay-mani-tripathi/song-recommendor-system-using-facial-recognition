import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

# Load the model safely
if os.path.exists("model.h5"):
    try:
        models = load_model("model.h5", compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error("model.h5 not found.")

# Load labels safely
if os.path.exists("labels.npy"):
    labels = np.load("labels.npy")
else:
    st.error("labels.npy not found.")

# Initialize Mediapipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Session state for controlling camera streaming
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load emotion if already saved
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

# Determine whether to start camera
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Define video processor
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = labels[np.argmax(models.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# UI for user input
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Show emotion if already detected
if emotion:
    st.info(f"Detected Emotion: **{emotion}**")

# Start camera only if input provided and emotion not yet captured
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Song recommendation logic
btn = st.button("Recommend me a song")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        search_url = f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
        st.success("Click below to see your song recommendations:")
        st.markdown(f"[🎵 Open YouTube Search]({search_url})", unsafe_allow_html=True)
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
