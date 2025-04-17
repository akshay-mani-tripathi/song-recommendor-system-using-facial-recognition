import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import os

st.title("🎵 Facial Emotion-Based Music Recommender")

# Load model
if os.path.exists("model.h5"):
    try:
        models = load_model("model.h5", compile=False)
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
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Emotion state
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Emotion Processor Class
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
                lst.extend([0.0]*42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0]*42)

            lst = np.array(lst).reshape(1, -1)

            pred = labels[np.argmax(models.predict(lst))]
            print(pred)

            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# User inputs
lang = st.text_input("Enter song language (e.g., English, Hindi)")
singer = st.text_input("Enter singer name (optional)")

# Start camera and capture emotion
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(
        key="key",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        }
    )

# Recommend song
if st.button("🎧 Recommend me a song"):
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        query = f"{lang} {emotion} song {singer}"
        st.success(f"Searching for: {query}")
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
