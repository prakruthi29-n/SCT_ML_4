import streamlit as st

try:
    import cv2
except:
    cv2 = None

import mediapipe as mp

mp_hands = mp.solutions.hands
from mediapipe.python.solutions import drawing_utils as mp_draw
import numpy as np
import joblib
import tempfile
import pandas as pd
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 18px;
    color: #B0B0B0;
}

.block-container {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.markdown("""
<div style='padding-top:10px; padding-bottom:20px;'>

<h1 style='
color:white;
font-size:48px;
font-weight:700;
margin-bottom:10px;
'>
🖐 Hand Gesture Recognition System
</h1>

<p style='
color:#B0B0B0;
font-size:18px;
margin-top:-10px;
'>
Professional ML-based real-time gesture recognition using OpenCV, MediaPipe & Streamlit
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------
# CLOUD WARNING
# --------------------------------
if cv2 is None:
    st.warning(
        "Live camera features are limited on Streamlit Cloud deployment."
    )

# --------------------------------
# LOAD MODEL
# --------------------------------
model = joblib.load("gesture_model.pkl")

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("⚙ Control Panel")

app_mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Live Camera", "Upload Video"]
)

st.sidebar.markdown("---")
st.sidebar.success("Model Loaded Successfully")

# --------------------------------
# SESSION STATE
# --------------------------------
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# --------------------------------
# GESTURE NAMES
# --------------------------------
gesture_names = {
    "01_palm": "Palm",
    "02_l": "L Sign",
    "03_fist": "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb": "Thumbs Up",
    "06_index": "Index",
    "07_ok": "OK",
    "08_palm_moved": "Palm Moved",
    "09_c": "C Sign",
    "10_down": "Down"
}

# --------------------------------
# MEDIAPIPE
# --------------------------------
if cv2 is not None:

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# --------------------------------
# FRAME PROCESS FUNCTION
# --------------------------------
def process_frame(frame):

    if cv2 is None:
        return frame, "OpenCV Not Available"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    detected_gesture = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(gray, (64, 64))

            flattened = resized.flatten().reshape(1, -1)

            prediction = model.predict(flattened)[0]

            detected_gesture = gesture_names.get(
                prediction,
                prediction
            )

            cv2.putText(
                frame,
                f"Gesture: {detected_gesture}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            st.session_state.detection_history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Gesture": detected_gesture
            })

    return frame, detected_gesture

# --------------------------------
# MAIN LAYOUT
# --------------------------------
col1, col2 = st.columns([3, 1])

# --------------------------------
# RIGHT PANEL
# --------------------------------
with col2:

    st.subheader("📊 Detection Info")

    total = len(st.session_state.detection_history)

    last_gesture = "None"

    if total > 0:
        last_gesture = st.session_state.detection_history[-1]["Gesture"]

    st.metric("Total Detections", total)

    st.metric("Last Gesture", last_gesture)

    st.markdown("---")

    st.subheader("📝 Detection History")

    if total > 0:

        history_df = pd.DataFrame(
            st.session_state.detection_history
        )

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

# --------------------------------
# LIVE CAMERA MODE
# --------------------------------
with col1:

    if app_mode == "Live Camera":

        if cv2 is None:

            st.warning(
                "Live Camera works best in local VS Code environment."
            )

        else:

            run = st.checkbox("Start Camera")

            FRAME_WINDOW = st.image([])

            if run:

                camera = cv2.VideoCapture(0)

                while run:

                    success, frame = camera.read()

                    if not success:
                        st.error("Camera not detected")
                        break

                    frame = cv2.flip(frame, 1)

                    processed_frame, gesture = process_frame(frame)

                    FRAME_WINDOW.image(
                        cv2.cvtColor(
                            processed_frame,
                            cv2.COLOR_BGR2RGB
                        )
                    )

                camera.release()

# --------------------------------
# VIDEO UPLOAD MODE
# --------------------------------
    elif app_mode == "Upload Video":

        if cv2 is None:

            st.warning(
                "Video processing is limited in cloud deployment."
            )

        else:

            uploaded_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "mov", "avi"]
            )

            if uploaded_file is not None:

                tfile = tempfile.NamedTemporaryFile(delete=False)

                tfile.write(uploaded_file.read())

                cap = cv2.VideoCapture(tfile.name)

                FRAME_WINDOW = st.image([])

                st.info("Processing Video...")

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    processed_frame, gesture = process_frame(frame)

                    FRAME_WINDOW.image(
                        cv2.cvtColor(
                            processed_frame,
                            cv2.COLOR_BGR2RGB
                        )
                    )

                cap.release()

                st.success("Video Processing Completed")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")

st.caption(
    "Developed by Prakruthi N | SkillCraft Technology ML Internship"
)