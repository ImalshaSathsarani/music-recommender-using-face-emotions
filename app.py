

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from keras.models import load_model
import pywhatkit as kit

# 1. Create a simple class to act as a bridge between threads
class EmotionState:
    def __init__(self):
        self.current_emotion = "Neutral"

# Initialize the bridge once
if 'emotion_bridge' not in st.session_state:
    st.session_state['emotion_bridge'] = EmotionState()

bridge = st.session_state['emotion_bridge']

@st.cache_resource
def load_my_model():
    try:
        return load_model('model_files.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

st.title("Mood-Sync: AI Music Recommender")

class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            reshaped = np.reshape(roi_gray, (1, 48, 48, 1))

            prediction = model.predict(reshaped)
            label = labels_dict[np.argmax(prediction)]

            # 2. Update the bridge object instead of session_state
            bridge.current_emotion = label

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img

webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# 3. Use the bridge to get the LATEST value when the button is clicked
if st.button("Recommend Music based on Emotion"):
    detected_mood = bridge.current_emotion
    st.success(f"Detected Emotion: {detected_mood}. Opening YouTube...")
    kit.playonyt(f"{detected_mood} mood songs")