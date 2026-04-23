# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import cv2
# import numpy as np
# from keras.models import load_model
# import os
# import av
# from huggingface_hub import hf_hub_download
# from twilio.rest import Client
# import os


# account_sid = os.getenv("TWILIO_ACCOUNT_SID")
# auth_token = os.getenv("TWILIO_AUTH_TOKEN")

# client = Client(account_sid, auth_token)
# token = client.tokens.create()

# RTC_CONFIGURATION = {
#     "iceServers": token.ice_servers
# }

# # RTC_CONFIGURATION = RTCConfiguration(
# #     {"iceServers": [
# #         {"urls": ["stun:stun.l.google.com:19302"]},
# #         {"urls": ["stun:stun1.l.google.com:19302"]},
# #         {"urls": ["stun:stun2.l.google.com:19302"]},
# #         {"urls": ["stun:stun3.l.google.com:19302"]},
# #         {"urls": ["stun:stun4.l.google.com:19302"]}
        
# #         ]}
# # )

# # --- 1. SETTINGS & CSS FOR VISUAL APPEAL ---
# st.set_page_config(page_title="Mood-Sync AI", page_icon="🎵", layout="centered")

# st.markdown("""
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 20px;
#         height: 3em;
#         background-color: #2e3136;
#         color: white;
#         border: 1px solid #4e5d6c;
#         transition: 0.3s;
#     }
#     .stButton>button:hover {
#         background-color: #ff4b4b;
#         border: 1px solid #ff4b4b;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # 1. Create a simple class to act as a bridge between threads
# class EmotionState:
#     def __init__(self):
#         self.current_emotion = "Neutral"

# # Initialize the bridge once
# if 'emotion_bridge' not in st.session_state:
#     st.session_state['emotion_bridge'] = EmotionState()

# bridge = st.session_state['emotion_bridge']

# @st.cache_resource
# def load_my_model():
#     try:
#        model_path = hf_hub_download(
#            repo_id="Imalsha123/emotion-model",
#            filename="model_files.keras"

#        )
#        return load_model(model_path)
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_my_model()
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# # --- ADDITIONAL FUNCTION: Fetch local music ---
# def get_local_songs(emotion):
#     base_path = "songs"
#     folder_name = emotion.lower()
#     if folder_name == "surprise" and not os.path.exists(os.path.join(base_path, "surprise")):
#         if os.path.exists(os.path.join(base_path, "suprise")):
#             folder_name = "suprise"
    
#     emotion_path = os.path.join(base_path, folder_name)
#     if os.path.exists(emotion_path):
#         songs = [f for f in os.listdir(emotion_path) if f.endswith(('.mp3', '.wav'))]
#         return songs, emotion_path
#     return [], None

# # --- UI HEADER ---
# st.title("🎵 Mood-Sync AI")
# st.write("Detect your mood and choose your music source below.")

# class EmotionProcessor(VideoProcessorBase):
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48))
#             roi_gray = roi_gray / 255.0
#             reshaped = np.reshape(roi_gray, (1, 48, 48, 1))

#             prediction = model.predict(reshaped)
#             label = labels_dict[np.argmax(prediction)]
#             bridge.current_emotion = label

#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 75, 75), 2)
#             cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # Camera Feed
# webrtc_streamer(
#     key="emotion-detection",
#     rtc_configuration=RTC_CONFIGURATION,
#     video_processor_factory=EmotionProcessor,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )

# st.markdown("### 🎧 Where should we play from?")

# # --- BUTTONS IN COLUMNS (No Sidebar) ---
# col1, col2 = st.columns(2)

# with col1:
#     if st.button("🚀 Search on YouTube"):
#         detected_mood = bridge.current_emotion
#         st.success(f"Mood: {detected_mood} | Redirecting to YouTube...")

#         search_query = f"{detected_mood}+mood+songs"
#         yt_url = f"https://www.youtube.com/results?search_query={search_query}"
#         st.link_button("Open YouTube", yt_url)
#        # kit.playonyt(f"{detected_mood} mood songs")

# with col2:
#     if st.button("🏠 Play from My Platform"):
#         detected_mood = bridge.current_emotion
#         st.info(f"Mood: {detected_mood} | Loading your collection...")
        
#         local_songs, folder_path = get_local_songs(detected_mood)
        
#         if local_songs:
#             st.markdown("---")
#             for song in local_songs:
#                 with st.expander(f"🎵 {song}"):
#                     try:
#                         audio_path = os.path.join(folder_path, song)
#                         audio_file = open(audio_path, 'rb')
#                         audio_bytes = audio_file.read()
#                         st.audio(audio_bytes, format='audio/mp3')
#                     except Exception as e:
#                         st.error(f"Error: {e}")
#         else:
#             st.warning(f"No local tracks found for '{detected_mood}'.")

# # --- FOOTER ---
# # st.markdown("<br><center><small>Powered by Deep Learning & Streamlit</small></center>", unsafe_allow_html=True)

import gradio as gr
import cv2
import numpy as np
from keras.models import load_model
import os
import webbrowser
from huggingface_hub import hf_hub_download

# -----------------------------
# LOAD MODEL (HF or local)
# -----------------------------
model_path = hf_hub_download(
    repo_id="YOUR_HF_USERNAME/YOUR_MODEL_REPO",
    filename="model_files.keras"
)

model = load_model(model_path)

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# -----------------------------
# UI STYLE (CUSTOM CSS)
# -----------------------------
custom_css = """
body {
    background-color: #0f1117;
    color: white;
}
h1 {
    text-align: center;
    color: #ff4b4b;
}
.gr-button {
    background-color: #ff4b4b !important;
    color: white !important;
    border-radius: 10px !important;
}
"""

# -----------------------------
# GET LOCAL SONGS
# -----------------------------
def get_local_songs(emotion):
    base_path = "songs"
    folder = emotion.lower()

    path = os.path.join(base_path, folder)

    if not os.path.exists(path):
        return []

    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith((".mp3", ".wav"))]


# -----------------------------
# YOUTUBE SEARCH
# -----------------------------
def open_youtube(emotion):
    query = f"https://www.youtube.com/results?search_query={emotion}+mood+songs"
    webbrowser.open(query)
    return f"Opened YouTube for {emotion} songs 🎵"


# -----------------------------
# EMOTION DETECTION FUNCTION
# -----------------------------
current_emotion = "Neutral"

def detect_emotion(frame):

    global current_emotion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_text = "No Face Detected"

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
        label = labels_dict[np.argmax(prediction)]

        current_emotion = label
        emotion_text = label

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, emotion_text


# -----------------------------
# SHOW LOCAL MUSIC
# -----------------------------
def show_music():
    songs = get_local_songs(current_emotion)

    if not songs:
        return "No local songs found for this mood ❌"

    return "\n".join([f"🎵 {os.path.basename(s)}" for s in songs])


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🎵 Mood Sync AI")
    gr.Markdown("### Real-time Emotion Detection + Music Recommendation System")

    with gr.Row():

        with gr.Column():
            webcam = gr.Image(source="webcam", streaming=True)
            emotion_box = gr.Textbox(label="Detected Emotion")

        with gr.Column():
            output_frame = gr.Image()

            youtube_btn = gr.Button("🔎 Open YouTube Songs")
            youtube_status = gr.Textbox(label="YouTube Status")

            local_btn = gr.Button("🎧 Show Local Songs")
            local_output = gr.Textbox(label="Local Playlist")

    # -----------------------------
    # LIVE STREAM
    # -----------------------------
    webcam.stream(
        fn=detect_emotion,
        inputs=webcam,
        outputs=[output_frame, emotion_box]
    )

    # -----------------------------
    # BUTTON ACTIONS
    # -----------------------------
    youtube_btn.click(
        fn=open_youtube,
        inputs=[],
        outputs=youtube_status
    )

    local_btn.click(
        fn=show_music,
        inputs=[],
        outputs=local_output
    )


# -----------------------------
# RUN APP
# -----------------------------
app.launch()