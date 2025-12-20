import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import time

DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=ztmY_cCtUl0"
DEFAULT_RESOLUTION = (854, 480)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


def get_stream_url(youtube_url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def main():
    st.title("🎥 YouTube Object Detection (YOLOv8)")

    url = st.text_input("YouTube URL", DEFAULT_YOUTUBE_URL)
    confidence = st.slider("Confidence", 0.1, 0.9, 0.5)
    start = st.button("Start")
    stop = st.button("Stop")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    if not st.session_state.run:
        st.info("Click Start")
        return

    model = load_model()
    cap = cv2.VideoCapture(get_stream_url(url))

    frame_box = st.empty()
    prev = time.time()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, device="cpu", verbose=False)
        frame = results[0].plot()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_box.image(frame, use_container_width=True)

        fps = 1 / (time.time() - prev)
        prev = time.time()
        st.caption(f"FPS: {fps:.2f}")

    cap.release()
