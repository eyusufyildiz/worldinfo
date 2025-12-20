import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

st.set_page_config(layout="wide")
st.title("📺 YouTube Live Object Detection (Streamlit Cloud)")

# ----------------------------
# Streamlit UI
# ----------------------------
youtube_url = st.text_input(
    "YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM"
)
confidence = st.slider("Detection confidence", 0.1, 0.9, 0.4)
start = st.button("▶ Start Detection")

# ----------------------------
# Load YOLO Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Lightweight & fast
    if torch.cuda.is_available():
        model.to("cuda")
        model.fuse()
    return model

import torch
model = load_model()

# ----------------------------
# Get stream URL
# ----------------------------
def get_stream_url(youtube_url):
    """Get direct stream URL from YouTube using yt-dlp"""
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4]", "-g", youtube_url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        st.error("Failed to get stream URL. Make sure yt-dlp and ffmpeg are installed.")
        return None

# ----------------------------
# Run Detection
# ----------------------------
if start and youtube_url:
    stream_url = get_stream_url(youtube_url)
    if stream_url:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            st.error("Failed to open video stream.")
        else:
            frame_placeholder = st.empty()
            frame_skip = 2
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Resize for speed
                frame = cv2.resize(frame, (1280, 768))

                # Run detection
                results = model(frame, conf=confidence)
                annotated = results[0].plot()

                # Convert BGR → RGB for Streamlit
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    annotated,
                    channels="RGB",
                    use_column_width=True
                )

                time.sleep(0.03)  # smooth playback

            cap.release()
            st.success("Detection completed ✔")
