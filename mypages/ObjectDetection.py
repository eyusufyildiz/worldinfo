import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
from PIL import Image
import torch
import time

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(layout="wide")
st.title("📺 YouTube Object Detection (Streamlit Cloud)")

# ----------------------------
# User Inputs
# ----------------------------
youtube_url = st.text_input(
    "YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM"
)
resolution = st.selectbox(
    "Select video resolution",
    ["144p", "240p", "360p", "480p", "720p", "1080p"],
    index=4
)
confidence = st.slider("Detection confidence", 0.1, 0.9, 0.4)
start = st.button("▶ Start Detection")

# ----------------------------
# Load YOLO model
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # lightweight & fast
    if torch.cuda.is_available():
        model.to("cuda")
        model.fuse()
    return model

model = load_model()

# ----------------------------
# Get YouTube stream URL
# ----------------------------
def get_stream_url(youtube_url, resolution):
    """Return direct stream URL using yt-dlp without printing logs."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f",
                f"bestvideo[height<={resolution.replace('p','')}]+bestaudio/best",
                "-g",
                youtube_url
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        st.error(
            "Failed to get stream URL. Make sure yt-dlp and ffmpeg are installed "
            "and the video is accessible."
        )
        return None

# ----------------------------
# Run object detection
# ----------------------------
if start and youtube_url:
    stream_url = get_stream_url(youtube_url, resolution)
    if stream_url:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            st.error("Failed to open video stream. Check the URL or resolution.")
        else:
            st.success("Stream opened! Running object detection...")

            frame_placeholder = st.empty()
            frame_skip = 2
            frame_count = 0

            # Pause / Resume button
            paused = False
            pause_button = st.button("⏸ Pause / Resume")

            while cap.isOpened():
                if pause_button:
                    paused = not paused

                if paused:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Resize frame for speed
                frame = cv2.resize(frame, (1280, 768))

                # YOLO detection
                try:
                    results = model(frame, conf=confidence)
                    annotated = results[0].plot()
                except Exception:
                    continue  # skip frame silently

                # Convert BGR → RGB for Streamlit display
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Display frame in Streamlit
                frame_placeholder.image(
                    annotated,
                    channels="RGB",
                    width=800  # use width instead of deprecated use_column_width
                )

                time.sleep(0.03)  # smooth playback

            cap.release()
            st.success("✅ Detection completed.")
