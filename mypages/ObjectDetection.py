import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
from PIL import Image
import torch
import time
import random

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(layout="wide")
st.title("📺 YouTube Object Detection (Streamlit Cloud)")

# ----------------------------
# User Inputs
# ----------------------------
safe_mode = st.checkbox("Enable Safe Mode (Anti-Ban)", value=True)

youtube_url = st.text_input(
    "YouTube URL", "https://www.youtube.com/watch?v=ztmY_cCtUl0"
)
resolution = st.selectbox(
    "Select video resolution",
    ["144p", "240p", "360p", "480p", "720p", "1080p"],
    index=4,
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
def get_stream_url(youtube_url, resolution, safe_mode=True):
    """Return direct stream URL using yt-dlp without printing logs."""
    cache_key = f"{youtube_url}_{resolution}"
    if cache_key in st.session_state:
        st.info("Using cached stream URL")
        return st.session_state[cache_key]

    if safe_mode:
        time.sleep(random.uniform(5, 10))
        cmd = [
            "yt-dlp",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "--sleep-interval", "1",
            "--max-sleep-interval", "3",
            "--sleep-requests", "1",
            "--retries", "3",
            "-f", f"bestvideo[height<={resolution.replace('p','')}]+bestaudio/best",
            "-g", youtube_url,
        ]
    else:
        cmd = [
            "yt-dlp",
            "-f", f"bestvideo[height<={resolution.replace('p','')}]+bestaudio/best",
            "-g", youtube_url,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        stream_url = result.stdout.strip()
        st.session_state[cache_key] = stream_url
        return stream_url
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        if any(word in error_msg.lower() for word in ["banned", "403", "429", "rate limit", "blocked"]):
            st.error("YouTube IP ban detected. Try using a VPN, proxy, or wait 10-30 minutes before retrying.")
        else:
            st.error(
                "Failed to get stream URL. Make sure yt-dlp and ffmpeg are installed "
                "and the video is accessible. Error: " + error_msg[:200]
            )
        return None


# ----------------------------
# Run object detection
# ----------------------------
if start and youtube_url:
    stream_url = get_stream_url(youtube_url, resolution, safe_mode)
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
                    width=800,  # use width instead of deprecated use_column_width
                )

                time.sleep(0.03)  # smooth playback

            cap.release()
            st.success("✅ Detection completed.")
