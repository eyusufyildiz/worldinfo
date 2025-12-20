# streamlit_yolo.py
import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time
import os

# --- Load YOLO model once ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# --- Helper to get stream URL ---
def get_stream_url(youtube_url, cookies_path=None):
    cmd = ["yt-dlp", "-f", "best[ext=mp4]", "-g", youtube_url]
    if cookies_path:
        cmd.insert(1, "--cookies")
        cmd.insert(2, cookies_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Error fetching stream URL:\n{result.stderr}")
        return None
    return result.stdout.strip()

# --- Main Streamlit app ---
def main():
    st.title("YouTube YOLO Object Detection")

    # User inputs
    youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    cookies_file = st.file_uploader("Upload cookies.txt (from browser, optional)", type=["txt"])

    model = load_model()

    if st.button("Start Detection"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL.")
            return

        # Save uploaded cookies file temporarily
        cookies_path = None
        if cookies_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(cookies_file.read())
                cookies_path = tmp.name

        # Get the video stream URL
        stream_url = get_stream_url(youtube_url, cookies_path)
        if not stream_url:
            return

        stframe = st.empty()
        cap = cv2.VideoCapture(stream_url)
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

            # YOLO detection
            results = model(frame, conf=0.4)
            annotated = results[0].plot()

            # Convert BGR → RGB
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated)

            stframe.image(img, use_column_width=True)
            time.sleep(0.03)

        cap.release()
        if cookies_path:
            os.remove(cookies_path)

# --- Entry point ---
if __name__ == "__main__":
    main()
