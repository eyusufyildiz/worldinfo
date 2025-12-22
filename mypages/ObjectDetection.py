import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
import torch
import time
import random
import numpy as np

# ----------------------------
# Configuration & Model Loading
# ----------------------------
@st.cache_resource
def load_model():
    # Use the Nano version for speed on Streamlit Cloud
    model = YOLO("yolov8n.pt") 
    if torch.cuda.is_available():
        model.to("cuda")
    return model

def get_stream_url(youtube_url, resolution, safe_mode=True):
    """Return direct stream URL using yt-dlp."""
    # Use sorting and better headers to avoid 403 Forbidden errors
    res_val = resolution.replace('p', '')
    
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--extractor-args", "youtube:player_client=ios,web,android",
        "-f", f"bestvideo[height<={res_val}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-g", youtube_url,
    ]

    if safe_mode:
        time.sleep(random.uniform(1, 3))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching stream: {e.stderr}")
        return None

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.set_page_config(page_title="YouTube YOLOv8", layout="wide")
    st.title("📺 YouTube Object Detection")
    st.markdown("This app uses **YOLOv8** to detect objects in real-time from a YouTube stream.")

    # Sidebar Settings
    st.sidebar.header("Settings")
    safe_mode = st.sidebar.checkbox("Anti-Ban Mode", value=True)
    resolution = st.sidebar.selectbox(
        "Resolution", ["144p", "240p", "360p", "480p", "720p"], index=2
    )
    confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
    
    # Input
    youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=smoU272Dv14")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        start_button = st.button("▶ Start")
    with col2:
        stop_button = st.button("⏹ Stop")

    # Load Model
    model = load_model()

    if start_button and youtube_url:
        with st.spinner("Bypassing YouTube filters and fetching stream..."):
            stream_url = get_stream_url(youtube_url, resolution, safe_mode)

        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                st.error("Could not open video stream.")
                return

            # Display area
            frame_placeholder = st.empty()
            fps_placeholder = st.sidebar.empty()
            
            # Processing Loop
            while cap.isOpened():
                if stop_button:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or interrupted.")
                    break

                t1 = time.time()
                
                # YOLO Inference
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()

                # Convert BGR to RGB
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # FPS Calculation
                t2 = time.time()
                fps = 1 / (t2 - t1)
                fps_placeholder.metric("FPS", f"{fps:.2f}")

            cap.release()
            st.info("Stream stopped.")

if __name__ == "__main__":
    main()
