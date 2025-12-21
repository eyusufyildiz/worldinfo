import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp
import numpy as np

def get_stream_info(url):
    """Extracts the best mp4 stream URL using yt-dlp with browser headers."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # Prioritize MP4
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info['url']
        except Exception as e:
            st.error(f"YouTube Error: {e}")
            return None

def main():
    st.set_page_config(page_title="YOLO Container Streamer", layout="wide")
    st.title("🎥 AI Object Detection Stream")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    video_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=MNn9q6cHTpw")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30)
    
    # Start/Stop Logic
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("🚀 Start"):
        st.session_state.is_running = True
    if col2.button("🛑 Stop"):
        st.session_state.is_running = False

    # --- Video Processing ---
    if st.session_state.is_running:
        model = YOLO(model_choice)
        stream_url = get_stream_info(video_url)
        
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            st_frame = st.empty()

            while cap.isOpened() and st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Attempting to reconnect or stream ended...")
                    break
                
                # Inference
                results = model.predict(frame, conf=conf_thresh, verbose=False)
                
                # Plot results
                annotated_frame = results[0].plot()
                
                # Convert BGR (OpenCV) to RGB (Streamlit)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
        else:
            st.error("Failed to retrieve stream. The video might be restricted or the URL is invalid.")

if __name__ == "__main__":
    main()
