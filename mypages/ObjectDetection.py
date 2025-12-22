import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time
import yt_dlp

def get_youtube_stream(url):
    """Get stream URL using yt-dlp with specific headers to avoid 429 errors."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        # Mimic a real browser to bypass rate limits
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"YouTube Access Error: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 Stream", layout="wide")
    st.title("🚀 YOLOv8 Detection (Bypass 429 Limit)")

    youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    
    if 'run' not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.columns(2)
    if col1.button("Start Detection"):
        st.session_state.run = True
    if col2.button("Stop"):
        st.session_state.run = False

    placeholder = st.empty()

    if st.session_state.run:
        # Load model only when starting
        model = YOLO("yolov8n.pt")
        
        stream_url = get_youtube_stream(youtube_url)
        
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            
            # Optimization: Lower resolution for cloud processing
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost stream connection. YouTube may have refreshed the URL.")
                    break

                # Inference
                results = model(frame, conf=0.4, verbose=False)
                annotated_frame = results[0].plot()

                # Display
                placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                
                # Yield for Streamlit UI
                time.sleep(0.01)

            cap.release()
