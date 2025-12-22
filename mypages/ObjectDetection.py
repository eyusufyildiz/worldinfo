import cv2
import streamlit as st
from ultralytics import YOLO
import yt_dlp
import numpy as np
import time

def get_static_mp4_url(youtube_url):
    """Try to get a direct MP4 link, avoiding the HLS/Live stream protocol."""
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Strictly ask for MP4
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"YouTube block detected: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 Fix", layout="centered")
    st.title("🛡️ YOLOv8 Bypass Mode")

    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    
    if st.button("Start Detection"):
        model = YOLO("yolov8n.pt")
        
        with st.spinner("Bypassing YouTube filters..."):
            static_url = get_static_mp4_url(url)
        
        if static_url:
            # Force FFMPEG and add a timeout
            cap = cv2.VideoCapture(static_url)
            
            # Reduce resolution immediately to save memory
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            video_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 2nd frame for speed
                results = model(frame, conf=0.4, verbose=False)
                annotated = results[0].plot()

                # Display
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Yield to UI
                time.sleep(0.01)

            cap.release()
        else:
            st.error("YouTube is blocking this Cloud Server IP. Try a non-live video URL.")

if __name__ == "__main__":
    main()
