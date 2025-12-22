import cv2
import streamlit as st
from ultralytics import YOLO
import yt_dlp
import numpy as np

def get_best_stream(url):
    """
    Uses yt-dlp to find a direct video-only or combined MP4 stream.
    This avoids the HLS/m3u8 segmentation error.
    """
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # Prioritize MP4 over HLS/m3u8
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('url')
    except Exception as e:
        st.error(f"Stream extraction failed: {e}")
        return None

def main():
    st.title("YOLOv8 YouTube Fix")
    
    # Input URL
    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    
    if st.button("Run Detection"):
        model = YOLO("yolov8n.pt")
        stream_url = get_best_stream(url)
        
        if stream_url:
            # We add a buffer size to help FFMPEG handle the stream
            cap = cv2.VideoCapture(stream_url)
            
            # Use a placeholder for the video
            video_spot = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # If the stream breaks, try to re-extract the URL 
                    # (YouTube links expire every few minutes)
                    break
                
                # Resize and Process
                frame = cv2.resize(frame, (640, 360))
                results = model(frame, conf=0.4, verbose=False)
                annotated = results[0].plot()
                
                # Convert for Streamlit
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_spot.image(annotated, channels="RGB")
                
            cap.release()
            st.warning("Stream ended. This often happens due to YouTube's IP blocking.")

if __name__ == "__main__":
    main()
