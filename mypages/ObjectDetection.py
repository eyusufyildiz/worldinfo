import cv2
import subprocess
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

def get_stream_url(youtube_url):
    """Uses yt-dlp to get the direct URL. Explicitly asks for mp4."""
    try:
        # Added --get-url and specific format handling
        command = [
            "yt-dlp", 
            "-f", "best[ext=mp4]", 
            "--get-url", 
            youtube_url
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        st.error(f"Failed to get stream: {e}")
        return None

def main():
    st.title("YOLOv8 Stream Fix")
    
    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    start_btn = st.button("Start")
    
    frame_placeholder = st.empty()

    if start_btn:
        model = YOLO("yolov8n.pt")
        stream_url = get_stream_url(url)
        
        if stream_url:
            # FORCE FFMPEG BACKEND HERE
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            # Check if connection is actually open
            if not cap.isOpened():
                st.error("OpenCV could not open the stream. The URL may have expired.")
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Processing logic
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, conf=0.4)
                annotated = results[0].plot()
                
                # Display
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated, channels="RGB")
                
            cap.release()

if __name__ == "__main__":
    main()
