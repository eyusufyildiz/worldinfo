import streamlit as st
import cv2
import yt_dlp
import os
from ultralytics import YOLO

def main():
    st.title("YouTube Object Detection (SABR Fix)")
    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=smoU272Dv14")
    
    if st.button("Start Processing"):
        # 1. Download settings (Low res for speed)
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': 'temp_video.mp4',
            'noplaylist': True,
        }

        with st.spinner("Downloading/Buffering video..."):
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # 2. Process with OpenCV
        cap = cv2.VideoCapture("temp_video.mp4")
        st_frame = st.empty()
        model = YOLO("yolo11n.pt") # Smallest model for speed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            results = model(frame, conf=0.4)
            annotated_frame = results[0].plot()
            
            # Display
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()
