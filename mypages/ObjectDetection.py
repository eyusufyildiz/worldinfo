import cv2
import streamlit as st
from ultralytics import YOLO
import yt_dlp
import numpy as np

def get_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'user_agent': 'Mozilla/5.0'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info['url']
        except:
            return None

st.title("YOLOv8 YouTube Debugger")

url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")

if st.button("Start"):
    real_url = get_url(url)
    if not real_url:
        st.error("YouTube blocked the request. Try a different video or run locally.")
    else:
        # We use CAP_FFMPEG to prevent the CAP_IMAGES error
        cap = cv2.VideoCapture(real_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            st.error("OpenCV cannot open this stream. The IP is likely banned.")
        else:
            model = YOLO("yolov8n.pt")
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, conf=0.4, verbose=False)
                annotated = results[0].plot()
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()
