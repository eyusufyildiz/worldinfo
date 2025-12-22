import cv2
import streamlit as st
from ultralytics import YOLO
import streamlink
import numpy as np

def get_stream_url(youtube_url):
    try:
        # Streamlink handles the 429 errors and HLS manifests much better than yt-dlp
        streams = streamlink.streams(youtube_url)
        if not streams:
            return None
        # Select the best available quality (usually 360p or 720p)
        stream_url = streams['best'].url
        return stream_url
    except Exception as e:
        st.error(f"Streamlink error: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 Live Fix")
    st.title("🚀 YOLOv8 YouTube Live Detection")

    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    
    if st.button("Start Live Stream"):
        model = YOLO("yolov8n.pt")
        stream_url = get_stream_url(url)
        
        if stream_url:
            # FORCE FFMPEG backend to avoid the CAP_IMAGES error
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            # Optimization for cloud processing
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            video_placeholder = st.empty()
            stop_button = st.button("Stop", key="stop_btn")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Buffer empty or stream disconnected.")
                    break
                
                # Resize to reduce CPU load on Streamlit Cloud
                frame = cv2.resize(frame, (640, 360))
                
                # Detection
                results = model(frame, conf=0.4, verbose=False)
                annotated = results[0].plot()
                
                # Display
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            cap.release()
        else:
            st.error("Could not resolve stream. YouTube might be blocking the Cloud IP.")

if __name__ == "__main__":
    main()
