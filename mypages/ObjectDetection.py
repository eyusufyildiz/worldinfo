import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np

@st.cache_resource
def load_yolo_model():
    # Downloads 'yolov8n.pt' automatically on first run
    return YOLO("yolov8n.pt") 

def main():
    st.set_page_config(page_title="YT Object Detector", layout="centered")
    st.title("🎯 YouTube Object Detection")

    # 1. Video URL Input
    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    # Placeholder for the video stream
    frame_placeholder = st.empty()

    # 2. Controls AFTER the video area
    st.markdown("---")
    st.subheader("Settings & Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Resolution mapping for yt-dlp
        res_choice = st.selectbox(
            "Stream Resolution", 
            ["360p", "480p", "720p"], 
            index=0,
            help="Lower resolution is faster for Cloud CPUs"
        )
        # Map choice to yt-dlp format string
        res_map = {"360p": "best[height<=360]", "480p": "best[height<=480]", "720p": "best[height<=720]"}

    with col2:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05)

    with col3:
        run_btn = st.checkbox("Start Detection", value=False)

    # 3. Execution Logic
    if run_btn and url:
        try:
            # Get the stream URL based on selected resolution
            ydl_opts = {'format': res_map[res_choice], 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            cap = cv2.VideoCapture(stream_url)
            model = load_yolo_model()

            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or resolution not available.")
                    break

                # Run YOLO Inference
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Annotate and Convert for Streamlit
                annotated_frame = results[0].plot() 
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update the placeholder
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        frame_placeholder.info("Enter a URL and check 'Start Detection' to begin.")

if __name__ == "__main__":
    main()
