import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np

# Page Configuration
st.set_page_config(page_title="YT Object Detector", layout="centered")

@st.cache_resource
def load_yolo_model():
    # Downloads 'yolov8n.pt' automatically on first run
    return YOLO("yolov8n.pt") 

def main():
    st.title("🎯 YouTube Object Detection")
    st.markdown("""
    This app uses **YOLOv8** to detect objects in real-time from a YouTube stream.
    *Note: If the stream doesn't start, try lowering the resolution.*
    """)

    # 1. Video URL Input
    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=smoU272Dv14")
    
    # Placeholder for the video stream
    frame_placeholder = st.empty()

    # 2. Controls
    st.markdown("---")
    st.subheader("Settings & Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        res_choice = st.selectbox(
            "Stream Resolution", 
            ["360p", "480p", "720p"], 
            index=0,
            help="Lower resolution is much faster for CPU processing."
        )
        # Map choice to yt-dlp format string
        res_map = {
            "360p": "best[height<=360]", 
            "480p": "best[height<=480]", 
            "720p": "best[height<=720]"
        }

    with col2:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.30, 0.05)

    with col3:
        # Using a button or checkbox to trigger the loop
        run_btn = st.checkbox("Start Detection", value=False)

    # 3. Execution Logic
    if run_btn and url:
        try:
            # Setup yt-dlp to get the direct stream URL
            ydl_opts = {
                'format': res_map[res_choice],
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            # CRITICAL FIX: Explicitly use CAP_FFMPEG to handle YouTube stream URLs
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                st.error("Failed to open video stream. YouTube might be blocking the request or FFmpeg is missing.")
                return

            model = load_yolo_model()

            # Loop for processing frames
            while run_btn:
                # To keep the stream "live" and avoid lag, we read two frames 
                # but only process the last one. This clears the OpenCV buffer.
                cap.grab() 
                ret, frame = cap.retrieve()
                
                if not ret:
                    st.warning("Stream ended or connection lost.")
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
            st.error(f"An error occurred: {e}")
            st.info("Try updating yt-dlp: pip install -U yt-dlp")
    else:
        frame_placeholder.info("Enter a URL and check 'Start Detection' to begin.")

if __name__ == "__main__":
    main()
