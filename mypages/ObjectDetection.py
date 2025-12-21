import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="YT Object Detector", layout="wide")

@st.cache_resource
def load_yolo_model():
    # Downloads 'yolov8n.pt' (Nano) which is best for CPU real-time performance
    return YOLO("yolov8n.pt") 

def main():
    st.title("🎯 YouTube Real-Time Object Detection")
    st.info("Ensure you have 'ffmpeg' installed on your system for this to run.")

    # 1. Sidebar - Configuration
    st.sidebar.header("Stream Settings")
    url = st.sidebar.text_input("YouTube URL:", "https://www.youtube.com/watch?v=smoU272Dv14")
    
    res_choice = st.sidebar.selectbox(
        "Resolution", 
        ["360p", "480p", "720p"], 
        index=0,
        help="Higher resolutions will cause significant lag on most CPUs."
    )
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    # 2. Main Area - Setup
    frame_placeholder = st.empty()
    run_btn = st.checkbox("🚀 Start Processing", value=False)

    # Resolution mapping for yt-dlp
    res_map = {
        "360p": "best[height<=360]", 
        "480p": "best[height<=480]", 
        "720p": "best[height<=720]"
    }

    if run_btn and url:
        try:
            # Step A: Extract fresh stream URL
            ydl_opts = {
                'format': res_map[res_choice],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            # Step B: Initialize OpenCV with FFMPEG backend
            # We use cv2.CAP_FFMPEG to ensure it handles the YouTube network stream
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to small to reduce lag
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                st.error("Error: Could not open video stream. This usually happens if FFmpeg is missing or the URL is blocked.")
                return

            model = load_yolo_model()

            # Step C: The Processing Loop
            while run_btn:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("Stream ended or failed to retrieve frame.")
                    break

                # Inference
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Plot results and convert BGR (OpenCV) to RGB (Streamlit)
                annotated_frame = results[0].plot() 
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display in Streamlit
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Critical Error: {e}")
    else:
        frame_placeholder.info("Click 'Start Processing' to begin the stream.")

if __name__ == "__main__":
    main()
