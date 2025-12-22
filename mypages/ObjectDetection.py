import cv2
import subprocess
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# Set page config
st.set_page_config(page_title="YOLOv8 YouTube Stream", layout="centered")

def get_stream_url(youtube_url):
    """Extracts the direct MP4 stream URL using yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4]/best", "-g", youtube_url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching video stream: {e}")
        return None

def main():
    st.title("🚀 YOLOv8 Real-time YouTube Detection")
    
    # Sidebar for Configuration
    st.sidebar.header("Settings")
    youtube_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
    frame_skip = st.sidebar.slider("Frame Skip (Speed)", 1, 10, 2)
    
    run_detection = st.sidebar.button("Start Detection")
    stop_detection = st.sidebar.button("Stop")

    # Placeholder for the video frame
    frame_placeholder = st.empty()

    if run_detection:
        # Load YOLO model
        with st.spinner("Loading Model..."):
            model = YOLO("yolov8n.pt")

        # Get Stream
        stream_url = get_stream_url(youtube_url)
        
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            frame_count = 0

            while cap.isOpened() and not stop_detection:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Finished or lost stream connection.")
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Resize for processing speed
                frame = cv2.resize(frame, (640, 480))

                # Run YOLO detection
                results = model(frame, conf=conf_threshold)
                
                # Plot results on frame
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit/PIL
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update the Streamlit UI
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Small delay to prevent CPU over-usage
                time.sleep(0.01)

            cap.release()
            st.success("Stream stopped.")

if __name__ == "__main__":
    main()
