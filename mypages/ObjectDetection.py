
import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np

# This function cache prevents the model from reloading on every frame
@st.cache_resource
def load_yolo_model():
    # Downloads 'yolov8n.pt' (nano version) automatically
    return YOLO("yolov8n.pt") 

def main():
    st.set_page_config(page_title="Cloud Object Detector", layout="wide")
    st.title("🎯 YouTube Object Detection (YOLOv8)")

    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    # UI Layout
    col1, col2 = st.columns([3, 1])
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)
        run_btn = st.checkbox("Run Detection", value=False)

    frame_placeholder = col1.empty()

    if run_btn and url:
        try:
            # 1. Get the direct stream URL using yt-dlp
            ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            # 2. Open Video Stream
            cap = cv2.VideoCapture(stream_url)
            model = load_yolo_model()

            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("End of stream or connection lost.")
                    break

                # 3. Run YOLO Detection
                # stream=True is more memory efficient for containers
                results = model(frame, conf=conf_threshold, verbose=False)

                # 4. Draw results on the frame
                annotated_frame = results[0].plot() 

                # 5. Display in Streamlit
                # Convert BGR (OpenCV) to RGB (Streamlit)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
