import streamlit as st
import cv2
from ultralytics import YOLO
import streamlink
import numpy as np

def get_stream_url(youtube_url, quality):
    """Uses streamlink to resolve the HLS manifest into a streamable URL."""
    try:
        streams = streamlink.streams(youtube_url)
        if quality in streams:
            return streams[quality].url
        return streams['best'].url
    except Exception as e:
        st.error(f"Streamlink Error: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
    
    st.title("🎥 YOLOv8 Real-Time YouTube Detection")
    st.caption("Using Streamlink to bypass OpenCV HLS/SABR errors.")

    # --- UI Controls (Main Page) ---
    col_url, col_res, col_conf = st.columns([2, 1, 1])
    
    with col_url:
        video_url = st.text_input("YouTube Video URL", value="https://www.youtube.com/watch?v=smoU272Dv14")
    
    with col_res:
        resolution = st.selectbox("Stream Resolution", ["360p", "480p", "720p", "best"], index=0)
    
    with col_conf:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.3)

    model_type = st.selectbox("Select YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

    # Start / Stop Buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    start_btn = col_btn1.button("▶️ Start Detection", use_container_width=True)
    stop_btn = col_btn2.button("⏹️ Stop", use_container_width=True)

    # Initialize session state for the loop
    if "run_detection" not in st.session_state:
        st.session_state.run_detection = False

    if start_btn:
        st.session_state.run_detection = True
    if stop_btn:
        st.session_state.run_detection = False

    # --- Video Processing Area ---
    st_frame = st.empty()

    if st.session_state.run_detection:
        # Load Model
        model = YOLO(model_type)
        
        # Get the stream link
        stream_url = get_stream_url(video_url, resolution)
        
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            
            # Optimization: Skip every other frame to keep up with the stream in a container
            frame_count = 0

            while cap.isOpened() and st.session_state.run_detection:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or failed to load.")
                    break

                frame_count += 1
                if frame_count % 2 != 0: # Process every 2nd frame
                    continue

                # YOLO Inference
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                
                # Annotate and Convert BGR to RGB
                annotated_frame = results[0].plot()
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the frame using 'stretch' for width
                st_frame.image(rgb_frame, channels="RGB", width="stretch")

            cap.release()
            st.session_state.run_detection = False
        else:
            st.error("Could not resolve video stream. Check if the URL is valid.")

if __name__ == "__main__":
    main()
