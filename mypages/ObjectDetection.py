import cv2
import streamlit as st
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
import time

def main():
    st.set_page_config(page_title="YOLOv8 YouTube Stream", layout="wide")
    st.title("🚀 YOLOv8 Real-time Detection")

    # Sidebar UI
    st.sidebar.header("Configuration")
    youtube_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4)
    
    # Use session state to handle the "Stop" button effectively
    if 'run' not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start"):
        st.session_state.run = True
    if col2.button("Stop"):
        st.session_state.run = False

    frame_placeholder = st.empty()

    if st.session_state.run:
        # Load Model
        model = YOLO("yolov8n.pt")

        # Open the stream using cap_from_youtube
        # '144p' or '360p' is recommended for Streamlit Cloud to save bandwidth/CPU
        cap = cap_from_youtube(youtube_url, '360p')

        if not cap.isOpened():
            st.error("Failed to open stream. The video might be private or restricted.")
            return

        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.write("Stream ended or failed.")
                break

            # Detection
            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            # Convert to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            # Essential for Streamlit's refresh cycle
            time.sleep(0.01)

        cap.release()
        st.session_state.run = False

if __name__ == "__main__":
    main()
