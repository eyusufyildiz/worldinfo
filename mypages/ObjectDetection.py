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
    # Added frame skip to help with Streamlit Cloud CPU limits
    frame_skip = st.sidebar.number_input("Process every Nth frame", min_value=1, value=2)
    
    if 'run' not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start"):
        st.session_state.run = True
    if col2.button("Stop"):
        st.session_state.run = False

    frame_placeholder = st.empty()

    if st.session_state.run:
        with st.spinner("Initializing stream..."):
            model = YOLO("yolov8n.pt")
            
            # --- Robust Stream Opening ---
            cap = None
            # Try 360p first for performance
            try:
                cap = cap_from_youtube(youtube_url, '360p')
            except ValueError:
                # Fallback to 'best' if 360p is missing
                try:
                    cap = cap_from_youtube(youtube_url, 'best')
                except Exception as e:
                    st.error(f"Could not open stream: {e}")
                    st.session_state.run = False
            
        if cap and cap.isOpened():
            frame_count = 0
            while st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or connection lost.")
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Run YOLO
                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()

                # Convert to RGB and display
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Small sleep to yield to the Streamlit event loop
                time.sleep(0.01)

            cap.release()
            st.session_state.run = False
            st.rerun()

if __name__ == "__main__":
    main()
