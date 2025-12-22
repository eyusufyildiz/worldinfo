import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

# Page Configuration
st.set_page_config(page_title="YouTube Object Detection", layout="wide")

st.title("🚀 YouTube Object Detection with YOLOv8")

# Model Selection - Loading YOLOv8 nano (fastest for CPU)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# User Inputs (No Sidebar)
url_input = st.text_input(
    "Enter YouTube URL:", 
    value="https://www.youtube.com/watch?v=smoU272Dv14"
)

conf_threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# Execution State
run_detection = st.button("Start Detection")
stop_detection = st.button("Stop")

# Placeholder for the video frame
frame_window = st.image([])

if run_detection:
    try:
        # Use cap_from_youtube to get the stream URL
        cap = cap_from_youtube(url_input, '720p')
        
        if not cap.isOpened():
            st.error("Error: Could not open video stream.")
        
        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of stream or error reading frame.")
                break

            # Perform Object Detection
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            # Plot results on the frame
            annotated_frame = results[0].plot()

            # Convert BGR (OpenCV) to RGB (Streamlit)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            frame_window.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

if stop_detection:
    st.info("Detection stopped.")
