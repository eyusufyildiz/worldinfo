import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

def main():
    st.set_page_config(page_title="YOLOv8 YouTube Detector", layout="wide")
    
    st.title("🚀 YOLOv8 YouTube Object Detection")
    st.sidebar.header("Settings")

    # 1. Select YOLO Model
    model_type = st.sidebar.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    )
    
    # Load model
    model = YOLO(model_type)

    # 2. Add YouTube Video URL
    video_url = st.sidebar.text_input(
        "YouTube Video URL",
        "https://www.youtube.com/watch?v=MNn9q6cHTpw"
    )

    # 3. Stream Resolution
    resolution = st.sidebar.selectbox(
        "Stream Resolution",
        ["360p", "480p", "720p", "1080p"],
        index=0
    )

    # 4. Detection Confidence
    conf_threshold = st.sidebar.slider(
        "Detection Confidence", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25
    )

    # 5. Start/Stop Button
    run_detection = st.sidebar.button("Start Detection")
    stop_detection = st.sidebar.button("Stop")

    st_frame = st.empty()

    if run_detection:
        # Capture video from YouTube
        cap = cap_from_youtube(video_url, resolution)
        
        if not cap.isOpened():
            st.error("Error: Could not open video stream.")
            return

        while cap.isOpened():
            if stop_detection:
                break
                
            ret, frame = cap.read()
            if not ret:
                st.write("Stream ended or failed.")
                break

            # Run YOLO detection
            results = model.predict(frame, conf=conf_threshold)
            
            # Plot results on the frame
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()
        st.success("Detection Stopped.")

if __name__ == "__main__":
    main()
