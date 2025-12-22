import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

def main():
    st.set_page_config(page_title="YouTube Object Detection", layout="wide")
    st.title("📺 YouTube Object Detection with YOLOv8")

    # 1. Input Section
    video_url = st.text_input(
        "Enter YouTube Video URL:", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.4, 
        step=0.05
    )

    # 2. Model Loading
    # Using 'yolov8n.pt' (nano) for faster performance in a web app
    model = YOLO('yolov8n.pt') 

    # 3. Execution Section
    if st.button("Start Detection"):
        try:
            # Use cap_from_youtube to get the stream URL directly
            cap = cap_from_youtube(video_url, '720p')
            
            if not cap.isOpened():
                st.error("Error: Could not open video stream.")
                return

            # Placeholder for the video frames
            frame_window = st.image([])
            stop_button = st.button("Stop Processing")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.info("Video stream ended or failed.")
                    break

                # Run YOLOv8 detection
                # stream=True handles memory more efficiently
                results = model(frame, conf=confidence_threshold, stream=True)

                # Process results and draw boxes
                for result in results:
                    annotated_frame = result.plot()  # Draws boxes and labels
                    
                    # Convert BGR (OpenCV) to RGB (Streamlit)
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    frame_window.image(annotated_frame)

            cap.release()
            st.success("Processing complete.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
