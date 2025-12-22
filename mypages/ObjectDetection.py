import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

def main():
    st.title("YouTube Object Detection")
    st.write("Detect objects in real-time from a YouTube URL using YOLOv8.")

    # Input section
    video_url = st.text_input(
        "YouTube Video URL", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.4, 
        step=0.05
    )

    # Load YOLO model (Small version for better cloud performance)
    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()

    if st.button("Start Detection"):
        try:
            # Use cap_from_youtube for efficient streaming
            cap = cap_from_youtube(video_url, "720p")
            
            # Placeholder for the video frames
            frame_placeholder = st.empty()
            stop_button = st.button("Stop")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.write("Video stream ended or failed.")
                    break

                # Run YOLOv8 inference
                results = model(frame, conf=confidence_threshold, verbose=False)

                # Visualize results on the frame
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the frame
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()
            
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
