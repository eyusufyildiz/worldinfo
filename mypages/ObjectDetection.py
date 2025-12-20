import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

def main():
    st.set_page_config(page_title="YouTube Object Detector", layout="wide")
    st.title("🚀 YouTube Video Object Detection")

    # --- Sidebar Configuration ---
    st.sidebar.header("Detection Settings")
    
    # Model Selection
    model_type = st.sidebar.selectbox("Select YOLO Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    
    # Confidence Threshold
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.45, 0.05)
    
    # Resolution Options
    resolution = st.sidebar.selectbox("Stream Resolution", ["360p", "480p", "720p", "1080p"], index=0)
    
    # YouTube URL Input
    url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=MNn9qKG2UFI")

    # --- Logic ---
    #if st.sidebar.button("Start Detection"):
    try:
        # Load Model
        model = YOLO(model_type)
        
        # Initialize YouTube Stream
        cap = cap_from_youtube(url, resolution)
        
        # Create a placeholder for the video frame
        st_frame = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("Finished processing or unable to fetch frame.")
                break

            # Run Inference
            results = model.predict(frame, conf=conf_threshold, verbose=False)

            # Plot results on frame
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()

    except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
