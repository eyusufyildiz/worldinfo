import streamlit as st
import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
import time

def main():
    st.title("🚀 YOLOv8 YouTube Object Detector")
    st.markdown("Forcing FFmpeg backend to bypass OpenCV pattern errors.")

    # Input Section
    video_url = st.text_input(
        "YouTube Video URL", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )
    
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    @st.cache_resource
    def load_yolo_model():
        return YOLO("yolov8n.pt")

    model = load_yolo_model()

    if st.button("Start Detection"):
        # We add "CAP_FFMPEG" to the parameters to bypass the CAP_IMAGES error
        options = {
            "STREAM_RESOLUTION": "480p",
            "THREADED_QUEUE_MODE": True,
            "CAP_PROP_HW_ACCELERATION": 0, # Disable HW accel to save cloud memory
        }
        
        try:
            # Initialize stream
            stream = CamGear(
                source=video_url, 
                stream_mode=True, 
                logging=True, 
                **options
            ).start()
            
            # Give the buffer a second to fill
            time.sleep(2)
            
            frame_placeholder = st.empty()
            stop_check = st.checkbox("Stop Stream")

            while not stop_check:
                frame = stream.read()
                
                if frame is None:
                    # If frame is None, the stream might still be loading or blocked
                    continue

                # Run Inference
                results = model.predict(frame, conf=conf_level, verbose=False)
                
                # Annotate
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Render
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            stream.stop()

        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.info("Note: Streamlit Cloud IPs are frequently throttled. If this fails, the stream URL is being blocked at the network level.")

if __name__ == "__main__":
    main()
