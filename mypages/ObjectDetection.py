import streamlit as st
import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear

def main():
    st.title("🚀 YOLOv8 YouTube Object Detector")
    
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
        # CamGear options: optimize for streaming and lower latency
        options = {
            "STREAM_RESOLUTION": "480p", # Lower resolution = smoother cloud performance
            "THREADED_QUEUE_MODE": True
        }
        
        try:
            # CamGear handles the YouTube stream extraction internally
            stream = CamGear(source=video_url, stream_mode=True, logging=True, **options).start()
            
            frame_placeholder = st.empty()
            stop_check = st.checkbox("Stop Stream")

            while not stop_check:
                frame = stream.read()
                
                if frame is None:
                    st.warning("No more frames or stream interrupted.")
                    break

                # Inference
                results = model.predict(frame, conf=conf_level, verbose=False)
                
                # Annotate and convert
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Render to Streamlit
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            stream.stop()

        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.info("Tip: If it still fails, YouTube may be temporarily blocking the cloud IP. Try again in a few minutes.")

if __name__ == "__main__":
    main()
