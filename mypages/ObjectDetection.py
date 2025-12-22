import streamlit as st
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube, list_video_streams

def main():
    st.set_page_config(page_title="YouTube Object Detection", layout="wide")
    st.title("📺 YouTube Object Detection with Resolution Select")

    # 1. Video URL Input
    video_url = st.text_input(
        "Enter YouTube Video URL:", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )

    # 2. Get Available Resolutions
    # We fetch these dynamically based on the URL provided
    try:
        streams, resolutions = list_video_streams(video_url)
        # resolutions is a numpy array of strings like ['144p', '360p', ...]
        selected_res = st.selectbox("Select Resolution", options=resolutions, index=len(resolutions)-1)
    except Exception as e:
        st.error(f"Could not fetch resolutions: {e}")
        selected_res = 'best'

    # 3. Confidence Slider
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )

    # 4. Model Loading (Cached)
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt') 
    
    model = load_model()

    # 5. Execution Section
    if st.button("Start Detection"):
        try:
            # Use the user-selected resolution
            cap = cap_from_youtube(video_url, selected_res)
            
            if cap is None or not cap.isOpened():
                st.error("Error: Could not open video stream.")
                return

            frame_placeholder = st.empty()
            stop_btn = st.button("Stop Processing")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO Inference
                results = model(frame, conf=confidence_threshold, stream=True)

                for result in results:
                    annotated_frame = result.plot()
                    # Convert BGR to RGB for Streamlit
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()
            st.success("Finished.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
