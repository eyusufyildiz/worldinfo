import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def main():
    st.title("🚀 YOLOv8 YouTube Object Detector")
    st.markdown("This app detects objects from a YouTube stream using YOLOv8.")

    # User Inputs
    video_url = st.text_input(
        "YouTube Video URL", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )
    
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    # Cache the model to avoid reloading on every interaction
    @st.cache_resource
    def load_yolo_model():
        return YOLO("yolov8n.pt")

    model = load_yolo_model()

    if st.button("Analyze Video"):
        # Configure yt-dlp to get a direct stream URL
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                stream_url = info['url']

            # Open video capture
            cap = cv2.VideoCapture(stream_url)
            frame_placeholder = st.empty()
            
            # Use a container for the stop button logic
            stop = st.checkbox("Stop Stream")

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # Object Detection
                results = model.predict(frame, conf=conf_level, verbose=False)
                
                # Annotate frame
                res_plotted = results[0].plot()
                
                # Streamlit uses RGB, OpenCV uses BGR
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                # Update the image in the placeholder
                frame_placeholder.image(res_rgb, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Error fetching video: {e}")
            st.info("YouTube sometimes blocks automated requests from cloud IPs. If this persists, try a different video or check for yt-dlp updates.")

if __name__ == "__main__":
    main()
