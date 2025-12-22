import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def main():
    st.title("🚀 YOLOv8 YouTube Object Detector")
    
    video_url = st.text_input(
        "YouTube Video URL", 
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )
    
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    @st.cache_resource
    def load_yolo_model():
        # Using the smallest model for cloud efficiency
        return YOLO("yolov8n.pt")

    model = load_yolo_model()

    if st.button("Analyze Video"):
        # Specific options to avoid M3U8/HLS and get a direct MP4 link
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Strictly prefer MP4
            'quiet': True,
            'no_warnings': True,
            'force_generic_extractor': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                # Select the direct URL
                stream_url = info['url']

            cap = cv2.VideoCapture(stream_url)
            
            # Check if cap is actually opened
            if not cap.isOpened():
                st.error("OpenCV could not open the video stream. This may be due to YouTube's bot protection.")
                return

            frame_placeholder = st.empty()
            stop = st.checkbox("Stop Stream")

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO Inference
                results = model.predict(frame, conf=conf_level, verbose=False)
                
                # Annotate and Convert BGR to RGB
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                # Update the image
                frame_placeholder.image(res_rgb, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Stream Error: {e}")

if __name__ == "__main__":
    main()
