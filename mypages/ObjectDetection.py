import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def get_stream_url(youtube_url):
    ydl_opts = {
        # Format 18 is the most stable 360p MP4 format for OpenCV
        'format': '18', 
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception:
        # Fallback to best if 18 isn't available
        ydl_opts = {'format': 'best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']

def main():
    st.set_page_config(page_title="YouTube Object Detection", layout="wide")
    st.title("📺 Robust YouTube Object Detection")

    video_url = st.text_input("YouTube URL:", value="https://www.youtube.com/watch?v=smoU272Dv14")
    conf_level = st.slider("Confidence", 0.0, 1.0, 0.4, 0.05)

    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt')
    model = load_model()

    if st.button("Start Detection"):
        stream_url = get_stream_url(video_url)
        
        if stream_url:
            # We remove the CAP_FFMPEG flag to let OpenCV choose the best available 
            # while providing a simpler single-stream URL
            cap = cv2.VideoCapture(stream_url)
            
            # Check if it opened, if not, try with a specific environment fix
            if not cap.isOpened():
                st.error("OpenCV backend error. Try running: pip install opencv-python-headless")
                return

            frame_placeholder = st.empty()
            stop_btn = st.checkbox("Stop Stream")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO Inference
                results = model(frame, conf=conf_level, stream=True)
                
                for r in results:
                    annotated_frame = r.plot()
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            st.write("Stream ended.")

if __name__ == "__main__":
    main()
