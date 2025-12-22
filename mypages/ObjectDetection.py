import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def get_stream_url(youtube_url):
    """
    Extracts a direct stream URL that is highly compatible with OpenCV.
    Forces a single 'ext:mp4' format to avoid DASH/SABR split-stream issues.
    """
    ydl_opts = {
        # '18' is the itag for 360p MP4 (Video+Audio combined) - very stable for CV
        # '22' is the itag for 720p MP4 (Video+Audio combined)
        # We try 720p first, then fall back to 360p
        'format': '22/18/best[ext=mp4]', 
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"yt-dlp error: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 YouTube Fix", layout="wide")
    st.title("🚀 High-Compatibility YouTube Detector")

    # 1. Inputs
    url = st.text_input("YouTube URL:", value="https://www.youtube.com/watch?v=smoU272Dv14")
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    # 2. Load Model
    @st.cache_resource
    def load_yolo():
        return YOLO('yolov8n.pt')
    model = load_yolo()

    # 3. Process
    if st.button("Start Detection"):
        stream_url = get_stream_url(url)
        
        if stream_url:
            # Force OpenCV to use the FFMPEG backend explicitly
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                st.error("OpenCV could not open this specific format. Try another video.")
                return

            frame_placeholder = st.empty()
            
            # Use a checkbox as a 'Stop' toggle to avoid button-state resets
            stop_process = st.checkbox("Stop Stream")

            while cap.isOpened() and not stop_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO Inference (stream=True for memory efficiency)
                results = model(frame, conf=conf_level, stream=True)
                
                for r in results:
                    annotated_frame = r.plot()
                    # Convert BGR to RGB for Streamlit
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            st.write("Stream ended.")

if __name__ == "__main__":
    main()
