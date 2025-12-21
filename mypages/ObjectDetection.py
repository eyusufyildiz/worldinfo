import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def get_youtube_stream_url(video_url, resolution):
    """Extracts the direct stream URL using yt-dlp."""
    # Mapping resolution to yt-dlp format codes
    res_map = {
        "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    }
    
    ydl_opts = {
        'format': res_map.get(resolution, "best"),
        'noplaylist': True,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info['url']

def main():
    st.set_page_config(page_title="YOLOv8 YouTube Detector", layout="wide")
    st.title("🚀 YOLOv8 YouTube Object Detection")
    
    # Sidebar Setup
    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox("Select YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    video_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=MNn9q6cHTpw")
    resolution = st.sidebar.selectbox("Stream Resolution", ["360p", "480p", "720p", "1080p"])
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)

    # State management for the Start/Stop toggle
    if 'run' not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start"):
        st.session_state.run = True
    if col2.button("Stop"):
        st.session_state.run = False

    # Load Model
    model = YOLO(model_type)
    st_frame = st.empty()

    if st.session_state.run:
        try:
            stream_url = get_youtube_stream_url(video_url, resolution)
            cap = cv2.VideoCapture(stream_url)
            
            while cap.isOpened() and st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or failed to load.")
                    break

                # Inference
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Render
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
            cap.release()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.run = False

if __name__ == "__main__":
    main()
