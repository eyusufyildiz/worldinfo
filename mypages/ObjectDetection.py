import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp

def get_youtube_stream_url(video_url):
    """
    Extracts the best streamable URL. 
    Using 'ext:mp4' helps avoid complex formats that require JS runtimes.
    """
    ydl_opts = {
        # 'best' is more reliable than specific resolution strings when JS is missing
        'format': 'best[ext=mp4]', 
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info['url']

def main():
    st.set_page_config(page_title="YOLOv8 AI Streamer", layout="wide")
    st.title("🛡️ YOLOv8 Real-Time Detection")

    # --- Sidebar Configuration ---
    st.sidebar.header("Control Panel")
    model_name = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])
    video_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=MNn9q6cHTpw")
    conf_level = st.sidebar.slider("Confidence", 0.0, 1.0, 0.3)
    
    # Session state to toggle detection
    if 'detecting' not in st.session_state:
        st.session_state.detecting = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Start"):
        st.session_state.detecting = True
    if col2.button("⏹️ Stop"):
        st.session_state.detecting = False

    # --- Detection Engine ---
    if st.session_state.detecting:
        model = YOLO(model_name)
        st_frame = st.empty()
        
        try:
            stream_url = get_youtube_stream_url(video_url)
            cap = cv2.VideoCapture(stream_url)

            while cap.isOpened() and st.session_state.detecting:
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                results = model.predict(frame, conf=conf_level, verbose=False)
                
                # Plot and Show
                annotated = results[0].plot()
                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
        except Exception as e:
            st.error(f"Stream Error: {e}")
            st.session_state.detecting = False

if __name__ == "__main__":
    main()
