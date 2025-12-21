import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp
import os

def get_stable_stream(url):
    """
    Forces yt-dlp to find a non-segmented, single-link MP4.
    This avoids the HLS/m3u8 errors in OpenCV.
    """
    ydl_opts = {
        # '18' is usually the code for 360p MP4 - highly compatible
        # '22' is 720p MP4
        'format': 'best[ext=mp4][protocol=https]', 
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            # Ensure we aren't getting a manifest/playlist URL
            return info.get('url')
        except Exception as e:
            st.error(f"yt-dlp error: {e}")
            return None

def main():
    st.set_page_config(page_title="Stable YOLO Streamer", layout="wide")
    
    st.sidebar.title("Settings")
    video_url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=MNn9q6cHTpw")
    model_name = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])
    conf_thresh = st.sidebar.slider("Confidence", 0.0, 1.0, 0.3)
    
    if 'run_state' not in st.session_state:
        st.session_state.run_state = False

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start"): st.session_state.run_state = True
    if col2.button("Stop"): st.session_state.run_state = False

    if st.session_state.run_state:
        model = YOLO(model_name)
        stream_url = get_stable_stream(video_url)
        
        if stream_url:
            # We add cv2.CAP_FFMPEG to force the use of the FFMPEG backend
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            # Set a timeout/buffer limit to prevent the app from freezing
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
            
            st_frame = st.empty()

            while cap.isOpened() and st.session_state.run_state:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost stream connection. Retrying...")
                    break
                
                # Perform Detection
                results = model.predict(frame, conf=conf_thresh, verbose=False)
                annotated_img = results[0].plot()
                
                # Display
                st_frame.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB")
                
            cap.release()
        else:
            st.error("Could not extract a compatible MP4 stream from this URL.")

if __name__ == "__main__":
    main()
