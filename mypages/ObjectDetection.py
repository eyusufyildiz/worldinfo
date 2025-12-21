import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp
import os

def get_direct_url(youtube_url):
    """Fetch a direct MP4 URL that is most compatible with OpenCV FFMPEG."""
    ydl_opts = {
        # Format 18 is 360p MP4, Format 22 is 720p MP4. 
        # These are 'flat' files, not HLS/Manifests, avoiding the 'capture by name' error.
        'format': 'best[ext=mp4]', 
        'noplaylist': True,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"Link Extraction Error: {e}")
        return None

def main():
    st.set_page_config(page_title="YOLOv8 Streamer", layout="wide")
    st.title("🎯 YOLOv8 Container Object Detection")

    model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt"])
    video_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=smoU272Dv14")
    conf_thresh = st.slider("Confidence", 0.0, 1.0, 0.25)
    
    # Updated width control
    use_stretch = st.checkbox("Stretch to Container Width", value=True)
    img_width = "stretch" if use_stretch else 720

    if 'running' not in st.session_state:
        st.session_state.running = False

    col1, col2 = st.columns(2)
    if col1.button("Start"):
        st.session_state.running = True
    if col2.button("Stop"):
        st.session_state.running = False

    # --- Processing Engine ---
    if st.session_state.running:
        model = YOLO(model_choice)
        direct_link = get_direct_url(video_url)
        
        if direct_link:
            # We use FFMPEG but pass the direct stream URL
            cap = cv2.VideoCapture(direct_link)
            st_frame = st.empty()

            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream end or connection lost.")
                    break
                
                # YOLO Inference
                results = model.predict(frame, conf=conf_thresh, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Streamlit Display with updated width parameter
                st_frame.image(rgb_frame, channels="RGB", width=img_width)

            cap.release()
        else:
            st.error("Could not extract a valid video stream.")

if __name__ == "__main__":
    main()
