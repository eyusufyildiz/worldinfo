import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
from PIL import Image
import torch
import time

# ----------------------------
# Load YOLO model
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # lightweight & fast
    if torch.cuda.is_available():
        model.to("cuda")
        model.fuse()
    return model

# ----------------------------
# Get YouTube stream URL
# ----------------------------
def get_stream_url(youtube_url, resolution):
    """Return direct stream URL using yt-dlp without printing logs."""
    try:
        # Clean the resolution string (e.g., '720p' -> '720')
        res_limit = resolution.replace('p', '')
        
        result = subprocess.run(
            [
                "yt-dlp", 
                "-f", f"best[height<={res_limit}][ext=mp4]/best[ext=mp4]/best", # Better compatibility
                "-g", youtube_url,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # yt-dlp -g can return two lines (video and audio), we take the first
        return result.stdout.strip().split('\n')[0]
    except subprocess.CalledProcessError:
        st.error(
            "Failed to get stream URL. Make sure yt-dlp and ffmpeg are installed "
            "and the video is accessible."
        )
        return None

def main():
    # ----------------------------
    # Streamlit page setup
    # ----------------------------
    st.set_page_config(layout="wide")
    st.title("📺 YouTube Object Detection (Streamlit Cloud)")

    # ----------------------------
    # Sidebar / User Inputs
    # ----------------------------
    youtube_url = st.text_input(
        "YouTube URL", "https://www.youtube.com/watch?v=BhBcICoRY6Q"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        resolution = st.selectbox(
            "Select video resolution",
            ["144p", "240p", "360p", "480p", "720p", "1080p"],
            index=4,
        )
    with col2:
        confidence = st.slider("Detection confidence", 0.1, 0.9, 0.4)
    
    start = st.button("▶ Start Detection")
    stop = st.button("🛑 Stop")

    model = load_model()

    # ----------------------------
    # Run object detection
    # ----------------------------
    if start and youtube_url:
        stream_url = get_stream_url(youtube_url, resolution)
        
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                st.error("Failed to open video stream. Check the URL or resolution.")
            else:
                st.success("Stream opened! Running object detection...")

                frame_placeholder = st.empty()
                frame_skip = 2
                frame_count = 0

                while cap.isOpened() and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    # Resize frame for speed
                    frame = cv2.resize(frame, (1280, 720))

                    # YOLO detection
                    try:
                        results = model(frame, conf=confidence, verbose=False)
                        annotated = results[0].plot()
                    except Exception as e:
                        continue  # skip frame silently

                    # Convert BGR → RGB for Streamlit display
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    # Display frame in Streamlit
                    frame_placeholder.image(
                        annotated,
                        channels="RGB",
                        use_container_width=True
                    )

                    # Small sleep to prevent CPU hogging
                    time.sleep(0.01)

                cap.release()
                st.info("✅ Detection stopped.")

if __name__ == "__main__":
    main()
