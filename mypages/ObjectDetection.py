import streamlit as st
import cv2
import tempfile
import os
import yt_dlp
import torch
from ultralytics import YOLO

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("📺 YouTube Object Detection (Fast & Accurate)")

youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=H3JYmEYC1pk")
resolution = st.selectbox(
    "Select video resolution",
    ["144p", "240p", "360p", "480p", "720p", "1080p"],
    index=4
)

confidence = st.slider("Detection confidence", 0.1, 0.9, 0.4)
start = st.button("▶ Start Detection")

# ----------------------------
# YOLO Model (Best Accuracy)
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8l.pt")   # best balance accuracy/speed
    if torch.cuda.is_available():
        model.to("cuda")
        model.fuse()
    return model

model = load_model()

# ----------------------------
# Download YouTube Video
# ----------------------------
def download_video(url, res):
    height = int(res.replace("p", ""))

    ydl_opts = {
        "format": f"bestvideo[height<={height}]+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename.replace(".webm", ".mp4")

# ----------------------------
# Run Detection
# ----------------------------
def app():
    if start and youtube_url:
        with st.spinner("Downloading video..."):
            video_path = download_video(youtube_url, resolution)
    
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        frame_placeholder = st.empty()
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
    
            # YOLO inference
            results = model.predict(
                frame,
                conf=confidence,
                imgsz=640,
                device=0 if torch.cuda.is_available() else "cpu",
                half=torch.cuda.is_available(),
                verbose=False
            )
    
            annotated = results[0].plot()
    
            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )
    
        cap.release()
        os.remove(video_path)
        st.success("Detection completed ✔")
