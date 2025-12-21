import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np
import psutil
import time

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt") 

def get_system_stats():
    stats = {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "net_sent": psutil.net_io_counters().bytes_sent / (1024 * 1024), 
        "net_recv": psutil.net_io_counters().bytes_recv / (1024 * 1024), 
    }
    return stats

def main():
    st.set_page_config(page_title="YT Object Detector", layout="wide")
    st.title("🎯 YouTube Object Detection")

    # Sidebar for System Metrics
    st.sidebar.title("💻 System Monitor")
    cpu_metric = st.sidebar.empty()
    ram_metric = st.sidebar.empty()
    disk_metric = st.sidebar.empty()
    net_metric = st.sidebar.empty()
    fps_metric = st.sidebar.empty()

    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=smoU272Dv14")
    frame_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Settings & Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        res_choice = st.selectbox("Stream Resolution", ["360p", "480p", "720p"], index=0)
        # Using specific format strings that are most compatible with OpenCV FFmpeg
        res_map = {
            "360p": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]",
            "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]",
            "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]"
        }

    with col2:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05)

    with col3:
        run_btn = st.checkbox("Start Detection", value=False)

    if run_btn and url:
        try:
            # We use yt-dlp to get the direct stream link
            ydl_opts = {
                'format': res_map[res_choice],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # yt-dlp might return a list of formats; we take the first 'url'
                stream_url = info.get('url', None)

            if not stream_url:
                st.error("Could not extract stream URL. Try another video.")
                return

            # Open with FFmpeg explicitly
            cap = cv2.VideoCapture(stream_url)
            
            # This is a key fix: If the first attempt fails, we try forcing the backend
            if not cap.isOpened():
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

            model = load_yolo_model()
            prev_time = 0
            
            while cap.isOpened() and run_btn:
                ret,
