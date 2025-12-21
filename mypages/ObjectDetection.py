import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np
import psutil
import time

@st.cache_resource
def load_yolo_model():
    # Downloads 'yolov8n.pt' automatically on first run
    return YOLO("yolov8n.pt") 

def get_system_stats():
    """Function to fetch current system resource usage."""
    stats = {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "net_sent": psutil.net_io_counters().bytes_sent / (1024 * 1024), # MB
        "net_recv": psutil.net_io_counters().bytes_recv / (1024 * 1024), # MB
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

    # 1. Video URL Input
    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=smoU272Dv14")
    
    # Placeholder for the video stream
    frame_placeholder = st.empty()

    # 2. Controls AFTER the video area
    st.markdown("---")
    st.subheader("Settings & Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        res_choice = st.selectbox(
            "Stream Resolution", 
            ["360p", "480p", "720p"], 
            index=0,
            help="Lower resolution is faster for Cloud CPUs"
        )
        res_map = {"360p": "best[height<=360]", "480p": "best[height<=480]", "720p": "best[height<=720]"}

    with col2:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05)

    with col3:
        run_btn = st.checkbox("Start Detection", value=False)

    # 3. Execution Logic
    if run_btn and url:
        try:
            ydl_opts = {'format': res_map[res_choice], 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            cap = cv2.VideoCapture(stream_url)
            model = load_yolo_model()

            # Variables for FPS calculation
            prev_time = 0
            
            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or resolution not available.")
                    break

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                # Run YOLO Inference
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Annotate and Convert for Streamlit
                annotated_frame = results[0].plot() 
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update Video Stream
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Update System Stats in Sidebar
                stats = get_system_stats()
                cpu_metric.metric("CPU Usage", f"{stats['cpu']}%")
                ram_metric.metric("RAM Usage", f"{stats['ram']}%")
                disk_metric.metric("Disk Usage", f"{stats['disk']}%")
                net_metric.write(f"🌐 Net: {stats['net_recv']:.1f}MB ↓ / {stats['net_sent']:.1f}MB ↑")
                fps_metric.metric("Processing Speed", f"{fps:.2f} FPS")

            cap.release()

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        frame_placeholder.info("Enter a URL and check 'Start Detection' to begin.")

if __name__ == "__main__":
    main()
