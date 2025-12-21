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
        res_choice = st.selectbox(
            "Stream Resolution", 
            ["360p", "480p", "720p"], 
            index=0
        )
        # Updated resolution mapping for better compatibility
        res_map = {
            "360p": "best[height<=360][ext=mp4]", 
            "480p": "best[height<=480][ext=mp4]", 
            "720p": "best[height<=720][ext=mp4]"
        }

    with col2:
        conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05)

    with col3:
        run_btn = st.checkbox("Start Detection", value=False)

    if run_btn and url:
        try:
            # Enhanced yt-dlp options to bypass SABR/Client issues
            ydl_opts = {
                'format': res_map[res_choice],
                'quiet': True,
                'no_warnings': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']

            # Create VideoCapture
            cap = cv2.VideoCapture(stream_url)
            
            # Check if video opened successfully
            if not cap.isOpened():
                st.error("OpenCV could not open the stream. YouTube may be throttling the request.")
                return

            model = load_yolo_model()
            prev_time = 0
            
            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret:
                    # Some streams require a brief pause or retry
                    time.sleep(0.1)
                    continue

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                results = model(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot() 
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                stats = get_system_stats()
                cpu_metric.metric("CPU Usage", f"{stats['cpu']}%")
                ram_metric.metric("RAM Usage", f"{stats['ram']}%")
                disk_metric.metric("Disk Usage", f"{stats['disk']}%")
                net_metric.write(f"🌐 Net: {stats['net_recv']:.1f}MB ↓ / {stats['net_sent']:.1f}MB ↑")
                fps_metric.metric("Processing Speed", f"{fps:.2f} FPS")

            cap.release()

        except Exception as e:
            st.error(f"Stream Error: {e}")
    else:
        frame_placeholder.info("Enter a URL and check 'Start Detection' to begin.")

if __name__ == "__main__":
    main()
