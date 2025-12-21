import streamlit as st
import cv2
import streamlink
import psutil
import time
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt") 

def get_system_stats():
    """Real-time hardware monitoring function."""
    stats = {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "net_sent": psutil.net_io_counters().bytes_sent / (1024 * 1024), 
        "net_recv": psutil.net_io_counters().bytes_recv / (1024 * 1024), 
    }
    return stats

def main():
    st.set_page_config(page_title="YT Object Detector", layout="wide")
    st.title("🎯 YouTube Object Detection")

    # Sidebar Metrics
    st.sidebar.title("💻 System Monitor")
    cpu_m = st.sidebar.empty()
    ram_m = st.sidebar.empty()
    disk_m = st.sidebar.empty()
    net_m = st.sidebar.empty()
    fps_m = st.sidebar.empty()

    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=smoU272Dv14")
    frame_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Settings & Controls")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        res_choice = st.selectbox("Resolution", ["360p", "480p", "720p"], index=0)
    with c2:
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.4, 0.05)
    with c3:
        run_btn = st.checkbox("Start Detection", value=False)

    if run_btn and url:
        try:
            # Use Streamlink to get the direct URL
            session = streamlink.Streamlink()
            session.set_option("http-headers", "User-Agent=Mozilla/5.0")
            streams = session.streams(url)
            
            if not streams:
                st.error("Streamlink couldn't find the stream. YouTube might be blocking the request.")
                return

            # Pick quality (fallback to 'best' if selection isn't available)
            stream_url = streams[res_choice].url if res_choice in streams else streams['best'].url
            
            cap = cv2.VideoCapture(stream_url)
            model = load_yolo_model()
            prev_time = 0
            
            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret: break

                # FPS Logic
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                # YOLO Processing
                results = model(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot() 
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update frame (Using 2025 'width' syntax)
                frame_placeholder.image(annotated_frame, channels="RGB", width='stretch')

                # Update Stats
                s = get_system_stats()
                cpu_m.metric("CPU", f"{s['cpu']}%")
                ram_m.metric("RAM", f"{s['ram']}%")
                disk_m.metric("Disk", f"{s['disk']}%")
                net_m.write(f"🌐 Net: {s['net_recv']:.1f}MB ↓ / {s['net_sent']:.1f}MB ↑")
                fps_m.metric("FPS", f"{fps:.2f}")

            cap.release()
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        frame_placeholder.info("Ready. Check 'Start Detection' to begin.")

if __name__ == "__main__":
    main()
