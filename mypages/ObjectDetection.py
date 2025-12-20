import streamlit as st
import cv2
import yt_dlp
import numpy as np
from ultralytics import YOLO
import time

# ================= CONFIG =================
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=ztmY_cCtUl0"
DEFAULT_RESOLUTION = (854, 480)  # safer for Streamlit Cloud
# =========================================


@st.cache_resource
def load_model():
    # YOLOv8 nano = fastest + lowest memory
    return YOLO("yolov8n.pt")


def get_stream_url(youtube_url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def main():
    st.set_page_config(
        page_title="YouTube Object Detection (YOLOv8)",
        layout="wide",
    )

    st.title("🎥 YouTube Live Object Detection (YOLOv8)")
    st.markdown("Runs fully inside **Streamlit Cloud** (CPU-only).")

    # ---------- Sidebar ----------
    with st.sidebar:
        youtube_url = st.text_input(
            "YouTube Video URL",
            value=DEFAULT_YOUTUBE_URL,
        )

        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )

        resolution = st.selectbox(
            "Resolution",
            options=[
                (640, 360),
                (854, 480),
                (1280, 720),
            ],
            index=1,
        )

        start = st.button("▶ Start Detection")
        stop = st.button("⏹ Stop")

    # ---------- State ----------
    if "running" not in st.session_state:
        st.session_state.running = False

    if start:
        st.session_state.running = True

    if stop:
        st.session_state.running = False

    # ---------- Video area ----------
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    if not st.session_state.running:
        st.info("Click **Start Detection** to begin.")
        return

    # ---------- Load model ----------
    model = load_model()

    # ---------- Open YouTube stream ----------
    try:
        stream_url = get_stream_url(youtube_url)
    except Exception as e:
        st.error(f"Failed to open YouTube stream: {e}")
        return

    cap = cv2.VideoCapture(stream_url)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    prev_time = time.time()

    # ---------- Streaming loop ----------
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream ended or cannot be read.")
            break

        frame = cv2.resize(frame, resolution)

        # YOLO inference (CPU-safe)
        results = model(
            frame,
            conf=confidence,
            imgsz=resolution[0],
            device="cpu",
            verbose=False,
        )

        annotated = results[0].plot()

        # Convert BGR → RGB for Streamlit
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            annotated,
            caption="YOLOv8 Detection",
            channels="RGB",
            use_container_width=True,
        )

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_placeholder.markdown(f"**FPS:** {fps:.2f}")

    cap.release()
    st.success("Detection stopped.")


if __name__ == "__main__":
    main()
