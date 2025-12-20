import streamlit as st
import cv2
import yt_dlp
import time
from ultralytics import YOLO

# ================= CONFIG =================
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=ztmY_cCtUl0"
DEFAULT_RESOLUTION = (854, 480)
MAX_RETRIES = 3
# =========================================


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano = fastest


def get_stream_url(youtube_url: str) -> str:
    """
    Always prefer HLS (m3u8) – most stable in Streamlit Cloud
    """
    ydl_opts = {
        "quiet": True,
        'cookies': 'cookies.txt',
        "format": "best",
        "noplaylist": True,
        "live_from_start": True,
        "extractor_args": {
            "youtube": {
                "skip": ["dash"],
            }
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

        if "hlsManifestUrl" in info:
            return info["hlsManifestUrl"]

        return info["url"]


def main():
    st.set_page_config(
        page_title="YouTube Live Object Detection (YOLOv8)",
        layout="wide",
    )

    st.title("🎥 YouTube Live Object Detection (YOLOv8)")
    st.caption("Runs fully inside Streamlit Cloud (CPU-only, HLS, auto-reconnect).")

    # -------- Sidebar controls --------
    #with st.sidebar:
    youtube_url = st.text_input(
        "YouTube URL",
        DEFAULT_YOUTUBE_URL,
    )

    confidence = st.slider(
        "Detection confidence",
        0.1,
        0.9,
        0.5,
        0.05,
    )

    resolution = st.selectbox(
        "Resolution",
        [(640, 360), (854, 480), (1280, 720)],
        index=1,
    )

    start = st.button("▶ Start")
    stop = st.button("⏹ Stop")

    # -------- State --------
    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True

    if stop:
        st.session_state.run = False

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    status_placeholder = st.empty()

    if not st.session_state.run:
        status_placeholder.info("Click ▶ Start to begin detection.")
        return

    # -------- Load model --------
    model = load_model()

    # -------- Open stream --------
    try:
        stream_url = get_stream_url(youtube_url)
    except Exception as e:
        st.error(f"Failed to extract stream URL: {e}")
        return

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    retries = 0
    prev_time = time.time()

    # -------- Streaming loop --------
    while st.session_state.run:
        ret, frame = cap.read()

        if not ret:
            retries += 1
            status_placeholder.warning("Stream expired – reconnecting...")
            cap.release()
            time.sleep(2)

            if retries > MAX_RETRIES:
                st.error("YouTube stream blocked or expired.")
                break

            stream_url = get_stream_url(youtube_url)
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            continue

        retries = 0

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
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            annotated,
            use_container_width=True,
        )

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        fps_placeholder.markdown(f"**FPS:** {fps:.2f}")

    cap.release()
    status_placeholder.success("Detection stopped.")


if __name__ == "__main__":
    main()
