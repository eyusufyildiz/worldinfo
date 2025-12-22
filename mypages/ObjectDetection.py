import streamlit as st
import cv2
import numpy as np
import tempfile
import subprocess
from ultralytics import YOLO

DEFAULT_URL = "https://www.youtube.com/watch?v=smoU272Dv14"

def extract_stream_url(youtube_url):
    """
    Extract direct stream URL using yt-dlp
    Works best-effort in Streamlit Cloud
    """
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-g",
        "--no-warnings",
        "--quiet",
        youtube_url
    ]

    try:
        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=15
        ).decode().strip()

        if not result.startswith("http"):
            raise RuntimeError("Invalid stream URL")

        return result

    except Exception as e:
        return None


def detect_objects(video_url, conf_threshold):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        st.error("❌ Unable to open video stream.")
        return

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=conf_threshold,
            verbose=False
        )

        annotated = results[0].plot()

        frame_placeholder.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

        info_placeholder.markdown(
            f"**Detections:** {len(results[0].boxes)} | "
            f"**Confidence ≥ {conf_threshold:.2f}**"
        )

    cap.release()


def main():
    st.set_page_config(
        page_title="YouTube Object Detection",
        layout="wide"
    )

    st.title("🎯 YouTube Object Detection (Streamlit Cloud)")
    st.caption("YOLOv8 · No sidebar · Cloud-safe")

    youtube_url = st.text_input(
        "YouTube Video URL",
        value=DEFAULT_URL
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

    if st.button("▶ Start Detection"):
        with st.spinner("Extracting video stream..."):
            stream_url = extract_stream_url(youtube_url)

        if not stream_url:
            st.warning(
                "⚠️ YouTube blocked stream access.\n\n"
                "This is common on Streamlit Cloud.\n"
                "Try another video or add cookies support."
            )
            return

        detect_objects(stream_url, confidence)


if __name__ == "__main__":
    main()
