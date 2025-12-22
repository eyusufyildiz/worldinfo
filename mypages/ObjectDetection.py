import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
import time


def get_youtube_stream_url(youtube_url: str) -> str | None:
    """
    Try to extract a playable stream URL without cookies or download.
    Uses Android client to reduce bot detection.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("url")
    except Exception as e:
        st.error("YouTube blocked this request from Streamlit Cloud.")
        st.caption(str(e))
        return None


def main():
    st.set_page_config(layout="wide")
    st.title("🎥 YouTube Object Detection (Streamlit Cloud)")

    youtube_url = st.text_input(
        "YouTube Video URL",
        value="https://www.youtube.com/watch?v=smoU272Dv14"
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

    start = st.button("Start Detection")

    if not start:
        return

    with st.spinner("Loading model..."):
        model = YOLO("yolov8n.pt")

    with st.spinner("Connecting to YouTube stream..."):
        stream_url = get_youtube_stream_url(youtube_url)

    if not stream_url:
        return

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        st.error("Failed to open video stream.")
        return

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Stream ended or blocked.")
            break

        results = model(frame, conf=confidence, verbose=False)

        annotated = results[0].plot()

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        fps_placeholder.caption(f"FPS: {fps:.2f}")

    cap.release()


if __name__ == "__main__":
    main()
