import streamlit as st
import subprocess
import numpy as np
import cv2
import time
from ultralytics import YOLO


WIDTH = 640
HEIGHT = 360


def start_ffmpeg_pipe(youtube_url):
    """
    Stream YouTube video frames as raw BGR via ffmpeg pipe
    """
    cmd = [
    "yt-dlp",
    "--quiet",
    "--no-warnings",
    "--merge-output-format", "mp4",
    "-f", "bv*[ext=mp4][height<=360]+ba[ext=m4a]/b[ext=mp4]/b",
    "--extractor-args", "youtube:player_client=android",
    "--no-playlist",
    "-o", "-",
    youtube_url
]


    ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "quiet",
    "-i", "pipe:0",
    "-an",
    "-vf", f"scale={WIDTH}:{HEIGHT},fps=10",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "pipe:1"
]


    ytdlp = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=ytdlp.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    return ffmpeg


def main():
    st.set_page_config(layout="wide")
    st.title("🎥 YouTube Object Detection (Streamlit Cloud – FIXED)")

    url = st.text_input(
        "YouTube URL",
        "https://www.youtube.com/watch?v=smoU272Dv14"
    )

    conf = st.slider("Confidence", 0.1, 1.0, 0.4)

    if not st.button("Start"):
        return

    model = YOLO("yolov8n.pt")

    with st.spinner("Starting stream..."):
        ffmpeg = start_ffmpeg_pipe(url)

    frame_size = WIDTH * HEIGHT * 3
    image_box = st.empty()
    fps_box = st.empty()

    prev = time.time()

    while True:
        raw = ffmpeg.stdout.read(frame_size)
        if len(raw) != frame_size:
            st.warning("Stream ended or blocked by YouTube.")
            break

        frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image_box.image(annotated, use_container_width=True)

        now = time.time()
        fps_box.caption(f"FPS: {1/(now-prev):.2f}")
        prev = now


if __name__ == "__main__":
    main()
