import streamlit as st
import subprocess
import numpy as np
import cv2
import time
from ultralytics import YOLO
import os

WIDTH = 640
HEIGHT = 360

def start_ffmpeg_pipe(youtube_url):
    """
    Stream YouTube video frames via ffmpeg pipe with anti-blocking headers.
    """
    # 1. Improved yt-dlp command to bypass blocks
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        # Use different clients to bypass 'bot detection'
        "--extractor-args", "youtube:player_client=ios,web,android", 
        # Spoof a real browser User-Agent
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # If running locally, you can use --cookies-from-browser chrome
        # On Streamlit Cloud, you might need a cookies.txt file uploaded
        # "--cookies", "cookies.txt", 
        "-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
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

    return ffmpeg, ytdlp

def main():
    st.set_page_config(layout="wide")
    st.title("🎥 YouTube Object Detection (Anti-Block Version)")

    st.info("Note: If the stream fails, YouTube may have blocked this IP. Try a different URL or use a cookies.txt file.")

    url = st.text_input(
        "YouTube URL",
        "https://www.youtube.com/watch?v=smoU272Dv14"
    )

    conf = st.slider("Confidence", 0.1, 1.0, 0.4)

    if not st.button("Start"):
        return

    model = YOLO("yolov8n.pt")

    with st.spinner("Initializing stream..."):
        ffmpeg_proc, ytdlp_proc = start_ffmpeg_pipe(url)

    frame_size = WIDTH * HEIGHT * 3
    image_box = st.empty()
    fps_box = st.empty()

    prev = time.time()

    try:
        while True:
            raw = ffmpeg_proc.stdout.read(frame_size)
            
            if len(raw) != frame_size:
                # Check for errors in yt-dlp
                st.error("Stream interrupted. YouTube is likely blocking this request.")
                break

            frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))

            # Run YOLO
            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()

            # Display
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            image_box.image(annotated, use_container_width=True)

            now = time.time()
            fps_box.caption(f"FPS: {1/(now-prev):.2f}")
            prev = now
    finally:
        ffmpeg_proc.terminate()
        ytdlp_proc.terminate()

if __name__ == "__main__":
    main()
