import streamlit as st
import subprocess
import numpy as np
import cv2
import time
from ultralytics import YOLO
import os

# Configuration
WIDTH = 640
HEIGHT = 360

def start_ffmpeg_pipe(youtube_url):
    """
    Stream YouTube frames using resilient format sorting and impersonation.
    """
    # Use sorting (-S) instead of strict format (-f) to avoid 'Format not available'
    # This prefers h264 video and aac audio at roughly 360p height
    ytdlp_cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        # 1. Impersonate multiple clients to bypass bot detection
        "--extractor-args", "youtube:player_client=ios,web,android;player_skip=webpage,configs",
        # 2. Use Format Sorting: 'Find the best h264/aac near 360p'
        "-S", "vcodec:h264,res:360,acodec:aac",
        # 3. Safety fallback: if sorting fails, just get the best single file available
        "-f", "best[ext=mp4]/best",
        "--no-playlist",
        "-o", "-",
        youtube_url
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-i", "pipe:0",  # Read from yt-dlp stdout
        "-an",           # Disable audio for processing
        "-vf", f"scale={WIDTH}:{HEIGHT},fps=10",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1"         # Output raw BGR to python
    ]

    # Launch processes
    ytdlp_proc = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=ytdlp_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    return ffmpeg_proc, ytdlp_proc

def main():
    st.set_page_config(page_title="YOLOv8 YouTube Stream", layout="wide")
    st.title("🎥 Resilient YouTube Object Detection")
    st.caption("Updated for 2025 YouTube Bot-Detection Bypass")

    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=smoU272Dv14")
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)
    
    if st.button("Start Detection"):
        model = YOLO("yolov8n.pt")
        
        with st.spinner("Connecting to YouTube..."):
            ffmpeg, ytdlp = start_ffmpeg_pipe(url)

        # UI Placeholders
        image_spot = st.empty()
        status_spot = st.empty()
        
        frame_size = WIDTH * HEIGHT * 3
        prev_time = time.time()

        try:
            while True:
                # Read raw bytes from ffmpeg
                raw_frame = ffmpeg.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    # Check if yt-dlp sent an error message
                    err = ytdlp.stderr.read().decode()
                    if err:
                        st.error(f"YouTube Error: {err}")
                    else:
                        st.warning("Stream ended unexpectedly.")
                    break

                # Convert to numpy and process
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # YOLO Inference
                results = model(frame, conf=conf, verbose=False)
                annotated_frame = results[0].plot()

                # Display in Streamlit
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                image_spot.image(display_frame, channels="RGB", use_container_width=True)

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                status_spot.text(f"Processing at {fps:.1f} FPS")
                prev_time = curr_time

        finally:
            ffmpeg.terminate()
            ytdlp.terminate()

if __name__ == "__main__":
    main()
