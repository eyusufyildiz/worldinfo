import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp
import os
import tempfile

def main():
    st.title("🔍 YOLOv8 Video Object Detector")
    
    # Selection for input source
    source_type = st.radio("Select Source:", ("YouTube URL", "Upload Video File"))
    
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()

    video_path = None

    if source_type == "YouTube URL":
        url = st.text_input("YouTube URL", value="https://www.youtube.com/watch?v=smoU272Dv14")
        if st.button("Download & Process"):
            with st.spinner("Downloading video (this bypasses stream blocks)..."):
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': 'temp_video.mp4',
                    'quiet': True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    video_path = "temp_video.mp4"
                except Exception as e:
                    st.error(f"YouTube Download Failed: {e}")
                    st.info("YouTube often blocks cloud servers. Please use the 'Upload' option below.")

    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    if video_path:
        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()
        stop_btn = st.button("Stop Processing")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Inference
            results = model.predict(frame, conf=conf_level, verbose=False)
            annotated_frame = results[0].plot()
            
            # Display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()
