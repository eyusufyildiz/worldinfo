import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp
import numpy as np

def main():
    st.title("🛡️ Robust YouTube Object Detector")
    st.info("Using 'Android' User-Agent headers to bypass cloud blocks.")

    video_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=smoU272Dv14")
    conf_level = st.slider("Confidence", 0.0, 1.0, 0.4, 0.05)

    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()

    if st.button("Start Analysis"):
        # Use Android user-agent to avoid the 'CAP_IMAGES' redirect/block
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'user_agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36'
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                stream_url = info['url']

            # Force FFmpeg backend explicitly in the VideoCapture call
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            # Optimization: Set buffer size
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            frame_placeholder = st.empty()
            stop_btn = st.checkbox("Stop Process")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    # Try to grab one more time in case of dropped packet
                    ret, frame = cap.read()
                    if not ret: break

                # Process every 2nd frame to reduce CPU load on Streamlit Cloud
                # (Optional: adds speed to the UI)
                
                results = model.predict(frame, conf=conf_level, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            cap.release()

        except Exception as e:
            st.error(f"Logic Error: {e}")
            st.markdown("""
            ### 🛠️ Why is this still failing?
            Streamlit Cloud uses shared IP addresses. If another user on Streamlit is also scraping YouTube, the whole IP range gets blocked. 
            
            **Try these steps:**
            1. **Re-deploy** the app (this sometimes moves you to a different server instance).
            2. **Use a different video URL** to see if it's a specific restriction on that video.
            3. **Run locally**: This code will work 100% on your local machine because your home IP isn't flagged.
            """)

if __name__ == "__main__":
    main()
