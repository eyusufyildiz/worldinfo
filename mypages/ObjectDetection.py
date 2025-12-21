import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt") 

def main():
    st.set_page_config(page_title="Fast YT Detector")
    st.title("⚡ High-Speed Object Detection")

    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    frame_placeholder = st.empty()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        res_choice = st.selectbox("Resolution", ["360p", "480p"], index=0)
    with col2:
        # LOWER THIS for speed: skipping 5 frames means only processing 6fps instead of 30fps
        frame_skip = st.slider("Frame Skip (Speedup)", 1, 10, 5) 
    with col3:
        run_btn = st.checkbox("Start")

    if run_btn and url:
        try:
            ydl_opts = {'format': 'best[height<=360]', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                stream_url = ydl.extract_info(url, download=False)['url']

            cap = cv2.VideoCapture(stream_url)
            model = load_yolo_model()
            
            count = 0
            while cap.isOpened() and run_btn:
                ret, frame = cap.read()
                if not ret: break

                # --- SPEED OPTIMIZATION: FRAME SKIPPING ---
                if count % frame_skip == 0:
                    # Resize frame slightly to speed up YOLO even more
                    # imgsz=320 is the "Nano" standard and very fast
                    results = model(frame, conf=0.4, imgsz=320, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Convert and display
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                count += 1

            cap.release()
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
