import streamlit as st
import cv2
import av
import streamlink
from ultralytics import YOLO
import numpy as np

def get_stream_url(youtube_url):
    try:
        session = streamlink.Streamlink()
        # Pretend to be a standard browser
        session.set_option("http-headers", "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        streams = session.streams(youtube_url)
        if streams:
            # Use 360p for stability on cloud
            return streams['360p'].url if '360p' in streams else streams['best'].url
    except Exception as e:
        st.error(f"Streamlink extraction failed: {e}")
    return None

def main():
    st.title("YOLOv8 + PyAV YouTube Stream")
    
    url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=j-hH64410UM")
    
    if "run" not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.columns(2)
    if col1.button("Start"): st.session_state.run = True
    if col2.button("Stop"): st.session_state.run = False

    output_image = st.empty()

    if st.session_state.run:
        model = YOLO("yolov8n.pt")
        stream_url = get_stream_url(url)
        
        if stream_url:
            try:
                # Use PyAV to open the stream instead of OpenCV
                container = av.open(stream_url)
                
                # Iterate through video frames
                for frame in container.decode(video=0):
                    if not st.session_state.run:
                        break
                    
                    # Convert PyAV frame to numpy array (RGB)
                    img = frame.to_image()
                    img_array = np.array(img)
                    
                    # Run YOLO
                    results = model(img_array, conf=0.4, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Display in Streamlit
                    output_image.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                container.close()
            except Exception as e:
                st.error(f"Streaming Error: {e}. YouTube is likely blocking this IP.")
                st.session_state.run = False

if __name__ == "__main__":
    main()
