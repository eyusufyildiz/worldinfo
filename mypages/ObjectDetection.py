import streamlit as st
import cv2
import numpy as np
import yt_dlp

# --- MODEL CONFIGURATION ---
# These are standard COCO class labels for MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

@st.cache_resource
def load_model():
    """Loads the pre-trained MobileNet-SSD model."""
    # Note: In a real repo, you would include these .prototxt and .caffemodel files
    net = cv2.dnn.readNetFromCaffe(
        cv2.samples.findFile("deploy.prototxt"), 
        cv2.samples.findFile("mobilenet_iter_73000.caffemodel")
    )
    return net

def main():
    st.set_page_config(page_title="YT Object Detector")
    st.title("🔍 YouTube Object Detection")

    url = st.text_input("YouTube URL:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    run_detection = st.checkbox("Start Detection")

    # Placeholder for the video frames
    frame_placeholder = st.empty()

    if run_detection and url:
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']

        cap = cv2.VideoCapture(stream_url)
        net = load_model()

        while cap.isOpened() and run_detection:
            ret, frame = cap.read()
            if not ret:
                break

            # Object Detection Logic
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw label and bounding box
                    label = f"{CLASSES[idx]}: {confidence:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert BGR to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

            if not run_detection:
                break
        
        cap.release()

if __name__ == "__main__":
    main()
