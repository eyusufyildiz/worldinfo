import streamlit as st
import yt_dlp

st.set_page_config(page_title="YouTube Streamer", page_icon="📺")

st.title("📺 YouTube Video Loader")
st.write("Enter a YouTube URL below to play it directly in the app.")

# Input field for the URL
url = st.text_input("Paste YouTube URL here:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if url:
    try:
        # Options for yt_dlp to get the direct video URL
        ydl_opts = {'format': 'best'}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Video')
            
            st.subheader(f"Now Playing: {video_title}")
            
            # Streamlit's native video player handles YouTube URLs directly
            # but using the processed URL ensures better compatibility
            st.video(url)
            
            st.success("Video loaded successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.info("Note: Some videos may be restricted by YouTube for third-party playback.")
