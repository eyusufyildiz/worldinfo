import streamlit as st
import yt_dlp

def fetch_video_info(url):
    """
    Helper function to extract video metadata using yt-dlp.
    """
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)

def main():
    # Page Configuration
    st.set_page_config(page_title="Streamlit YT Player", page_icon="🎥")

    # UI Header
    st.title("🎥 YouTube Cloud Player")
    st.markdown("---")

    # Sidebar for settings/info
    st.sidebar.header("About")
    st.sidebar.info("This app runs in a Streamlit Cloud container and uses `yt-dlp` to process YouTube content.")

    # User Input
    url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

    if url:
        with st.spinner("Fetching video details..."):
            try:
                info = fetch_video_info(url)
                
                # Display Video Title and Thumbnail
                st.subheader(info.get('title', 'Video Preview'))
                
                # Streamlit's built-in video widget
                st.video(url)
                
                # Optional: Show Metadata in an expander
                with st.expander("View Video Metadata"):
                    st.json({
                        "Channel": info.get("uploader"),
                        "Views": info.get("view_count"),
                        "Duration": f"{info.get('duration')} seconds",
                        "Upload Date": info.get("upload_date")
                    })

            except Exception as e:
                st.error(f"Error: Could not retrieve video. {e}")
    else:
        st.write("Please enter a valid link to begin.")

# The entry point for the Streamlit container
if __name__ == "__main__":
    main()
