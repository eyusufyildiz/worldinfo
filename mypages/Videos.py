import streamlit as st
import streamlit.components.v1 as components
from utils import tools as tool

#tool.streamlit_config(page_title="📡 Starlink / GPS Satellites", page_icon="📡")

def video_music():
    VIDEO_URL = "https://youtu.be/tW4KBk_JL7M"
    st.video(VIDEO_URL)
    st.audio("mypages/Audio.mpeg", format="audio/mpeg", loop=True)
