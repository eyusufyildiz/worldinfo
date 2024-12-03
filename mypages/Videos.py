import streamlit as st
import streamlit.components.v1 as components
from utils import tools as tool

#tool.streamlit_config(page_title="📡 Starlink / GPS Satellites", page_icon="📡")

def music_video():
    st.video("https://youtu.be/tW4KBk_JL7M")
     st.video("https://www.youtube.com/watch?v=tVWOBcG62HY")
    st.audio("mypages/Audio.mpeg", format="audio/mpeg", loop=True)
    st.audio("mypages/mzk/Altan Urag - No Mercy [ ezmp3.cc ].mp3", format="audio/mpeg", loop=True)
