import streamlit as st
import streamlit.components.v1 as components
from utils import tools as tool

#tool.streamlit_config(page_title="📡 Starlink / GPS Satellites", page_icon="📡")

def satellites():
    st.audio("Audio.mpeg", format="audio/mpeg", loop=True)
    st.container()
    sat = '<iframe src="https://satellitemap.space/" width="800px" height="600px" frameborder="0" title="Starlink Satellite Map"></iframe>'
    components.html(sat, height=600)


