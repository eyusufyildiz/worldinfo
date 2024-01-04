# https://github.com/pahlisch/streamlit-app-volcano
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import tools as tool

#tool.streamlit_config(page_title="🌋 Volcanos", page_icon="🌋")

def get_volcanos():
    url1="https://volcano.oregonstate.edu/volcano_table"
    url2="https://volcano.oregonstate.edu/volcano_table?page=1"
    url3="https://volcano.oregonstate.edu/volcano_table?page=2"
    volcanos = pd.DataFrame()
    df1 = pd.read_html(url1)[0]
    df2 = pd.read_html(url2)[0]
    df3 = pd.read_html(url3)[0]

    volcanos = pd.concat([df1, df2, df3], ignore_index=True)
    #volnanos = volnanos.sort_values(by=['Country'])
    st.write(f"Number of volcanos 🌋: {len(volcanos)} from {url1}")

    fig = tool.plotly_map(volcanos, lat="Latitude (dd)", lon="Longitude d(d)", 
                        title= "Volcanos", hover_name="Volcano Name", 
                        hover_data=["Country",  "Elevation (m)"])

    st.write(fig)
    with st.expander("Volcano list"):
        st.dataframe(volcanos)
