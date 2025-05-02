# https://github.com/pahlisch/streamlit-app-volcano
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from utils import tools as tool

#tool.streamlit_config(page_title="🌋 Volcanos", page_icon="🌋")

def http_requests(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    return requests.get(url, headers=headers)

def _get_volcanos():
    url1="https://volcano.oregonstate.edu/volcano_table"
    url2="https://volcano.oregonstate.edu/volcano_table?page=1"
    url3="https://volcano.oregonstate.edu/volcano_table?page=2"
    volcanos = pd.DataFrame()
    df1 = pd.read_html(http_requests(url1).text)[0]
    df2 = pd.read_html(http_requests(url2).text)[0]
    df3 = pd.read_html(http_requests(url3).text)[0]

    volcanos = pd.concat([df1, df2, df3], ignore_index=True)
    #volnanos = volnanos.sort_values(by=['Country'])
    st.write(f"Number of volcanos 🌋: {len(volcanos)} from {url1}")

    fig = tool.plotly_map(volcanos, lat="Latitude (dd)", lon="Longitude (dd)", 
                        title= "Volcanos", hover_name="Volcano Name", 
                        hover_data=["Country",  "Elevation (m)"])

    st.write(fig)

    with st.expander("Volcano list"):
        st.dataframe(volcanos)




def get_volcanos():
    volcan_file="mypages/mzk/GVP_Volcano_List_Holocene_202505011922.csv"
    volcanos = pd.read_csv(volcan_file)
    st.write(f"Number of volcanos 🌋: {len(volcanos)}")

    fig = tool.plotly_map(volcanos, lat="Latitude", lon="Longitude", 
                        title= "Volcanos", hover_name="Volcano Name", 
                        hover_data=["Country",  "Elevation (m)"])

    st.write(fig)

    with st.expander("Volcano list"):
        st.dataframe(volcanos)
