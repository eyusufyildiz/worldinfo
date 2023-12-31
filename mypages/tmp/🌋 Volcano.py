# https://github.com/pahlisch/streamlit-app-volcano
import pandas as pd
import json, requests
import streamlit as st

st.title("🌋 Volcanos")

url = "https://raw.githubusercontent.com/pahlisch/streamlit-app-volcano/master/data/clean_volcano_data.csv"
volnanos = pd.read_csv(url)

volnanos.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

st.write(f"Number of volcanos 🌋: {len(volnanos)}")
st.map(volnanos)
st.dataframe(volnanos)


