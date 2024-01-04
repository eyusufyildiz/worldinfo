import pandas as pd
import json, requests
import streamlit as st
from datetime import datetime
from utils import tools as tool
import time

#tool.streamlit_config(page_title="🌀 Eartquakes", page_icon="🌀")

def quakes():
    opt_list={
        "1h - Significant Earthquakes": "significant_hour.geojson",
        "1h - M4.5+ Earthquakes": "4.5_hour.geojson",
        "1h - M2.5+ Earthquakes": "2.5_hour.geojson",
        "1h - M1.0+ Earthquakes": "1.0_hour.geojson",
        "1h - All Earthquakes": "all_hour.geojson",

        "1d - Significant Earthquakes": "significant_day.geojson",
        "1d - M4.5+ Earthquakes": "4.5_day.geojson",
        "1d - M2.5+ Earthquakes": "2.5_day.geojson",
        "1d - M1.0+ Earthquakes": "1.0_day.geojson",
        "1d - All Earthquakes": "all_day.geojson",

        "1w - Significant Earthquakes": "significant_week.geojson",
        "1w - M4.5+ Earthquakes": "4.5_week.geojson",
        "1w - M2.5+ Earthquakes": "2.5_week.geojson",
        "1w - M1.0+ Earthquakes": "1.0_week.geojson",
        "1w - All Earthquakes": "all_week.geojson",

        "30d - Significant Earthquakes": "significant_month.geojson",
        "30d - M4.5+ Earthquakes": "4.5_month.geojson",
        "30d - M2.5+ Earthquakes": "2.5_month.geojson",
        "30d - M1.0+ Earthquakes": "1.0_month.geojson",
        "30d - All Earthquakes": "all_month.geojson" 
    }


    col1, col2 = st.columns(2)

    with col1:
        past = st.select_slider("Last", options=["1h", "1d", "1w", "30d"], value=('1d'))
    with col2:
        mgn = st.select_slider("Magnitute", options=["Significant", "M4.5+", "M2.5+", "M1.0+", "All"], value=('M2.5+') )

    #option = opt_list[opt in opt_list if past in opt and mgn in opt][0]

    for opt in opt_list:
        if past in opt and mgn in opt:
            option = opt_list[opt]


    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{option}"

    res = requests.get(url).json()['features']
    quakes =[]

    if res:
        for q in res :
            x= q['properties']
            x['lat'] = q['geometry']['coordinates'][1]
            x['lon'] = q['geometry']['coordinates'][0]
            x['depth'] = q['geometry']['coordinates'][2]
            quakes.append( x )
    
        quakes=pd.DataFrame(quakes)
        quakes = quakes.filter(['mag', 'place', 'lat', 'lon', 'time', 'url'], axis=1)
        quakes['time'] =  pd.to_datetime(quakes['time'], unit='ms')
    
        st.write(f"Number of {mgn} eartquake(s) in {past}: {len(quakes)}")
    
        if len(quakes):
            st.map(quakes)
            with st.expander("Earthquakes list"):
                quakes.sort_values(by=['mag'], ascending=False)
                st.write(quakes)

        
