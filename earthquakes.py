import streamlit as st
import pandas as pd
import json, requests


opt_list={
    "Past Hour - Significant Earthquakes": "significant_hour.geojson",
    "Past Hour - M4.5+ Earthquakes": "4.5_hour.geojson",
    "Past Hour - M2.5+ Earthquakes": "2.5_hour.geojson",
    "Past Hour - M1.0+ Earthquakes": "1.0_hour.geojson",
    "Past Hour - All Earthquakes": "all_hour.geojson",

    "Past Day - Significant Earthquakes": "significant_day.geojson",
    "Past Day - M4.5+ Earthquakes": "4.5_day.geojson",
    "Past Day - M2.5+ Earthquakes": "2.5_day.geojson",
    "Past Day - M1.0+ Earthquakes": "1.0_day.geojson",
    "Past Day - All Earthquakes": "all_day.geojson",

    "Past Week - Significant Earthquakes": "significant_week.geojson",
    "Past Week - M4.5+ Earthquakes": "4.5_week.geojson",
    "Past Week - M2.5+ Earthquakes": "2.5_week.geojson",
    "Past Week - M1.0+ Earthquakes": "1.0_week.geojson",
    "Past Week - All Earthquakes": "all_week.geojson",

    "Past 30 Days - Significant Earthquakes": "significant_month.geojson",
    "Past 30 Days - M4.5+ Earthquakes": "4.5_month.geojson",
    "Past 30 Days - M2.5+ Earthquakes": "2.5_month.geojson",
    "Past 30 Days - M1.0+ Earthquakes": "1.0_month.geojson",
    "Past 30 Days - All Earthquakes": "all_month.geojson" 
}


option = st.selectbox( "Select Period: ", set(opt_list.keys()), )
opt=opt_list[option]
url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{opt}"
res = requests.get(url).json()['features']
quakes =[]


for q in res :
    x= q['properties']
    x['lat'] = q['geometry']['coordinates'][1]
    x['lon'] = q['geometry']['coordinates'][0]
    x['depth'] = q['geometry']['coordinates'][2]
    quakes.append( x )


quakes = pd.json_normalize(quakes)

st.write(f"Number of {option}: {len(quakes)}")
st.map(quakes)

if len(quakes):
    sorted_quakes = quakes.sort_values(by=['mag'], ascending=False)
    st.write(sorted_quakes)
