import streamlit as st
import pandas as pd
import json, requests

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
res = requests.get(url).json()['features']
quakes =[]

for q in res :
    x= q['properties']
    x['lat'] = q['geometry']['coordinates'][1]
    x['lon'] = q['geometry']['coordinates'][0]
    x['depth'] = q['geometry']['coordinates'][2]
    quakes.append( x )
    if x['mag'] >= 6: x['color'] = 'red'
    elif 5<= x['mag'] < 6: x['color'] = 'orange'
    elif 4.5<= x['mag'] < 5: x['color'] = 'yellow'
    elif 4<= x['mag'] < 4.5: x['color'] = 'blue'
    elif 3.5<= x['mag'] < 4: x['color'] = 'cyan'
    else: x['color'] = 'green'

quakes = pd.json_normalize(quakes)
qa=[quakes.lat, quakes.lon, quakes.place

st.write(f"Number of earthquakes (>4.5) in last week: ** {len(quakes)} **")

    st.map(qa)

sorted_quakes = quakes.sort_values(by=['mag'], ascending=False)
st.write(sorted_quakes)
