import streamlit as st
import pandas as pd
import json, requests

colors={"mag >= 6": 'red',
        "5 <= mag <6": 'orange',
        "4.5 <= mag <5": 'yellow',
        "4 <= mag < 4.5": 'blue',
        "3.5 <= mag < 4": 'cyan',
        "mag < 3.5": 'green'}

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

st.write(f"Number of earthquakes (>4.5) in last week: {len(quakes)}")


sorted_quakes = quakes.sort_values(by=['mag'], ascending=False)
#st.line_chart(sorted_quakes)
st.print(sorted_quakes)
