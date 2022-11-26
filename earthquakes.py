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
fig = px.scatter_mapbox(quakes, lat="lat", lon="lon", 
                        title= "Number of earthquakes (>4.5) in last week:",
                        hover_name="place", size="mag", 
                        #animation_frame = 'time', animation_group = 'place', 
                        #color_continuous_scale=px.colors.cyclical.HSV,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        hover_data=['title', "mag", "depth", "time", 'alert', "type", "tsunami"],
                        color_discrete_sequence=[quakes.color], zoom=2, height=500, color='mag')
                        #color_discrete_sequence=[quakes.color], zoom=2, height=500)
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.update_layout(autosize=True, width=1500, height=700)

st.markdown(f"Number of earthquakes (>4.5) in last week: ** {len(quakes)} **")
st.pyplot(fig )

sorted_quakes = quakes.sort_values(by=['mag'], ascending=False)
st.write(sorted_quakes)
