import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from utils import tools as tool

tool.streamlit_config(page_title="🛰️ ISS (International Space Station) Now", page_icon="🛰️")

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
# count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

def number_of_people_now():
    url2 = "http://api.open-notify.org/astros.json"
    # st.write(url2)
    obj2 = tool.http_requests(url2, type="json")
    obj2 = obj2["people"]
    number_of_people = len(obj2)

    st.write(f"""#### People in Space Right Now:\n
      There are currently {number_of_people} humans in space. They are:""")
    
    data1  = pd.json_normalize(obj2)
    st.write(data1)

def iss_now1():
    st_autorefresh(interval=5000)
    url1 = "http://api.open-notify.org/iss-now.json"
    obj1 = tool.http_requests(url1, type="json")
    ts = obj1["timestamp"] 
    dt = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    obj1["timestamp"] = dt
    
    pos={"lat": float(obj1["iss_position"]['latitude']), "lon": float(obj1["iss_position"]['longitude']) }
    pd_pos = pd.json_normalize(pos)

    st.markdown("### 🛰️ Current ISS Location")
    st.markdown("The International Space Station is moving at close to 28,000 km/h so its location changes really fast! Where is it right now?")

    fig = tool.plotly_map(pd_pos, lat="lat", lon="lon", title= "ISS now" )
    zaman = obj1["timestamp"]
    lon = obj1["iss_position"]["longitude"]
    lat = obj1["iss_position"]["latitude"]
    st.success( f"**{zaman}**  [{lat}, {lon}]")
    
    if tool.geo_reverse(lat, lon):
        st.write("ISS is now on below address/place:")
        tbl = pd.json_normalize( tool.geo_reverse(lat, lon) )
        st.write( tbl )

    # st.write(fig)
    st.map(pd_pos, zoom=3, use_container_width=True)

def iss():
    st.container()
    # st_autorefresh(interval=5000)
    iss_now1()
    
    with st.expander("People in Space Right Now"):
        number_of_people_now()
