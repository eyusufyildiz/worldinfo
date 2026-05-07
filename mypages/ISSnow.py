import streamlit as st
import pandas as pd
from datetime import datetime
import requests

try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = None

st.set_page_config(page_title="🛰️ ISS (International Space Station) Now", page_icon="🛰️")

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
# count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

ISS_POSITION_URL = "http://api.open-notify.org/iss-now.json"
ASTROS_URL = "http://api.open-notify.org/astros.json"


@st.cache_data(ttl=25, show_spinner=False)
def get_iss_position():
    response = requests.get(ISS_POSITION_URL, timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def get_people_in_space():
    response = requests.get(ASTROS_URL, timeout=5)
    response.raise_for_status()
    return response.json().get("people", [])


@st.cache_data(ttl=3600, show_spinner=False)
def reverse_geocode_position(lat, lon):
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="worldinfo_iss_now")
    rounded_lat = round(float(lat), 1)
    rounded_lon = round(float(lon), 1)

    try:
        location = geolocator.reverse(f"{rounded_lat}, {rounded_lon}", language="en", timeout=5)
    except Exception:
        return None

    if not location:
        return None

    return location.raw.get("address")


def number_of_people_now():
    people = get_people_in_space()
    number_of_people = len(people)

    st.write(f"""#### People in Space Right Now:\n
      There are currently {number_of_people} humans in space. They are:""")

    data1  = pd.json_normalize(people)
    st.write(data1)

def iss_now1():
    obj1 = get_iss_position()
    ts = obj1["timestamp"] 
    dt = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    obj1["timestamp"] = dt
    
    pos={"lat": float(obj1["iss_position"]['latitude']), "lon": float(obj1["iss_position"]['longitude']) }
    pd_pos = pd.json_normalize(pos)

    st.markdown("### 🛰️ Current ISS Location")
    st.markdown("The International Space Station is moving at close to 28,000 km/h so its location changes really fast! Where is it right now?")

    zaman = obj1["timestamp"]
    lon = obj1["iss_position"]["longitude"]
    lat = obj1["iss_position"]["latitude"]
    st.success( f"**{zaman}**  [{lat}, {lon}]")

    address = reverse_geocode_position(lat, lon)
    if address:
        st.write("ISS is now on below address/place:")
        tbl = pd.json_normalize(address)
        st.write( tbl )

    st.map(pd_pos, zoom=3, use_container_width=True)
    if st_autorefresh:
        st_autorefresh(interval=30000, key="iss-refresh")


def iss():
    iss_now1()
    with st.expander("People in Space Right Now"):
        number_of_people_now()
    
