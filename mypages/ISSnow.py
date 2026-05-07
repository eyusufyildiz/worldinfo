import streamlit as st
import pandas as pd
from datetime import datetime
import pydeck as pdk
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
ISS_TRAIL_LIMIT = 24


@st.cache_data(ttl=4, show_spinner=False)
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

    try:
        location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=5)
    except Exception:
        return None

    if not location:
        return None

    return location.raw.get("address")


def update_iss_trail(lat, lon, timestamp):
    trail = st.session_state.setdefault("iss_trail", [])
    if not trail or trail[-1]["timestamp"] != timestamp:
        trail.append({"lat": lat, "lon": lon, "timestamp": timestamp})
        st.session_state["iss_trail"] = trail[-ISS_TRAIL_LIMIT:]

    return pd.DataFrame(st.session_state["iss_trail"])


def draw_live_iss_map(current_position, trail):
    current_layer = pdk.Layer(
        "ScatterplotLayer",
        data=current_position,
        get_position="[lon, lat]",
        get_radius=180000,
        get_fill_color=[240, 62, 62, 220],
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
        pickable=True,
    )
    trail_layer = pdk.Layer(
        "ScatterplotLayer",
        data=trail,
        get_position="[lon, lat]",
        get_radius=55000,
        get_fill_color=[25, 113, 194, 120],
    )
    view_state = pdk.ViewState(
        latitude=float(current_position.iloc[0]["lat"]),
        longitude=float(current_position.iloc[0]["lon"]),
        zoom=1.8,
        pitch=0,
    )
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[trail_layer, current_layer],
        tooltip={"text": "ISS\nLat: {lat}\nLon: {lon}\nTime: {timestamp}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


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
    
    pos = {
        "lat": float(obj1["iss_position"]['latitude']),
        "lon": float(obj1["iss_position"]['longitude']),
        "timestamp": dt,
    }
    pd_pos = pd.json_normalize(pos)
    trail = update_iss_trail(pos["lat"], pos["lon"], pos["timestamp"])

    st.markdown("### 🛰️ Current ISS Location")
    st.markdown("The International Space Station is moving at close to 28,000 km/h so its location changes really fast! Where is it right now?")

    zaman = obj1["timestamp"]
    lon = obj1["iss_position"]["longitude"]
    lat = obj1["iss_position"]["latitude"]
    st.success( f"**{zaman}**  [{lat}, {lon}]")

    address = reverse_geocode_position(round(float(lat), 1), round(float(lon), 1))
    if address:
        st.write("ISS is now on below address/place:")
        tbl = pd.json_normalize(address)
        st.write( tbl )

    draw_live_iss_map(pd_pos, trail)
    if st_autorefresh:
        st_autorefresh(interval=5000, key="iss-refresh")


def iss():
    iss_now1()
    with st.expander("People in Space Right Now"):
        number_of_people_now()
    
