import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import folium
import requests
from streamlit_folium import st_folium

st.set_page_config(page_title="🛰️ ISS (International Space Station) Now", page_icon="🛰️")

# Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
# after it's been refreshed 100 times.
# count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

ISS_POSITION_URL = "http://api.open-notify.org/iss-now.json"
ASTROS_URL = "http://api.open-notify.org/astros.json"
ISS_TRAIL_FILE = Path(__file__).with_name("iss_positions.json")
ISS_TRAIL_SECONDS = 60 * 60


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


def read_iss_trail():
    if not ISS_TRAIL_FILE.exists():
        return []

    try:
        trail = json.loads(ISS_TRAIL_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(trail, list):
        return []

    return recent_iss_trail(trail)


def write_iss_trail(trail):
    try:
        ISS_TRAIL_FILE.write_text(json.dumps(recent_iss_trail(trail)))
    except OSError:
        pass


def recent_iss_trail(trail):
    now = datetime.utcnow()
    recent_trail = []

    for position in trail:
        try:
            timestamp = datetime.strptime(position["timestamp"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, TypeError, ValueError):
            continue

        if (now - timestamp).total_seconds() <= ISS_TRAIL_SECONDS:
            recent_trail.append(position)

    return recent_trail


def update_iss_trail(lat, lon, timestamp):
    trail = recent_iss_trail(st.session_state.setdefault("iss_trail", read_iss_trail()))
    if not trail or trail[-1]["timestamp"] != timestamp:
        trail.append({"lat": lat, "lon": lon, "timestamp": timestamp})
        trail = recent_iss_trail(trail)
        st.session_state["iss_trail"] = trail
        write_iss_trail(trail)

    return pd.DataFrame(trail)


def build_path_segments(trail):
    if len(trail) < 2:
        return pd.DataFrame(columns=["from_lon", "from_lat", "to_lon", "to_lat"])

    segments = []
    points = trail.to_dict("records")
    for start, end in zip(points, points[1:]):
        segments.append(
            {
                "from_lon": start["lon"],
                "from_lat": start["lat"],
                "to_lon": end["lon"],
                "to_lat": end["lat"],
            }
        )

    return pd.DataFrame(segments)


def previous_positions(trail):
    if len(trail) <= 1:
        return trail.iloc[0:0]

    return trail.iloc[:-1]


def draw_live_iss_map(current_position, trail):
    path_segments = build_path_segments(trail)
    previous_trail = previous_positions(trail)
    current = current_position.iloc[0]
    iss_map = folium.Map(
        location=[float(current["lat"]), float(current["lon"])],
        zoom_start=3,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    if len(path_segments):
        path_points = [
            [float(row["from_lat"]), float(row["from_lon"])]
            for _, row in path_segments.iterrows()
        ]
        last_segment = path_segments.iloc[-1]
        path_points.append([float(last_segment["to_lat"]), float(last_segment["to_lon"])])
        folium.PolyLine(path_points, color="#1971c2", weight=3, opacity=0.8).add_to(iss_map)

    for _, position in previous_trail.iterrows():
        folium.CircleMarker(
            location=[float(position["lat"]), float(position["lon"])],
            radius=3,
            color="#1971c2",
            fill=True,
            fill_color="#1971c2",
            fill_opacity=0.55,
            tooltip=f"Previous ISS position<br>{position['timestamp']}",
        ).add_to(iss_map)

    folium.CircleMarker(
        location=[float(current["lat"]), float(current["lon"])],
        radius=5,
        color="#c92a2a",
        fill=True,
        fill_color="#f03e3e",
        fill_opacity=0.9,
        tooltip=(
            f"Current ISS position<br>"
            f"Lat: {current['lat']}<br>"
            f"Lon: {current['lon']}<br>"
            f"Time: {current['timestamp']}"
        ),
    ).add_to(iss_map)

    st_folium(
        iss_map,
        height=500,
        use_container_width=True,
        returned_objects=[],
        key="iss-live-map",
    )


def number_of_people_now():
    people = get_people_in_space()
    number_of_people = len(people)

    st.write(f"""#### People in Space Right Now:\n
      There are currently {number_of_people} humans in space. They are:""")

    data1  = pd.json_normalize(people)
    st.write(data1)

@st.fragment(run_every="5s")
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


def iss():
    iss_now1()
    with st.expander("People in Space Right Now"):
        number_of_people_now()
    
