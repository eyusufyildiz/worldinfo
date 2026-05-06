import pandas as pd
import json, requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import time

#tool.streamlit_config(page_title="🌀 Eartquakes", page_icon="🌀")

CACHE_FILE = Path(__file__).with_name("earthquakes_cache.json")
CACHE_TTL_SECONDS = 10 * 60


def read_earthquake_cache():
    if not CACHE_FILE.exists():
        return {"feeds": {}}

    try:
        return json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"feeds": {}}


def write_earthquake_cache(cache):
    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except OSError:
        pass


def get_earthquake_features(url):
    cache = read_earthquake_cache()
    feeds = cache.setdefault("feeds", {})
    cached_feed = feeds.get(url)
    now = time.time()

    if cached_feed and now - cached_feed.get("fetched_at", 0) < CACHE_TTL_SECONDS:
        return cached_feed.get("features", [])

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        features = response.json()["features"]
    except (requests.RequestException, ValueError, KeyError):
        if cached_feed:
            return cached_feed.get("features", [])
        raise

    feeds[url] = {
        "fetched_at": now,
        "features": features,
    }
    write_earthquake_cache(cache)
    return features


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_earthquake_data(url):
    features = get_earthquake_features(url)
    quakes = []

    for quake in features:
        properties = quake["properties"].copy()
        properties["lat"] = quake["geometry"]["coordinates"][1]
        properties["lon"] = quake["geometry"]["coordinates"][0]
        properties["depth"] = quake["geometry"]["coordinates"][2]
        quakes.append(properties)

    if not quakes:
        return pd.DataFrame(columns=["mag", "place", "lat", "lon", "time", "url"])

    quakes = pd.DataFrame(quakes)
    quakes = quakes.filter(["mag", "place", "lat", "lon", "time", "url"], axis=1)
    quakes["time"] = pd.to_datetime(quakes["time"], unit="ms")
    return quakes


def build_earthquake_map(quakes):
    center = [float(quakes["lat"].mean()), float(quakes["lon"].mean())]
    quake_map = folium.Map(
        location=center,
        zoom_start=2,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    for quake in quakes.itertuples(index=False):
        magnitude = float(quake.mag) if pd.notna(quake.mag) else 0.0
        tooltip = (
            f"Magnitude: {magnitude}<br>"
            f"Place: {quake.place}<br>"
            f"Time: {quake.time}<br>"
            f"URL: {quake.url}"
        )
        color = "#c92a2a" if magnitude >= 5 else "#f08c00" if magnitude >= 3 else "#1971c2"
        folium.CircleMarker(
            location=[float(quake.lat), float(quake.lon)],
            radius=max(4, magnitude * 2),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=tooltip,
        ).add_to(quake_map)

    return quake_map


def filter_quakes_by_bounds(quakes, bounds):
    if not bounds:
        return quakes

    south = bounds["_southWest"]["lat"]
    north = bounds["_northEast"]["lat"]
    west = bounds["_southWest"]["lng"]
    east = bounds["_northEast"]["lng"]

    lat_filter = quakes["lat"].between(south, north)
    if west <= east:
        lon_filter = quakes["lon"].between(west, east)
    else:
        lon_filter = (quakes["lon"] >= west) | (quakes["lon"] <= east)

    return quakes[lat_filter & lon_filter]


def find_feed_option(opt_list, past, mgn):
    for label, option in opt_list.items():
        if past in label and mgn in label:
            return option

    return None

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

    option = find_feed_option(opt_list, past, mgn)

    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{option}"
    quakes = load_earthquake_data(url)
    st.write(f"Number of {mgn} eartquake(s) in {past}: {len(quakes)}")

    if len(quakes):
        map_data = st_folium(
            build_earthquake_map(quakes),
            height=500,
            use_container_width=True,
            key=f"earthquake-map-{past}-{mgn}",
            returned_objects=["bounds"],
        ) or {}
        visible_quakes = filter_quakes_by_bounds(quakes, map_data.get("bounds")).copy()

        with st.expander("Earthquakes list"):
            st.write(f"Visible earthquake(s): {len(visible_quakes)}")
            visible_quakes = visible_quakes.sort_values(by=["mag"], ascending=False)
            st.dataframe(
                visible_quakes,
                column_config={
                    "url": st.column_config.LinkColumn(
                        "url",
                        display_text="USGS details",
                    )
                },
                hide_index=True,
                use_container_width=True,
            )
