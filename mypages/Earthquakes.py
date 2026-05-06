import pandas as pd
import json, requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime
from pathlib import Path
from utils import tools as tool
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


def build_earthquake_map(quakes):
    center = [quakes["lat"].mean(), quakes["lon"].mean()]
    quake_map = folium.Map(location=center, zoom_start=2, tiles="OpenStreetMap")

    for _, quake in quakes.iterrows():
        popup = folium.Popup(
            f"""
            <b>Magnitude:</b> {quake['mag']}<br>
            <b>Place:</b> {quake['place']}<br>
            <b>Time:</b> {quake['time']}<br>
            <a href="{quake['url']}" target="_blank">USGS details</a>
            """,
            max_width=300,
        )
        folium.CircleMarker(
            location=[quake["lat"], quake["lon"]],
            radius=max(4, quake["mag"] * 2),
            color="#d9480f",
            fill=True,
            fill_color="#f08c00",
            fill_opacity=0.75,
            popup=popup,
            tooltip=f"M{quake['mag']} - {quake['place']}",
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
    res = get_earthquake_features(url)
    quakes =[]

    if res:
        for q in res :
            x= q['properties'].copy()
            x['lat'] = q['geometry']['coordinates'][1]
            x['lon'] = q['geometry']['coordinates'][0]
            x['depth'] = q['geometry']['coordinates'][2]
            quakes.append( x )
    
        quakes=pd.DataFrame(quakes)
        quakes = quakes.filter(['mag', 'place', 'lat', 'lon', 'time', 'url'], axis=1)
        quakes['time'] =  pd.to_datetime(quakes['time'], unit='ms')
        st.write(f"Number of {mgn} eartquake(s) in {past}: {len(quakes)}")
    
        if len(quakes):
            map_data = st_folium(
                build_earthquake_map(quakes),
                height=500,
                use_container_width=True,
                key=f"earthquake-map-{past}-{mgn}",
            ) or {}
            visible_quakes = filter_quakes_by_bounds(quakes, map_data.get("bounds")).copy()

            with st.expander("Earthquakes list"):
                st.write(f"Visible earthquake(s): {len(visible_quakes)}")
                visible_quakes = visible_quakes.sort_values(by=['mag'], ascending=False)
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
