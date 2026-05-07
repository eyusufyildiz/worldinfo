# https://github.com/pahlisch/streamlit-app-volcano
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path

#tool.streamlit_config(page_title="🌋 Volcanos", page_icon="🌋")

VOLCANO_FILE = Path(__file__).parent / "mzk" / "GVP_Volcano_List_Holocene_202505011922.csv"


@st.cache_data(show_spinner=False)
def load_volcano_data():
    volcanos = pd.read_csv(VOLCANO_FILE)
    volcanos = volcanos.dropna(subset=["Latitude", "Longitude"])
    volcanos["url"] = volcanos["Volcano Number"].apply(
        lambda volcano_number: f"https://volcano.si.edu/volcano.cfm?vn={volcano_number}"
    )
    return volcanos


def build_volcano_map(volcanos):
    center = [float(volcanos["Latitude"].mean()), float(volcanos["Longitude"].mean())]
    volcano_map = folium.Map(
        location=center,
        zoom_start=2,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    for _, volcano in volcanos.iterrows():
        tooltip = (
            f"Volcano: {volcano['Volcano Name']}<br>"
            f"Country: {volcano['Country']}<br>"
            f"Elevation: {volcano['Elevation (m)']} m<br>"
            f"Last Known Eruption: {volcano['Last Known Eruption']}<br>"
            f"URL: {volcano['url']}"
        )
        folium.CircleMarker(
            location=[float(volcano["Latitude"]), float(volcano["Longitude"])],
            radius=5,
            color="#c92a2a",
            fill=True,
            fill_color="#f03e3e",
            fill_opacity=0.75,
            tooltip=tooltip,
        ).add_to(volcano_map)

    return volcano_map


def filter_volcanos_by_bounds(volcanos, bounds):
    if not bounds:
        return volcanos

    south = bounds["_southWest"]["lat"]
    north = bounds["_northEast"]["lat"]
    west = bounds["_southWest"]["lng"]
    east = bounds["_northEast"]["lng"]

    lat_filter = volcanos["Latitude"].between(south, north)
    if west <= east:
        lon_filter = volcanos["Longitude"].between(west, east)
    else:
        lon_filter = (volcanos["Longitude"] >= west) | (volcanos["Longitude"] <= east)

    return volcanos[lat_filter & lon_filter]


def get_volcanos():
    volcanos = load_volcano_data()
    st.write(f"Number of volcanos 🌋: {len(volcanos)}")

    if len(volcanos):
        map_data = st_folium(
            build_volcano_map(volcanos),
            height=500,
            use_container_width=True,
            key="volcano-map",
            returned_objects=["bounds"],
        ) or {}
        visible_volcanos = filter_volcanos_by_bounds(volcanos, map_data.get("bounds")).copy()

        with st.expander("Volcano list"):
            st.write(f"Visible volcano(s): {len(visible_volcanos)}")
            visible_volcanos = visible_volcanos.sort_values(by=["Volcano Name"])
            st.dataframe(
                visible_volcanos,
                column_config={
                    "url": st.column_config.LinkColumn(
                        "url",
                        display_text="GVP details",
                    )
                },
                hide_index=True,
                use_container_width=True,
            )
