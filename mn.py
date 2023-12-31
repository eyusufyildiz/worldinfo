import streamlit as st
from streamlit_option_menu import option_menu
from mypages import LocationFinder as location_finder
from mypages import ISSnow as issnow
from mypages import Earthquakes as earthquakes
from mypages import StarlinkGps as starlink
from mypages import Volcanos as volcanos


with st.sidebar:
    selected = option_menu(None, ["IpLocationFinder", "Earthquakes",  "Volcanos", '🧿Issnow', 'StarlinkGPS'], 
        icons=['geo-alt', 'cloud-upload', "list-task", 'gear', 'broadcast-pin'], 
        menu_icon="gear", 
        default_index=1, 
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )


if selected == "IpLocationFinder":
    location_finder.get_ip_location()
elif selected == "Earthquakes":
    earthquakes.quakes()
elif selected == "Volcanos":
    volcanos.get_volcanos()
elif selected =='Issnow':
    issnow.iss()
elif selected =='StarlinkGPS':
    starlink.satellites()
