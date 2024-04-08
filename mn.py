import streamlit as st
from streamlit_option_menu import option_menu
from mypages import IpLocation as location_finder
from mypages import ISSnow as issnow
from mypages import Earthquakes as earthquakes
from mypages import Satellites as starlink
from mypages import Volcanos as volcanos
from mypages import Tests as tests
from utils import tools as tool

# tool.hide_stremlit()

def bg():
    # https://www.magicpattern.design/tools/css-backgrounds
    page_bg_img = """
    <style>
background-color: #e5e5f7;
opacity: 0.8;
background-image: radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
background-size: 10px 10px;
</style>
    """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

bg()

with st.sidebar:
    selected = option_menu(None, ["IpLocation", "Earthquakes",  "Volcanos", 'Issnow', 'Satellites'], 
        icons=['geo-alt', 'cloud-upload', "list-task", 'gear', 'broadcast-pin'], 
        menu_icon="gear", 
        default_index=1, 
        #styles={
        #    "container": {"padding": "0!important", "background-color": "#fafafa"},
        #    "icon": {"color": "orange", "font-size": "18px"}, 
        #    "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        #    "nav-link-selected": {"background-color": "green"},
        #}
    )
    with st.container(border=True):
        if st.button("by Bilge"):
            st.snow()
        if st.button("by Berkehan"):
            st.balloons()

if selected == "IpLocation":
    location_finder.get_ip_location()
elif selected == "Earthquakes":
    earthquakes.quakes()
elif selected == "Volcanos":
    volcanos.get_volcanos()
elif selected =='Issnow':
    issnow.iss()
elif selected =='Satellites':
    starlink.satellites()

# tool.show_globe()
