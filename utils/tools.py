import requests, json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from datetime import datetime

def streamlit_config(page_title="", page_icon=None):
    st.set_page_config(
    page_title=page_title,
    page_icon=page_icon,
)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style> """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def hide_stremlit():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style> """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
 
def timestamp(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def plotly_map(data, title, hover_name=None, hover_data=None, 
    lat="Latitude", lon="Longitude",
    mapbox_style="stamen-terrain", zoom=3, height=400, size=None ):
    
        # data: Dataframe 
    # hover_data: List
    
    ##  fig = px.scatter_mapbox(volnanos, lat="Latitude", lon="Longitude", 
    ##                          title= "Volcanos",
    ##                          #size="Elev", 
    ##                          hover_name="Volcano Name",
    ##                          hover_data=["Country", "Region", "Elev", "Type", "Status", "Last Known", "lat3", "lon2", "Population (2020)"],
    ##                          color_discrete_sequence=["red"],
    ##                          zoom=3, height=400)
    ##  #fig.update_layout(mapbox_style="open-street-map")
    ##  fig.update_layout(mapbox_style="stamen-terrain")
    ##  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig = px.scatter_mapbox(data, lat=lat, lon=lon, title=title,
                        size=size,
                        hover_name=hover_name,
                        hover_data=hover_data,
                        # color_discrete_sequence="red",
                        zoom=zoom, height=height)
    fig.update_layout(mapbox_style=mapbox_style)  # "open-street-map"
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig


def clear_cache():
    st.runtime.legacy_caching.clear_cache()
    #st.legacy_caching.caching.clear_cache()


def http_requests(url, type=None, username=None, password=None):
    if username:
        #basic = HTTPBasicAuth(username, password)
        #st.write(url, type)
        response= requests.get(url, auth=(username, password))
        #response= requests.get('url', auth=basic)
    else:
        response = requests.get(url)
    
    if type == 'json':
        return json.loads(response.content)
    else:
        return response.content

    
def http_requests_csv(url, username=None, password=None):
    import csv
    
    if username:
        response= requests.get(url, auth=(username, password), stream=True, verify=False)
        return response


def show_globe():
    html_string = f'''<html><body>
            <script type="text/javascript" src="//rf.revolvermaps.com/0/0/6.js?i=54i1ewhl3lw&amp;m=7&amp;c=ff0000&amp;cr1=ffffff&amp;f=arial&amp;l=0&amp;s=341&amp;lx=280&amp;ly=-80&amp;rs=0" async="async"></script>
            </body></html>'''
    components.html(html_string, height=400)
    
    
def geo_reverse(lat, lon):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="geoapiExercises")
    geolocator = Nominatim(user_agent="geoapiIssNow")
    location    = geolocator.reverse(str(lat) + ", " + str(lon))
    location_en = geolocator.reverse(str(lat) + ", " + str(lon), language='en')
    #location    = geolocator.reverse(f"{lat}, {lon}")
    #location_en = geolocator.reverse(f"{lat}, {lon}", language='en')
    
    try:
        address = location.raw['address']
        address_en = location_en.raw['address']
        return address, address_en
    except:
        return None

    
def ip_addess_location(ip_or_dom):
    import domain_utils as du
    base_url = "http://ip-api.com/json"
    ip_or_dom = du.get_etld1(ip_or_dom)
    url = f"{base_url}/{ip_or_dom}"

    return http_requests(url, type="json")
    

def client_ip():
    from streamlit_javascript import st_javascript

    url = 'https://api.ipify.org?format=json'
    script = (f'await fetch("{url}").then('
                'function(response) {'
                    'return response.json();'
                '})')
    
    try:
        result = st_javascript(script)
        
        if isinstance(result, dict) and 'ip' in result:
            return result['ip']
    except:
        pass
