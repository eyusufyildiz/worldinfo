import streamlit as st
import pandas as pd
from utils import tools as tool

#tool.streamlit_config(page_title="🕵 Ip Adddress or Domain Name Location Finder", page_icon="🕵")

def get_ip_location():
    st.markdown(f"### 🕵 Ip Adddress/Domain Name Location Finder")
    #client_ip = st.session_state.get("client_ip")
    client_ip = tools.client_ip()

    if client_ip:
        st.write(f"Your Ip Adress: {client_ip}")
    else:
        client_ip = "http://www.fortinet.com"
        
    ip_addr = st.text_input("Enter Ip Address/Domain Name here:", client_ip)
    ip_info  = tool.ip_addess_location(ip_addr)
    ip_info.pop("status")
    ipInf = pd.json_normalize(ip_info)
    st.write(ipInf)
    
    with st.expander("Details in json"):
        st.json(ip_info)

    if not ip_info.get('message') : 
        st.map(ipInf, zoom=4, use_container_width=True)
