import streamlit as st

from utils.logger import setup_logging

setup_logging()

home_page = st.Page("pages/home.py", title="KabuAI Home", icon="🤖")
ticker_info_page = st.Page("pages/ticker_info.py", title="Explore Tickers", icon="🎟️")

pg = st.navigation([home_page, ticker_info_page])
pg.run()
