import streamlit as st

home_page = st.Page("pages/home.py", title="VentureAI Home", icon="🤖")
ticker_info_page = st.Page("pages/ticker_info.py", title="Explore Tickers", icon="🎟️")

pg = st.navigation([home_page, ticker_info_page])
pg.run()
