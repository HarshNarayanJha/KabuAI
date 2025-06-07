from time import sleep

import streamlit as st

st.title("VentureAI")
st.text("Explore Stocks and Tickers")

query = st.text_input("Query")

if st.button("Search!"):
    with st.spinner("Searching..."):
        sleep(5)
        st.text("Apple!")
