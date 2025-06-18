import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# TODO: fix this impORT DELIMMA
from src.graph.boss_state import StockBossState
from src.main import invoke_agent
from utils.st_callable_util import get_streamlit_cb

st.title("VentureAI")
st.text("Explore Stocks and Tickers")

prompt = st.chat_input()

initial_state: StockBossState = {
    "messages": [AIMessage("Hey! How can I help you today?")],
    "ticker": None,
    "stock_data": None,
    "stock_summary": None,
    "next": "",
}

if "state" not in st.session_state:
    st.session_state["state"] = initial_state

for msg in st.session_state.state["messages"]:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt:
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.container())
        response = invoke_agent(st.session_state.state, [st_callback])
        st.session_state.state.update(response)
