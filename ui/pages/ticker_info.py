import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ui.utils.prompt import SYSTEM_PROMPT
from ui.utils.st_callable_util import get_streamlit_cb
from ventureai.graph.boss_state import StockBossState
from ventureai.main import invoke_agent

st.title("VentureAI")
st.text("Explore Stocks and Tickers")

prompt = st.chat_input()

initial_state: StockBossState = {
    "messages": [SystemMessage(SYSTEM_PROMPT), AIMessage("Hey! I am VentureAI. How can I help you today?")],
    "ticker": None,
    "stock_data": None,
    "stock_summary": None,
    "next": "",
}

if "state" not in st.session_state:
    st.session_state.state = initial_state

for msg in st.session_state.state["messages"]:
    if msg.type == "ai":
        st.chat_message("assistant").write(msg.content)
    elif msg.type == "human":
        st.chat_message("user").write(msg.content)

if prompt:
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.empty())
        response = invoke_agent(st.session_state.state, [st_callback])
        not_yet_messages = [msg for msg in response["messages"] if msg not in st.session_state.state["messages"]]
        st.session_state.state.update(response)

    for msg in not_yet_messages:
        st.chat_message(msg.type).write(msg.content)
