import json
from collections.abc import Iterator

import requests
import streamlit as st
from requests_sse import EventSource, InvalidContentTypeError, InvalidStatusCodeError
from utils.prompt import SYSTEM_PROMPT

from constants.agents import SUPERVISOR_NAME
from models.api import APIState, Message, Request, Response

URL = "http://localhost:8000/chat"
HEADERS = {"Content-Type": "application/json"}

st.title("VentureAI")
st.text("Explore Stocks and Tickers")


class ControlledSpinner:
    def __init__(self, text: str = "In progress..."):
        self.text: str = text
        self._spinner: Iterator[None] | None = None

    def _start(self):
        with st.spinner(self.text, show_time=True):
            yield

    def start(self):
        self._spinner = iter(self._start())
        next(self._spinner)

    def stop(self):
        if self._spinner:
            next(self._spinner, None)

    def set_text(self, text: str):
        self.text = text


initial_state = APIState(
    messages=[
        Message(type="system", content=SYSTEM_PROMPT),
        Message(type="ai", content="Hey! I am VentureAI. How can I help you today?", name=SUPERVISOR_NAME),
    ],
    ticker=None,
    stock_data=None,
    stock_summary=None,
    next="",
)

if "state" not in st.session_state:
    st.session_state.state = initial_state

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Existing messages
for msg in st.session_state.state.messages:
    if msg.type == "ai":
        st.chat_message("assistant").text(msg.content)
    elif msg.type == "human":
        st.chat_message("user").text(msg.content)

if (prompt := st.chat_input("Type your message...")) and not st.session_state.awaiting_response:
    st.session_state.awaiting_response = True

    st.session_state.state.messages.append(Message(type="human", content=prompt))

    with st.chat_message("user"):
        st.text(prompt)

    with st.chat_message("assistant"):
        handoff_section = st.empty()
        tool_section = st.empty()

        handoff_spinner = ControlledSpinner("Calling agent...")
        tool_spinner = ControlledSpinner("Calling tool...")

        message_placeholder = st.empty()

        state = Request(state=st.session_state.state).model_dump()

        full_response = ""
        last_chunk = ""
        got_final_message = False

        # print(f"Posting data to server: {state}")

        with EventSource(URL, method="POST", headers=HEADERS, json=state, timeout=5) as event_source:
            try:
                for event in event_source:
                    data = Response(**json.loads(event.data or ""))

                    print(f"GOT DATA: {data}")

                    match data.type:
                        case "handoff":
                            if data.arguments is None:
                                continue

                            if data.arguments.get("next") == "FINISH":
                                continue

                            with handoff_section.expander("Delegating Task"):
                                st.code(f"Asking {data.arguments.get('next')} for help", language=None)

                            handoff_spinner.set_text(data.arguments.get("message", "Working..."))
                            handoff_spinner.start()

                        case "tool":
                            with tool_section.expander(data.name or "Some Tool", expanded=True):
                                st.code(data.arguments)

                                tool_spinner.set_text("Calling tool...")
                                tool_spinner.start()

                        case "update":
                            handoff_spinner.stop()
                            tool_spinner.stop()

                            if data.state:
                                if data.state.messages:
                                    for msg in data.state.messages:
                                        if msg in st.session_state.state.messages:
                                            continue
                                        st.session_state.state.messages.append(msg)

                                if data.state.next:
                                    st.session_state.state.next = data.state.next
                                    if data.state.next == "__end__":
                                        got_final_message = True
                                        break

                                if data.state.ticker:
                                    st.session_state.state.ticker = data.state.ticker

                                if data.state.stock_data:
                                    st.session_state.state.stock_data = data.state.stock_data

                                if data.state.stock_summary:
                                    st.session_state.state.stock_summary = data.state.stock_summary

                        case "chunk":
                            handoff_spinner.stop()
                            tool_spinner.stop()

                            if data.content and data.content != full_response:
                                full_response += data.content or ""
                                message_placeholder.text(full_response + "| ")

                message_placeholder.empty()

            except InvalidStatusCodeError as e:
                st.error(f"Error: Invalid Status Code {e.status_code}")
                st.stop()
            except InvalidContentTypeError as e:
                st.error(f"Error: Invalid Content Type: {e.content_type}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"Network error: {e}")
                st.stop()
            finally:
                handoff_spinner.stop()
                tool_spinner.stop()
                st.session_state.awaiting_response = False

        # TODO; Need to fix some stuff here

        # message_placeholder.text(full_response)
