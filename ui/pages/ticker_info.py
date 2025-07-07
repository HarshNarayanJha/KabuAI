import json
import os
import re
from collections.abc import Iterator

import humanize
import requests
import streamlit as st
from requests_sse import EventSource, InvalidContentTypeError, InvalidStatusCodeError

from constants.agents import SUPERVISOR_NAME
from models.api import APIState, Message, Request, Response
from utils.prompt import SYSTEM_PROMPT

URL = os.getenv("API_URL", "")
HEADERS = {"Content-Type": "application/json"}

INITIAL_MESSAGE = os.getenv("INITIAL_MESSAGE", "Hey! I am KabuAI. How can I help you today?")

# st.html(
#     """
#     <style>
#     [data-testid="stMetricDelta"] svg {
#         display: none;
#     }

#     [data-testid="stMetricDelta"] > div:before {
#         content: "◯";
#         font-weight: bold;
#     }

#     </style>
#     """
# )

st.title("KabuAI")
st.subheader("Explore Stocks and Tickers")


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


def escape_markdown(text: str) -> str:
    return re.sub(r"(?<!\\)\$", r"\\$", text)


initial_state = APIState(
    next="",
    messages=[
        Message(type="system", content=SYSTEM_PROMPT),
        Message(type="ai", content=INITIAL_MESSAGE, name=SUPERVISOR_NAME),
    ],
    ticker=None,
    stock_data=None,
    stock_summary=None,
    search_query=None,
    search_results=[],
    search_summary=None,
    analysis_result=None,
    analysis_score=None,
)

if "state" not in st.session_state:
    st.session_state.state = initial_state

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Existing messages
for msg in st.session_state.state.messages:
    if msg.type == "ai":
        st.chat_message("assistant").markdown(escape_markdown(msg.content))
    elif msg.type == "human":
        st.chat_message("user").markdown(escape_markdown(msg.content))

if (
    prompt := st.chat_input("Type your message...", disabled=st.session_state.awaiting_response)
) and not st.session_state.awaiting_response:
    st.session_state.awaiting_response = True

    st.session_state.state.messages.append(Message(type="human", content=prompt))

    with st.chat_message("user"):
        st.markdown(escape_markdown(prompt))

    with st.chat_message("assistant"):
        handoff_section = st.empty()
        tool_section = st.empty()

        handoff_spinner = ControlledSpinner("Calling agent...")
        tool_spinner = ControlledSpinner("Calling tool...")

        stock_placeholder = st.empty()
        sources_placeholder = st.empty()

        message_placeholder = st.empty()
        full_response = ""

        # print(f"Posting data to server: {st.session_state.state}")
        state = Request(state=st.session_state.state).model_dump_json()

        with EventSource(URL, method="POST", headers=HEADERS, data=state, timeout=30) as event_source:
            try:
                for event in event_source:
                    data = Response(**json.loads(event.data or "{}"))

                    print(f"GOT DATA: {data}")
                    match data.type:
                        case "handoff":
                            if data.arguments is None:
                                continue

                            if data.arguments.get("next") == "FINISH":
                                # grab and show the message, that is the AI response
                                message_placeholder.markdown(escape_markdown(data.arguments["message"]) + "| ")
                                continue

                            with handoff_section.expander("Delegating Task"):
                                st.code(f"Asking {data.arguments.get('next', 'agent')} for help", language=None)

                            handoff_spinner.set_text(data.arguments.get("message", "Working..."))
                            handoff_spinner.start()

                        case "tool":
                            with tool_section.expander(data.name or "Some Tool", expanded=True):
                                st.code(data.arguments)

                                tool_spinner.set_text("Calling tool...")
                                tool_spinner.start()

                        case "update":
                            # handoff_spinner.stop()
                            # tool_spinner.stop()

                            if data.state:
                                if data.state.messages:
                                    for msg in data.state.messages:
                                        # TODO: use ids to prevent duplicates just in case
                                        st.session_state.state.messages.append(msg)
                                        message_placeholder.empty()
                                        st.markdown(escape_markdown(msg.content.strip()))

                                if data.state.next:
                                    handoff_spinner.stop()
                                    st.session_state.state.next = data.state.next
                                    # essential to stop the SSE streaming, otherwise it will keep sending messages!
                                    if data.state.next == "__end__":
                                        break

                                if data.state.ticker is not None:
                                    st.session_state.state.ticker = data.state.ticker

                                if (
                                    data.state.stock_data is not None
                                    and data.state.stock_data != st.session_state.state.stock_data
                                ):
                                    st.session_state.state.stock_data = data.state.stock_data
                                    if data.state.stock_data.prices:
                                        stock_placeholder.empty()
                                        with stock_placeholder:
                                            col1, col2 = st.columns(2)

                                            _label = f"**{data.state.stock_data.metadata.symbol} _{data.state.stock_data.company.longName}_**"
                                            _latest_price = f"${data.state.stock_data.prices[-1].close:.2f}"
                                            _delta = f"{
                                                (
                                                    data.state.stock_data.prices[-1].close
                                                    - data.state.stock_data.prices[-1].open
                                                ):.2f}"

                                            with col1:
                                                st.text("")
                                                st.metric(_label, _latest_price, _delta)
                                                st.text("")

                                            _label = f"**{data.state.stock_data.metadata.symbol} _Market Capital_**"
                                            _market_cap = f"${humanize.intword(data.state.stock_data.metadata.market_cap or 'N/A', format='%.3f').title()}"
                                            _market_cap_full = f"${humanize.intcomma(data.state.stock_data.metadata.market_cap or 'N/A')}"

                                            with col2:
                                                st.text("")
                                                st.metric(_label, _market_cap, _market_cap_full, delta_color="off")
                                                st.text("")

                                if data.state.stock_summary is not None:
                                    st.session_state.state.stock_summary = data.state.stock_summary

                                if data.state.search_query is not None:
                                    st.session_state.state.search_query = data.state.search_query

                                if (
                                    data.state.search_results
                                    and data.state.search_results != st.session_state.state.search_results
                                ):
                                    st.session_state.state.search_results = data.state.search_results
                                    sources_placeholder.empty()
                                    with sources_placeholder.expander("Sources", expanded=True, icon="📃"):
                                        for res in data.state.search_results:
                                            print(f"Rendering search results: {res.link}")
                                            color = "blue"
                                            if res.sentiment_score >= 0.25 and res.confidence >= 0.25:
                                                color = "green"
                                            elif res.sentiment_score <= -0.25 and res.confidence >= 0.25:
                                                color = "orange"

                                            st.badge(f"{res.source} ({res.link})", icon="🌐", color=color)

                                if data.state.search_summary is not None:
                                    st.session_state.state.search_summary = data.state.search_summary

                                if data.state.analysis_result is not None:
                                    st.session_state.state.analysis_result = data.state.analysis_result

                                if data.state.analysis_score is not None:
                                    st.session_state.state.analysis_score = data.state.analysis_score

                        case "chunk":
                            # stop handoff spinner on state.next update instead
                            # handoff_spinner.stop()
                            tool_spinner.stop()

                            if data.content and data.content.strip():
                                full_response += data.content.strip()
                                message_placeholder.markdown(escape_markdown(full_response) + "| ")

                        case "task":
                            # Not of any use right now, will use
                            print(f"Task update: {data.direction} -> {data.name}")

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
                message_placeholder.empty()
                st.session_state.awaiting_response = False
