import json
import logging
import os
import re
from collections.abc import Iterator
from typing import Any, cast

import humanize
import requests
import streamlit as st
from requests_sse import EventSource, InvalidContentTypeError, InvalidStatusCodeError
from streamlit.delta_generator import DeltaGenerator

from constants.agents import SUPERVISOR_NAME
from models.api import APIState, Message, Request, Response
from models.chat import ChatEntry
from models.search import SearchResult
from models.stock import StockData
from utils.prompt import SYSTEM_PROMPT

URL = os.getenv("API_URL", "")
HEADERS = {"Content-Type": "application/json"}

INITIAL_MESSAGE = os.getenv("INITIAL_MESSAGE", "Hey! I am KabuAI. How can I help you today?")

logger = logging.getLogger(__name__)


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


def draw_tool_call(
    tool_name: str,
    tool_args: dict[str, Any] | None,
    container: DeltaGenerator,
    spinner: ControlledSpinner | None = None,
):
    """
    Draws a tool call expander in the container.
    If spinner is passed, inserts and starts the spinner.
    """
    container.empty()
    with container.expander(tool_name, expanded=spinner is not None):
        st.code(tool_args)

        if spinner is not None:
            spinner.set_text("Calling tool...")
            spinner.start()


def draw_stock_cards(stock_data: StockData, container: DeltaGenerator):
    """
    Draws Stock Price and Market Cap side by side
    """

    container.empty()
    with container:
        col1, col2 = st.columns(2)

        _label = f"**{stock_data.metadata.symbol} _{stock_data.company.longName}_**"
        _latest_price = f"${stock_data.prices[-1].close:.2f}"
        _delta = f"{(stock_data.prices[-1].close - stock_data.prices[-1].open):.2f}"

        with col1:
            st.text("")
            st.metric(_label, _latest_price, _delta)
            st.text("")

        _label = f"**{stock_data.metadata.symbol} _Market Capital_**"
        _market_cap = f"${humanize.intword(stock_data.metadata.market_cap or 'N/A', format='%.3f').title()}"
        _market_cap_full = f"${humanize.intcomma(stock_data.metadata.market_cap or 'N/A')}"

        with col2:
            st.text("")
            st.metric(_label, _market_cap, _market_cap_full, delta_color="off")
            st.text("")


def draw_news_sources(news_sources: list[SearchResult], container: DeltaGenerator):
    """
    Draws news sources in a list
    """

    container.empty()
    with container.expander("Sources", expanded=True, icon="📃"):
        for res in news_sources:
            logger.debug(f"Rendering search results: {res.link}")
            color = "blue"
            if res.sentiment_score >= 0.25 and res.confidence >= 0.25:
                color = "green"
            elif res.sentiment_score <= -0.25 and res.confidence >= 0.25:
                color = "orange"

            st.badge(f"{res.source} ({res.link})", icon="🌐", color=color)


initial_state = APIState(
    next="",
    plan=[],
    step=-1,
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

initial_chat_entries = [
    ChatEntry(entry_type="message", message=initial_state.messages[-1]),
]

if "state" not in st.session_state:
    st.session_state.state = initial_state

if "chat_entries" not in st.session_state:
    st.session_state.chat_entries = initial_chat_entries

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# with st.sidebar:
#     st.code(f"AWAITING RESPONSE: {st.session_state.awaiting_response}")
#     st.code(f"NEXT: {st.session_state.state.next}")
#     st.code(f"STOCK TICKER: {st.session_state.state.ticker}")
#     st.code(f"ANALYSIS SCORE: {st.session_state.state.analysis_score}")

# Existing chat entries
for entry in cast(list[ChatEntry], st.session_state.chat_entries):
    match entry.entry_type:
        case "message":
            if entry.message is None:
                continue

            if entry.message.type == "ai":
                st.chat_message("assistant").markdown(escape_markdown(entry.message.content))
            elif entry.message.type == "human":
                st.chat_message("user").markdown(escape_markdown(entry.message.content))

        case "stock_card":
            if entry.stock_data is None:
                continue

            with st.chat_message("assistant"):
                draw_stock_cards(entry.stock_data, st.empty())

        case "news_items":
            if entry.news_items is None:
                continue

            with st.chat_message("assistant"):
                draw_news_sources(entry.news_items, st.empty())

        case "tool":
            if entry.tool_name is None:
                continue

            with st.chat_message("assistant"):
                draw_tool_call(entry.tool_name, entry.tool_args, st.empty())

        case "chart":
            pass


if (
    prompt := st.chat_input("Type your message...", disabled=st.session_state.awaiting_response)
    or st.session_state.awaiting_response
):
    if not st.session_state.awaiting_response:
        msg = Message(type="human", content=prompt)
        st.session_state.state.messages.append(msg)
        st.session_state.chat_entries.append(ChatEntry(entry_type="message", message=msg))

        with st.chat_message("user"):
            st.markdown(escape_markdown(prompt))
        st.session_state.awaiting_response = True
        st.rerun()

    with st.chat_message("assistant"):
        handoff_spinner = ControlledSpinner("Calling agent...")
        tool_spinner = ControlledSpinner("Calling tool...")

        handoff_section = st.empty()
        tool_section = st.empty()

        stock_placeholder = st.empty()
        sources_placeholder = st.empty()

        message_placeholder = st.empty()
        full_response = ""

        logger.debug(f"Posting data to server: {st.session_state.state}")
        state = Request(state=st.session_state.state).model_dump_json()

        with EventSource(URL, method="POST", headers=HEADERS, data=state, timeout=30) as event_source:
            try:
                for event in event_source:
                    logger.debug(f"GOT DATA: {event.data}")
                    data = Response(**json.loads(event.data or "{}"))

                    match data.type:
                        case "handoff":
                            if data.arguments is None:
                                continue

                            if data.arguments.get("next") == "__end__":
                                # grab and show the message, that is the AI response
                                message_placeholder.markdown(escape_markdown(data.arguments["message"]) + "| ")
                                continue

                            handoff_spinner.set_text(data.arguments.get("message", "Working..."))
                            handoff_spinner.start()

                            with handoff_section.expander("Delegating Task"):
                                st.code(f"Asking {data.arguments.get('next', 'agent')} for help", language=None)

                        case "tool":
                            st.session_state.chat_entries.append(
                                ChatEntry(
                                    entry_type="tool", tool_name=data.name or "Some Tool", tool_args=data.arguments
                                )
                            )
                            draw_tool_call(data.name or "Some Tool", data.arguments, tool_section, tool_spinner)

                        case "update":
                            # handoff_spinner.stop()
                            # tool_spinner.stop()

                            if data.state:
                                if data.state.messages:
                                    for msg in data.state.messages:
                                        # TODO: use ids to prevent duplicates just in case
                                        st.session_state.state.messages.append(msg)
                                        st.session_state.chat_entries.append(
                                            ChatEntry(entry_type="message", message=msg)
                                        )
                                        message_placeholder.empty()
                                        st.markdown(escape_markdown(msg.content.strip()))

                                if data.state.next:
                                    # handoff_spinner.stop()
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
                                        st.session_state.chat_entries.append(
                                            ChatEntry(entry_type="stock_card", stock_data=data.state.stock_data)
                                        )
                                        draw_stock_cards(data.state.stock_data, stock_placeholder)

                                if data.state.stock_summary is not None:
                                    st.session_state.state.stock_summary = data.state.stock_summary

                                if data.state.search_query is not None:
                                    st.session_state.state.search_query = data.state.search_query

                                if (
                                    data.state.search_results
                                    and data.state.search_results != st.session_state.state.search_results
                                ):
                                    # TODO:
                                    # needs a better way to identify same news list update
                                    # probably the same search query to be included in the results itself
                                    if (
                                        st.session_state.state.search_results
                                        and st.session_state.state.search_results[0].title
                                        == data.state.search_results[0].title
                                        and st.session_state.state.search_results[0].link
                                        == data.state.search_results[0].link
                                    ):
                                        # same is being updated with sentiment scores
                                        # remove the last news item
                                        st.session_state.chat_entries = [
                                            chat_entry
                                            for chat_entry in st.session_state.chat_entries
                                            if chat_entry.news_items != st.session_state.state.search_results
                                        ]

                                    st.session_state.state.search_results = data.state.search_results
                                    st.session_state.chat_entries.append(
                                        ChatEntry(entry_type="news_items", news_items=data.state.search_results)
                                    )
                                    draw_news_sources(data.state.search_results, sources_placeholder)

                                if data.state.search_summary is not None:
                                    st.session_state.state.search_summary = data.state.search_summary

                                if data.state.analysis_result is not None:
                                    st.session_state.state.analysis_result = data.state.analysis_result

                                if data.state.analysis_score is not None:
                                    st.session_state.state.analysis_score = data.state.analysis_score

                        case "chunk":
                            handoff_spinner.stop()
                            tool_spinner.stop()

                            if data.content and data.content.strip():
                                full_response += data.content.strip()
                                message_placeholder.markdown(escape_markdown(full_response) + "| ")

                        case "task":
                            # Not of any use right now, will use
                            logger.debug(f"Task update: {data.direction} -> {data.name}")

            except InvalidStatusCodeError as e:
                st.error(f"Error: Invalid Status Code {e.status_code}")
                logger.error(f"Invalid Status Code: {e.status_code}")
                st.stop()
            except InvalidContentTypeError as e:
                st.error(f"Error: Invalid Content Type: {e.content_type}")
                logger.error(f"Invalid Content Type: {e.content_type}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"Network error: {e}")
                logger.error(f"Network error: {e}")
                st.stop()
            finally:
                handoff_spinner.stop()
                tool_spinner.stop()
                message_placeholder.empty()

    st.session_state.awaiting_response = False
    st.rerun()
