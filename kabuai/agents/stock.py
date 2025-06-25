import os
from pprint import pprint
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from constants.agents import STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.stock_state import StockAgentState
from models.stock import StockData
from prompts.stock import fetch_prompt, summary_prompt
from tools.stock import fetch_stock_details

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT") or ""
CHAT_MODEL_HEAVY = os.getenv("CHAT_MODEL_HEAVY") or ""

DEBUG = os.getenv("DEBUG", "0") == "1"
SUMMARY_LENGTH: Literal["short", "medium", "long"] = "medium"

llm = init_chat_model(model=CHAT_MODEL, temperature=0, max_tokens=2048)
llm_light = init_chat_model(model=CHAT_MODEL_LIGHT, temperature=0, max_tokens=4096)
llm_heavy = init_chat_model(model=CHAT_MODEL_HEAVY, temperature=0)


class StockDetailsResponseFormat(BaseModel):
    ticker_or_name: str | None = Field(description="Ticker symbol of the stock or the company name.")


def stock_details_node(state: StockAgentState) -> dict | Command:
    """
    Process stock details request.
    """

    if DEBUG:
        print("ENTERING stock_details NODE with state:")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": state["stock_summary"],
                "ticker": state["ticker"],
            }
        )

    try:
        messages = [
            SystemMessage(fetch_prompt),
            *state["messages"],
        ]

        ticker_response = cast(
            StockDetailsResponseFormat,
            llm.with_structured_output(StockDetailsResponseFormat).invoke(messages),
        )

        if DEBUG:
            print("TickerResponse:", ticker_response.ticker_or_name)

        if not ticker_response.ticker_or_name:
            return {"ticker": None, "stock_data": None, "stock_summary": None}

        if state["ticker"] == ticker_response.ticker_or_name:
            # same ticker, we already have that data, no need to fetch
            return {}

        response: StockData = fetch_stock_details.invoke(ticker_response.ticker_or_name)

        if DEBUG:
            print(f"EXITING stock_details NODE with response {response.metadata}...")

        return {
            "ticker": response.metadata.symbol,
            "stock_data": response,
        }

    except Exception as e:
        err = "I encountered an error while fetching stock details"
        if DEBUG:
            print(f"ERROR in stock_details NODE: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={"messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)], "stock_data": None},
            graph=Command.PARENT,
        )


def stock_summary_node(state: StockAgentState) -> dict | Command:
    """
    Summarize stock data.
    """

    if DEBUG:
        print("ENTERING stock_summary NODE with state:")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": state["stock_summary"],
                "ticker": state["ticker"],
            }
        )

    try:
        ticker = state["ticker"]
        stock_data = state["stock_data"]
        if ticker is None:
            return {"stock_summary": None}

        if stock_data is None:
            return {"stock_summary": f"No stock data available for {state['ticker']}"}

        messages = [
            SystemMessage(
                summary_prompt.format(
                    summary_length=SUMMARY_LENGTH,
                ),
            ),
            SystemMessage(stock_data.model_dump_json()),
            *state["messages"],
        ]

        response = llm.invoke(messages)

        if DEBUG:
            print(f"EXITING stock_summary NODE with response {response.content[:30]}...")

        return {"stock_summary": str(response.content)}

    except Exception as e:
        if DEBUG:
            print(f"ERROR in stock_summary NODE: {e}")
        err = "I'm sorry, but I encountered an error while generating the stock summary"
        return Command(
            goto=SUPERVISOR_NAME,
            update={"messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)], "stock_summary": None},
            graph=Command.PARENT,
        )


stock_agent = (
    StateGraph(StockAgentState)
    .add_node(
        "stock_details_node",
        stock_details_node,
        destinations=("stock_summary_node", END),
    )
    .add_node(
        "stock_summary_node",
        stock_summary_node,
        destinations=(END,),
    )
    .set_entry_point("stock_details_node")
    .add_edge("stock_details_node", "stock_summary_node")
    .add_edge("stock_details_node", END)
    .set_finish_point("stock_summary_node")
    .compile(name=STOCK_AGENT_NAME, debug=DEBUG)
)

if DEBUG:
    with open("stock_graph.png", "wb") as fp:
        fp.write(stock_agent.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    state: StockAgentState = {"messages": [], "ticker": None, "stock_data": None, "stock_summary": None}
    while True:
        print("========\n\n")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": state["stock_summary"],
                "ticker": state["ticker"],
            }
        )
        print("\n\n========")

        query = input(f"{STOCK_AGENT_NAME}> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(StockAgentState, stock_agent.invoke(state))

        state.update(result)
