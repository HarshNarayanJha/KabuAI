import logging
import os
from pprint import pprint
from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from ai_models.chat import chat_model, chat_model_heavy, chat_model_light  # noqa: F401
from ai_models.llm import llm, llm_heavy, llm_light  # noqa: F401
from constants.agents import STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.stock_state import StockAgentState
from models.stock import StockData
from prompts.stock import fetch_prompt, summary_prompt
from tools.stock import fetch_stock_details

DEBUG = os.getenv("DEBUG", "0") == "1"
SUMMARY_LENGTH: Literal["short", "medium", "long"] = "medium"

logger = logging.getLogger(__name__)
logger.info(f"SUMMARY_LENGTH set to {SUMMARY_LENGTH}")


class StockDetailsResponseFormat(BaseModel):
    ticker_or_name: str | None = Field(description="Ticker symbol of the stock or the company name.")


def stock_details_node(state: StockAgentState) -> dict | Command:
    """
    Process stock details request.
    """

    logger.debug("Entering stock_details_node in stock agent")
    try:
        messages = [
            SystemMessage(fetch_prompt),
            *state["messages"],
        ]

        ticker_response = cast(
            StockDetailsResponseFormat,
            chat_model_heavy.with_structured_output(StockDetailsResponseFormat).invoke(messages),
        )

        logger.debug(f"TickerResponse:, {ticker_response.ticker_or_name}")

        if not ticker_response or not ticker_response.ticker_or_name:
            err = f"Did not get a valid ticker response: {ticker_response}"
            logger.error(f"ERROR: {err}")
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)],
                    "stock_data": None,
                    "stock_summary": None,
                },
                graph=Command.PARENT,
            )

        if state["stock_data"] and (
            state["ticker"] == ticker_response.ticker_or_name
            or ticker_response.ticker_or_name in state["stock_data"].company.longName
        ):
            # same ticker, we already have that data, no need to fetch
            logger.debug("Leaving stock_details_node since ticker is the same")
            return {}

        response: StockData | None = fetch_stock_details.invoke(ticker_response.ticker_or_name)
        if not response:
            err = "Unable to fetch stock data. Please try again"
            logger.error(f"ERROR: {err}")
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)],
                    "ticker": ticker_response.ticker_or_name,
                    "stock_data": None,
                    "stock_summary": None,
                },
                graph=Command.PARENT,
            )

        logger.debug("leaving stock_details_node with data")
        return {
            "ticker": response.metadata.symbol,
            "stock_data": response,
        }

    except Exception as e:
        err = "I encountered an error while fetching stock details"
        logger.error(f"ERROR in stock_details NODE: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={
                "messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)],
                "ticker": None,
                "stock_data": None,
                "stock_summary": None,
            },
            graph=Command.PARENT,
        )


def stock_summary_node(state: StockAgentState) -> dict | Command:
    """
    Summarize stock data.
    """

    logger.debug("Entering stock_summary_node in stock agent")
    try:
        ticker = state["ticker"]
        stock_data = state["stock_data"]
        if not ticker:
            logger.debug("Leaving stock_summary_node since no ticker")
            return {"stock_summary": None}

        if not stock_data:
            logger.debug("Leaving stock_summary_node since no stock data")
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

        response = chat_model.invoke(messages)
        if not response:
            err = "Unable to summarize stock data. Please try again"
            logger.error(f"ERROR: {err}")
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=STOCK_AGENT_NAME)],
                    "stock_summary": None,
                },
                graph=Command.PARENT,
            )

        logger.debug("Leaving stock_summary_node with data")
        return {"stock_summary": str(response.content)}

    except Exception as e:
        err = "I'm sorry, but I encountered an error while generating the stock summary"
        logger.error(f"ERROR in stock_summary NODE: {e}")
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
