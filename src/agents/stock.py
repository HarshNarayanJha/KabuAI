import os
from pprint import pprint
from typing import Any, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph

from models.stock import StockData
from prompts.stock import prompt
from tools.stock import fetch_stock_details


class AgentState(TypedDict):
    ticker: str
    stock_data: StockData | None
    stock_summary: str | None


CHAT_MODEL = os.getenv("CHAT_MODEL") or ""


def stock_details_node(state: AgentState):
    stock_data: StockData = fetch_stock_details.invoke({"ticker": state["ticker"]})
    return {"stock_data": stock_data}


def stock_summary_node(state: AgentState) -> dict[str, Any]:
    stock_data = state["stock_data"]
    if stock_data is None:
        return {}

    llm = init_chat_model(CHAT_MODEL, temperature=0, max_tokens=2048)

    messages = [
        SystemMessage(prompt),
        HumanMessage(stock_data.model_dump_json()),
    ]

    response = llm.invoke(messages)

    return {"stock_summary": str(response.content)}


graph = StateGraph(AgentState)

graph.add_node("stock_details_node", stock_details_node)
graph.add_node("stock_summary_node", stock_summary_node)

graph.set_entry_point("stock_details_node")

graph.add_edge("stock_details_node", "stock_summary_node")

graph.set_finish_point("stock_summary_node")

stock_agent = graph.compile(name="stock_agent")

if __name__ == "__main__":
    state: AgentState = {"ticker": "", "stock_data": None, "stock_summary": None}
    while True:
        print("========\n\n")
        pprint(state)
        print("\n\n========")

        query = input("You> ").strip()
        state["ticker"] = query

        result = cast(AgentState, stock_agent.invoke(state))

        state.update(result)
