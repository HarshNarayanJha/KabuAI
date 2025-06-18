from typing import cast

from langchain_core.messages import HumanMessage

from agents.boss import boss
from graph.boss_state import StockBossState


def invoke_agent(state: StockBossState, callables: list) -> StockBossState:
    return cast(StockBossState, boss.invoke(state, config={"callbacks": callables, "configurable": {"thread_id": "1"}}))


if __name__ == "__main__":
    state: StockBossState = {"messages": [], "stock_data": None, "stock_summary": None, "ticker": None, "next": ""}
    while True:
        query = input("You> ").strip()
        state["messages"].append(HumanMessage(query))
        result = invoke_agent(state, [])
        print(result["messages"])
        state.update(result)
