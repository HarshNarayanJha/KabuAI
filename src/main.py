from agents.boss import boss


def invoke_agent(state, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    return boss.invoke(state, config={"callbacks": callables, "configurable": {"thread_id": "1"}})


if __name__ == "__main__":
    state = {"messages": [], "stock_data": None, "stock_summary": None, "ticker": None, "next": ""}
    while True:
        query = input("You> ").strip()
        state["messages"].append(query)
        result = invoke_agent(state, [])
        print(result["messages"])
        state.update(result)
