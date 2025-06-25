import os
from pprint import pprint
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from agents.search import search_agent
from agents.stock import stock_agent
from constants.agents import MEMBERS, SEARCH_AGENT_NAME, STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.boss_state import StockBossState
from graph.search_state import SearchAgentState
from graph.stock_state import StockAgentState
from prompts.boss import DONE_PROMPT, supervisor_prompt_template

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT") or ""
CHAT_MODEL_HEAVY = os.getenv("CHAT_MODEL_HEAVY") or ""
DEBUG = os.getenv("DEBUG", "0") == "1"

llm = init_chat_model(model=CHAT_MODEL, temperature=0, max_tokens=2048)
llm_light = init_chat_model(model=CHAT_MODEL_LIGHT, temperature=0, max_tokens=4096)
llm_heavy = init_chat_model(model=CHAT_MODEL_HEAVY, temperature=0)


checkpointer = InMemorySaver()

OPTIONS = MEMBERS + ["FINISH"]


class Router(BaseModel):
    next: Literal["stock_agent", "search_agent", "FINISH"] = Field(
        description="Agent to route to next. If no agents needed, route to FINISH."
    )
    message: str = Field(..., description="Response Message to the user.")
    system_instruction: str = Field(..., description="Very Brief system instruction to the agent you are routing to.")


def boss_node(state: StockBossState) -> Command:
    if DEBUG:
        print("ENTERING Boss Node with state:")
        pprint(
            {
                "messages": [message.content for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": f"{state['stock_summary'][:30]}..." if state["stock_summary"] else None,
                "ticker": state["ticker"],
                "next": state["next"],
                "search_query": state["search_query"],
                "search_results": len(state["search_results"]),
            }
        )

    try:
        # Let's check if we already have the summary
        # In the final version we will check for the final verdict
        last_message = state["messages"][-1] if state["messages"] else None
        if (
            last_message
            # last message is from an agent
            and last_message.type == "ai"
            and last_message.name in MEMBERS
            and (
                # any of the state vars from any agent is there
                (state["stock_summary"] and state["stock_data"])
                or (state["search_results"] and state["search_summary"])
            )
        ):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", DONE_PROMPT),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            supervisor_response = llm.invoke(prompt.invoke({"messages": state["messages"]}))

            # completed
            return Command(
                goto=END,
                update={
                    "next": END,
                    "messages": [
                        AIMessage(
                            supervisor_response.content,
                            name=SUPERVISOR_NAME,
                        ),
                    ],
                },
            )

        messages = supervisor_prompt_template.invoke(
            {"messages": state["messages"], "options": ", ".join(OPTIONS), "members": ", ".join(MEMBERS)}
        )

        response = cast(Router, llm.with_structured_output(Router).invoke(messages))

        if DEBUG:
            print(f"Got router response {response}")

        if not response or not response.next:
            return Command(
                goto=END,
                update={
                    "next": END,
                    "messages": [AIMessage("I have encountered an error. Please try again.", name=SUPERVISOR_NAME)],
                },
            )

        goto = response.next
        if goto == "FINISH":
            finishing_update: dict = {"next": END}
            if response.message:
                finishing_update["messages"] = [AIMessage(response.message, name=SUPERVISOR_NAME)]

            return Command(goto=END, update=finishing_update)

        if DEBUG:
            print(f"GOING TO {goto}... {response}")

        return Command(
            goto=goto,
            update={
                "next": Send(
                    goto,
                    {
                        "messages": [
                            HumanMessage(state["messages"][-1].content),
                            SystemMessage(response.system_instruction, name=SUPERVISOR_NAME),
                        ]
                    },
                ),
            },
        )
    except Exception as e:
        error_msg = "I encountered an error while processing your query"
        if DEBUG:
            print(error_msg + str(e))

        return Command(
            goto=END,
            update={
                "next": END,
                "messages": [AIMessage(error_msg, name=SUPERVISOR_NAME)],
            },
        )


def call_stock_agent(state: StockBossState) -> dict:
    # if DEBUG:
    # print("ENTERING Stock Agent Handler with state:")
    # pprint(state)

    try:
        send: Send = cast(Send, state["next"])
        stock_state = {
            "messages": send.arg["messages"],
            "stock_data": state["stock_data"],
            "stock_summary": state["stock_summary"],
            "ticker": state["ticker"],
        }

        stock_result: StockAgentState = cast(StockAgentState, stock_agent.invoke(stock_state))

        summary = stock_result["stock_summary"]
        if stock_result.get("stock_data") is None or summary is None:
            error_msg = "I couldn't find any stock data. Could you please provide a valid stock symbol or company name?"
            return {
                "next": SUPERVISOR_NAME,
                "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)],
            }

        return {
            "stock_data": stock_result["stock_data"],
            "stock_summary": stock_result["stock_summary"],
            "ticker": stock_result["ticker"],
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(summary, name=STOCK_AGENT_NAME)],
        }

    except Exception as e:
        error_msg = "I encountered an error while fetching stock information"
        if DEBUG:
            print(error_msg + str(e))

        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)],
        }


def call_search_agent(state: StockBossState) -> dict:
    # if DEBUG:
    # print("ENTERING Search Agent Handler with state:")
    # pprint(state)

    try:
        send: Send = cast(Send, state["next"])
        search_state = {
            "ticker": state["ticker"],
            "stock_summary": state["stock_summary"],
            "messages": send.arg["messages"],
        }

        search_result: SearchAgentState = cast(SearchAgentState, search_agent.invoke(search_state))

        results = search_result["search_results"]
        summary = search_result["search_summary"]
        if search_result.get("search_query") is None or not results or not summary:
            error_msg = "I couldn't find any search results. Could you ask something more specific?"
            return {
                "next": SUPERVISOR_NAME,
                "messages": [AIMessage(error_msg, name=SEARCH_AGENT_NAME)],
            }

        return {
            "next": SUPERVISOR_NAME,
            "search_query": search_result["search_query"],
            "search_results": results,
            "search_summary": summary,
            "messages": [AIMessage(summary, name=SEARCH_AGENT_NAME)],
        }

    except Exception as e:
        error_msg = "I encountered an error while fetching search information"
        if DEBUG:
            print(error_msg + str(e))

        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=SEARCH_AGENT_NAME)],
        }


boss = (
    StateGraph(StockBossState)
    .add_node(
        SUPERVISOR_NAME,
        boss_node,
        destinations=(STOCK_AGENT_NAME,),
    )
    .add_node(
        STOCK_AGENT_NAME,
        call_stock_agent,
        destinations=(SUPERVISOR_NAME,),
    )
    .add_node(
        SEARCH_AGENT_NAME,
        call_search_agent,
        destinations=(SUPERVISOR_NAME,),
    )
    #
    .set_entry_point(SUPERVISOR_NAME)
    .add_edge(STOCK_AGENT_NAME, SUPERVISOR_NAME)
    .add_edge(SEARCH_AGENT_NAME, SUPERVISOR_NAME)
    # end would be the final verdict agent
    .set_finish_point(SUPERVISOR_NAME)
    .compile(debug=DEBUG)
)

if DEBUG:
    with open("graph.png", "wb") as fp:
        fp.write(boss.get_graph(xray=1).draw_mermaid_png())
    with open("boss_graph.png", "wb") as fp:
        fp.write(boss.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    state: StockBossState = {
        "messages": [],
        "stock_data": None,
        "stock_summary": None,
        "ticker": None,
        "next": "",
        "search_query": None,
        "search_results": [],
        "search_summary": None,
    }

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        print("========\n\n")
        pprint(
            {
                "messages": [(message.type, message.content, message.name) for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": f"{state['stock_summary'][:30]}..." if state["stock_summary"] else None,
                "ticker": state["ticker"],
                "next": state["next"],
                "search_query": state["search_query"],
                "search_results": len(state["search_results"]),
                "search_summary": f"{state['search_summary'][:30]}..." if state["search_summary"] else None,
            }
        )
        print("\n\n========")

        query = input("Boss> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(StockBossState, boss.invoke(state, config=config))
        state.update(result)
