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

from agents.stock import stock_agent
from constants.agents import MEMBERS, STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.boss_state import StockBossState
from graph.stock_state import StockAgentState
from prompts.boss import DONE_PROMPT, supervisor_prompt_template

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT") or ""
CHAT_MODEL_HEAVY = os.getenv("CHAT_MODEL_HEAVY") or ""

llm = init_chat_model(model=CHAT_MODEL, temperature=0, max_tokens=2048)
llm_light = init_chat_model(model=CHAT_MODEL_LIGHT, temperature=0, max_tokens=4096)
llm_heavy = init_chat_model(model=CHAT_MODEL_HEAVY, temperature=0)

DEBUG = False

checkpointer = InMemorySaver()

OPTIONS = MEMBERS + ["FINISH"]


# remember to keep updating this
class Router(BaseModel):
    next: Literal["stock_agent", "supervisor", "FINISH"] = Field(
        description="Agent to route to next. If no agents needed, route to FINISH."
    )
    message: str = Field(..., description="Message to the user.")
    system_instruction: str = Field(..., description="Brief system instruction to the agent")


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
            }
        )

    # Let's check if we already have the summary
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.name == STOCK_AGENT_NAME and state["stock_summary"] and state["stock_data"]:
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

    response = cast(Router, llm_heavy.with_structured_output(Router).invoke(messages))
    goto = response.next
    if goto == "FINISH":
        return Command(goto=END, update={"next": END, "messages": [AIMessage(response.message, name=SUPERVISOR_NAME)]})

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
            "messages": [AIMessage(response.message, name=SUPERVISOR_NAME)],
        },
    )


def call_stock_agent(state: StockBossState):
    if DEBUG:
        print("ENTERING Stock Agent Handler with state:")
        pprint(state)

    try:
        send: Send = cast(Send, state["next"])
        stock_state = {
            "messages": send.arg["messages"],
            "stock_data": state["stock_data"],
            "stock_summary": state["stock_summary"],
            "ticker": state["ticker"],
        }

        stock_result: StockAgentState = cast(StockAgentState, stock_agent.invoke(stock_state))

        if stock_result.get("stock_data") is None or stock_result.get("stock_summary") is None:
            error_msg = "I couldn't find any stock data. Could you please provide a valid stock symbol or company name?"
            return {**state, "next": SUPERVISOR_NAME, "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)]}

        summary = cast(str, stock_result["stock_summary"])

        return {
            "stock_data": stock_result["stock_data"],
            "stock_summary": stock_result["stock_summary"],
            "ticker": stock_result["ticker"],
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(summary, name=STOCK_AGENT_NAME)],
        }
    except Exception as e:
        error_msg = f"I encountered an error while fetching stock information: {str(e)}"
        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)],
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
    #
    .set_entry_point(SUPERVISOR_NAME)
    .add_edge(STOCK_AGENT_NAME, SUPERVISOR_NAME)
    # end would be the final verdict agent
    .set_finish_point(SUPERVISOR_NAME)
    .compile(debug=DEBUG)
)

with open("boss_graph.png", "wb") as fp:
    fp.write(boss.get_graph(xray=1).draw_mermaid_png())

if __name__ == "__main__":
    state: StockBossState = {"messages": [], "stock_data": None, "stock_summary": None, "ticker": None, "next": ""}

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        print("========\n\n")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "stock_data": f"{state['stock_data'].metadata}..." if state["stock_data"] else None,
                "stock_summary": f"{state['stock_summary'][:30]}..." if state["stock_summary"] else None,
                "ticker": state["ticker"],
                "next": state["next"],
            }
        )
        print("\n\n========")

        query = input("Boss> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(StockBossState, boss.invoke(state, config=config))
        state.update(result)
