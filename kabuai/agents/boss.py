import logging
import os
from datetime import datetime
from pprint import pprint
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel

from agents.analyzer import analyzer_agent
from agents.search import search_agent
from agents.stock import stock_agent
from ai_models.chat import chat_model, chat_model_heavy, chat_model_light  # noqa: F401
from ai_models.llm import llm, llm_heavy, llm_light  # noqa: F401
from constants.agents import (
    ANALYZER_AGENT_NAME,
    MEMBERS,
    MEMBERS_DESCRIPTIONS,
    SEARCH_AGENT_NAME,
    STOCK_AGENT_NAME,
    SUPERVISOR_NAME,
)
from graph.analyzer_state import AnalyzerAgentState
from graph.boss_state import PlanStep, StockBossState
from graph.search_state import SearchAgentState
from graph.stock_state import StockAgentState
from prompts.boss import DONE_PROMPT, supervisor_prompt_template

DEBUG = os.getenv("DEBUG", "0") == "1"
checkpointer = InMemorySaver()
OPTIONS = MEMBERS + ["FINISH"]

logger = logging.getLogger(__name__)


class Router(BaseModel):
    plan: list[PlanStep]


def boss_node(state: StockBossState) -> Command | dict:
    logger.debug("Entering boss_node in supervisor")
    writer = get_stream_writer()

    try:
        # check if we have a plan
        if state["plan"] and state["step"] >= 0:
            logger.debug("Continuing the plan...")

            if len(state["plan"]) == state["step"] + 1:
                # means last step in plan wasn't FINISH
                logger.warning(
                    "No more steps, should have routed to end already. Routing to end. Leaving boss_node in supervisor."
                )
                writer({"handoff": {"next": END}})
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", DONE_PROMPT),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                supervisor_response = chat_model_light.invoke(prompt.invoke({"messages": state["messages"]}))

                return {
                    "messages": [
                        AIMessage(
                            supervisor_response.content,
                            name=SUPERVISOR_NAME,
                        ),
                    ],
                    "next": END,
                    "plan": [],
                    "step": -1,
                }

            next_step = state["plan"][state["step"] + 1]
            goto = next_step.agent
            if goto == "FINISH":
                logger.debug(
                    f"Leaving boss_node in supervisor since routing to FINISH as part of the plan step {state['step'] + 1}",
                )
                writer({"handoff": {"next": END, "message": next_step.message}})

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", DONE_PROMPT),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                supervisor_response = chat_model_light.invoke(prompt.invoke({"messages": state["messages"]}))

                return {
                    "messages": [
                        AIMessage(
                            supervisor_response.content,
                            name=SUPERVISOR_NAME,
                        ),
                    ],
                    "next": END,
                    "plan": [],
                    "step": -1,
                }

            logger.debug(
                f"Leaving boss_node in supervisor since routing to {goto} as part of the plan step {state['step'] + 1}"
            )
            writer(
                {
                    "handoff": {
                        "next": goto,
                        "system_instruction": next_step.system_instruction,
                        "message": next_step.message,
                    }
                }
            )

            return Command(
                goto=goto,
                update={
                    "step": state["step"] + 1,
                    "next": Send(
                        goto,
                        {
                            "messages": [
                                HumanMessage(content=next_step.request),
                                SystemMessage(content=next_step.system_instruction, name=SUPERVISOR_NAME),
                            ],
                        },
                    ),
                },
            )

        # no plan. create one.
        messages = supervisor_prompt_template.invoke(
            {
                "messages": state["messages"],
                "options": ", ".join(OPTIONS),
                "members": ", ".join(MEMBERS),
                "members_descriptions": "\n".join([f"{k} - {v}" for k, v in MEMBERS_DESCRIPTIONS.items()]),
                "today": datetime.today().isoformat(),
            }
        )

        response = cast(Router, chat_model_heavy.with_structured_output(Router).invoke(messages))
        logger.debug(f"Got router response {response}")

        if not response or not response.plan:
            logger.error("Leaving boss_node in supervisor since no plan was created")
            return {
                "next": END,
                "messages": [AIMessage("I have encountered an error. Please try again.", name=SUPERVISOR_NAME)],
            }

        goto = response.plan[0].agent
        if goto == "FINISH":
            finishing_update: dict = {"next": END, "step": -1, "plan": []}
            if response.plan[0].message:
                finishing_update["messages"] = [AIMessage(response.plan[0].message, name=SUPERVISOR_NAME)]

            logger.debug("Leaving boss_node in supervisor since routed to FINISH with no plan.")
            writer({"handoff": {"next": END, "message": response.plan[0].message}})

            return finishing_update

        # goto to the first agent in plan
        logger.debug(f"Leaving boss_node in supervisor since going to {goto} as the first step in plan.")
        writer(
            {
                "handoff": {
                    "next": goto,
                    "message": response.plan[0].message,
                    "system_instruction": response.plan[0].system_instruction,
                }
            }
        )

        return Command(
            goto=goto,
            update={
                "step": 0,
                "plan": response.plan,
                "next": Send(
                    goto,
                    {
                        "messages": [
                            HumanMessage(response.plan[0].request),
                            SystemMessage(response.plan[0].system_instruction, name=SUPERVISOR_NAME),
                        ]
                    },
                ),
            },
        )
    except Exception as e:
        error_msg = "I encountered an error while processing your query"
        logger.error(f"{error_msg}. ERROR: {e}")

        return {
            "next": END,
            "messages": [AIMessage(error_msg, name=SUPERVISOR_NAME)],
        }


def call_stock_agent(state: StockBossState) -> dict:
    logger.debug("Entering call_stock_agent in supervisor")

    try:
        send: Send = cast(Send, state["next"])
        stock_state: StockAgentState = {
            "messages": send.arg["messages"],
            "stock_data": state["stock_data"],
            "stock_summary": state["stock_summary"],
            "ticker": state["ticker"],
        }

        stock_result: StockAgentState = cast(StockAgentState, stock_agent.invoke(stock_state))

        summary = stock_result["stock_summary"]
        if stock_result.get("stock_data") is None or summary is None:
            error_msg = "I couldn't find any stock data. Could you please provide a valid stock symbol or company name?"
            logger.error(error_msg)
            return {
                "next": SUPERVISOR_NAME,
                "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)],
            }

        logger.debug("Leaving call_stock_agent with data")
        return {
            "stock_data": stock_result["stock_data"],
            "stock_summary": stock_result["stock_summary"],
            "ticker": stock_result["ticker"],
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(summary, name=STOCK_AGENT_NAME)],
        }

    except Exception as e:
        error_msg = "I encountered an error while fetching stock information"
        logger.error(error_msg + str(e))

        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=STOCK_AGENT_NAME)],
        }


def call_search_agent(state: StockBossState) -> dict:
    logger.debug("Entering call_search_agent in supervisor")
    try:
        send: Send = cast(Send, state["next"])
        search_state: SearchAgentState = {
            "ticker": state["ticker"],
            "stock_summary": state["stock_summary"],
            "messages": send.arg["messages"],
            "search_query": state["search_query"],
            "search_results": state["search_results"],
            "search_summary": state["search_summary"],
        }

        search_result: SearchAgentState = cast(SearchAgentState, search_agent.invoke(search_state))

        results = search_result["search_results"]
        summary = search_result["search_summary"]
        if search_result.get("search_query") is None or not results or not summary:
            error_msg = "I couldn't find any search results. Could you ask something more specific?"
            logger.error(error_msg)
            return {
                "next": SUPERVISOR_NAME,
                "messages": [AIMessage(error_msg, name=SEARCH_AGENT_NAME)],
            }

        logger.debug("Leaving call_search_agent with data")
        return {
            "next": SUPERVISOR_NAME,
            "search_query": search_result["search_query"],
            "search_results": results,
            "search_summary": summary,
            "messages": [AIMessage(summary, name=SEARCH_AGENT_NAME)],
        }

    except Exception as e:
        error_msg = "I encountered an error while fetching search information"
        logger.error(error_msg + str(e))

        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=SEARCH_AGENT_NAME)],
        }


def call_analyzer_agent(state: StockBossState) -> dict:
    logger.debug("Entering analyzer_agent in supervisor")
    try:
        send: Send = cast(Send, state["next"])
        analyzer_state: AnalyzerAgentState = {
            "ticker": state["ticker"],
            "stock_data": state["stock_data"],
            "stock_summary": state["stock_summary"],
            "messages": send.arg["messages"],
            "search_results": state["search_results"],
            "search_summary": state["search_summary"],
            "analysis_result": state["analysis_result"],
            "analysis_score": state["analysis_score"],
        }

        analysis_result: AnalyzerAgentState = cast(AnalyzerAgentState, analyzer_agent.invoke(analyzer_state))

        analysis = analysis_result["analysis_result"]
        if not analysis:
            error_msg = "I was unable to analyze the provided data. Could you please try again?"
            logger.error(error_msg)
            return {
                "next": SUPERVISOR_NAME,
                "messages": [AIMessage(error_msg, name=ANALYZER_AGENT_NAME)],
            }

        logger.debug("Leaving analyzer_agent with data")
        return {
            "next": SUPERVISOR_NAME,
            "analysis_result": analysis,
            "analysis_score": analysis_result["analysis_score"],
            "messages": [AIMessage(analysis, name=ANALYZER_AGENT_NAME)],
        }

    except Exception as e:
        error_msg = "I encountered an error while fetching search information"
        logger.error(error_msg + str(e))

        return {
            "next": SUPERVISOR_NAME,
            "messages": [AIMessage(error_msg, name=ANALYZER_AGENT_NAME)],
        }


boss = (
    StateGraph(StockBossState)
    .add_node(
        SUPERVISOR_NAME,
        boss_node,
        destinations=(STOCK_AGENT_NAME, SEARCH_AGENT_NAME, ANALYZER_AGENT_NAME, END),
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
    .add_node(
        ANALYZER_AGENT_NAME,
        call_analyzer_agent,
        destinations=(SUPERVISOR_NAME,),
    )
    #
    .set_entry_point(SUPERVISOR_NAME)
    .add_edge(STOCK_AGENT_NAME, SUPERVISOR_NAME)
    .add_edge(SEARCH_AGENT_NAME, SUPERVISOR_NAME)
    .add_edge(ANALYZER_AGENT_NAME, SUPERVISOR_NAME)
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
        "plan": [],
        "next": "",
        "step": -1,
        "search_query": None,
        "search_results": [],
        "search_summary": None,
        "analysis_result": None,
        "analysis_score": None,
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
                "plan": ", ".join([x.agent for x in state["plan"]]),
                "step": state["step"],
                "next": state["next"],
                "search_query": state["search_query"],
                "search_results": len(state["search_results"]),
                "search_summary": f"{state['search_summary'][:30]}..." if state["search_summary"] else None,
                "analysis_result": f"{state['analysis_result'][:30]}..." if state["analysis_result"] else None,
            }
        )
        print("\n\n========")

        query = input("Boss> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(StockBossState, boss.invoke(state, config=config))
        state.update(result)
