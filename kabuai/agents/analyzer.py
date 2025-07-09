import json
import logging
import os
from pprint import pprint
from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from ai_models.chat import chat_model, chat_model_heavy, chat_model_light  # noqa: F401
from ai_models.llm import llm, llm_heavy, llm_light  # noqa: F401
from constants.agents import ANALYZER_AGENT_NAME, SUPERVISOR_NAME
from graph.analyzer_state import AnalyzerAgentState
from prompts.analyzer import analysis_prompt_template
from tools.search import search_web
from utils.search import calculate_overall_sentiment_score

DEBUG = os.getenv("DEBUG", "0") == "1"
SUMMARY_LENGTH: Literal["short", "medium", "long"] = "medium"

logger = logging.getLogger(__name__)
logger.info(f"SUMMARY_LENGTH set to {SUMMARY_LENGTH}")


class AnalysisResponseFormat(BaseModel):
    analysis_result: str = Field(description="Detailed analysis result")
    analysis_score: float = Field(
        description="Overall score for the stock analysis between 0.000 and 1.000 with 3 decimal points, i.e. 0.524",
        default=0.000,
    )


def perform_analysis_node(state: AnalyzerAgentState) -> dict | Command:
    """
    Performs stock analysis based on the provided data
    """

    logger.debug("Entering perform_analysis_node in analyzer agent")
    try:
        if (
            not state["stock_data"]
            or not state["stock_summary"]
            or not state["ticker"]
            or not state["search_results"]
            or not state["search_summary"]
        ):
            logger.debug("Leaving perform_analysis_node since missing data")
            err = "Some of the required data was not provided. Please provide me the latest data."
            logger.error(err)
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=ANALYZER_AGENT_NAME)],
                    "analysis_result": None,
                },
                graph=Command.PARENT,
            )

        sentiment_score = calculate_overall_sentiment_score(state["search_results"])
        messages = analysis_prompt_template.invoke(
            {
                "messages": state["messages"],
                "ticker": state["ticker"],
                "stock_data": state["stock_data"].model_dump_json(exclude={"news"}),
                "stock_summary": state["stock_summary"],
                "search_results": json.dumps(
                    [res.model_dump_json(exclude={"link"}) for res in state["search_results"]]
                ),
                "sentiment_score": sentiment_score,
                "search_summary": state["search_summary"],
                "summary_length": SUMMARY_LENGTH,
            }
        )

        sub_agent = create_react_agent(
            model=chat_model,
            tools=[search_web],
            response_format=AnalysisResponseFormat,
            name=f"{ANALYZER_AGENT_NAME}-react-agent",
        )

        analysis_response = sub_agent.invoke(messages)

        logger.debug(f"Analysis Response: {analysis_response}")

        # return to supervisor
        if not analysis_response or not analysis_response["structured_response"]:
            err = "I was unable to perform analysis"
            logger.error(err)
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=ANALYZER_AGENT_NAME)],
                    "analysis_result": None,
                    "analysis_score": None,
                },
                graph=Command.PARENT,
            )

        structured_response: AnalysisResponseFormat = analysis_response["structured_response"]

        logger.debug("Leaving perform_analysis_node")
        return {
            "analysis_result": structured_response.analysis_result,
            "analysis_score": structured_response.analysis_score,
        }

    except Exception as e:
        err = "I'm sorry, but I encountered an error while analyzing the data"
        logger.error(f"ERROR in perform_analysis_node: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={
                "messages": [AIMessage(content=err, name=ANALYZER_AGENT_NAME)],
                "analysis_result": None,
                "analysis_score": None,
            },
            graph=Command.PARENT,
        )


analyzer_agent = (
    StateGraph(AnalyzerAgentState)
    .add_node(
        "perform_analysis_node",
        perform_analysis_node,
        destinations=(END,),
    )
    .set_entry_point("perform_analysis_node")
    .set_finish_point("perform_analysis_node")
    .compile(name=ANALYZER_AGENT_NAME, debug=DEBUG)
)

if DEBUG:
    with open("analyzer_graph.png", "wb") as fp:
        fp.write(analyzer_agent.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    state: AnalyzerAgentState = {
        "ticker": None,
        "messages": [],
        "stock_data": None,
        "stock_summary": None,
        "search_results": [],
        "search_summary": None,
        "analysis_result": None,
        "analysis_score": None,
    }
    while True:
        print("========\n\n")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "analysis_result": state["analysis_result"],
            }
        )
        print("\n\n========")

        query = input(f"{ANALYZER_AGENT_NAME}> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(AnalyzerAgentState, analyzer_agent.invoke(state))

        state.update(result)
