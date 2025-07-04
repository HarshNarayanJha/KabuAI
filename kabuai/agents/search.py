import json
import os
from pprint import pprint
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from constants.agents import SEARCH_AGENT_NAME, SUPERVISOR_NAME
from graph.search_state import SearchAgentState
from prompts.search import search_prompt, sentiment_prompt, summary_prompt_template
from tools.search import search_web

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT") or ""
CHAT_MODEL_HEAVY = os.getenv("CHAT_MODEL_HEAVY") or ""

DEBUG = os.getenv("DEBUG", "0") == "1"

llm = init_chat_model(model=CHAT_MODEL, temperature=0, max_tokens=2048)
llm_light = init_chat_model(model=CHAT_MODEL_LIGHT, temperature=0, max_tokens=4096)
llm_heavy = init_chat_model(model=CHAT_MODEL_HEAVY, temperature=0)


class SearchQueryResponseFormat(BaseModel):
    query: str = Field(description="Query to search for. Must be 3-5 words")


def search_news_node(state: SearchAgentState) -> dict | Command:
    """
    Searches the web based on the system instruction and user's query
    """

    # if DEBUG:
    #     print("ENTERING search_news NODE with state:")
    #     pprint(
    #         {
    #             **state,
    #             "messages": [(message.type, message.content) for message in state["messages"]],
    #         }
    #     )

    try:
        messages = [
            SystemMessage(search_prompt.format(ticker=state["ticker"], stock_summary=state["stock_summary"])),
            *state["messages"],
        ]

        query_response = cast(
            SearchQueryResponseFormat, llm.with_structured_output(SearchQueryResponseFormat).invoke(messages)
        )

        if DEBUG:
            print(f"Query Response: {query_response}")

        # return to supervisor
        if not query_response or not query_response.query:
            err = "I was unable to generate a search query"
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
                    "search_results": [],
                    "search_query": None,
                },
                graph=Command.PARENT,
            )

        response = search_web(query_response.query)

        # if DEBUG:
        #     print(f"EXITING search_news NODE with response {response}")

        # continue to sentiment node
        return {"search_query": query_response.query, "search_results": response}

    except Exception as e:
        err = "I'm sorry, but I encountered an error while searching for news"
        print(f"ERROR in search_news NODE: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={
                "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
                "search_results": [],
                "search_query": None,
            },
            graph=Command.PARENT,
        )


class SentimentResultsResponseFormat(BaseModel):
    sentiment_scores: list[float] = Field(description="Sentiment scores for each search result, in order")
    confidence_scores: list[float] = Field(description="Confidence scores for each sentiment score, in order")


def sentiment_news_node(state: SearchAgentState) -> dict | Command:
    """
    Performs Sentiment Analysis on the search results
    """

    # if DEBUG:
    #     print("ENTERING sentiment_news NODE with state:")
    #     pprint(
    #         {
    #             **state,
    #             "messages": [(message.type, message.content) for message in state["messages"]],
    #         }
    #     )

    try:
        if not state["search_results"]:
            return {}

        messages = [
            SystemMessage(sentiment_prompt),
            HumanMessage(
                content=json.dumps(
                    [
                        {"title": res.title, "snippet": res.snippet, "date": res.date.isoformat(), "source": res.source}
                        for res in state["search_results"]
                    ]
                )
            ),
        ]

        if DEBUG:
            print(f"Asking for sentiment scores with messages: {messages}")

        response = cast(
            SentimentResultsResponseFormat,
            llm_light.with_structured_output(SentimentResultsResponseFormat).invoke(messages),
        )

        if DEBUG:
            print(f"GOT response in sentiment_news NODE: {response}")

        # if there is any kind of mistake in the response, empty or lengths do not match, or values out of range
        if (
            not response
            or not response.sentiment_scores
            or not response.confidence_scores
            or len(response.sentiment_scores) != len(response.confidence_scores)
            or not len(response.sentiment_scores) == len(state["search_results"])
            or any([x > 1 or x < -1 for x in response.sentiment_scores])
            or any([x > 1 or x < 0 for x in response.confidence_scores])
        ):
            err = "I was unable to generate sentiment scores"
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
                },
                graph=Command.PARENT,
            )

        updated_search_results = []

        for i, res in enumerate(state["search_results"]):
            mod_result = res.model_copy()
            mod_result.sentiment_score = response.sentiment_scores[i]
            mod_result.confidence = response.confidence_scores[i]

            updated_search_results.append(mod_result)

        return {"search_results": updated_search_results}

    except Exception as e:
        err = "I'm sorry, but I encountered an error while analyzing sentiment"
        print(f"ERROR in sentiment_news NODE: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={
                "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
            },
            graph=Command.PARENT,
        )


def news_summary_node(state: SearchAgentState) -> dict | Command:
    """
    Summarizes the search results to answer user's initial query.
    """

    # if DEBUG:
    #     print("ENTERING news_summary NODE with state:")
    #     pprint(
    #         {
    #             **state,
    #             "messages": [(message.type, message.content) for message in state["messages"]],
    #         }
    #     )

    try:
        if not state["search_results"]:
            return {}

        messages = summary_prompt_template.invoke(
            {
                "messages": state["messages"],
                "data": json.dumps(
                    [
                        {"title": res.title, "snippet": res.snippet, "date": res.date.isoformat(), "source": res.source}
                        for res in state["search_results"]
                    ]
                ),
            }
        )

        if DEBUG:
            print(f"Asking for news summary with messages: {messages}")

        response = llm.invoke(messages)

        if DEBUG:
            print(f"GOT response in news_summary NODE: {response}")

        if not response or not response.content:
            err = "I was unable to generate the answer."
            return Command(
                goto=SUPERVISOR_NAME,
                update={
                    "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
                    "search_summary": None,
                },
                graph=Command.PARENT,
            )

        return {"search_summary": response.content}

    except Exception as e:
        err = "I'm sorry, but I encountered an error while summarizing news"
        print(f"ERROR in news_summary NODE: {e}")
        return Command(
            goto=SUPERVISOR_NAME,
            update={
                "messages": [AIMessage(content=err, name=SEARCH_AGENT_NAME)],
                "search_summary": None,
            },
            graph=Command.PARENT,
        )


search_agent = (
    StateGraph(SearchAgentState)
    .add_node(
        "search_news_node",
        search_news_node,
        destinations=("sentiment_news_node",),
    )
    .add_node(
        "sentiment_news_node",
        sentiment_news_node,
        destinations=("news_summary_node",),
    )
    .add_node(
        "news_summary_node",
        news_summary_node,
        destinations=(END,),
    )
    .set_entry_point("search_news_node")
    .add_edge("search_news_node", "sentiment_news_node")
    .add_edge("sentiment_news_node", "news_summary_node")
    .set_finish_point("news_summary_node")
    .compile(name=SEARCH_AGENT_NAME, debug=DEBUG)
)

if DEBUG:
    with open("search_graph.png", "wb") as fp:
        fp.write(search_agent.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    state: SearchAgentState = {
        "messages": [],
        "search_query": None,
        "search_results": [],
        "search_summary": None,
        "stock_summary": None,
        "ticker": None,
    }
    while True:
        print("========\n\n")
        pprint(
            {
                "messages": [(message.type, message.content) for message in state["messages"]],
                "search_query": state["search_query"],
                "search_results": [
                    {
                        "title": x.title,
                        "sentiment_score": x.sentiment_score,
                        "confidence": x.confidence,
                    }
                    for x in state["search_results"]
                ],
                "search_summary": state["search_summary"],
            }
        )
        print("\n\n========")

        query = input(f"{SEARCH_AGENT_NAME}> ").strip()
        state["messages"].append(HumanMessage(query))

        result = cast(SearchAgentState, search_agent.invoke(state))

        state.update(result)
