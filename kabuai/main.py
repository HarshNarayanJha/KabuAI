import json
import logging
import os
from pprint import pprint
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AnyMessage, BaseMessage, BaseMessageChunk, HumanMessage
from langgraph.types import Send

from agents.boss import boss
from constants.agents import ANALYZER_AGENT_NAME, SEARCH_AGENT_NAME, STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.boss_state import StockBossState
from models.api import Request, Response
from utils.logger import setup_logging

DEBUG = os.getenv("DEBUG", "0") == "1"


def invoke_agent(state: StockBossState, callables: list) -> StockBossState:
    return cast(StockBossState, boss.invoke(state, config={"callbacks": callables, "configurable": {"thread_id": "1"}}))


setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Starting with DEBUG: {DEBUG}")

app = FastAPI()

origin_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in origin_env.split(",") if origin_env.strip()]

logger.info(f"Allowed Origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def sse_format(payload: Response) -> str:
    logger.info(f"Yielding {payload.type}")
    return f"data: {payload.model_dump_json(exclude_unset=True)}\n\n"


@app.get("/")
async def root():
    return {"message": "Welcome to the KabuAI API!"}


@app.get("/health")
async def health():
    return {"message": "Health Check!"}


@app.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    state: StockBossState = {
        "messages": request.state.messages,
        "stock_data": request.state.stock_data,
        "stock_summary": request.state.stock_summary,
        "ticker": request.state.ticker,
        "next": request.state.next,
        "plan": request.state.plan,
        "step": request.state.step,
        "search_query": request.state.search_query,
        "search_results": request.state.search_results,
        "search_summary": request.state.search_summary,
        "analysis_result": request.state.analysis_result,
        "analysis_score": request.state.analysis_score,
    }

    async def stream_generator():
        # we are streaming these modes
        # messages -> to get chunk by chunk streaming messages
        # updates -> for tool calls and state updates
        # tasks -> for context changes

        async for namespace, mode, data in boss.astream(
            state, stream_mode=["messages", "updates", "tasks", "custom"], subgraphs=True
        ):
            logger.debug(namespace)
            logger.debug(mode)
            logger.debug(data)
            logger.debug("---")

            if mode == "messages":
                token, metadata = cast(tuple[AnyMessage, dict[str, Any]], data)
                node_name = metadata["langgraph_node"]

                # identify tool call
                if token.additional_kwargs and "function_call" in token.additional_kwargs:
                    if node_name == SUPERVISOR_NAME:
                        if token.additional_kwargs["function_call"]["name"] == "Router":
                            # now we need to use custom events for handoff events
                            logger.warning(
                                "Ignoring Router function call in supervisor. custom event 'handoff' is handling these now."
                            )
                        else:
                            logger.warning("Shouldn't have happen, another tool call in supervisor.")
                            # no longer true, since it is a plan now
                            # it is the routing call
                            # yield handoff message
                            yield sse_format(
                                Response(
                                    type="handoff",
                                    arguments=json.loads(token.additional_kwargs["function_call"]["arguments"]),
                                )
                            )

                    elif node_name in (STOCK_AGENT_NAME, "stock_details_node"):
                        # it is the ticker tool call
                        # yield tool call message
                        yield sse_format(
                            Response(
                                type="tool",
                                name="fetch_stock_details",
                                arguments=json.loads(token.additional_kwargs["function_call"]["arguments"]),
                            )
                        )

                    elif node_name in (SEARCH_AGENT_NAME, "search_news_node"):
                        # it is the search tool call
                        # yield tool call message
                        yield sse_format(
                            Response(
                                type="tool",
                                name="web_search",
                                arguments=json.loads(token.additional_kwargs["function_call"]["arguments"]),
                            )
                        )

                    elif node_name in (ANALYZER_AGENT_NAME, "perform_analysis_node"):
                        # it is the search tool call
                        # yield tool call message
                        yield sse_format(
                            Response(
                                type="tool",
                                name="web_analysis",
                                arguments=json.loads(token.additional_kwargs["function_call"]["arguments"]),
                            )
                        )

                elif isinstance(token, BaseMessageChunk) and token.content:
                    # message chunk
                    agent_name = node_name
                    if namespace and namespace[0] and ":" in namespace[0]:
                        agent_name = namespace[0].split(":")[0]

                    yield sse_format(
                        Response(
                            type="chunk",
                            name=agent_name,
                            content=str(token.content),
                        )
                    )

            elif mode == "updates":
                # need to convert messages to json
                # and handle next with Send

                data = cast(dict[str, Any], data)

                if SUPERVISOR_NAME in data:
                    updated_state = cast(dict[str, Any], data[SUPERVISOR_NAME])

                    if "next" in updated_state and isinstance(updated_state["next"], Send):
                        updated_state["next"] = updated_state["next"].node

                    if "messages" in updated_state:
                        updated_messages = []
                        for msg in updated_state["messages"]:
                            if isinstance(msg, BaseMessage):
                                updated_messages.append({"type": msg.type, "content": msg.content, "name": msg.name})
                            else:
                                updated_messages.append(msg)

                        updated_state["messages"] = updated_messages

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )

                elif STOCK_AGENT_NAME in data:
                    updated_state = cast(dict[str, Any], data[STOCK_AGENT_NAME])

                    if "next" in updated_state and isinstance(updated_state["next"], Send):
                        updated_state["next"] = updated_state["next"].node

                    if "messages" in updated_state:
                        updated_messages = []
                        for msg in updated_state["messages"]:
                            if isinstance(msg, BaseMessage):
                                updated_messages.append({"type": msg.type, "content": msg.content, "name": msg.name})
                            else:
                                updated_messages.append(msg)

                        updated_state["messages"] = updated_messages

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "stock_details_node" in data:
                    updated_state = cast(dict[str, Any], data["stock_details_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "stock_summary_node" in data:
                    updated_state = cast(dict[str, Any], data["stock_summary_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )

                elif SEARCH_AGENT_NAME in data:
                    updated_state = cast(dict[str, Any], data[SEARCH_AGENT_NAME])

                    if "next" in updated_state and isinstance(updated_state["next"], Send):
                        updated_state["next"] = updated_state["next"].node

                    if "messages" in updated_state:
                        updated_messages = []
                        for msg in updated_state["messages"]:
                            if isinstance(msg, BaseMessage):
                                updated_messages.append({"type": msg.type, "content": msg.content, "name": msg.name})
                            else:
                                updated_messages.append(msg)

                        updated_state["messages"] = updated_messages

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "search_news_node" in data:
                    updated_state = cast(dict[str, Any], data["search_news_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "sentiment_news_node" in data:
                    updated_state = cast(dict[str, Any], data["sentiment_news_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "news_summary_node" in data:
                    updated_state = cast(dict[str, Any], data["news_summary_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )

                elif ANALYZER_AGENT_NAME in data:
                    updated_state = cast(dict[str, Any], data[ANALYZER_AGENT_NAME])

                    if "next" in updated_state and isinstance(updated_state["next"], Send):
                        updated_state["next"] = updated_state["next"].node

                    if "messages" in updated_state:
                        updated_messages = []
                        for msg in updated_state["messages"]:
                            if isinstance(msg, BaseMessage):
                                updated_messages.append({"type": msg.type, "content": msg.content, "name": msg.name})
                            else:
                                updated_messages.append(msg)

                        updated_state["messages"] = updated_messages

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
                elif "perform_analysis_node" in data:
                    updated_state = cast(dict[str, Any], data["perform_analysis_node"])

                    if "messages" in updated_state:
                        # don't allow messages update from this node
                        del updated_state["messages"]

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )

                elif "process_analysis_node" in data:
                    updated_state = cast(dict[str, Any], data["process_analysis_node"])

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )

            elif mode == "tasks":
                # Emit task changes (not in use right now)
                data = cast(dict[str, Any], data)

                direction = None
                if "input" in data:
                    direction = "enter"
                elif "result" in data:
                    direction = "leave"

                yield sse_format(
                    Response(
                        type="task",
                        name=data["name"],
                        direction=direction,
                    )
                )
            elif mode == "custom":
                # emit custom events
                data = cast(dict[str, Any], data)
                if handoff := data.get("handoff"):
                    yield sse_format(
                        Response(
                            type="handoff",
                            arguments=handoff,
                        )
                    )
                else:
                    logger.warning("Unknown custom event detected.")

            else:
                logger.warning(f"Unknown mode detected {mode}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    state: StockBossState = {
        "messages": [],
        "stock_data": None,
        "stock_summary": None,
        "ticker": None,
        "next": "",
        "plan": [],
        "step": -1,
        "search_query": None,
        "search_results": [],
        "search_summary": None,
        "analysis_result": None,
        "analysis_score": None,
    }
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

        query = input("You> ").strip()
        state["messages"].append(HumanMessage(query))
        result = invoke_agent(state, [])
        state.update(result)
