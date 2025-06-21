import json
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AnyMessage, BaseMessage, BaseMessageChunk, HumanMessage
from langgraph.types import Send

from agents.boss import boss
from constants.agents import STOCK_AGENT_NAME, SUPERVISOR_NAME
from graph.boss_state import StockBossState
from models.api import Request, Response


def invoke_agent(state: StockBossState, callables: list) -> StockBossState:
    return cast(StockBossState, boss.invoke(state, config={"callbacks": callables, "configurable": {"thread_id": "1"}}))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["127.0.0.1:5000"],
    allow_methods=["GET", "POST"],
)


def sse_format(payload: Response) -> str:
    return f"data: {payload.model_dump_json(exclude_unset=True)}\n\n"


@app.get("/")
async def root():
    return {"message": "Welcome to the VentureAI API!"}


@app.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    state: StockBossState = {
        "messages": request.state.messages or [],
        "stock_data": request.state.stock_data or None,
        "stock_summary": request.state.stock_summary or None,
        "ticker": request.state.ticker or None,
        "next": "",
    }

    async def stream_generator():
        # we are streaming both modes
        # messages -> to get chunk by chunk streaming messages
        # updates -> for tool calls and state updates
        #
        async for mode, data in boss.astream(state, stream_mode=["messages", "updates"], subgraphs=False):
            if mode == "messages":
                token, metadata = cast(tuple[AnyMessage, dict[str, Any]], data)
                node_name = metadata["langgraph_node"]

                # identify tool call
                if token.additional_kwargs and "function_call" in token.additional_kwargs:
                    if node_name == SUPERVISOR_NAME:
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

                elif isinstance(token, BaseMessageChunk):
                    # message chunk
                    yield sse_format(
                        Response(
                            type="chunk",
                            name=node_name,
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

                    yield sse_format(
                        Response(
                            type="update",
                            state=updated_state,
                        )
                    )
            else:
                raise ValueError(f"Invalid Mode {mode}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    state: StockBossState = {"messages": [], "stock_data": None, "stock_summary": None, "ticker": None, "next": ""}
    while True:
        query = input("You> ").strip()
        state["messages"].append(HumanMessage(query))
        result = invoke_agent(state, [])
        print(result["messages"])
        state.update(result)
