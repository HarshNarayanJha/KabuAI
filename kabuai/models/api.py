from typing import Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from graph.boss_state import PlanStep
from models.search import SearchResult
from models.stock import StockData


class APIState(BaseModel):
    next: str = Field(default="")
    plan: list[PlanStep] = Field(default=[])
    step: int = Field(default=-1)
    messages: list[AnyMessage] = Field(default=[])
    # stock agent
    ticker: str | None = Field(default=None)
    stock_data: StockData | None = Field(default=None)
    stock_summary: str | None = Field(default=None)
    # search agent
    search_query: str | None = Field(default=None)
    search_results: list[SearchResult] = Field(default=[])
    search_summary: str | None = Field(default=None)
    # analysis_agent
    analysis_result: str | None = Field(default=None)
    analysis_score: float | None = Field(default=None)


class Request(BaseModel):
    state: APIState


class Response(BaseModel):
    type: Literal["handoff", "tool", "chunk", "update", "task"]
    # type = "handoff" | "tool"
    arguments: dict[str, Any] | None = Field(default=None)
    # type = "tool" | "task"
    name: str | None = Field(default=None)
    # type = "chunk"
    content: str | None = Field(default=None)
    # type = "update"
    state: dict[str, Any] | None = Field(default=None)
    # type = "task"
    direction: Literal["enter", "leave"] | None = Field(default=None)
