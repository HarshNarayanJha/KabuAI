from typing import Any, Literal

from pydantic import BaseModel, Field

from models.search import SearchResult
from models.stock import StockData


class Message(BaseModel):
    type: Literal["ai", "human", "system", "tool"]
    content: str
    name: str = Field(default="")


class APIState(BaseModel):
    next: str = Field(default="")
    messages: list[Message] = Field(default=[])
    # stock agent
    ticker: str | None = Field(default=None)
    stock_data: StockData | None = Field(default=None)
    stock_summary: str | None = Field(default=None)
    # search agent
    search_query: str | None = Field(default=None)
    search_results: list[SearchResult] = Field(default=[])
    search_summary: str | None = Field(default=None)


class Request(BaseModel):
    state: APIState


class Response(BaseModel):
    type: Literal["handoff", "tool", "chunk", "update"]
    # type = "handoff" | "tool"
    arguments: dict[str, Any] | None = Field(default=None)
    # type = "tool"
    name: str | None = Field(default=None)
    # type = "chunk"
    content: str | None = Field(default=None)
    # type = "update"
    state: APIState | None = Field(default=None)
