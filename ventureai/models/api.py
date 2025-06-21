from typing import Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from models.stock import StockData


class APIState(BaseModel):
    messages: list[AnyMessage]
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None


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
    state: dict[str, Any] | None = Field(default=None)
