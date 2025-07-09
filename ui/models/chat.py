from typing import Any, Literal

from pydantic import BaseModel, Field

from models.api import Message
from models.search import SearchResult
from models.stock import StockData


class ChatEntry(BaseModel):
    entry_type: Literal["message", "stock_card", "news_items", "chart", "tool"]
    message: Message | None = Field(default=None)
    stock_data: StockData | None = Field(default=None)
    news_items: list[SearchResult] | None = Field(default=None)
    tool_name: str | None = Field(default=None)
    tool_args: dict[str, Any] | None = Field(default=None)
