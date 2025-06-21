from typing import TypedDict

from langchain_core.messages import AnyMessage

from models.stock import StockData


class APIState(TypedDict):
    messages: list[AnyMessage]
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
