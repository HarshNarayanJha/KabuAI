from langgraph.graph import MessagesState

from models.stock import StockData


class StockAgentState(MessagesState):
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
