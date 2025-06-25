from langgraph.graph import MessagesState
from langgraph.types import Send

from models.search import SearchResult
from models.stock import StockData


class StockBossState(MessagesState):
    next: str | Send
    # stock agent
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
    # search agent
    search_query: str | None
    search_results: list[SearchResult]
    search_summary: str | None
    # analysis: str
    # prediction: str
    # risk_summary: str
    # risk_score: float
    # advice: str
