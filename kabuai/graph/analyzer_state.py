from langgraph.graph import MessagesState

from models.search import SearchResult
from models.stock import StockData


class AnalyzerAgentState(MessagesState):
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
    search_results: list[SearchResult]
    search_summary: str | None
    # outputs
    analysis_result: str | None
    analysis_score: float | None
