from langgraph.graph import MessagesState

from models.search import SearchResult


class SearchAgentState(MessagesState):
    ticker: str | None
    stock_summary: str | None
    search_query: str | None
    search_results: list[SearchResult]
    search_summary: str | None
