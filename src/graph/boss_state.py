from langgraph.graph import MessagesState
from langgraph.types import Send

from models.stock import StockData


class StockBossState(MessagesState):
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
    next: str | Send
    # news_data: NewsData
    # analysis: str
    # prediction: str
    # risk_summary: str
    # risk_score: float
    # advice: str
