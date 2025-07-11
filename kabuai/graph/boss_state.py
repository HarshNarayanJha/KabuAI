from typing import Literal

from langgraph.graph import MessagesState
from langgraph.types import Send
from pydantic import BaseModel, Field

from models.search import SearchResult
from models.stock import StockData


class PlanStep(BaseModel):
    agent: Literal["stock_agent", "search_agent", "analyzer_agent", "FINISH"] = Field(
        description="Agent to route to. If not agents needed, route to FINISH."
    )
    request: str = Field(description="User's request to this agent. Extract from user's message.")
    message: str = Field(description="Response Message to the user.")
    system_instruction: str = Field(description="Very Brief system instruction to the agent.")


class StockBossState(MessagesState):
    next: str | Send
    plan: list[PlanStep]
    step: int
    # stock agent
    ticker: str | None
    stock_data: StockData | None
    stock_summary: str | None
    # search agent
    search_query: str | None
    search_results: list[SearchResult]
    search_summary: str | None
    # analyzer agent
    analysis_result: str | None
    analysis_score: float | None
    # prediction: str
    # risk_summary: str
    # risk_score: float
    # advice: str
