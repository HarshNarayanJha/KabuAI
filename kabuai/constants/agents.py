from typing import Final

STOCK_AGENT_NAME: Final[str] = "stock_agent"
SEARCH_AGENT_NAME: Final[str] = "search_agent"
ANALYZER_AGENT_NAME: Final[str] = "analyzer_agent"

SUPERVISOR_NAME: Final[str] = "supervisor"

MEMBERS: Final[list[str]] = [STOCK_AGENT_NAME, SEARCH_AGENT_NAME, ANALYZER_AGENT_NAME]

MEMBERS_DESCRIPTIONS: Final[dict[str, str]] = {
    STOCK_AGENT_NAME: "Fetches stock metrics and summary. Provides realtime stock data.",
    SEARCH_AGENT_NAME: "Searches the internet for news and latest information and provides sentiment scores for news items.",
    ANALYZER_AGENT_NAME: "Provides detailed stock and financial analysis of information fetched by the stock and search.",
}
