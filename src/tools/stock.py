from datetime import datetime
from typing import NamedTuple, cast

import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError
from langchain_core.tools import tool
from pandas import Timestamp

from models.stock import Financials, News, StockData, StockMetadata, StockPrice


class HistoryRow(NamedTuple):
    Index: Timestamp
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int
    Dividends: float
    Stock_Splits: float


def search_stock(query: str) -> yf.Ticker:
    print(f"Search Stock called with query: {query}")
    results = yf.Search(query)
    if not results.quotes:
        raise Exception(f"No stock found with the following query {query}")
    data = yf.Ticker(results.quotes[0]["symbol"])
    return data


@tool
def fetch_stock_details(ticker: str) -> StockData:
    """
    Fetches stock details for a given ticker symbol.
    If the given symbol is not a valid symbol, searches for the term and uses the first result.

    Args:
        ticker (str): The ticker symbol of the stock.

    Returns:
        StockData: An object containing the stock details.
    """

    print(f"Fetch stock details tool used {ticker}")

    data = yf.Ticker(ticker)
    try:
        # try loading the data
        _ = data.info
    except HTTPError as e:
        if "404" in str(e):
            # ticker not found, search it
            data = search_stock(query=ticker)
        else:
            raise
    except Exception as e:
        raise Exception(f"Failed to fetch stock details: {e}")

    # --- Metadata ---
    info: dict[str, str] = data.info
    metadata = StockMetadata(
        symbol=info["symbol"],
        company_name=info.get("longName") or info.get("shortName", ""),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
        market_cap=int(info.get("marketCap", 0)) or None,
        pe_ratio=float(info.get("trailingPE", 0)) or None,
        dividend_yield=float(info.get("dividendYield", 0)) * 100,
        beta=float(info.get("beta", 0)),
    )

    # --- Price History ---
    hist = data.history(period="6mo")
    prices = [
        StockPrice(
            date=cast(Timestamp, index).to_pydatetime(),
            open=cast(float, row.get("Open", 0)),
            high=cast(float, row.get("High", 0)),
            low=cast(float, row.get("Low", 0)),
            close=cast(float, row.get("Close", 0)),
            adjusted_close=cast(float, row.get("Adj Close", row.get("Close", 0))),
            volume=cast(int, row.get("Volume", 0)),
        )
        for index, row in hist.iterrows()
    ]

    # --- Financials ---
    income = data.income_stmt
    balance = data.balance_sheet
    financials = Financials(
        revenue=income.loc["Total Revenue"].dropna().iloc[0] if "Total Revenue" in income else None,
        gross_profit=income.loc["Gross Profit"].dropna().iloc[0] if "Gross Profit" in income else None,
        operating_income=income.loc["Operating Income"].dropna().iloc[0] if "Operating Income" in income else None,
        net_income=income.loc["Net Income"].dropna().iloc[0] if "Net Income" in income else None,
        total_assets=balance.loc["Total Assets"].dropna().iloc[0] if "Total Assets" in balance else None,
        total_liabilities=balance.loc["Total Liab"].dropna().iloc[0] if "Total Liab" in balance else None,
        shareholders_equity=balance.loc["Total Stockholder Equity"].dropna().iloc[0]
        if "Total Stockholder Equity" in balance
        else None,
        current_ratio=float(info.get("currentRatio", 0)),
        quick_ratio=float(info.get("quickRatio", 0)),
        return_on_equity=float(info.get("returnOnEquity", 0)),
        return_on_assets=float(info.get("returnOnAssets", 0)),
    )

    # --- News ---
    news = [
        News(
            date=datetime.fromisoformat(n.get("content", {}).get("pubDate", "")),
            headline=n.get("content", {}).get("title", ""),
            summary=n.get("content", {}).get("summary", ""),
            content_type=n.get("content", {}).get("contentType", ""),
            url=n.get("content", {}).get("canonicalUrl", {}).get("url", ""),
            region=n.get("content", {}).get("canonicalUrl", {}).get("region", ""),
            provider=n.get("content", {}).get("provider", {}).get("displayName", ""),
        )
        for n in data.news[:10]
    ]

    return StockData(metadata=metadata, prices=prices, financials=financials, news=news)
