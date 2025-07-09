import logging
from datetime import datetime
from typing import NamedTuple, cast

import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError
from langchain_core.tools import tool
from pandas import Timestamp
from pydantic import BaseModel, Field

from models.stock import CompanyDetails, CompanyOfficer, Financials, News, StockData, StockMetadata, StockPrice

logger = logging.getLogger(__name__)


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
    logger.debug(f"Search Stock called with query: {query}")
    results = yf.Search(query)
    if not results.quotes:
        raise Exception(f"No stock found with the following query {query}")
    data = yf.Ticker(results.quotes[0]["symbol"])
    return data


class FetchStockDetailsInput(BaseModel):
    ticker_or_name: str = Field(description="The ticker symbol of the stock or  name of the company")


@tool("fetch_stock_details", args_schema=FetchStockDetailsInput)
def fetch_stock_details(ticker_or_name: str) -> StockData | str:
    """
    Fetches stock details for a given ticker symbol.
    If the given symbol is not a valid symbol, searches for the term and uses the first result.
    Do not pass None or no Value

    Args:
        ticker_or_name (str): The ticker symbol of the stock or name of the company.

    Returns:
        StockData | str: An object containing the stock details or an error message.
    """

    logger.debug(f"Fetch stock details tool used {ticker_or_name}")

    try:
        data = yf.Ticker(ticker_or_name)
        _ = data.info
    except HTTPError as e:
        if "404" not in str(e):
            raise

        try:
            data = search_stock(query=ticker_or_name)
        except Exception as e:
            logger.error(f"Failed to fetch stock details. Error: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to fetch stock details. Error: {e}")
        raise

    # --- Metadata ---
    info: dict[str, str] = data.info
    officers: list[CompanyOfficer] = []

    for officer in cast(list[dict], info.get("companyOfficers", [])):
        officers.append(
            CompanyOfficer(
                name=officer.get("name", "N/A"),
                title=officer.get("title", "N/A"),
                age=officer.get("age", 0),
                fiscalYear=officer.get("fiscalYear", 0),
                totalPay=officer.get("totalPay", 0),
            )
        )

    company_details = CompanyDetails(
        longName=info.get("longName", ""),
        symbol=info.get("symbol", ""),
        address1=info.get("address1", ""),
        city=info.get("city", ""),
        state=info.get("state", ""),
        zip=info.get("zip", ""),
        country=info.get("country", ""),
        phone=info.get("phone", ""),
        website=info.get("website", ""),
        industry=info.get("industry", ""),
        sector=info.get("sector", ""),
        longBusinessSummary=info.get("longBusinessSummary", ""),
        fullTimeEmployees=int(info.get("fullTimeEmployees", 0)),
        companyOfficers=officers,
        currentPrice=float(info.get("currentPrice", 0)),
        marketCap=int(info.get("marketCap", 0)),
        sharesOutstanding=int(info.get("sharesOutstanding", 0)),
        profitMargins=float(info.get("profitMargins", 0)),
        returnOnEquity=float(info.get("returnOnEquity", 0)),
        totalRevenue=int(info.get("totalRevenue", 0)),
        grossProfits=int(info.get("grossProfits", 0)),
        freeCashflow=int(info.get("freeCashflow", 0)),
        operatingCashflow=int(info.get("operatingCashflow", 0)),
        totalCash=int(info.get("totalCash", 0)),
        totalDebt=int(info.get("totalDebt", 0)),
        revenueGrowth=int(info.get("revenueGrowth", 0)),
        lastFiscalYearEnd=int(info.get("lastFiscalYearEnd", 0)),
        mostRecentQuarter=int(info.get("mostRecentQuarter", 0)),
        earningsTimestamp=int(info.get("earningsTimestamp", 0)),
    )

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
            # summary=n.get("content", {}).get("summary", ""),
            content_type=n.get("content", {}).get("contentType", ""),
            # url=n.get("content", {}).get("canonicalUrl", {}).get("url", ""),
            region=n.get("content", {}).get("canonicalUrl", {}).get("region", ""),
            provider=n.get("content", {}).get("provider", {}).get("displayName", ""),
        )
        for n in data.news[:5]
    ]

    logger.debug("Stock details fetch complete")

    return StockData(company=company_details, metadata=metadata, prices=prices, financials=financials, news=news)


if __name__ == "__main__":
    ticker = input("Ticker Or Company Name> ")
    print(fetch_stock_details.invoke(ticker))
