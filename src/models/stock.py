from datetime import datetime

from pydantic import BaseModel, Field


class StockMetadata(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    company_name: str | None = Field(None, description="Full name of the company")
    sector: str | None = Field(None, description="Market sector of the company")
    industry: str | None = Field(None, description="Specific industry classification")
    market_cap: int | None = Field(None, description="Market capitalization in billions")
    pe_ratio: float | None = Field(None, description="Price to earnings ratio")
    dividend_yield: float | None = Field(None, description="Annual dividend yield percentage")
    beta: float | None = Field(None, description="Measure of stock volatility relative to market")


class StockPrice(BaseModel):
    date: datetime = Field(description="Date of the stock price data")
    open: float = Field(description="Opening price of the stock")
    high: float = Field(description="Highest price during trading session")
    low: float = Field(description="Lowest price during trading session")
    close: float = Field(description="Closing price of the stock")
    adjusted_close: float = Field(description="Closing price adjusted for corporate actions")
    volume: int = Field(description="Number of shares traded")


class Financials(BaseModel):
    revenue: float | None = Field(None, description="Total revenue from sales")
    gross_profit: float | None = Field(None, description="Revenue minus cost of goods sold")
    operating_income: float | None = Field(None, description="Profit from operations")
    net_income: float | None = Field(None, description="Total profit after all expenses")
    total_assets: float | None = Field(None, description="Sum of all company assets")
    total_liabilities: float | None = Field(None, description="Sum of all company debts and obligations")
    shareholders_equity: float | None = Field(None, description="Net worth of the company")
    current_ratio: float | None = Field(None, description="Current assets divided by current liabilities")
    quick_ratio: float | None = Field(None, description="Liquid assets divided by current liabilities")
    return_on_equity: float | None = Field(None, description="Net income divided by shareholders equity")
    return_on_assets: float | None = Field(None, description="Net income divided by total assets")


class News(BaseModel):
    date: datetime = Field(description="Date of the news article")
    headline: str = Field(description="News article headline")
    # summary: str = Field(description="Summary of the article")
    content_type: str = Field(description="Type of content")
    # url: str = Field(description="URL of the news article")
    region: str | None = Field(None, description="Region where the news originated")
    provider: str | None = Field(None, description="Provider of the news")


class StockData(BaseModel):
    metadata: StockMetadata = Field(description="General information about the stock")
    prices: list[StockPrice] = Field(description="Historical price data")
    financials: Financials = Field(description="Financial metrics and ratios")
    news: list[News] = Field(description="Related news articles")
