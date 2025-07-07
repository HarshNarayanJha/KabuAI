from datetime import datetime

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    snippet: str = Field(description="Snippet of the search item", repr=False)
    title: str = Field(description="Title of the search item")
    link: str = Field(description="URL of the search item", repr=False)
    date: datetime = Field(description="Publication date of the search item", repr=False)
    source: str = Field(description="Source of the search item")
    sentiment_score: float = Field(
        default=0.0,
        description="Sentiment analysis score for the search item. Ranges from -1.0 (most negative) to 1.0 (most positive)",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for the sentiment analysis. Ranges from 0.0 (least confident) to 1.0 (most confident)",
    )
