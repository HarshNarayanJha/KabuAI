import json
import logging
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models.search import SearchResult

logger = logging.getLogger(__name__)


class SearchWebInput(BaseModel):
    query: str = Field(description="The query to search for. Small but to the point and specific")
    what: Literal["news", "text"] = Field(
        description="The type of content to search for. 'news' is better for news styled results. 'text' is better for simple information gathering. Defaults to 'news'."
    )


@tool("search_web", args_schema=SearchWebInput)
def search_web(query: str, what: Literal["news", "text"] = "news") -> list[SearchResult]:
    """
    Searches Duck Duck Go for news or text content for the given query.
    Args:
        query (str): The search query.
        what (Literal["news", "text"], optional): The type of content to search for. Defaults to "news".

    Returns:
        list[SearchResult]: A list of search results.
    """

    logger.warning("Changing what to news even if text was passed, since text output is... uh..")
    what = "news"

    search = DuckDuckGoSearchResults(output_format="json", backend=what, num_results=5)
    results = search.invoke(query)

    try:
        results = [SearchResult(**x) for x in json.loads(results)]
        return results
    except Exception as e:
        logger.error(f"Failed to call the search_web tool: {e}")
        return []


if __name__ == "__main__":
    query = input("Query> ").strip()
    results = search_web(query)
    print(results)
