import json
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults

from models.search import SearchResult


def search_web(query: str, what: Literal["news", "text"] = "news") -> list[SearchResult]:
    """
    Searches Duck Duck Go for news or text content for the given query.
    Args:
        query (str): The search query.
        what (Literal["news", "text"], optional): The type of content to search for. Defaults to "news".

    Returns:
        list[SearchResult]: A list of search results.
    """

    search = DuckDuckGoSearchResults(output_format="json", backend=what)
    results = search.invoke(query)

    try:
        results = [SearchResult(**x) for x in json.loads(results)]
        return results
    except Exception as e:
        print(f"Failed to call the search_web tool: {e}")
        return []


if __name__ == "__main__":
    query = input("Query> ").strip()
    results = search_web(query)
    print(results)
