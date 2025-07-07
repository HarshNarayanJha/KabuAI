from models.search import SearchResult


def calculate_overall_sentiment_score(search_results: list[SearchResult]) -> float:
    score = 0.0
    for s in search_results:
        score += s.sentiment_score * s.confidence

    return score
