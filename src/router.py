from dataclasses import dataclass
from typing import Literal


Intent = Literal["rag", "sql", "mixed", "visualization"]


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    reason: str


VISUALIZATION_MARKERS = {
    "chart",
    "graph",
    "plot",
    "visualize",
    "distribution",
    "statistics",
    "stats",
    "top countries",
    "top genres",
    "ratings",
}

SQL_MARKERS = {
    "all",
    "available",
    "containing",
    "director",
    "cast",
    "actor",
    "actors",
    "country",
    "from",
    "genre",
    "genres",
    "list",
    "movies",
    "released",
    "series",
    "shows",
    "start with",
    "starts with",
    "title",
    "titles",
    "year",
}

RAG_MARKERS = {
    "about",
    "description",
    "describe",
    "summary",
    "what do you know",
    "what genre",
    "genre is",
    "what rating",
    "rating is",
    "who directed",
    "who stars",
    "duration",
}


def classify_intent(question: str) -> RouteDecision:
    normalized = question.lower()

    if any(marker in normalized for marker in VISUALIZATION_MARKERS):
        return RouteDecision("visualization", "The question asks for a chart or catalog statistics.")

    has_sql_signal = any(marker in normalized for marker in SQL_MARKERS)
    has_rag_signal = any(marker in normalized for marker in RAG_MARKERS)

    if has_sql_signal and has_rag_signal:
        return RouteDecision("mixed", "The question combines structured filters with descriptive details.")
    if has_sql_signal:
        return RouteDecision("sql", "The question can be answered with structured catalog filters.")
    return RouteDecision("rag", "The question is best answered from retrieved semantic context.")


def train_baseline_classifier():
    """Return a tiny TF-IDF classifier used as an experimental routing baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    examples = [
        ("Give me the description of Bird Box", "rag"),
        ("What do you know about Fate/Zero?", "rag"),
        ("Who directed Bird Box?", "rag"),
        ("movies released in 2020", "sql"),
        ("series from Japan released in 2020", "sql"),
        ("movies with director containing John", "sql"),
        ("all titles that start with the legend of", "sql"),
        ("show me a chart of titles by year", "visualization"),
        ("top countries chart", "visualization"),
    ]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit([text for text, _ in examples], [label for _, label in examples])
    return model
