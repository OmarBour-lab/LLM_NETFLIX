from src.chain import FALLBACK_MESSAGE, try_sql_answer


def test_nonexistent_title_query_returns_fallback_without_broad_search():
    assert try_sql_answer("Tell me about a title that does not exist xyzabc123") == FALLBACK_MESSAGE

