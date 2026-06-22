from src.router import classify_intent


def test_routes_description_to_rag():
    assert classify_intent("Give me the description of Bird Box").intent == "rag"


def test_routes_filtered_list_to_sql():
    assert classify_intent("movies released in 2020").intent == "sql"


def test_routes_visualization_to_visualization():
    assert classify_intent("show me a chart of titles by year").intent == "visualization"

