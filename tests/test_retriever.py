from src.retriever import (
    analyze_query,
    detect_requested_type,
    detect_requested_year,
    normalize_text,
)


def test_normalize_text_removes_accents():
    assert normalize_text("Réalisateur John") == "realisateur john"


def test_detect_requested_type_for_movies():
    assert detect_requested_type("movies released in 2020") == "Movie"


def test_detect_requested_year():
    assert detect_requested_year("series from Japan released in 2020") == 2020


def test_title_contains_query_cleans_filler_words():
    analysis = analyze_query("movies with pokemon in their titre")

    assert analysis.field_contains == ("title", "pokemon")
    assert analysis.title_keyword == "pokemon"


def test_genre_contains_query_removes_release_year_from_keyword():
    analysis = analyze_query("movies with genre containing horror released in 2019")

    assert analysis.field_contains == ("genres", "horror")
    assert analysis.requested_year == 2019
