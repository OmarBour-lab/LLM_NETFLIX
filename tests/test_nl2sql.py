from src.nl2sql import question_to_sql, validate_sql


def test_generates_sql_for_movies_released_in_2020():
    sql, params = question_to_sql("movies released in 2020")

    assert validate_sql(sql)
    assert "FROM netflix_titles" in sql
    assert "type = :type" in sql
    assert "release_year = :release_year" in sql
    assert params["type"] == "Movie"
    assert params["release_year"] == 2020


def test_rejects_unsafe_sql():
    assert not validate_sql("DROP TABLE netflix_titles")
    assert not validate_sql("SELECT * FROM netflix_titles; DELETE FROM netflix_titles")


def test_generates_title_prefix_filter():
    sql, params = question_to_sql("all titles that start with the legend of")

    assert validate_sql(sql)
    assert "LOWER(title) LIKE :title_prefix" in sql
    assert params["title_prefix"] == "the legend of%"


def test_extracts_field_contains_value_without_field_name():
    sql, params = question_to_sql("movies with director containing John")

    assert validate_sql(sql)
    assert "LOWER(director)" in sql
    assert params["field_contains"] == "%john%"


def test_extracts_title_keyword_from_mixed_french_title_word():
    sql, params = question_to_sql("movies with pokemon in their titre")

    assert validate_sql(sql)
    assert "LOWER(title)" in sql
    assert params["field_contains"] == "%pokemon%"


def test_field_contains_removes_release_year_from_keyword():
    sql, params = question_to_sql("movies with genre containing horror released in 2019")

    assert validate_sql(sql)
    assert "LOWER(listed_in)" in sql
    assert params["release_year"] == 2019
    assert params["field_contains"] == "%horror%"


def test_unfiltered_sql_returns_no_rows_clause():
    sql, params = question_to_sql("Tell me about a title that does not exist xyzabc123")

    assert validate_sql(sql)
    assert "WHERE 1 = 0" in sql
    assert params == {}


def test_rating_filter_normalizes_punctuation():
    sql, params = question_to_sql("movies with rating containing TV-MA")

    assert validate_sql(sql)
    assert "regexp_replace" in sql
    assert params["field_contains"] == "%tv ma%"
