import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import DATABASE_URL
except ImportError:
    from config import DATABASE_URL


load_dotenv()

BASE_SELECT = (
    'SELECT show_id, type, title, director, "cast", country, date_added, '
    "release_year, rating, duration, listed_in, description "
    "FROM netflix_titles"
)
ORDER_BY = " ORDER BY title ASC, release_year ASC"
FORBIDDEN_SQL = re.compile(
    r"\b(drop|delete|update|insert|alter|truncate|create|grant|revoke|copy|execute)\b",
    re.IGNORECASE,
)

COUNTRY_ALIASES = {
    "South Korea": ["south korea", "korea", "korean", "coree", "coreen"],
    "India": ["india", "indian", "inde", "indien", "indienne"],
    "United States": ["united states", "usa", "american", "etats unis"],
    "United Kingdom": ["united kingdom", "british", "royaume uni"],
    "Japan": ["japan", "japanese", "japon", "japonais"],
}

FIELD_ALIASES = {
    "director": ["director", "directed by", "realisateur", "réalisateur"],
    "cast": ["cast", "actor", "actors", "acteur", "acteurs"],
    "country": ["country", "countries", "pays"],
    "listed_in": ["genre", "genres", "listed in", "category", "categorie"],
    "rating": ["rating", "classification"],
    "duration": ["duration", "duree", "durée"],
    "title": ["title", "titre"],
}


@dataclass
class SQLQueryResult:
    sql: str
    params: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


def normalize_text(text_value: str) -> str:
    normalized = unicodedata.normalize("NFKD", text_value.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def detect_requested_type(question: str) -> str | None:
    normalized = normalize_text(question)
    wants_show = re.search(r"\b(series|show|shows|tv show|serie)\b", normalized)
    wants_movie = re.search(r"\b(movie|movies|film|films)\b", normalized)
    if wants_show and not wants_movie:
        return "TV Show"
    if wants_movie and not wants_show:
        return "Movie"
    return None


def detect_year(question: str) -> int | None:
    match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", question)
    return int(match.group(1)) if match else None


def detect_country(question: str) -> str | None:
    normalized = normalize_text(question)
    for country, aliases in COUNTRY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(normalize_text(alias))}\b", normalized) for alias in aliases):
            return country
    return None


def detect_title_prefix(question: str) -> str | None:
    normalized = normalize_text(question)
    match = re.search(r"(?:start|starts|begin|begins) with ([a-z0-9 ]+)", normalized)
    if not match:
        return None
    prefix = re.sub(
        r"\b(movie|movies|series|shows|available|catalog|catalogue)\b",
        "",
        match.group(1),
    ).strip()
    return prefix or None


def clean_keyword(value: str) -> str:
    filler = {
        "a",
        "all",
        "available",
        "catalog",
        "catalogue",
        "containing",
        "find",
        "for",
        "give",
        "in",
        "list",
        "me",
        "movie",
        "movies",
        "of",
        "series",
        "show",
        "shows",
        "that",
        "the",
        "their",
        "title",
        "titre",
        "titles",
        "with",
    }
    return " ".join(word for word in normalize_text(value).split() if word not in filler)


def clean_field_keyword(value: str, question: str) -> str:
    detected_year = detect_year(question)
    ignored = {"released", "release", "year", "annee", "sorti", "sortie"}
    if detected_year:
        ignored.add(str(detected_year))
    return " ".join(word for word in clean_keyword(value).split() if word not in ignored)


def detect_field_contains(question: str) -> tuple[str, str] | None:
    normalized = normalize_text(question)
    detected_field = None
    detected_aliases: list[str] = []
    for field_name, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(normalize_text(alias))}\b", normalized):
                detected_field = field_name
                detected_aliases = aliases
                break
        if detected_field:
            break
    if not detected_field:
        return None

    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    if quoted:
        return detected_field, normalize_text(quoted.group(1) or quoted.group(2))

    for alias in detected_aliases:
        normalized_alias = normalize_text(alias)
        field_match = re.search(
            rf"\b{re.escape(normalized_alias)}\b\s+(?:contains|containing|with|avec|par|by)\s+([a-z0-9 ]+)",
            normalized,
        )
        if field_match:
            keyword = clean_field_keyword(field_match.group(1), question)
            if keyword:
                return detected_field, keyword

    match = re.search(r"(?:contains|containing|with|by|avec|par)\s+([a-z0-9 ]+)", normalized)
    if not match:
        return None
    keyword = clean_field_keyword(match.group(1), question)
    return (detected_field, keyword) if keyword else None


def detect_title_keyword(question: str) -> str | None:
    normalized = normalize_text(question)
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    if quoted:
        return normalize_text(quoted.group(1) or quoted.group(2))

    patterns = [
        r"(?:list|find|show me|give me)(?: all)? (?:titles|movie|movies|series|shows) (?:with|containing|about|called|named) ([a-z0-9 ]+)",
        r"(?:all|which|what) (?:titles|movie|movies|series|shows) (?:with|containing|about|called|named) ([a-z0-9 ]+)",
        r"(?:all|find|give me all)(?: the)? ([a-z0-9 ]+?) (?:movie|movies|series|shows)(?: available)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            keyword = clean_keyword(match.group(1))
            if keyword:
                return keyword
    return None


def column_sql(field_name: str) -> str:
    return '"cast"' if field_name == "cast" else field_name


def normalized_column_sql(field_name: str) -> str:
    return f"regexp_replace(LOWER({column_sql(field_name)}), '[^a-z0-9]+', ' ', 'g')"


def question_to_sql(question: str, limit: int | None = None) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    requested_type = detect_requested_type(question)
    if requested_type:
        clauses.append("type = :type")
        params["type"] = requested_type

    year = detect_year(question)
    if year:
        clauses.append("release_year = :release_year")
        params["release_year"] = year

    country = detect_country(question)
    if country:
        clauses.append("country ILIKE :country")
        params["country"] = f"%{country}%"

    prefix = detect_title_prefix(question)
    if prefix:
        clauses.append("LOWER(title) LIKE :title_prefix")
        params["title_prefix"] = f"{prefix.lower()}%"

    field_contains = detect_field_contains(question)
    if field_contains:
        field_name, keyword = field_contains
        clauses.append(f"{normalized_column_sql(field_name)} LIKE :field_contains")
        params["field_contains"] = f"%{keyword.lower()}%"

    title_keyword = detect_title_keyword(question)
    if title_keyword and not prefix and not field_contains:
        clauses.append("LOWER(title) LIKE :title_keyword")
        params["title_keyword"] = f"%{title_keyword.lower()}%"

    sql = BASE_SELECT
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    else:
        sql += " WHERE 1 = 0"
    sql += ORDER_BY
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = limit
    return sql, params


def validate_sql(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped.lower().startswith("select "):
        return False
    if ";" in stripped:
        return False
    if FORBIDDEN_SQL.search(stripped):
        return False
    return " from netflix_titles" in stripped.lower()


def execute_sql(sql: str, params: dict[str, Any] | None = None) -> SQLQueryResult:
    start = time.perf_counter()
    if not validate_sql(sql):
        return SQLQueryResult(sql=sql, params=params or {}, error="Unsafe SQL query rejected.")

    try:
        engine = create_engine(DATABASE_URL, future=True)
        with engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            rows = [dict(row._mapping) for row in result]
        latency_ms = (time.perf_counter() - start) * 1000
        return SQLQueryResult(sql=sql, params=params or {}, rows=rows, latency_ms=latency_ms)
    except SQLAlchemyError as error:
        latency_ms = (time.perf_counter() - start) * 1000
        return SQLQueryResult(
            sql=sql,
            params=params or {},
            latency_ms=latency_ms,
            error=str(error),
        )


def query_catalog(question: str, limit: int | None = None) -> SQLQueryResult:
    sql, params = question_to_sql(question, limit=limit)
    return execute_sql(sql, params)
