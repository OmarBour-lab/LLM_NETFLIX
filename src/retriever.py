import re
import unicodedata
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


load_dotenv()

PERSIST_DIRECTORY = "./chroma_db/"
COLLECTION_NAME = "netflix_catalog"
EMBEDDING_MODEL = "nomic-embed-text"

COUNTRY_ALIASES = {
    "South Korea": [
        "coree",
        "coreen",
        "coreens",
        "coreenne",
        "coreennes",
        "korea",
        "korean",
        "koreans",
        "south korea",
    ],
    "India": ["inde", "indien", "indienne", "india", "indian"],
    "United States": ["etats-unis", "usa", "united states", "american"],
    "United Kingdom": ["royaume-uni", "united kingdom", "british"],
    "Japan": ["japon", "japonais", "japanese", "japan"],
}

QUERY_FILLER_WORDS = {
    "a",
    "about",
    "all",
    "any",
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
    "titles",
    "with",
}

FIELD_ALIASES = {
    "director": ["director", "directed by", "realisateur", "réalisateur"],
    "cast": ["actor", "actors", "cast", "acteur", "acteurs"],
    "country": ["country", "countries", "pays"],
    "genres": ["genre", "genres", "listed in", "categorie", "catégorie"],
    "title": ["title", "titre"],
    "description": ["description", "summary", "resume", "résumé"],
    "rating": ["rating", "classification"],
    "duration": ["duration", "duree", "durée"],
}


@dataclass
class QueryAnalysis:
    query: str
    title_prefix: str | None
    title_keyword: str | None
    requested_type: str | None
    requested_country: str | None
    requested_year: int | None
    field_contains: tuple[str, str] | None

    @property
    def has_structured_filters(self) -> bool:
        return any(
            [
                self.requested_type,
                self.requested_country,
                self.requested_year,
                self.field_contains,
            ]
        )


def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def get_retriever(k: int = 10):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def title_matches_query(title: str, query: str) -> bool:
    normalized_title = normalize_text(title)
    normalized_query = normalize_text(query)
    return bool(
        normalized_title
        and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_title)}(?![a-z0-9])",
            normalized_query,
        )
    )


def documents_from_rows(rows: dict) -> list[Document]:
    return [
        Document(page_content=page_content, metadata=metadata)
        for page_content, metadata in zip(rows["documents"], rows["metadatas"])
    ]


def normalize_key(key: str) -> str:
    normalized = unicodedata.normalize("NFKD", key.strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def parse_document_fields(page_content: str) -> dict[str, str]:
    fields = {}
    for line in page_content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[normalize_key(key)] = value.strip()
    return fields


def get_field_value(document: Document, field: str) -> str:
    fields = parse_document_fields(document.page_content)
    metadata = document.metadata
    values = {
        "title": metadata.get("title") or fields.get("titre", ""),
        "type": metadata.get("type") or fields.get("type", ""),
        "director": fields.get("realisateur", ""),
        "cast": fields.get("acteurs", ""),
        "country": metadata.get("country") or fields.get("pays", ""),
        "year": str(metadata.get("release_year") or fields.get("annee", "")),
        "rating": metadata.get("rating", ""),
        "duration": fields.get("duree", ""),
        "genres": metadata.get("listed_in") or fields.get("genres", ""),
        "description": fields.get("description", ""),
    }
    return str(values.get(field, ""))


def get_all_records(vectorstore: Chroma) -> list[Document]:
    rows = vectorstore.get(include=["documents", "metadatas"])
    return documents_from_rows(rows)


def find_exact_title_documents(vectorstore: Chroma, query: str) -> list[Document]:
    records = get_all_records(vectorstore)
    matches = [
        document
        for document in records
        if len(normalize_text(document.metadata.get("title", ""))) >= 4
        and title_matches_query(document.metadata.get("title", ""), query)
    ]
    matches.sort(key=lambda document: len(normalize_text(document.metadata.get("title", ""))), reverse=True)

    filtered_matches: list[Document] = []
    longer_titles: list[str] = []
    for document in matches:
        title = normalize_text(document.metadata.get("title", ""))
        if any(title != longer_title and title in longer_title for longer_title in longer_titles):
            continue
        longer_titles.append(title)
        filtered_matches.append(document)

    return filtered_matches


def detect_title_prefix(query: str) -> str | None:
    normalized_query = normalize_text(query)
    patterns = [
        r"(?:start|starts|begin|begins) with ([a-z0-9 ]+)",
        r"titles? that (?:start|starts|begin|begins) with ([a-z0-9 ]+)",
    ]

    quoted_match = re.search(r'"([^"]+)"', query)
    if quoted_match and re.search(r"\b(start|starts|begin|begins)\b", normalized_query):
        return normalize_text(quoted_match.group(1))

    for pattern in patterns:
        match = re.search(pattern, normalized_query)
        if match:
            phrase = match.group(1).strip()
            return re.sub(r"\b(movie|movies|series|shows|available|catalogue|catalog)\b", "", phrase).strip()

    return None


def find_title_prefix_documents(vectorstore: Chroma, query: str, k: int) -> list[Document]:
    prefix = detect_title_prefix(query)
    if not prefix:
        return []

    matches = [
        document
        for document in get_all_records(vectorstore)
        if normalize_text(document.metadata.get("title", "")).startswith(prefix)
    ]
    matches.sort(key=lambda document: document.metadata.get("title", ""))
    return matches[:k]


def detect_title_keyword(query: str) -> str | None:
    normalized_query = normalize_text(query)

    quoted_match = re.search(r'"([^"]+)"', query)
    if quoted_match:
        return normalize_text(quoted_match.group(1))

    patterns = [
        r"(?:all|find|give me all)(?: the)? ([a-z0-9 ]+?) (?:movie|movies|series|show|shows)(?: available| in the catalog| in the catalogue)?",
        r"(?:list|find|give me|show me)(?: all)? (?:titles|movie|movies|series|show|shows) (?:with|containing|about|called|named) ([a-z0-9 ]+)",
        r"(?:all|which|what) (?:titles|movie|movies|series|show|shows) (?:with|containing|about|called|named) ([a-z0-9 ]+)",
        r"(?:movie|movies|series|show|shows) (?:called|named|about) ([a-z0-9 ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_query)
        if match:
            keyword = clean_keyword(match.group(1))
            if keyword:
                return keyword

    words = [
        word
        for word in normalized_query.split()
        if len(word) >= 4 and word not in QUERY_FILLER_WORDS
    ]
    if words:
        return " ".join(words)

    return None


def clean_keyword(keyword: str) -> str:
    words = [
        word
        for word in normalize_text(keyword).split()
        if word not in QUERY_FILLER_WORDS
    ]
    return " ".join(words).strip()


def find_title_keyword_documents(vectorstore: Chroma, query: str, k: int) -> list[Document]:
    keyword = detect_title_keyword(query)
    if not keyword:
        return []

    requested_type = detect_requested_type(query)
    matches = [
        document
        for document in get_all_records(vectorstore)
        if keyword_matches_title(keyword, document.metadata.get("title", ""))
        and (not requested_type or document.metadata.get("type") == requested_type)
    ]
    matches.sort(
        key=lambda document: (
            document.metadata.get("title", ""),
            int(document.metadata.get("release_year", 0)),
        )
    )
    return matches[:k]


def keyword_matches_title(keyword: str, title: str) -> bool:
    normalized_title = normalize_text(title)
    if len(keyword) <= 4:
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                normalized_title,
            )
        )
    return keyword in normalized_title


def detect_requested_type(query: str) -> str | None:
    normalized_query = normalize_text(query)
    wants_show = re.search(r"(?<![a-z0-9])(serie|series|show|shows|tv show)(?![a-z0-9])", normalized_query)
    wants_movie = re.search(r"(?<![a-z0-9])(film|films|movie|movies)(?![a-z0-9])", normalized_query)

    if wants_show and wants_movie:
        return None
    if wants_show:
        return "TV Show"
    if wants_movie:
        return "Movie"
    return None


def detect_requested_country(query: str) -> str | None:
    normalized_query = normalize_text(query)
    for country, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(normalize_text(alias))}(?![a-z0-9])",
                normalized_query,
            ):
                return country
    return None


def detect_requested_year(query: str) -> int | None:
    normalized_query = normalize_text(query)
    match = re.search(
        r"\b(?:released|sorti|sortie|sortis|sorties|year|annee|année)?\s*(19[0-9]{2}|20[0-9]{2})\b",
        normalized_query,
    )
    if match:
        return int(match.group(1))
    return None


def detect_field_contains(query: str) -> tuple[str, str] | None:
    normalized_query = normalize_text(query)

    detected_field = None
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            if re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
                normalized_query,
            ):
                detected_field = field
                break
        if detected_field:
            break

    if not detected_field:
        return None

    quoted_match = re.search(r'"([^"]+)"|' r"'([^']+)'", query)
    if quoted_match:
        return detected_field, normalize_text(quoted_match.group(1) or quoted_match.group(2))

    field_alias_pattern = "|".join(
        re.escape(normalize_text(alias))
        for aliases in FIELD_ALIASES.values()
        for alias in aliases
    )
    field_contains_match = re.search(
        rf"(?:{field_alias_pattern})\s+(?:contains|containing|contient|contenant|with|avec)\s+([a-z0-9 ]+)",
        normalized_query,
    )
    if field_contains_match:
        value = clean_keyword(field_contains_match.group(1))
        if value:
            return detected_field, value

    contains_match = re.search(
        r"(?:contains|containing|contient|contenant|with|avec|by|par)\s+([a-z0-9 ]+)",
        normalized_query,
    )
    if contains_match:
        value = clean_keyword(contains_match.group(1))
        if value:
            return detected_field, value

    return None


def find_structured_filter_documents(vectorstore: Chroma, query: str, k: int) -> list[Document]:
    requested_type = detect_requested_type(query)
    requested_country = detect_requested_country(query)
    requested_year = detect_requested_year(query)
    field_contains = detect_field_contains(query)

    if not any([requested_type, requested_country, requested_year, field_contains]):
        return []

    matches: list[Document] = []
    for document in get_all_records(vectorstore):
        if requested_type and get_field_value(document, "type") != requested_type:
            continue
        if requested_country and requested_country not in get_field_value(document, "country"):
            continue
        if requested_year and get_field_value(document, "year") != str(requested_year):
            continue
        if field_contains:
            field, expected_value = field_contains
            actual_value = normalize_text(get_field_value(document, field))
            if expected_value not in actual_value:
                continue
        matches.append(document)

    matches.sort(
        key=lambda document: (
            get_field_value(document, "title"),
            get_field_value(document, "year"),
        )
    )
    return matches[:k]


def find_metadata_filter_documents(vectorstore: Chroma, query: str, k: int) -> list[Document]:
    requested_type = detect_requested_type(query)
    requested_country = detect_requested_country(query)

    if not requested_type and not requested_country:
        return []

    matches: list[Document] = []
    for document in get_all_records(vectorstore):
        metadata = document.metadata
        if requested_type and metadata.get("type") != requested_type:
            continue
        if requested_country and requested_country not in metadata.get("country", ""):
            continue
        matches.append(document)

    matches.sort(
        key=lambda document: (
            int(document.metadata.get("release_year", 0)),
            document.metadata.get("title", ""),
        ),
        reverse=True,
    )
    return matches[:k]


def analyze_query(query: str) -> QueryAnalysis:
    return QueryAnalysis(
        query=query,
        title_prefix=detect_title_prefix(query),
        title_keyword=detect_title_keyword(query),
        requested_type=detect_requested_type(query),
        requested_country=detect_requested_country(query),
        requested_year=detect_requested_year(query),
        field_contains=detect_field_contains(query),
    )


def exact_title_matches(records: list[Document], query: str) -> list[Document]:
    matches = [
        document
        for document in records
        if len(normalize_text(document.metadata.get("title", ""))) >= 4
        and title_matches_query(document.metadata.get("title", ""), query)
    ]
    matches.sort(
        key=lambda document: len(normalize_text(document.metadata.get("title", ""))),
        reverse=True,
    )

    filtered_matches: list[Document] = []
    longer_titles: list[str] = []
    for document in matches:
        title = normalize_text(document.metadata.get("title", ""))
        if any(title != longer_title and title in longer_title for longer_title in longer_titles):
            continue
        longer_titles.append(title)
        filtered_matches.append(document)

    return filtered_matches


def record_matches_structured_filters(document: Document, analysis: QueryAnalysis) -> bool:
    if analysis.requested_type and get_field_value(document, "type") != analysis.requested_type:
        return False
    if analysis.requested_country and analysis.requested_country not in get_field_value(document, "country"):
        return False
    if analysis.requested_year and get_field_value(document, "year") != str(analysis.requested_year):
        return False
    if analysis.field_contains:
        field, expected_value = analysis.field_contains
        actual_value = normalize_text(get_field_value(document, field))
        if expected_value not in actual_value:
            return False
    return True


def sort_by_title_and_year(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: (
            get_field_value(document, "title"),
            get_field_value(document, "year"),
        ),
    )


def search_catalog(vectorstore: Chroma, analysis: QueryAnalysis, k: int) -> list[Document]:
    records = get_all_records(vectorstore)

    if analysis.title_prefix:
        matches = [
            document
            for document in records
            if normalize_text(document.metadata.get("title", "")).startswith(analysis.title_prefix)
        ]
        documents = unique_documents(sort_by_title_and_year(matches), k)
        if documents:
            return documents

    if analysis.title_keyword:
        matches = [
            document
            for document in records
            if keyword_matches_title(analysis.title_keyword, document.metadata.get("title", ""))
            and (
                not analysis.requested_type
                or document.metadata.get("type") == analysis.requested_type
            )
        ]
        documents = unique_documents(sort_by_title_and_year(matches), k)
        if documents:
            return documents

    if analysis.has_structured_filters:
        matches = [
            document
            for document in records
            if record_matches_structured_filters(document, analysis)
        ]
        documents = unique_documents(sort_by_title_and_year(matches), k)
        if documents:
            return documents

    exact_matches = exact_title_matches(records, analysis.query)
    if exact_matches:
        return unique_documents(exact_matches, k)

    vector_documents = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(analysis.query)
    return unique_documents(vector_documents, k)


def unique_documents(documents: list[Document], k: int) -> list[Document]:
    unique: list[Document] = []
    seen_ids: set[str] = set()

    for document in documents:
        show_id = document.metadata.get("show_id", document.page_content)
        if show_id in seen_ids:
            continue
        seen_ids.add(show_id)
        unique.append(document)
        if len(unique) == k:
            break

    return unique


def retrieve_documents(query: str, k: int = 10) -> list[Document]:
    vectorstore = get_vectorstore()
    analysis = analyze_query(query)
    return search_catalog(vectorstore, analysis, k)
