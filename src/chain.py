import unicodedata
import json
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

try:
    from .config import LLM_MODEL, OLLAMA_BASE_URL
    from .nl2sql import query_catalog
    from .retriever import has_exact_title_match, retrieve_documents
    from .router import classify_intent
except ImportError:
    from config import LLM_MODEL, OLLAMA_BASE_URL
    from nl2sql import query_catalog
    from retriever import has_exact_title_match, retrieve_documents
    from router import classify_intent


load_dotenv()

FALLBACK_MESSAGE = "I cannot confirm that with the retrieved Netflix context."
DEFAULT_RETRIEVAL_K = 10
LIST_RETRIEVAL_K = 10000
MEMORY_MESSAGES = 6
REQUIRED_SYSTEM_INSTRUCTION = (
    "Always answer in English, using natural language and complete sentences. "
    "Never return raw CSV, JSON, Markdown tables, or a raw bullet list. "
    "Mention every relevant title found in the context without omitting any."
)

DETAIL_PROMPT = """You rewrite one database row into a direct English answer.
The row is valid and relevant.
Use only the row.
Use the row's description when asked for description, summary, or what you know.
Use the row's director when asked who directed it.
Use the row's cast when asked who stars in it or who acts in it.
Use only the row's year when asked when it was released.
Do not say that information is missing if the row has a description.
Do not use outside knowledge.
Do not add extra notes about other rows or missing rows.
{required_instruction}"""

LIST_PROMPT = """You answer list questions using only the source title list.
Answer in natural language with complete sentences.
Mention every source title exactly once.
Do not use CSV, JSON, Markdown tables, or a raw bullet list.
Do not add titles that are not in the source title list.
{required_instruction}"""

GENERAL_PROMPT = """You are a strict database question-answering assistant.
Use only the retrieved rows.

Rules:
- Answer in English.
- The retrieved rows are valid database rows.
- The rows are JSON objects with keys: title, type, director, cast, country, year, duration, genres, description.
- If a value appears in the "title" key, that title exists in the database.
- If the "description" value is not "Inconnu", the description exists. Never say there is no description in that case.
- If the question asks for a description, summary, or "what do you know", use the matching row's "description" value.
- If the question asks for titles, list every "title" value from every retrieved row.
- If the question asks for titles that start with a phrase, list only TITLE values that start with that phrase.
- You may include TYPE, DIRECTOR, COUNTRY, YEAR, DURATION, and GENRES when useful.
- If DIRECTOR or CAST is "Inconnu", say "unknown".
- Do not use outside knowledge.
- Do not mention websites or external services.
- Do not add extra notes about unrelated or missing rows.
- Do not mention record counts, row counts, or prompt metadata.
- If the retrieved rows do not contain the answer, say that the retrieved rows do not contain enough information.
{required_instruction}"""


def is_list_question(question: str) -> bool:
    normalized_question = question.lower()
    if any(
        detail_marker in normalized_question
        for detail_marker in [
            "description of",
            "description for",
            "who directed",
            "what do you know",
            "when was",
            "when is",
        ]
    ):
        return False

    if any(
        marker in normalized_question
        for marker in [
            "all ",
            "available",
            "ceux",
            "celles",
            "contient",
            "contenant",
            "films",
            "movies",
            "which ",
            "what titles",
            "list",
            "liste",
            "sorti",
            "sortie",
            "titles that start",
            "qui ont",
            "réalisateur",
            "realisateur",
        ]
    ):
        return True

    return bool(
        re.search(
            r"\b(?:movies|films|series|shows|tv shows?)\b\s+"
            r"(?:from|released|with|containing|by|in|that|where|whose)\b",
            normalized_question,
        )
    )


def system_prompt_for(question: str, documents) -> str:
    if is_list_question(question):
        return LIST_PROMPT.format(required_instruction=REQUIRED_SYSTEM_INSTRUCTION)
    if len(documents) == 1:
        return DETAIL_PROMPT.format(required_instruction=REQUIRED_SYSTEM_INSTRUCTION)
    return GENERAL_PROMPT.format(required_instruction=REQUIRED_SYSTEM_INSTRUCTION)


def trim_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    return history[-MEMORY_MESSAGES:]


def format_history(history: list[dict[str, str]] | None) -> str:
    recent_history = trim_history(history)
    if not recent_history:
        return "No previous messages."

    lines = []
    for message in recent_history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def retrieval_query(question: str, history: list[dict[str, str]] | None) -> str:
    recent_history = trim_history(history)
    if not recent_history or is_list_question(question) or has_exact_title_match(question):
        return question

    return (
        "Recent conversation:\n"
        + format_history(recent_history)
        + "\n\nCurrent question:\n"
        + question
    )


def infer_last_discussed_title(history: list[dict[str, str]] | None) -> str | None:
    if not history:
        return None

    patterns = [
        r"^(.+?) is a (?:Movie|TV Show) released",
        r"^(.+?) is in the retrieved Netflix context",
        r"^(.+?) was released in",
        r"^The duration of (.+?) is",
        r"^The cast of (.+?) includes",
        r"^(.+?) directed (.+?)\.",
        r"^The director of (.+?) is",
    ]

    for message in reversed(history):
        content = message.get("content", "").strip()
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            if pattern.startswith("^(.+?) directed"):
                return match.group(2).strip()
            return match.group(1).strip()
    return None


def resolve_follow_up_question(question: str, history: list[dict[str, str]] | None) -> str:
    if not re.search(r"\b(it|this title|that title)\b", question.lower()):
        return question

    title = infer_last_discussed_title(history)
    if not title:
        return question

    resolved = re.sub(r"\bthis title\b", title, question, flags=re.IGNORECASE)
    resolved = re.sub(r"\bthat title\b", title, resolved, flags=re.IGNORECASE)
    resolved = re.sub(r"\bit\b", title, resolved, flags=re.IGNORECASE)
    return resolved


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


def format_documents(documents) -> str:
    if not documents:
        return ""

    records = []
    for index, document in enumerate(documents, start=1):
        fields = parse_document_fields(document.page_content)
        record = {
            "record": index,
            "title": fields.get("titre", document.metadata.get("title", "Inconnu")),
            "description": fields.get("description", "Inconnu"),
            "type": fields.get("type", document.metadata.get("type", "Inconnu")),
            "director": fields.get("realisateur", "Inconnu"),
            "country": fields.get("pays", document.metadata.get("country", "Inconnu")),
            "year": fields.get("annee", document.metadata.get("release_year", "Inconnu")),
            "duration": fields.get("duree", "Inconnu"),
            "genres": fields.get("genres", document.metadata.get("listed_in", "Inconnu")),
            "cast": fields.get("acteurs", "Inconnu"),
        }
        records.append(
            {key: value for key, value in record.items() if value != "Inconnu"}
        )

    return json.dumps(records, ensure_ascii=False, indent=2)


def format_title_list(documents) -> str:
    titles = []
    for document in documents:
        title = document.metadata.get("title")
        if title:
            titles.append(title)
    return "\n".join(f"{index}. {title}" for index, title in enumerate(titles, start=1))


def answer_title_list(documents) -> str:
    titles = []
    seen_titles = set()
    for document in documents:
        title = document.metadata.get("title")
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)

    if not titles:
        return FALLBACK_MESSAGE

    if len(titles) == 1:
        return f"The relevant title found in the retrieved context is {titles[0]}."

    return (
        "The relevant titles found in the retrieved context are "
        + ", ".join(titles[:-1])
        + f" and {titles[-1]}."
    )


def answer_sql_rows(question: str, rows: list[dict]) -> str | None:
    if not rows:
        return FALLBACK_MESSAGE

    normalized_question = question.lower()
    if len(rows) == 1:
        row = rows[0]
        title = row.get("title", "this title")
        if "who directed" in normalized_question or "director" in normalized_question:
            director = row.get("director") or "unknown"
            return f"{director} directed {title}."
        if any(marker in normalized_question for marker in ["description", "summary", "what do you know", "about"]):
            description = row.get("description") or "no description is available"
            year = row.get("release_year")
            title_type = row.get("type")
            details = []
            if title_type:
                details.append(f"a {title_type}")
            if year:
                details.append(f"released in {year}")
            intro = f"{title} is " + " ".join(details) + "." if details else f"{title} is in the catalog."
            return f"{intro} Its description is: {description}"
        if "cast" in normalized_question or "actor" in normalized_question:
            cast = row.get("cast") or "unknown"
            return f"The cast of {title} includes {cast}."
        if "released" in normalized_question or "year" in normalized_question:
            year = row.get("release_year") or "unknown"
            return f"{title} was released in {year}."

    titles = []
    seen = set()
    for row in rows:
        title = row.get("title")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)

    if not titles:
        return FALLBACK_MESSAGE
    if len(titles) == 1:
        return f"The relevant title found in the PostgreSQL catalog is {titles[0]}."
    return (
        "The relevant titles found in the PostgreSQL catalog are "
        + ", ".join(titles[:-1])
        + f" and {titles[-1]}."
    )


def try_sql_answer(question: str) -> str | None:
    if re.search(r"\b(does not exist|do not exist|not exist|nonexistent)\b", question.lower()):
        return FALLBACK_MESSAGE

    route = classify_intent(question)
    if route.intent not in {"sql", "mixed"}:
        return None

    result = query_catalog(question)
    if result.error:
        return None
    if not result.rows:
        return None
    return answer_sql_rows(question, result.rows)


def is_unknown(value: str | None) -> bool:
    return not value or str(value).strip() == "Inconnu"


def format_comma_values(value: str) -> str:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        return "unknown"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def answer_single_record(question: str, document) -> str | None:
    fields = parse_document_fields(document.page_content)
    title = fields.get("titre", document.metadata.get("title", "this title"))
    title = str(title)
    normalized_question = question.lower()

    record = {
        "type": fields.get("type", document.metadata.get("type", "Inconnu")),
        "director": fields.get("realisateur", "Inconnu"),
        "cast": fields.get("acteurs", "Inconnu"),
        "country": fields.get("pays", document.metadata.get("country", "Inconnu")),
        "year": fields.get("annee", document.metadata.get("release_year", "Inconnu")),
        "duration": fields.get("duree", "Inconnu"),
        "genres": fields.get("genres", document.metadata.get("listed_in", "Inconnu")),
        "description": fields.get("description", "Inconnu"),
    }

    if "who directed" in normalized_question or re.search(r"\bdirector\b", normalized_question):
        if is_unknown(record["director"]):
            return f"The director of {title} is unknown in the retrieved Netflix context."
        return f"{record['director']} directed {title}."

    if any(marker in normalized_question for marker in ["who stars", "who acts", "cast", "actor", "actors"]):
        if is_unknown(record["cast"]):
            return f"The cast of {title} is unknown in the retrieved Netflix context."
        return f"The cast of {title} includes {format_comma_values(record['cast'])}."

    if any(marker in normalized_question for marker in ["when was", "when is", "release year", "released"]):
        if is_unknown(str(record["year"])):
            return f"The release year of {title} is unknown in the retrieved Netflix context."
        return f"{title} was released in {record['year']}."

    if "duration" in normalized_question or "how long" in normalized_question:
        if is_unknown(record["duration"]):
            return f"The duration of {title} is unknown in the retrieved Netflix context."
        return f"The duration of {title} is {record['duration']}."

    if any(marker in normalized_question for marker in ["genre", "category", "listed in"]):
        if is_unknown(record["genres"]):
            return f"The genres for {title} are unknown in the retrieved Netflix context."
        return f"{title} is listed in {format_comma_values(record['genres'])}."

    if any(
        marker in normalized_question
        for marker in ["description", "summary", "what do you know", "about"]
    ):
        details = []
        if not is_unknown(record["type"]):
            details.append(f"a {record['type']}")
        if not is_unknown(str(record["year"])):
            details.append(f"released in {record['year']}")
        if not is_unknown(record["duration"]):
            details.append(f"with a duration of {record['duration']}")

        intro = f"{title} is " + " ".join(details) + "." if details else f"{title} is in the retrieved Netflix context."
        if is_unknown(record["description"]):
            return f"{intro} The retrieved Netflix context does not include a description."
        return f"{intro} Its description is: {record['description']}"

    return None


def ask_netflix(question: str, history: list[dict[str, str]] | None = None) -> str:
    recent_history = trim_history(history)
    effective_question = resolve_follow_up_question(question, recent_history)
    sql_answer = try_sql_answer(effective_question)
    if sql_answer:
        return sql_answer

    retrieval_k = LIST_RETRIEVAL_K if is_list_question(effective_question) else DEFAULT_RETRIEVAL_K
    documents = retrieve_documents(retrieval_query(effective_question, recent_history), k=retrieval_k)
    list_mode = is_list_question(effective_question)
    context = format_title_list(documents) if list_mode else format_documents(documents)

    if not context:
        return FALLBACK_MESSAGE

    if list_mode:
        return answer_title_list(documents)

    if len(documents) == 1:
        single_record_answer = answer_single_record(effective_question, documents[0])
        if single_record_answer:
            return single_record_answer

    user_message = (
        f"Recent conversation, for resolving follow-up references only:\n"
        f"{format_history(recent_history)}\n\n"
        f"Question:\n{effective_question}\n\n"
        f"Retrieved database rows:\n{context}\n\n"
        "If there is one row, it is the matching row. "
        "Answer using only the retrieved database rows."
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=1024 if list_mode else 512,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt_for(effective_question, documents)),
            HumanMessage(content=user_message),
        ]
    )

    return response.content.strip()
