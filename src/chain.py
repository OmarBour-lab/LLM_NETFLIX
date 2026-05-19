import unicodedata
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

try:
    from .retriever import retrieve_documents
except ImportError:
    from retriever import retrieve_documents


load_dotenv()

LLM_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
FALLBACK_MESSAGE = "Je ne peux pas confirmer ça avec le contexte Netflix récupéré."
DEFAULT_RETRIEVAL_K = 4
LIST_RETRIEVAL_K = 100

DETAIL_PROMPT = """You rewrite one database row into a direct English answer.
The row is valid and relevant.
Use only the row.
Use the row's description when asked for description, summary, or what you know.
Do not say that information is missing if the row has a description.
Do not use outside knowledge."""

LIST_PROMPT = """You list titles from database rows.
Use only the source title list.
Copy every source title exactly once.
Return one bullet per source title.
Do not add titles that are not in the source title list.
Do not add explanations."""

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
- If the retrieved rows do not contain the answer, say that the retrieved rows do not contain enough information.
"""


def is_list_question(question: str) -> bool:
    normalized_question = question.lower()
    return any(
        marker in normalized_question
        for marker in [
            "all ",
            "available",
            "ceux",
            "celles",
            "contient",
            "contenant",
            "films",
            "film",
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
    )


def system_prompt_for(question: str, documents) -> str:
    if is_list_question(question):
        return LIST_PROMPT
    if len(documents) == 1:
        return DETAIL_PROMPT
    return GENERAL_PROMPT


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

    return "\n".join(f"* {title}" for title in titles)


def ask_netflix(question: str) -> str:
    retrieval_k = LIST_RETRIEVAL_K if is_list_question(question) else DEFAULT_RETRIEVAL_K
    documents = retrieve_documents(question, k=retrieval_k)
    list_mode = is_list_question(question)
    context = format_title_list(documents) if list_mode else format_documents(documents)

    if not context:
        return FALLBACK_MESSAGE

    if list_mode:
        return answer_title_list(documents)

    user_message = (
        f"Question:\n{question}\n\n"
        f"Number of retrieved rows: {len(documents)}\n\n"
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
            SystemMessage(content=system_prompt_for(question, documents)),
            HumanMessage(content=user_message),
        ]
    )

    return response.content.strip()
