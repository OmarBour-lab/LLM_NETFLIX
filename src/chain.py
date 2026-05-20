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
- If the retrieved rows do not contain the answer, say that the retrieved rows do not contain enough information.
{required_instruction}"""


def is_list_question(question: str) -> bool:
    normalized_question = question.lower()
    if "when was" in normalized_question or "when is" in normalized_question:
        return False

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
            "movie",
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
    if not recent_history:
        return question

    return (
        "Recent conversation:\n"
        + format_history(recent_history)
        + "\n\nCurrent question:\n"
        + question
    )


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


def ask_netflix(question: str, history: list[dict[str, str]] | None = None) -> str:
    recent_history = trim_history(history)
    retrieval_k = LIST_RETRIEVAL_K if is_list_question(question) else DEFAULT_RETRIEVAL_K
    documents = retrieve_documents(retrieval_query(question, recent_history), k=retrieval_k)
    list_mode = is_list_question(question)
    context = format_title_list(documents) if list_mode else format_documents(documents)

    if not context:
        return FALLBACK_MESSAGE

    if list_mode:
        return answer_title_list(documents)

    user_message = (
        f"Recent conversation, for resolving follow-up references only:\n"
        f"{format_history(recent_history)}\n\n"
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
