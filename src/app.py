from pathlib import Path

try:
    from .chain import ask_netflix
except ImportError:
    from chain import ask_netflix


PERSIST_DIRECTORY = Path("chroma_db")


def chroma_db_is_missing() -> bool:
    return not PERSIST_DIRECTORY.exists() or not any(PERSIST_DIRECTORY.iterdir())


def print_startup_help() -> None:
    print("Local Netflix RAG chatbot")
    print("Ask a question in English about the Netflix catalog.")
    print("Type /quit to exit.\n")


def explain_error(error: Exception) -> str:
    message = str(error).lower()

    if chroma_db_is_missing():
        return "ChromaDB is missing. Run first: uv run python src/ingest.py"

    if (
        "connection refused" in message
        or "failed to connect" in message
        or "connecterror" in message
        or "winerror 10061" in message
    ):
        return "Ollama does not seem to be running. Start Ollama and try again."

    if "model" in message and (
        "not found" in message or "pull" in message or "does not exist" in message
    ):
        return (
            "An Ollama model is missing. Run: "
            "ollama pull llama3.2:1b and ollama pull nomic-embed-text"
        )

    return f"Error: {error}"


def main() -> None:
    print_startup_help()

    while True:
        try:
            question = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if question.lower() == "/quit":
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            answer = ask_netflix(question)
            print(f"Assistant: {answer}\n")
        except Exception as error:
            print(explain_error(error))
            print()


if __name__ == "__main__":
    main()
