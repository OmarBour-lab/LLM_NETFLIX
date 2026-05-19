from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


load_dotenv()

CSV_PATH = Path("data/netflix_titles.csv")
PERSIST_DIRECTORY = "./chroma_db/"
COLLECTION_NAME = "netflix_catalog"
EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 500

DOCUMENT_TEMPLATE = """Titre: {title}
Type: {type}
Réalisateur: {director}
Acteurs: {cast}
Pays: {country}
Année: {release_year}
Durée: {duration}
Genres: {listed_in}
Description: {description}"""


def load_dataset() -> pd.DataFrame:
    """Charge le CSV Netflix sans jamais le modifier."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {CSV_PATH}. Placez netflix_titles.csv dans data/."
        )

    df = pd.read_csv(CSV_PATH)

    cols_with_nan = ["director", "cast", "country", "date_added"]
    df[cols_with_nan] = df[cols_with_nan].fillna("Inconnu")

    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    documents: list[Document] = []

    for _, row in df.iterrows():
        page_content = DOCUMENT_TEMPLATE.format(
            title=row["title"],
            type=row["type"],
            director=row["director"],
            cast=row["cast"],
            country=row["country"],
            release_year=row["release_year"],
            duration=row["duration"],
            listed_in=row["listed_in"],
            description=row["description"],
        )

        metadata = {
            "show_id": row["show_id"],
            "title": row["title"],
            "type": row["type"],
            "release_year": int(row["release_year"]),
            "country": row["country"],
            "listed_in": row["listed_in"],
            "rating": row["rating"],
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def collection_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


def ingest() -> None:
    vectorstore = get_vectorstore()
    existing_count = collection_count(vectorstore)

    if existing_count > 0:
        print(
            f"La collection '{COLLECTION_NAME}' contient déjà {existing_count} documents."
        )
        print("Ingestion ignorée pour éviter les doublons.")
        return

    df = load_dataset()
    documents = build_documents(df)

    print(f"Ingestion de {len(documents)} titres Netflix dans ChromaDB...")
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        ids = [document.metadata["show_id"] for document in batch]
        vectorstore.add_documents(batch, ids=ids)
        print(f"Lot ingéré: {start + len(batch)}/{len(documents)}")

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    print("Ingestion terminée.")


if __name__ == "__main__":
    ingest()
