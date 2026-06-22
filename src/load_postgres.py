from sqlalchemy import create_engine, text

try:
    from .config import DATABASE_URL, DATASET_PATH
    from .ingest import load_dataset
except ImportError:
    from config import DATABASE_URL, DATASET_PATH
    from ingest import load_dataset


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS netflix_titles (
    show_id TEXT PRIMARY KEY,
    type TEXT,
    title TEXT,
    director TEXT,
    "cast" TEXT,
    country TEXT,
    date_added TEXT,
    release_year INTEGER,
    rating TEXT,
    duration TEXT,
    listed_in TEXT,
    description TEXT
)
"""


def load_postgres() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = load_dataset()
    records = df.where(df.notna(), None).to_dict(orient="records")
    engine = create_engine(DATABASE_URL, future=True)

    insert_sql = text(
        """
        INSERT INTO netflix_titles (
            show_id, type, title, director, "cast", country, date_added,
            release_year, rating, duration, listed_in, description
        )
        VALUES (
            :show_id, :type, :title, :director, :cast, :country, :date_added,
            :release_year, :rating, :duration, :listed_in, :description
        )
        ON CONFLICT (show_id) DO UPDATE SET
            type = EXCLUDED.type,
            title = EXCLUDED.title,
            director = EXCLUDED.director,
            "cast" = EXCLUDED."cast",
            country = EXCLUDED.country,
            date_added = EXCLUDED.date_added,
            release_year = EXCLUDED.release_year,
            rating = EXCLUDED.rating,
            duration = EXCLUDED.duration,
            listed_in = EXCLUDED.listed_in,
            description = EXCLUDED.description
        """
    )

    with engine.begin() as connection:
        connection.execute(text(CREATE_TABLE_SQL))
        connection.execute(insert_sql, records)

    print(f"Loaded {len(records)} Netflix titles into PostgreSQL.")


if __name__ == "__main__":
    load_postgres()

