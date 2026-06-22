# LLM_NETFLIX - Local Netflix RAG + NL2SQL Platform

LLM_NETFLIX is a local question-answering platform for the Netflix catalog in
`data/netflix_titles.csv`. It combines semantic RAG with ChromaDB and structured
NL2SQL-style filtering through PostgreSQL. The CSV is read only and is never
modified.

The user-facing chatbot answers in English.

## Stack

- Python
- LangChain
- Ollama with `llama3.2:1b`
- Ollama embeddings with `nomic-embed-text`
- ChromaDB for persistent vector search
- PostgreSQL for structured catalog queries
- FastAPI for the backend API
- Streamlit for the web UI
- Plotly for visualizations
- Scikit-learn for routing/evaluation experiments
- pytest and httpx for tests
- Docker Compose for PostgreSQL, FastAPI, and Streamlit

## Project Structure

```text
data/netflix_titles.csv
src/ingest.py
src/load_postgres.py
src/retriever.py
src/nl2sql.py
src/router.py
src/chain.py
src/app.py
src/web_app.py
src/backend/main.py
src/backend/auth.py
src/backend/schemas.py
evaluation/questions.json
evaluation/evaluate.py
docs/uml/
tests/
frontend/README.md
Dockerfile
docker-compose.yml
```

## Environment

Create or update `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/netflix_db
SECRET_KEY=change-this-local-development-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Install

```powershell
uv pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

## Ingest Data

Create the ChromaDB vector store:

```powershell
uv run python src/ingest.py
```

Load the CSV into PostgreSQL:

```powershell
uv run python src/load_postgres.py
```

If PostgreSQL is not running, the chatbot still falls back to the existing
ChromaDB/Python retrieval path.

## Run

CLI:

```powershell
uv run python src/app.py
```

Streamlit:

```powershell
uv run streamlit run src/web_app.py
```

FastAPI:

```powershell
uv run uvicorn src.backend.main:app --reload
```

Docker Compose:

```powershell
docker-compose up --build
```

## API

- `GET /health`
- `POST /token`
- `POST /chat`
- `GET /history?session_id=...`
- `POST /feedback`
- `GET /stats`
- `WS /ws/{session_id}`

## Test

```powershell
uv run pytest tests/ -v
```

Current validation status:

- 20 automated tests pass.
- A 50-question manual validation suite passed with 50/50 successful answers.

## Evaluate

```powershell
uv run python evaluation/evaluate.py
```

The evaluation script writes:

- `exports/evaluation_results.csv`
- `exports/netflix_stats.csv`

These files can be opened in Power BI for external analysis.

## Frontend Scope

The implemented web UI is Streamlit. The React.js + TypeScript frontend is
documented as a production perspective in `frontend/README.md`; it is not a full
implemented frontend in this version.

## Example Questions

```text
Give me the description of Bird Box
Who directed Bird Box?
What do you know about Fate/Zero?
give me all the nova movies available
all titles that start with the legend of
movies with director containing John
movies with cast containing Sandra
movies released in 2020
series from Japan released in 2020
list movies containing love
all titles with christmas
```
