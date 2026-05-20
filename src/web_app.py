from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    from .app import chroma_db_is_missing, explain_error
    from .chain import ask_netflix
except ImportError:
    from app import chroma_db_is_missing, explain_error
    from chain import ask_netflix


MEMORY_MESSAGES = 6
DATASET_PATH = Path("data/netflix_titles.csv")


APP_TITLE = "Netflix RAG Chatbot"
APP_SUBTITLE = (
    "Ask questions against the local Netflix catalog. "
    "The app can retrieve titles, filter fields, and summarize records with a local LLM."
)
EXAMPLE_OPTIONS = [
    "Give me the description of League of Legends Origins",
    "Who directed Bird Box?",
    "give me all the nova movies available",
    "all titles that start with the legend of",
    "movies with director containing John",
    "movies released in 2020",
]


def default_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hi. Ask me in English about the local Netflix catalog. "
                "Try title descriptions, title lists, director/cast filters, countries, years, or genres."
            ),
        }
    ]


def start_new_chat() -> None:
    st.session_state.messages = default_messages()
    st.session_state.pending_question = None
    st.session_state.page = "Chat"
    st.session_state.page_selector = "Chat"


def queue_example_question() -> None:
    st.session_state.pending_question = st.session_state.example_question
    st.session_state.page = "Chat"
    st.session_state.page_selector = "Chat"


def setup_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="N",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --netflix-red: #e50914;
                --bg: #0b0f19;
                --panel: #121826;
                --panel-soft: #192132;
                --text: #f8fafc;
                --muted: #9ca3af;
                --line: rgba(255,255,255,0.11);
            }

            .stApp {
                background:
                    radial-gradient(circle at 20% 12%, rgba(229, 9, 20, 0.18), transparent 30rem),
                    linear-gradient(135deg, #090d16 0%, #0f172a 48%, #16120f 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: rgba(9, 13, 22, 0.92);
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
            }

            .main .block-container {
                max-width: 1120px;
                padding-top: 2rem;
                padding-bottom: 6rem;
            }

            .hero {
                border: 1px solid var(--line);
                background: linear-gradient(135deg, rgba(18,24,38,0.94), rgba(25,33,50,0.88));
                border-radius: 18px;
                padding: 2rem;
                box-shadow: 0 24px 80px rgba(0,0,0,0.32);
            }

            .hero h1 {
                margin: 0;
                font-size: 2.5rem;
                letter-spacing: 0;
                color: var(--text);
            }

            .hero p {
                margin: 0.75rem 0 0;
                max-width: 720px;
                color: #cbd5e1;
                font-size: 1rem;
                line-height: 1.65;
            }

            .status-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 1rem 0 1.25rem;
            }

            .status-card {
                border: 1px solid var(--line);
                background: rgba(18,24,38,0.72);
                border-radius: 12px;
                padding: 1rem;
            }

            .status-card span {
                display: block;
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .status-card strong {
                display: block;
                margin-top: 0.3rem;
                color: var(--text);
                font-size: 1rem;
            }

            .stChatMessage {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: rgba(18,24,38,0.76);
            }

            [data-testid="stChatInput"] {
                border-top: 1px solid var(--line);
                background: rgba(9,13,22,0.72);
                backdrop-filter: blur(14px);
            }

            div.stButton > button {
                border: 1px solid rgba(229, 9, 20, 0.42);
                background: rgba(229, 9, 20, 0.12);
                color: #fff;
                border-radius: 10px;
                min-height: 2.65rem;
                font-weight: 600;
            }

            div.stButton > button:hover {
                border-color: rgba(229, 9, 20, 0.95);
                background: rgba(229, 9, 20, 0.22);
                color: #fff;
            }

            .small-note {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.55;
            }

            .section-title {
                margin-top: 1.5rem;
                margin-bottom: 0.5rem;
            }

            @media (max-width: 760px) {
                .hero {
                    padding: 1.25rem;
                    border-radius: 14px;
                }

                .hero h1 {
                    font-size: 1.75rem;
                }

                .status-row {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    cols_with_nan = ["director", "cast", "country", "date_added"]
    df[cols_with_nan] = df[cols_with_nan].fillna("Inconnu")
    return df


def split_and_count(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    values = values[values != ""]
    values = values[values != "Inconnu"]
    counts = values.value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    return counts


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = default_messages()
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    if "page_selector" not in st.session_state:
        st.session_state.page_selector = st.session_state.page
    if "example_question" not in st.session_state:
        st.session_state.example_question = EXAMPLE_OPTIONS[0]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Navigation")
        st.radio(
            "Choose a page",
            ["Chat", "Dataset Overview"],
            key="page_selector",
            label_visibility="collapsed",
        )
        st.session_state.page = st.session_state.page_selector
        st.divider()

        st.markdown("## Runtime")
        db_status = "Ready" if not chroma_db_is_missing() else "Needs ingest"
        st.markdown(
            f"""
            <div class="status-card">
                <span>Vectorstore</span>
                <strong>{db_status}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="small-note">
                LLM model: <strong>llama3.2:1b</strong><br>
                Embeddings: <strong>nomic-embed-text</strong><br>
                Base Ollama: <strong>localhost:11434</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.button("New chat", use_container_width=True, on_click=start_new_chat)
        st.markdown(
            """
            <div class="small-note">
                If ChromaDB is empty, run:
                <code>uv run python src/ingest.py</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header() -> None:
    st.markdown(
        f"""
        <section class="hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </section>
        <div class="status-row">
            <div class="status-card"><span>Catalog</span><strong>8,807 titles</strong></div>
            <div class="status-card"><span>Retrieval</span><strong>ChromaDB + field filters</strong></div>
            <div class="status-card"><span>Runtime</span><strong>100% local</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(df: pd.DataFrame) -> None:
    total_titles = len(df)
    movie_count = int((df["type"] == "Movie").sum())
    show_count = int((df["type"] == "TV Show").sum())
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())
    country_count = split_and_count(df, "country", top_n=10000)["country"].nunique()
    genre_count = split_and_count(df, "listed_in", top_n=10000)["listed_in"].nunique()

    columns = st.columns(4)
    columns[0].metric("Total titles", f"{total_titles:,}")
    columns[1].metric("Movies", f"{movie_count:,}")
    columns[2].metric("TV shows", f"{show_count:,}")
    columns[3].metric("Years covered", f"{min_year}-{max_year}")

    columns = st.columns(2)
    columns[0].metric("Countries represented", f"{country_count:,}")
    columns[1].metric("Genres/categories", f"{genre_count:,}")


def render_dataset_overview() -> None:
    st.markdown("### Dataset Overview")
    st.markdown(
        """
        <div class="small-note">
            These charts are generated directly from <code>data/netflix_titles.csv</code>.
            The CSV is read only and never modified.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_catalog()
    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        return

    render_metric_cards(df)

    st.markdown("#### Movies vs TV Shows")
    type_counts = df["type"].value_counts().rename_axis("type").reset_index(name="count")
    st.bar_chart(type_counts, x="type", y="count", color="#e50914")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Titles by Release Year")
        year_counts = (
            df.groupby("release_year")
            .size()
            .reset_index(name="count")
            .sort_values("release_year")
        )
        min_year = int(year_counts["release_year"].min())
        max_year = int(year_counts["release_year"].max())
        year_chart = (
            alt.Chart(year_counts)
            .mark_line(color="#e50914", strokeWidth=3)
            .encode(
                x=alt.X(
                    "release_year:Q",
                    title="Release year",
                    scale=alt.Scale(domain=[min_year, max_year], zero=False),
                ),
                y=alt.Y("count:Q", title="Titles"),
                tooltip=["release_year", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(year_chart, use_container_width=True)

    with right:
        st.markdown("#### Top Ratings")
        rating_counts = df["rating"].value_counts().head(10).rename_axis("rating").reset_index(name="count")
        st.bar_chart(rating_counts, x="rating", y="count", color="#f97316")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top Countries")
        country_counts = split_and_count(df, "country", top_n=10)
        st.bar_chart(country_counts, x="country", y="count", color="#38bdf8")
        st.dataframe(country_counts, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Top Genres")
        genre_counts = split_and_count(df, "listed_in", top_n=10)
        st.bar_chart(genre_counts, x="listed_in", y="count", color="#22c55e")
        st.dataframe(genre_counts, use_container_width=True, hide_index=True)

    st.markdown("#### Sample Catalog Rows")
    st.dataframe(
        df[["title", "type", "release_year", "director", "country", "listed_in"]].head(25),
        use_container_width=True,
        hide_index=True,
    )


def render_examples() -> None:
    st.markdown("### Try an Example")
    st.markdown(
        """
        <div class="small-note">
            Pick one compact example, or type your own question below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([4, 1])
    with left:
        st.selectbox(
            "Example question",
            EXAMPLE_OPTIONS,
            key="example_question",
            label_visibility="collapsed",
        )
    with right:
        st.button("Try example", use_container_width=True, on_click=queue_example_question)


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def answer_question(question: str) -> None:
    history = st.session_state.messages[-MEMORY_MESSAGES:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the local Netflix catalog..."):
            try:
                answer = ask_netflix(question, history=history)
            except Exception as error:
                answer = explain_error(error)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    setup_page()
    inject_styles()
    init_state()
    render_sidebar()
    render_header()

    if st.session_state.page == "Chat":
        render_examples()
        render_messages()
    else:
        render_dataset_overview()

    if st.session_state.page == "Chat":
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        chat_question = st.chat_input("Ask about titles, directors, cast, countries, years, or genres...")
        if chat_question:
            question = chat_question

        if question:
            answer_question(question.strip())


if __name__ == "__main__":
    main()
