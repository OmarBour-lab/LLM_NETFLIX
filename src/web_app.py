import streamlit as st

try:
    from .app import chroma_db_is_missing, explain_error
    from .chain import ask_netflix
except ImportError:
    from app import chroma_db_is_missing, explain_error
from chain import ask_netflix


MEMORY_MESSAGES = 6


APP_TITLE = "Netflix RAG Chatbot"
APP_SUBTITLE = (
    "Ask questions against the local Netflix catalog. "
    "The app can retrieve titles, filter fields, and summarize records with a local LLM."
)
EXAMPLE_GROUPS = [
    (
        "Title search",
        [
            "Give me the description of League of Legends Origins",
            "What do you know about Fate/Zero?",
            "Who directed Bird Box?",
        ],
    ),
    (
        "List titles",
        [
            "give me all the nova movies available",
            "all pokemon movies or series that are available",
            "all titles that start with the legend of",
        ],
    ),
    (
        "Filter fields",
        [
            "movies with director containing John",
            "movies with cast containing Sandra",
            "series from Japan released in 2020",
        ],
    ),
    (
        "Catalog filters",
        [
            "movies released in 2020",
            "all titles with christmas",
            "list movies containing love",
        ],
    ),
]


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
                width: 100%;
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


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi. Ask me in English about the local Netflix catalog. "
                    "Try title descriptions, title lists, director/cast filters, countries, years, or genres."
                ),
            }
        ]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_sidebar() -> None:
    with st.sidebar:
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
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()
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


def render_examples() -> None:
    st.markdown("### Try What It Can Do")
    st.markdown(
        """
        <div class="small-note">
            Use English questions for best results with <strong>llama3.2:1b</strong>.
            List-style questions are filtered directly against the catalog; description questions use the retrieved context and the local LLM.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for group_title, questions in EXAMPLE_GROUPS:
        st.markdown(f"#### {group_title}")
        columns = st.columns(3)
        for column, question in zip(columns, questions):
            with column:
                if st.button(question, key=f"example-{group_title}-{question}"):
                    st.session_state.pending_question = question
                    st.rerun()


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
    render_examples()
    render_messages()

    question = st.session_state.pending_question
    st.session_state.pending_question = None

    chat_question = st.chat_input("Ask about titles, directors, cast, countries, years, or genres...")
    if chat_question:
        question = chat_question

    if question:
        answer_question(question.strip())


if __name__ == "__main__":
    main()
