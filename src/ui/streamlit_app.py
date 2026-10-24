
import streamlit as st
import tempfile
import uuid
import os
import shutil
import logging
import traceback
import html
from typing import List, Dict, Any

from src.rag.pipelines import RAGPipeline
from src.exception import MyException


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-DOC Project",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "status" not in st.session_state:
    st.session_state.status = "idle"

if "loaded_documents" not in st.session_state:
    st.session_state.loaded_documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp(
        prefix="rag_uploads_"
    )

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1f2937;
    }

    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: #334155;
    }

    .sub-text {
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 0.7rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
    }

    .answer-title {
        color: #2563eb;
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
    }

    .source-citation {
        background: #f1f5f9;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        overflow-wrap: anywhere;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def display_name_from_path(path: str) -> str:
    """Return a readable filename without its UUID prefix."""

    if not path:
        return "Unknown source"

    # Handle URLs separately.
    if path.startswith(("http://", "https://")):
        return path

    basename = os.path.basename(path)
    parts = basename.split("_", 1)

    if (
        len(parts) == 2
        and len(parts[0]) == 32
    ):
        return parts[1]

    return basename


def cleanup_resources():
    """Clear the current pipeline, documents and chat history."""

    try:
        pipeline = st.session_state.get("pipeline")

        if pipeline is not None:
            if hasattr(pipeline, "vector_store"):
                pipeline.vector_store = None

            if hasattr(pipeline, "retriever"):
                pipeline.retriever = None

        st.session_state.pipeline = None

        old_tmp_dir = st.session_state.get("tmp_dir")

        if old_tmp_dir and os.path.exists(old_tmp_dir):
            shutil.rmtree(old_tmp_dir, ignore_errors=True)

        st.session_state.tmp_dir = tempfile.mkdtemp(
            prefix="rag_uploads_"
        )

        st.session_state.status = "idle"
        st.session_state.loaded_documents = []
        st.session_state.messages = []
        st.session_state.pending_query = None

        st.success("Documents and chat history cleared.")
        st.rerun()

    except Exception as exc:
        logger.exception("Cleanup failed")
        st.error(
            f"Cleanup failed: {type(exc).__name__}: {exc}"
        )


# --------------------------------------------------
# DOCUMENT INDEXING
# --------------------------------------------------

def index_documents(uploaded_files, url_input):
    """Save uploaded files and index the selected documents."""

    try:
        st.session_state.status = "processing"

        # Initialize the real pipeline.
        if st.session_state.pipeline is None:
            st.session_state.pipeline = RAGPipeline(
                config_dir="configs"
            )

        pipeline = st.session_state.pipeline

        # Ensure the upload directory exists.
        os.makedirs(
            st.session_state.tmp_dir,
            exist_ok=True
        )

        file_paths: List[str] = []
        documents: List[Dict[str, Any]] = []

        # Save uploaded files.
        for uploaded_file in uploaded_files or []:
            original_name = os.path.basename(
                uploaded_file.name
            )

            unique_name = (
                f"{uuid.uuid4().hex}_{original_name}"
            )

            file_path = os.path.join(
                st.session_state.tmp_dir,
                unique_name
            )

            with open(file_path, "wb") as output_file:
                output_file.write(
                    uploaded_file.getvalue()
                )

            file_paths.append(file_path)

        # Prepare document configuration.
        for file_path in file_paths:
            documents.append(
                {
                    "path": file_path,
                    "enabled": True,
                    "name": display_name_from_path(
                        file_path
                    ),
                }
            )

        # Add the URL if supplied.
        if url_input and url_input.strip():
            documents.append(
                {
                    "path": url_input.strip(),
                    "enabled": True,
                    "name": url_input.strip(),
                }
            )

        if not documents:
            raise ValueError(
                "No valid documents or URLs were provided."
            )

        # Configure and run the real indexing pipeline.
        pipeline.config["documents"] = documents

        with st.spinner(
            "Processing and indexing documents..."
        ):
            pipeline.prepare_vector_store()

        st.session_state.loaded_documents = [
            {
                "name": document["name"],
                "path": document["path"],
            }
            for document in documents
        ]

        st.session_state.status = "ready"

        st.sidebar.success(
            f"Successfully indexed {len(documents)} source(s)."
        )

    except Exception as exc:
        logger.exception("Document indexing failed")

        st.session_state.status = "error"

        st.sidebar.error(
            f"Indexing failed: {type(exc).__name__}: {exc}"
        )

        with st.sidebar.expander(
            "View error details",
            expanded=True
        ):
            st.code(
                traceback.format_exc(),
                language="text"
            )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

def sidebar():
    st.sidebar.title("⚙️ RAG System Controls")

    st.sidebar.header("📚 Load Documents")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        key="document_uploader",
    )

    url_input = st.sidebar.text_input(
        "Or provide a URL:",
        placeholder="https://example.com/report.pdf",
        key="document_url",
    )

    if st.sidebar.button(
        "🚀 Index Documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files and not url_input.strip():
            st.sidebar.warning(
                "Upload a document or provide a URL first."
            )
        else:
            index_documents(
                uploaded_files,
                url_input
            )

    st.sidebar.markdown("---")
    st.sidebar.header("Current Context Status")

    status = st.session_state.status
    loaded_documents = st.session_state.loaded_documents

    if status == "ready":
        st.sidebar.success(
            f"Context ready: {len(loaded_documents)} source(s)."
        )
    elif status == "processing":
        st.sidebar.info("Indexing in progress...")
    elif status == "error":
        st.sidebar.error(
            "Indexing failed. Expand View error details above."
        )
    else:
        st.sidebar.warning(
            "Idle. No documents indexed."
        )

    if loaded_documents:
        with st.sidebar.expander(
            f"Indexed Sources ({len(loaded_documents)})",
            expanded=True,
        ):
            for document in loaded_documents:
                st.caption(
                    f"📄 {document['name']}"
                )

    st.sidebar.markdown("---")
    st.sidebar.header("Context Management")

    if st.sidebar.button(
        "🔥 Clear ALL Indexed Documents",
        type="secondary",
        use_container_width=True,
    ):
        cleanup_resources()


# --------------------------------------------------
# SOURCE DISPLAY
# --------------------------------------------------

def display_sources(sources):
    """Display the sources returned by the RAG pipeline."""

    if not sources:
        st.caption(
            "No specific sources were returned."
        )
        return

    with st.expander(
        f"📚 Sources Cited ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            source_path = source.get("path", "Unknown source")
            source_name = display_name_from_path(source_path)
            page_number = source.get(
                "page_info",
                source.get("page", "Not specified"),
            )
            snippet = source.get("snippet", "")

            # Escape untrusted source content before HTML rendering.
            safe_name = html.escape(str(source_name))
            safe_path = html.escape(str(source_path))
            safe_page = html.escape(str(page_number))
            safe_snippet = html.escape(str(snippet))

            st.markdown(
                f"""
                <div class="source-citation">
                    <strong>Source {index}: {safe_name}</strong>
                    <p>📍 Location: {safe_page}</p>
                    <p>📄 Path: {safe_path}</p>
                    <p><strong>Snippet:</strong></p>
                    <div>{safe_snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# CHAT AREA
# --------------------------------------------------

def chat_area():
    st.markdown(
        '<div class="title-text">'
        '🪐 Multi-Doc RAG Q&A Interface'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-text">'
        'Ask questions grounded in your indexed documents.'
        '</div>',
        unsafe_allow_html=True,
    )

    rag_ready = (
        st.session_state.status == "ready"
        and st.session_state.pipeline is not None
    )

    # Render existing conversation.
    for message in st.session_state.messages:
        with st.chat_message(
            message["role"],
            avatar="👤" if message["role"] == "user" else "🤖",
        ):
            content = message["content"]

            if message["role"] == "user":
                st.markdown(content)
            else:
                if isinstance(content, dict):
                    answer = content.get(
                        "answer",
                        "No answer was returned.",
                    )
                    sources = content.get("sources", [])

                    st.markdown(
                        '<div class="answer-card">'
                        '<div class="answer-title">'
                        '🤖 AI Response'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(answer)
                    display_sources(sources)
                else:
                    st.markdown(str(content))

    # Chat input form.
    with st.form(
        "chat_form",
        clear_on_submit=True,
    ):
        query = st.text_input(
            "Ask a question about your documents",
            placeholder=(
                "e.g., Summarize the key findings "
                "from the uploaded report."
            ),
            disabled=not rag_ready,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([1, 6])

        with col1:
            submitted = st.form_submit_button(
                "Send",
                type="primary",
                disabled=not rag_ready,
                use_container_width=True,
            )

        with col2:
            clear_chat = st.form_submit_button(
                "Clear Chat History",
                type="secondary",
                use_container_width=True,
            )

    if clear_chat:
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    if submitted and query.strip():
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query.strip(),
            }
        )

        st.session_state.pending_query = query.strip()
        st.rerun()

    # Process the submitted query after displaying it.
    pending_query = st.session_state.pending_query

    if pending_query:
        try:
            pipeline = st.session_state.pipeline

            if pipeline is None:
                raise RuntimeError(
                    "Pipeline is not initialized. "
                    "Index documents before asking questions."
                )

            with st.spinner("Searching documents and generating an answer..."):
                result = pipeline.answer_with_sources(
                    pending_query
                )

            if not isinstance(result, dict):
                result = {
                    "answer": str(result),
                    "sources": [],
                }

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

        except MyException as exc:
            logger.exception("RAG query failed")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": (
                            f"RAG error: {type(exc).__name__}: {exc}"
                        ),
                        "sources": [],
                    },
                }
            )

        except Exception as exc:
            logger.exception("Query processing failed")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": (
                            f"Query failed: {type(exc).__name__}: {exc}"
                        ),
                        "sources": [],
                    },
                }
            )

        finally:
            st.session_state.pending_query = None

        st.rerun()


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

def main():
    sidebar()
    chat_area()


if __name__ == "__main__":
    main()
