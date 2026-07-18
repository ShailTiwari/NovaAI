from html import escape
from pathlib import Path

import streamlit as st

from src.chunker import chunk_pages
from src.config import get_settings
from src.llm import MissingApiKeyError, get_llm
from src.logger import get_logger
from src.pdf_loader import (
    PDFValidationError,
    extract_pdf_pages,
    save_uploaded_pdf,
    validate_pdf,
    validate_pdf_upload,
)
from src.rag_chain import answer_question
from src.utils import ensure_directories, generate_document_id
from src.vector_store import (
    add_chunks_to_vector_store,
    collection_name_for_hash,
    get_vector_store,
)


logger = get_logger(__name__)

SUMMARY_QUESTION = (
    "In 3-4 sentences, summarize what this document is about and list its "
    "main topics. Write it as a neutral overview, not a list of instructions."
)

# Small friendly robot mark, used both as the brand logo and the assistant avatar.
ROBOT_SVG = """
<svg width="22" height="22" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="4" width="8" height="8" rx="4" fill="white"/>
  <rect x="22.5" y="10" width="3" height="6" fill="white"/>
  <rect x="6" y="16" width="36" height="28" rx="10" fill="white"/>
  <circle cx="17" cy="30" r="4.2" fill="#3B82F6"/>
  <circle cx="31" cy="30" r="4.2" fill="#3B82F6"/>
  <path d="M17 38 Q24 42 31 38" stroke="#3B82F6" stroke-width="2.4" stroke-linecap="round" fill="none"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def apply_custom_styles() -> None:
    """Dark, minimal NovaAI theme - two clean panels, chat-app restraint."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #0a0e17;
                --surface: #10141f;
                --surface-soft: #151a27;
                --border: #212739;
                --border-soft: #1a2030;
                --text: #e8ecf4;
                --muted: #7c869e;
                --accent: #3b82f6;
                --accent-dim: #1d4ed8;
                --accent-soft: rgba(59, 130, 246, 0.12);
                --user-bubble: #2f6fed;
            }

            html, body, .stApp {
                background: var(--bg) !important;
                color: var(--text);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .block-container {
                max-width: 1180px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: var(--surface);
                border-right: 1px solid var(--border);
            }
            [data-testid="stSidebar"] * { color: var(--text); }
            [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
                color: var(--muted) !important;
            }

            /* Top brand bar */
            .brand-bar {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 4px 0 22px 0;
                border-bottom: 1px solid var(--border-soft);
                margin-bottom: 22px;
            }
            .brand-mark {
                width: 38px;
                height: 38px;
                border-radius: 10px;
                background: linear-gradient(135deg, var(--accent), #6d28d9);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }
            .brand-name {
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--text);
                letter-spacing: -0.01em;
                line-height: 1.1;
            }
            .brand-name span { color: var(--accent); }
            .brand-sub {
                font-size: 0.8rem;
                color: var(--muted);
                margin-top: 1px;
            }

            /* Panel titles */
            .panel-title {
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--muted);
                margin: 2px 0 12px 0;
            }

            /* Document card */
            .doc-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 16px 18px;
                margin-bottom: 14px;
            }
            .doc-card .doc-name {
                font-weight: 650;
                font-size: 0.98rem;
                color: var(--text);
                word-break: break-word;
                margin-bottom: 10px;
            }
            .doc-stats {
                display: flex;
                gap: 10px;
            }
            .doc-stat {
                flex: 1;
                background: var(--surface-soft);
                border: 1px solid var(--border-soft);
                border-radius: 8px;
                padding: 8px 10px;
                text-align: center;
            }
            .doc-stat .n {
                font-size: 1.15rem;
                font-weight: 750;
                color: var(--text);
                line-height: 1.1;
            }
            .doc-stat .l {
                font-size: 0.68rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.03em;
                margin-top: 2px;
            }

            .summary-card {
                background: var(--accent-soft);
                border: 1px solid rgba(59, 130, 246, 0.28);
                border-radius: 12px;
                padding: 16px 18px;
                margin-bottom: 14px;
            }
            .summary-card .s-label {
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--accent);
                margin-bottom: 6px;
            }
            .summary-card .s-text {
                font-size: 0.92rem;
                line-height: 1.55;
                color: var(--text);
            }

            .empty-panel {
                background: var(--surface);
                border: 1px dashed var(--border);
                border-radius: 12px;
                padding: 26px 20px;
                text-align: center;
                color: var(--muted);
                font-size: 0.88rem;
            }

            /* Fixed-height scrollable chat container (keeps chat_input pinned) */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stChatMessage"]),
            div[data-testid="stContainer"] {
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 14px !important;
            }

            /* Chat bubbles */
            [data-testid="stChatMessage"] {
                background: transparent;
                padding: 4px 0;
            }
            div[data-testid="stChatMessageContent"] {
                background: var(--surface-soft);
                border: 1px solid var(--border-soft);
                border-radius: 14px;
                padding: 10px 14px;
            }
            div[data-testid="stChatMessageContent"] p,
            div[data-testid="stChatMessageContent"] li,
            div[data-testid="stChatMessageContent"] span,
            div[data-testid="stChatMessageContent"] strong,
            div[data-testid="stChatMessageContent"] em {
                color: var(--text) !important;
            }
            div[data-testid="stChatMessageContent"] a {
                color: var(--accent) !important;
            }

            /* Assistant avatar circle in NovaAI blue */
            div[data-testid="stChatMessageAvatarAssistant"],
            div[data-testid="stChatMessageAvatarCustom"] {
                background: var(--accent) !important;
                border: none !important;
            }

            /* User bubble in blue */
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
              div[data-testid="stChatMessageContent"] {
                background: var(--user-bubble);
                border-color: var(--user-bubble);
            }
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
              div[data-testid="stChatMessageContent"] p,
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
              div[data-testid="stChatMessageContent"] span {
                color: white !important;
            }

            .source-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
            .source-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                background: var(--accent-soft);
                border: 1px solid rgba(59, 130, 246, 0.3);
                color: #9dc0ff;
                font-size: 0.72rem;
                font-weight: 650;
                padding: 3px 9px;
            }

            /* File uploader - dropzone + browse button */
            div[data-testid="stFileUploader"] section {
                background: var(--surface) !important;
                border: 1px dashed var(--border) !important;
                border-radius: 10px;
            }
            div[data-testid="stFileUploader"] section > div,
            div[data-testid="stFileUploaderDropzoneInstructions"] span,
            div[data-testid="stFileUploaderDropzoneInstructions"] small {
                color: var(--muted) !important;
            }
            div[data-testid="stFileUploaderDropzone"] button,
            div[data-testid="stBaseButton-secondary"] {
                background: var(--surface-soft) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
            }

            .stButton > button {
                border-radius: 8px;
                border: 1px solid var(--border);
                background: var(--surface-soft);
                color: var(--text);
                font-weight: 600;
            }
            .stButton > button:hover {
                border-color: var(--accent);
                color: var(--accent);
            }
            .stButton > button p { color: inherit !important; }

            /* Chat input bar + its floating container */
            [data-testid="stBottomBlockContainer"],
            [data-testid="stChatInputContainer"],
            .stChatFloatingInputContainer {
                background: var(--bg) !important;
                border-top: 1px solid var(--border-soft) !important;
            }
            div[data-testid="stChatInput"] {
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
            }
            div[data-testid="stChatInput"] textarea {
                background: transparent !important;
                color: var(--text) !important;
            }
            div[data-testid="stChatInput"] textarea::placeholder {
                color: var(--muted) !important;
            }
            div[data-testid="stChatInput"] button {
                background: var(--accent) !important;
                border-radius: 8px !important;
            }
            div[data-testid="stChatInput"] button svg { fill: white !important; }

            div[data-testid="stExpander"] {
                background: var(--surface);
                border: 1px solid var(--border-soft);
                border-radius: 10px;
            }
            div[data-testid="stExpander"] summary { color: var(--muted); }
            div[data-testid="stExpander"] p { color: var(--text) !important; }

            hr { border-color: var(--border-soft); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def source_badges(pages: list[int]) -> str:
    if not pages:
        return ""
    pills = "".join(f'<span class="source-pill">Page {page}</span>' for page in pages)
    return f'<div class="source-wrap">{pills}</div>'


def render_brand_bar() -> None:
    st.markdown(
        f"""
        <div class="brand-bar">
            <div class="brand-mark">{ROBOT_SVG}</div>
            <div>
                <div class="brand-name">Nova<span>AI</span></div>
                <div class="brand-sub">Your intelligent document assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "messages": [],
        "document_id": None,
        "collection_name": None,
        "vector_store": None,
        "page_count": 0,
        "chunk_count": 0,
        "file_name": None,
        "document_summary": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def embedding_key(settings) -> str:
    model = (
        settings.ollama_embed_model
        if settings.embedding_provider == "ollama"
        else settings.sentence_transformer_model
    )
    return f"{settings.embedding_provider}:{model}:chunk-{settings.chunk_size}:overlap-{settings.chunk_overlap}"


def document_collection_name(document_id: str, settings) -> str:
    return collection_name_for_hash(document_id, embedding_key(settings))


# ---------------------------------------------------------------------------
# Indexing + summary
# ---------------------------------------------------------------------------

def generate_summary_on_demand(settings) -> str | None:
    """Generate a document summary only when the user asks for it."""
    vector_store = st.session_state.vector_store
    document_id = st.session_state.document_id
    if vector_store is None or document_id is None:
        return None
    try:
        sample_docs = vector_store.similarity_search(
            "overview summary main topics",
            k=min(6, settings.top_k + 2),
            filter={"document_id": document_id},
        )
        llm = get_llm(settings)
        result = answer_question(SUMMARY_QUESTION, sample_docs, llm)
        return result.get("answer")
    except Exception:
        logger.exception("Summary generation failed")
        return None


def save_and_index_pdf(uploaded_file, settings) -> dict:
    """Run the full PDF indexing pipeline for one uploaded file."""
    original_name = getattr(uploaded_file, "name", "uploaded.pdf")
    logger.info("PDF uploaded: %s", original_name)
    validate_pdf_upload(uploaded_file, settings.max_pdf_mb)

    pdf_path, _ = save_uploaded_pdf(uploaded_file, settings.upload_dir)
    validate_pdf(str(pdf_path), settings.max_pdf_mb)
    logger.info("PDF validated: %s", pdf_path)

    document_id = generate_document_id(str(pdf_path))
    collection_name = document_collection_name(document_id, settings)

    pages = extract_pdf_pages(str(pdf_path), document_id=document_id)
    logger.info("PDF pages extracted: document_id=%s pages=%s", document_id, len(pages))
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Text chunks created: document_id=%s chunks=%s", document_id, len(chunks))

    logger.info("Embedding started: document_id=%s collection=%s", document_id, collection_name)
    added_ids = add_chunks_to_vector_store(chunks, collection_name)
    vector_store = get_vector_store(collection_name)
    logger.info(
        "Embedding completed: document_id=%s total_chunks=%s new_chunks=%s",
        document_id,
        len(chunks),
        len(added_ids),
    )

    return {
        "document_id": document_id,
        "collection_name": collection_name,
        "vector_store": vector_store,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "added_count": len(added_ids),
        "file_name": original_name or Path(pdf_path).name,
    }


def reset_current_document() -> None:
    document_id = st.session_state.get("document_id")
    collection_name = st.session_state.get("collection_name")
    if not document_id or not collection_name:
        st.warning("Upload and index a PDF before resetting its vector data.")
        return

    try:
        store = get_vector_store(collection_name)
        store._collection.delete(where={"document_id": document_id})
    except Exception:
        logger.exception("Vector database reset failed")
        st.error("Vector database reset failed. Check that Chroma is installed and available.")
        return

    for key in (
        "vector_store", "collection_name", "document_id",
        "page_count", "chunk_count", "file_name", "document_summary",
    ):
        st.session_state[key] = None if key != "page_count" and key != "chunk_count" else 0
    st.session_state.messages = []
    st.success("Document cleared.")


# ---------------------------------------------------------------------------
# Sidebar (settings only - kept slim)
# ---------------------------------------------------------------------------

def render_sidebar(settings) -> None:
    with st.sidebar:
        st.markdown("**Settings**")
        st.caption(f"LLM: {settings.llm_provider}")
        st.caption(f"Embeddings: {settings.embedding_provider}")
        st.caption(f"Chunk size: {settings.chunk_size} · overlap: {settings.chunk_overlap}")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset document", use_container_width=True):
            reset_current_document()
            st.rerun()


# ---------------------------------------------------------------------------
# Left panel: document
# ---------------------------------------------------------------------------

def render_document_panel(settings) -> None:
    st.markdown('<div class="panel-title">Document</div>', unsafe_allow_html=True)

    if st.session_state.document_id:
        name = escape(st.session_state.file_name or "Indexed PDF")
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-name">{name}</div>
                <div class="doc-stats">
                    <div class="doc-stat"><div class="n">{st.session_state.page_count}</div><div class="l">Pages</div></div>
                    <div class="doc-stat"><div class="n">{st.session_state.chunk_count}</div><div class="l">Chunks</div></div>
                    <div class="doc-stat"><div class="n">{len(st.session_state.messages)}</div><div class="l">Turns</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.document_summary:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="s-label">Summary</div>
                    <div class="s-text">{escape(st.session_state.document_summary)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.button("Generate summary", use_container_width=True):
                with st.spinner("Reading the document..."):
                    summary = generate_summary_on_demand(settings)
                if summary:
                    st.session_state.document_summary = summary
                else:
                    st.error("Could not generate a summary for this document.")
                st.rerun()

        if st.button("Upload a different PDF", use_container_width=True):
            reset_current_document()
            st.rerun()
        return

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is None:
        st.markdown(
            '<div class="empty-panel">Upload a PDF to build a searchable index and get an instant summary.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Indexing PDF..."):
            result = save_and_index_pdf(uploaded_file, settings)
    except PDFValidationError as exc:
        logger.exception("PDF upload validation failed")
        st.error(str(exc))
    except ValueError as exc:
        logger.exception("PDF indexing validation failed")
        message = str(exc)
        if "OCR" in message or "extractable text" in message:
            st.error("This PDF has no readable text. It may be scanned and needs OCR first.")
        else:
            st.error(message)
    except MissingApiKeyError as exc:
        logger.exception("LLM configuration failed")
        st.error(str(exc))
    except RuntimeError as exc:
        logger.exception("PDF indexing runtime error")
        message = str(exc)
        if "Ollama server is not running" in message:
            st.error(message)
        elif "Chroma" in message or "vector" in message.lower():
            st.error("Vector DB error. Check Chroma installation and the vector_db folder.")
        else:
            st.error(message)
    except Exception:
        logger.exception("PDF indexing failed")
        st.error("Unknown error while indexing the PDF. Check the terminal logs.")
    else:
        st.session_state.document_id = result["document_id"]
        st.session_state.collection_name = result["collection_name"]
        st.session_state.vector_store = result["vector_store"]
        st.session_state.page_count = result["page_count"]
        st.session_state.chunk_count = result["chunk_count"]
        st.session_state.file_name = result["file_name"]
        st.session_state.document_summary = None
        st.rerun()


# ---------------------------------------------------------------------------
# Right panel: chat
# ---------------------------------------------------------------------------

def render_retrieved_chunks(chunks) -> None:
    if not chunks:
        return
    with st.expander("Retrieved chunks"):
        for index, chunk in enumerate(chunks, start=1):
            page = chunk.metadata.get("page_number", chunk.metadata.get("page", "unknown"))
            st.markdown(f"**Chunk {index} · Page {page}**")
            st.write(chunk.page_content)


def render_assistant_message(message: dict) -> None:
    answer = message.get("answer") or message.get("content", "")
    st.markdown(answer)


def render_chat_history() -> None:
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def answer_user_question(question: str, settings) -> dict:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        raise RuntimeError("No PDF has been indexed yet. Upload a PDF first.")

    logger.info("Retrieval started: document_id=%s", st.session_state.document_id)
    with st.spinner("Searching PDF..."):
        docs = vector_store.similarity_search(
            question,
            k=settings.top_k,
            filter={"document_id": st.session_state.document_id},
        )
    logger.info("Retrieval completed: document_id=%s chunks=%s", st.session_state.document_id, len(docs))

    for doc in docs:
        if "page" not in doc.metadata and "page_number" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page_number"]

    logger.info("LLM answer started: document_id=%s", st.session_state.document_id)
    with st.spinner("Generating answer..."):
        llm = get_llm(settings)
        result = answer_question(question, docs, llm)
    logger.info("LLM answer completed: document_id=%s", st.session_state.document_id)

    return {
        "content": result["answer"],
        "answer": result["answer"],
        "source_pages": result["source_pages"],
        "chunks": docs,
    }


def render_chat_panel(settings) -> None:
    st.markdown('<div class="panel-title">Chat</div>', unsafe_allow_html=True)

    if not st.session_state.document_id:
        st.markdown(
            '<div class="empty-panel">Ask anything about your documents once one is indexed.</div>',
            unsafe_allow_html=True,
        )
        return

    chat_box = st.container(height=560, border=True)
    with chat_box:
        render_chat_history()

    question = st.chat_input("Ask anything about your document...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with chat_box:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            assistant_message = answer_user_question(question, settings)
        except MissingApiKeyError as exc:
            logger.exception("LLM configuration failed")
            assistant_message = {"content": str(exc), "answer": str(exc), "source_pages": [], "chunks": []}
        except RuntimeError as exc:
            logger.exception("Chat runtime error")
            message = str(exc)
            if "Ollama server is not running" in message:
                answer = message
            elif "API request failed" in message or "Model not found" in message:
                answer = message
            else:
                answer = "Vector DB error or LLM error. Check the terminal logs."
            assistant_message = {"content": answer, "answer": answer, "source_pages": [], "chunks": []}
        except Exception:
            logger.exception("Chat failed")
            answer = "Unknown error while answering. Check the terminal logs."
            assistant_message = {"content": answer, "answer": answer, "source_pages": [], "chunks": []}

        st.session_state.messages.append({"role": "assistant", **assistant_message})
        with st.chat_message("assistant", avatar="🤖"):
            render_assistant_message(assistant_message)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NovaAI - Chat with PDF", page_icon="🌀", layout="wide")
    apply_custom_styles()

    settings = get_settings()
    ensure_directories(settings.upload_dir, settings.processed_dir, settings.chroma_persist_dir)
    init_session_state()

    render_brand_bar()
    render_sidebar(settings)

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        render_document_panel(settings)
    with right:
        render_chat_panel(settings)


if __name__ == "__main__":
    main()