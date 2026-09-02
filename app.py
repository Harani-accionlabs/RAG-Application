import hashlib
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from simple_rag.chat_history import (
    create_conversation,
    list_conversations,
    load_messages,
    save_message,
    clear_conversation,
)
from simple_rag.exceptions import IndexBuildError, IndexNotBuiltError, QueryError
from simple_rag.config import settings
from simple_rag.embedding import Embedding
from simple_rag.exceptions import (
    IndexBuildError,
    IndexNotBuiltError,
    QueryError,
)
from simple_rag.rag import RAG

st.set_page_config(
    page_title="Marginalia — Ask Your PDF",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(ttl=3600)
def cleanup_stale_sessions():
    tmp_root = Path(tempfile.gettempdir())
    now = time.time()
    for folder in tmp_root.glob("rag_data_*"):
        try:
            if folder.is_dir() and (now - folder.stat().st_mtime) > 3600:
                shutil.rmtree(folder, ignore_errors=True)
        except Exception:
            pass
    return True


cleanup_stale_sessions()


def initialize_session():
    if "session_id" not in st.session_state:
        query_session_id = st.query_params.get("sid")
        if query_session_id:
            st.session_state.session_id = query_session_id
        else:
            st.session_state.session_id = uuid.uuid4().hex
            st.query_params["sid"] = st.session_state.session_id

    defaults = {
        "index_ready": False,
        "current_file_hash": None,
        "conversation_id": None,
        "messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


@st.cache_resource(show_spinner="Loading AI models...")
def load_shared_models():

    embeddings = Embedding(settings.embedding_model)

    llm = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
    )

    reranker = None

    if settings.enable_reranker:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(
            settings.reranker_model
        )

    return embeddings, llm, reranker


embeddings, llm, reranker = load_shared_models()


if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAG(embeddings, llm, reranker)

rag_service = st.session_state.rag_service


def start_new_conversation(label: str):
    """Starts a fresh conversation thread tied to a newly indexed document."""
    new_id = uuid.uuid4().hex
    create_conversation(new_id, st.session_state.session_id, label)
    st.session_state.conversation_id = new_id
    st.session_state.messages = []


with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
    )

    if uploaded_file:

        if st.button(
            "Index Document",
            type="primary",
            use_container_width=True,
        ):

            current_hash = get_file_hash(uploaded_file)

            if current_hash == st.session_state.current_file_hash:
                st.info("This document has already been indexed.")

            else:

                with st.spinner("Indexing document..."):

                    tmp_path = None

                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        chunks = rag_service.build_index_from_pdf(tmp_path)

                        st.session_state.index_ready = True
                        st.session_state.current_file_hash = current_hash
                        start_new_conversation(label=uploaded_file.name)

                        st.success(f"Indexed successfully ({chunks} chunks).")

                    except IndexBuildError as e:
                        st.error(f"Indexing failed: {e}")

                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)

    st.divider()

    st.subheader("Configuration")
    st.caption(f"LLM: {settings.llm_model}")
    st.caption(f"Embedding: {settings.embedding_model}")
    st.caption(f"Top K: {settings.top_k}")
    st.caption(f"Reranker: {'Enabled' if settings.enable_reranker else 'Disabled'}")

    st.divider()

    mode = st.radio("Mode", ["Ask", "Find exact phrase"])

    if st.session_state.index_ready:
        if st.button("📐 Extract all formulas & tables", use_container_width=True):
            user_msg = "Extract all formulas and tables from this document"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            save_message(st.session_state.conversation_id, "user", user_msg)
            with st.spinner("Scanning the entire document..."):
                try:
                    result = rag_service.extract_formulas_and_tables()
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    save_message(st.session_state.conversation_id, "assistant", result)
                except (IndexNotBuiltError, QueryError) as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Extraction failed: {e}"})
            st.rerun()

    st.divider()

    st.subheader("Past conversations")
    conversations = list_conversations(st.session_state.session_id)

    if not conversations:
        st.caption("No past conversations yet.")
    else:
        for conv in conversations:
            is_current = conv["id"] == st.session_state.conversation_id
            button_label = f"{'• ' if is_current else ''}{conv['label']}"
            if st.button(button_label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                st.session_state.messages = load_messages(conv["id"])
                if conv["id"] != st.session_state.get("current_conversation_indexed"):
                    st.session_state.index_ready = False
                st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        if st.session_state.conversation_id:
            clear_conversation(st.session_state.conversation_id)
        st.session_state.messages = []
        st.rerun()

    if st.button("📄 Remove Document", use_container_width=True):
        st.session_state.index_ready = False
        st.session_state.current_file_hash = None
        try:
            rag_service.vectorstore = None
        except Exception:
            pass
        st.rerun()

    st.divider()

    if st.session_state.index_ready:
        st.success("Document indexed and ready.")
    else:
        st.info("Upload and index a PDF.")


st.title("📚 Marginalia")
st.caption("Upload a PDF and ask questions grounded only in the document.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder = "Ask a question about the document..." if mode == "Ask" else "Type an exact phrase to locate..."
user_input = st.chat_input(placeholder, disabled=not st.session_state.index_ready)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(st.session_state.conversation_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if mode == "Ask":
            with st.spinner("Thinking..."):
                try:
                    answer = rag_service.generate_answer(user_input, k=settings.top_k)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    save_message(st.session_state.conversation_id, "assistant", answer)
                except IndexNotBuiltError as e:
                    st.error(str(e))
                except QueryError as e:
                    st.error(f"Something went wrong: {e}")
        else:
            with st.spinner("Searching..."):
                try:
                    pages = rag_service.find_all_pages(user_input)
                    if pages:
                        result_text = f"Found on {len(pages)} page(s): " + ", ".join(str(p) for p in pages)
                    else:
                        result_text = f'No exact match for "{user_input}" found in the document.'
                    st.markdown(result_text)
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
                    save_message(st.session_state.conversation_id, "assistant", result_text)
                except IndexNotBuiltError as e:
                    st.error(str(e))
                except QueryError as e:
                    st.error(f"Something went wrong: {e}")