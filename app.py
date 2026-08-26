import tempfile
import uuid
from pathlib import Path

import streamlit as st

from simple_rag.config import settings
from simple_rag.embedding import Embedding
from simple_rag.rag import RAG
from simple_rag.exceptions import IndexBuildError, IndexNotBuiltError, QueryError

from langchain_groq import ChatGroq

st.set_page_config(page_title="Marginalia — Ask Your PDF", page_icon="§", layout="centered")


@st.cache_resource(show_spinner="Loading models (embedding + reranker)...")
def load_shared_models():
    embeddings = Embedding(settings.embedding_model)
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)

    reranker = None
    if settings.enable_reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(settings.reranker_model)

    return embeddings, llm, reranker


embeddings, llm, reranker = load_shared_models()


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "rag_service" not in st.session_state:
    session_dir = Path(tempfile.gettempdir()) / f"rag_data_{st.session_state.session_id}"
    st.session_state.rag_service = RAG(embeddings, llm, reranker, persist_directory=session_dir)

rag_service = st.session_state.rag_service

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("§ Marginalia")
st.caption("Upload a PDF and ask questions grounded only in its content.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index this document", type="primary", use_container_width=True):
            with st.spinner("Indexing document — this can take a minute..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    chunks = rag_service.build_index_from_pdf(tmp_path)
                    st.session_state.index_ready = True
                    st.session_state.messages = []
                    st.success(f"Indexed — {chunks} chunks ready.")
                except IndexBuildError as e:
                    st.error(f"Indexing failed: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    if st.session_state.index_ready:
        st.success("Document ready — ask away.")
    else:
        st.info("Upload and index a PDF to get started.")

    st.divider()
    mode = st.radio("Mode", ["Ask", "Find exact phrase"], help="Ask uses semantic search + an LLM answer. Find locates every page containing an exact phrase.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

placeholder = "Ask a question about the document..." if mode == "Ask" else "Type an exact phrase to locate..."
user_input = st.chat_input(placeholder, disabled=not st.session_state.index_ready)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        if mode == "Ask":
            with st.spinner("Thinking..."):
                try:
                    answer = rag_service.generate_answer(user_input, k=5)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
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
                    st.write(result_text)
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
                except IndexNotBuiltError as e:
                    st.error(str(e))
                except QueryError as e:
                    st.error(f"Something went wrong: {e}")