import hashlib
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq

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



def initialize_session():
    defaults = {
        "session_id": uuid.uuid4().hex,
        "index_ready": False,
        "messages": [],
        "current_file_hash": None,
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

    session_dir = (
        Path(tempfile.gettempdir())
        / f"rag_data_{st.session_state.session_id}"
    )

    st.session_state.rag_service = RAG(
        embeddings,
        llm,
        reranker,
        persist_directory=session_dir,
    )

rag_service = st.session_state.rag_service


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

            current_hash = get_file_hash(
                uploaded_file
            )

            if (
                current_hash
                == st.session_state.current_file_hash
            ):
                st.info(
                    "This document has already been indexed."
                )

            else:

                with st.spinner(
                    "Indexing document..."
                ):

                    tmp_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf",
                        ) as tmp:

                            tmp.write(
                                uploaded_file.getvalue()
                            )

                            tmp_path = tmp.name

                        chunks = (
                            rag_service
                            .build_index_from_pdf(
                                tmp_path
                            )
                        )

                        st.session_state.index_ready = True
                        st.session_state.messages = []

                        st.session_state.current_file_hash = (
                            current_hash
                        )

                        st.success(
                            f"Indexed successfully ({chunks} chunks)."
                        )

                    except IndexBuildError as e:
                        st.error(
                            f"Indexing failed: {e}"
                        )

                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(
                                missing_ok=True
                            )

    st.divider()

    st.subheader("Configuration")

    st.caption(
        f"LLM: {settings.llm_model}"
    )

    st.caption(
        f"Embedding: {settings.embedding_model}"
    )

    st.caption(
        f"Top K: {settings.top_k}"
    )

    st.caption(
        f"Reranker: {'Enabled' if settings.enable_reranker else 'Disabled'}"
    )

    st.divider()

    mode = st.radio(
        "Mode",
        [
            "Ask",
            "Find exact phrase",
        ],
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "📄 Remove Document",
        use_container_width=True,
    ):

        st.session_state.index_ready = False
        st.session_state.current_file_hash = None
        st.session_state.messages = []

        try:
            rag_service.vectorstore = None
        except Exception:
            pass

        st.rerun()

    st.divider()

    if st.session_state.index_ready:
        st.success(
            "Document indexed and ready."
        )
    else:
        st.info(
            "Upload and index a PDF."
        )



st.title("📚 Marginalia")

st.caption(
    "Upload a PDF and ask questions grounded only in the document."
)


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

placeholder = "Ask a question about the document..." if mode == "Ask" else "Type an exact phrase to locate..."
user_input = st.chat_input(placeholder, disabled=not st.session_state.index_ready)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if mode == "Ask":
            with st.spinner("Thinking..."):
                try:
                    answer = rag_service.generate_answer(user_input, k=settings.top_k)
                    st.markdown(answer)
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
                    st.markdown(result_text)
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
                except IndexNotBuiltError as e:
                    st.error(str(e))
                except QueryError as e:
                    st.error(f"Something went wrong: {e}")