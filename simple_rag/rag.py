import logging
import shutil
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .exceptions import IndexBuildError, IndexNotBuiltError, QueryError

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class RAG:
    """Holds one user's index/session state. Expensive shared resources (the
    embedding model, LLM client, and reranker) are passed in from outside —
    typically loaded once and reused across sessions — while persist_directory
    should be UNIQUE per user/session to avoid concurrent writes to the same
    underlying Chroma database."""

    def __init__(
        self,
        embeddings,
        llm,
        reranker=None,
        persist_directory=None,
    ):
        self.embeddings = embeddings
        self.llm = llm
        self.reranker = reranker
        self.persist_directory = persist_directory
        self.vectorstore = None

    def build_index_from_pdf(self, pdf_path: str) -> int:
        """Build the vector index from a PDF file. Returns the number of chunks indexed."""
        try:
            if not Path(pdf_path).is_file():
                raise IndexBuildError(f"PDF file not found: {pdf_path}")

            if self.persist_directory.exists():
                logger.info(f"Clearing existing index at {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore = None

            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            MAX_PAGES = 300
            if len(documents) > MAX_PAGES:
                raise IndexBuildError(
                    f"This PDF has {len(documents)} pages, which exceeds the {MAX_PAGES}-page "
                    "limit for this demo (to avoid running out of storage). Try a smaller document."
                )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            docs = splitter.split_documents(documents)
            if not docs:
                raise IndexBuildError(
                    "No extractable text found in this PDF. It may be a scanned/"
                    "image-based document — try a PDF with selectable text, or use "
                    "OCR to convert it first."
                )
            logger.info(f"Building index from {len(docs)} chunks ({len(documents)} pages)")
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            logger.info("Index built successfully")
            return len(docs)
        except IndexBuildError:
            raise
        except Exception as e:
            logger.error(f"Failed to build index from PDF: {e}")
            raise IndexBuildError(str(e)) from e

    def build_index(self, data: List[dict]) -> None:
        """Build the vector index from Q&A-style records (kept for backward compatibility)."""
        try:
            if self.persist_directory.exists():
                logger.info(f"Clearing existing index at {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore = None

            df = pd.DataFrame(data)
            df["content"] = "Question: " + df["question"] + "\nAnswer: " + df["answer"]
            loader = DataFrameLoader(df, page_content_column="content")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            docs = splitter.split_documents(documents)
            logger.info(f"Building index from {len(docs)} chunks")
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            logger.info("Index built successfully")
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise IndexBuildError(str(e)) from e

    def load_existing_index(self) -> bool:
        """Load a previously persisted index from disk, if one exists for this session."""
        if not self.persist_directory.exists():
            return False
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )
        logger.info("Loaded existing index from disk")
        return True

    def query(self, query: str, k: Optional[int] = None) -> List[Any]:
        """Retrieve the top-k most relevant chunks. Uses embedding search followed by
        cross-encoder reranking when a reranker is set; otherwise returns the top-k
        embedding-similarity matches directly."""
        if self.vectorstore is None:
            raise IndexNotBuiltError("Index has not been built or loaded yet")
        try:
            top_k = k or settings.top_k

            if self.reranker is None:
                return self.vectorstore.similarity_search(query, k=top_k)

            pool_size = max(settings.rerank_candidate_pool, top_k)
            candidates = self.vectorstore.similarity_search(query, k=pool_size)

            if not candidates:
                return []

            pairs = [(query, doc.page_content) for doc in candidates]
            scores = self.reranker.predict(pairs)

            reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in reranked[:top_k]]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(str(e)) from e

    def find_all_pages(self, phrase: str) -> List[Any]:
        """Exhaustive keyword search across every indexed chunk. Unlike query(), which
        returns the top-k most semantically relevant chunks, this returns every page
        where the exact phrase literally appears — a plain substring search, not a
        similarity search."""
        if self.vectorstore is None:
            raise IndexNotBuiltError("Index has not been built or loaded yet")
        try:
            phrase_lower = phrase.lower()
            all_docs = self.vectorstore.get()
            pages = []
            for content, metadata in zip(all_docs["documents"], all_docs["metadatas"]):
                if phrase_lower in content.lower():
                    pages.append(metadata.get("page_label", metadata.get("page")))

            seen = set()
            unique_pages = []
            for p in pages:
                if p not in seen:
                    seen.add(p)
                    unique_pages.append(p)
            return unique_pages
        except Exception as e:
            logger.error(f"find_all_pages failed: {e}")
            raise QueryError(str(e)) from e

    def generate_answer(self, query: str, k: Optional[int] = None) -> str:
        docs = self.query(query, k)
        context = "\n\n".join(
            f"[Page {doc.metadata.get('page_label', doc.metadata.get('page', '?'))}]\n{doc.page_content}"
            for doc in docs
        )

        prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below. Each excerpt is labeled "
            "with its page number in the source document — cite the page number(s) in "
            "your answer when relevant. Note that the context only contains a handful of "
            "the most relevant excerpts, not the entire document, so if asked to find every "
            "occurrence of something, say you can only confirm what's in the context shown, "
            "not the whole document.\n\n"
            "If the document contains a mathematical formula relevant to the answer, "
            "reproduce it using LaTeX syntax — wrap inline formulas in single dollar "
            "signs (e.g. $x^2$) and standalone/block formulas in double dollar signs "
            "(e.g. $$PE_{{(pos,2i)}} = \\sin(pos/10000^{{2i/d_{{model}}}})$$).\n\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": query})
        return response.content