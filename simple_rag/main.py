import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .exceptions import IndexBuildError, IndexNotBuiltError, QueryError
from .rag import RAG

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=settings.log_level,
)
logger = logging.getLogger(__name__)

rag_service = RAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    if not rag_service.load_existing_index():
        logger.warning("No existing index found. Call POST /index-pdf before querying.")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Simple RAG Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = None


class QueryResult(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    results: List[QueryResult]


class AskResponse(BaseModel):
    answer: str


class IndexPdfResponse(BaseModel):
    status: str
    chunks_indexed: int


class FindResponse(BaseModel):
    phrase: str
    pages_found: List[str]


@app.post("/index-pdf", response_model=IndexPdfResponse, status_code=201)
async def index_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        chunks = rag_service.build_index_from_pdf(tmp_path)
        return IndexPdfResponse(status="success", chunks_indexed=chunks)
    except IndexBuildError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        results = rag_service.query(request.query, request.k)
        return QueryResponse(
            results=[
                QueryResult(content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]
        )
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except QueryError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask(request: QueryRequest):
    try:
        answer = rag_service.generate_answer(request.query, request.k)
        return AskResponse(answer=answer)
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except QueryError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/find", response_model=FindResponse)
async def find_pages(phrase: str):
    """Exhaustive keyword search — returns every page where the exact phrase appears,
    unlike /query or /ask which only return the most semantically relevant matches."""
    try:
        pages = rag_service.find_all_pages(phrase)
        return FindResponse(phrase=phrase, pages_found=[str(p) for p in pages])
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except QueryError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": rag_service.vectorstore is not None}