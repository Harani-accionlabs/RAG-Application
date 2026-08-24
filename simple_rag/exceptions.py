class RAGError(Exception):
    """Base exception for all RAG-related errors."""


class IndexNotBuiltError(RAGError):
    """Raised when a query is attempted before the index has been built or loaded."""


class IndexBuildError(RAGError):
    """Raised when building the vector index fails."""


class QueryError(RAGError):
    """Raised when a query against the vector store fails."""