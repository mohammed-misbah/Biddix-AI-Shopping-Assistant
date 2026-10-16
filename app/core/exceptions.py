class AppServiceError(Exception):
    """
    Base class for expected backend service errors.
    """

    pass


class LLMTimeoutError(AppServiceError):
    """
    Gemini took too long to respond.
    """

    pass


class LLMServiceError(AppServiceError):
    """
    Gemini failed for a reason other than timeout.
    """

    pass


class RetrievalServiceError(AppServiceError):
    """
    Product retrieval / FAISS / embedding failed.
    """

    pass


class RedisServiceError(AppServiceError):
    """
    Redis/Memurai operation failed.
    """

    pass