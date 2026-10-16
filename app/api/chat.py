import logging

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
)

from schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.core.config import settings

from app.core.exceptions import (
    LLMServiceError,
    LLMTimeoutError,
    RedisServiceError,
    RetrievalServiceError,
)

from app.services.chat_service import (
    chat_service,
)

from app.services.redis_service import (
    redis_service,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: Request,
    payload: ChatRequest,
    response: Response,
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    try:

        (
            allowed,
            current_count,
            retry_after_seconds,
        ) = await redis_service.check_rate_limit(
            identifier=client_ip,
            max_requests=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )

        response.headers[
            "X-RateLimit-Limit"
        ] = str(
            settings.RATE_LIMIT_REQUESTS
        )

        remaining = max(
            settings.RATE_LIMIT_REQUESTS
            - current_count,
            0,
        )

        response.headers[
            "X-RateLimit-Remaining"
        ] = str(
            remaining
        )

        if not allowed:

            logger.warning(
                "RATE LIMIT BLOCKED | "
                "request_id=%s | "
                "client_ip=%s | "
                "count=%s | "
                "limit=%s | "
                "retry_after=%s",
                request_id,
                client_ip,
                current_count,
                settings.RATE_LIMIT_REQUESTS,
                retry_after_seconds,
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many chat requests. "
                    "Please try again shortly."
                ),
                headers={
                    "Retry-After": str(
                        retry_after_seconds
                    ),
                    "X-RateLimit-Limit": str(
                        settings.RATE_LIMIT_REQUESTS
                    ),
                    "X-RateLimit-Remaining": "0",
                },
            )

        logger.info(
            "RATE LIMIT OK | "
            "request_id=%s | "
            "client_ip=%s | "
            "count=%s | "
            "limit=%s | "
            "remaining=%s",
            request_id,
            client_ip,
            current_count,
            settings.RATE_LIMIT_REQUESTS,
            remaining,
        )

        result = await chat_service.chat(
            message=payload.message,
            session_id=payload.session_id,
            request_id=request_id,
        )

        return ChatResponse(
            **result
        )

    except LLMTimeoutError:

        logger.warning(
            "LLM TIMEOUT | "
            "request_id=%s | "
            "client_ip=%s",
            request_id,
            client_ip,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is taking too long "
                "to respond. Please try again shortly."
            ),
        )

    except LLMServiceError:

        logger.error(
            "LLM SERVICE ERROR | "
            "request_id=%s | "
            "client_ip=%s",
            request_id,
            client_ip,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again shortly."
            ),
        )

    except RedisServiceError:

        logger.error(
            "REDIS SERVICE ERROR | "
            "request_id=%s | "
            "client_ip=%s",
            request_id,
            client_ip,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The chat service is temporarily unavailable. "
                "Please try again shortly."
            ),
        )

    except RetrievalServiceError:

        logger.error(
            "RETRIEVAL SERVICE ERROR | "
            "request_id=%s | "
            "client_ip=%s",
            request_id,
            client_ip,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Product search is temporarily unavailable. "
                "Please try again shortly."
            ),
        )

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "CHAT ENDPOINT ERROR | "
            "request_id=%s | "
            "client_ip=%s | "
            "error_type=%s | "
            "error=%s",
            request_id,
            client_ip,
            type(exc).__name__,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process chat request."
            ),
        ) from exc