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

    # =====================================================
    # REQUEST ID
    # =====================================================

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    # =====================================================
    # CLIENT IP
    #
    # For local testing:
    # 127.0.0.1
    #
    # Later in production, if you are behind a trusted
    # reverse proxy, we can handle forwarded IPs properly.
    # =====================================================

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    try:

        # =================================================
        # RATE LIMIT CHECK
        #
        # Example:
        #
        # 20 requests
        # during
        # 60 seconds
        # per IP
        # =================================================

        (
            allowed,
            current_count,
            retry_after_seconds,
        ) = await redis_service.check_rate_limit(
            identifier=client_ip,
            max_requests=(
                settings.RATE_LIMIT_REQUESTS
            ),
            window_seconds=(
                settings.RATE_LIMIT_WINDOW_SECONDS
            ),
        )

        # =================================================
        # RATE LIMIT RESPONSE HEADERS
        #
        # These are useful for debugging and later
        # frontend/API clients.
        # =================================================

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

        # =================================================
        # BLOCK REQUEST
        #
        # IMPORTANT:
        # We stop here BEFORE chat_service and Gemini.
        # =================================================

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
                    "Retry-After":
                        str(
                            retry_after_seconds
                        ),
                    "X-RateLimit-Limit":
                        str(
                            settings
                            .RATE_LIMIT_REQUESTS
                        ),
                    "X-RateLimit-Remaining":
                        "0",
                },
            )

        # =================================================
        # OPTIONAL RATE-LIMIT LOG
        #
        # This lets you see the counter while testing.
        # =================================================

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

        # =================================================
        # NORMAL CHAT
        #
        # Nothing about your chat logic changes here.
        # =================================================

        result = await chat_service.chat(
            message=payload.message,
            session_id=payload.session_id,
            request_id=request_id,
        )

        return ChatResponse(
            **result
        )

    # =====================================================
    # IMPORTANT:
    #
    # Keep intentional HTTP errors such as 429.
    #
    # Without this block, the 429 could be converted
    # into a 500 by the generic exception handler.
    # =====================================================

    except HTTPException:
        raise

    # =====================================================
    # UNEXPECTED SERVER ERROR
    # =====================================================

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
                "Unable to process "
                "chat request."
            ),
        ) from exc