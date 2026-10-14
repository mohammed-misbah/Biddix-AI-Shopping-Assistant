from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import settings
from app.services.conversation_service import conversation_service


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # STARTUP

    logger.info(
        "Checking Redis/Memurai connection..."
    )

    redis_ok = conversation_service.ping()

    if not redis_ok:

        logger.critical(
            "Redis/Memurai is unavailable. "
            "Application startup aborted."
        )

        raise RuntimeError(
            "Redis/Memurai connection failed."
        )

    logger.info(
        "Redis/Memurai connection successful."
    )

    try:

        yield

    finally:

        # SHUTDOWN

        logger.info(
            "Closing Redis/Memurai connection..."
        )

        conversation_service.close()

        logger.info(
            "Redis/Memurai connection closed."
        )


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered product assistant for Biddix."
    ),
    lifespan=lifespan,
)


# =====================================================
# REQUEST ID + REQUEST TIMING
# =====================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = (
        request_id
    )

    start_time = (
        time.perf_counter()
    )

    logger.info(
        "REQUEST START | "
        "request_id=%s | "
        "method=%s | "
        "path=%s | "
        "client_ip=%s",
        request_id,
        request.method,
        request.url.path,
        (
            request.client.host
            if request.client
            else "unknown"
        ),
    )

    try:

        response = await call_next(
            request
        )

        total_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        logger.info(
            "REQUEST END | "
            "request_id=%s | "
            "method=%s | "
            "path=%s | "
            "status=%s | "
            "total_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            total_ms,
        )

        return response

    except Exception as exc:

        total_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )

        logger.exception(
            "REQUEST ERROR | "
            "request_id=%s | "
            "method=%s | "
            "path=%s | "
            "error_type=%s | "
            "total_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            type(exc).__name__,
            total_ms,
        )

        raise


# ROUTES

app.include_router(
    chat_router,
    prefix="/api/v1",
)


# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():

    return {
        "message":
            "Biddix AI Product Assistant API",

        "docs":
            "/docs",
    }