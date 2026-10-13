from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import settings
from app.services.conversation_service import conversation_service


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