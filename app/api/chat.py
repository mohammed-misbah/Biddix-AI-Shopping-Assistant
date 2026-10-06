import logging

from fastapi import APIRouter, HTTPException

from schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        result = await chat_service.chat(
            message=request.message,
            session_id=request.session_id,
        )

        return ChatResponse(**result)

    except Exception as exc:
        logger.exception(
            "Chat endpoint failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process chat request.",
        ) from exc