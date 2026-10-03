from fastapi import APIRouter, HTTPException
from google.genai.errors import ServerError

from schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service


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

    except ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily busy. Please try again in a moment.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc