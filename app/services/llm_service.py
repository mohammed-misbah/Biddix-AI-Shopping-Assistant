import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.prompts import build_product_assistant_prompt


logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=float(settings.HARD_TIMEOUT_SECONDS),
                write=10.0,
                pool=5.0,
            )
        )

    async def generate_response(
        self,
        user_message: str,
        products: list[dict[str, Any]],
    ) -> str:

        product_context = json.dumps(
            products,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        prompt = build_product_assistant_prompt(
            user_message=user_message,
            product_context=product_context,
            product_count=len(products),
        )

        url = (
            f"{settings.API_BASE_URL}/"
            f"{settings.GEMINI_MODEL}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ],
                }
            ],

            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "MINIMAL"
                }
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.GEMINI_API_KEY,
        }

        try:
            response = await self.client.post(
                url,
                headers=headers,
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

            candidates = data.get("candidates", [])

            if not candidates:
                logger.error(
                    "Gemini returned no candidates: %s",
                    data,
                )
                return self._fallback()

            candidate = candidates[0]

            # VERY IMPORTANT FOR DEBUGGING
            finish_reason = candidate.get(
                "finishReason"
            )

            usage = data.get(
                "usageMetadata",
                {}
            )

            logger.info(
                "Gemini finish_reason=%s usage=%s",
                finish_reason,
                usage,
            )

            parts = (
                candidate
                .get("content", {})
                .get("parts", [])
            )

            answer = "".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            ).strip()

            if not answer:
                logger.error(
                    "Gemini returned empty answer. "
                    "finish_reason=%s data=%s",
                    finish_reason,
                    data,
                )

                return self._fallback()

            return answer

        except httpx.TimeoutException:
            logger.error(
                "Gemini timed out after %s seconds.",
                settings.HARD_TIMEOUT_SECONDS,
            )

            return self._fallback()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Gemini HTTP %s: %s",
                exc.response.status_code,
                exc.response.text,
            )

            return self._fallback()

        except Exception as exc:
            logger.exception(
                "Unexpected Gemini error: %s",
                exc,
            )

            return self._fallback()

    @staticmethod
    def _fallback() -> str:
        return (
            "I'm having trouble preparing the full recommendation "
            "right now. Please try again in a moment."
        )


llm_service = LLMService()