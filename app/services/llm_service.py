import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import (
    LLMServiceError,
    LLMTimeoutError,
)
from app.core.prompts import (
    build_assistant_prompt,
)


logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self):

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,

                read=float(
                    settings.HARD_TIMEOUT_SECONDS
                ),

                write=10.0,

                pool=5.0,
            )
        )

    async def generate_response(
        self,
        user_message: str,
        products: list[
            dict[str, Any]
        ],
        mode: str,
        conversation_history: str = "",
        response_style: str = "normal",
    ) -> str:

        # =================================================
        # PREPARE SAFE PRODUCT DATA FOR GEMINI
        # =================================================

        safe_products = (
            self._prepare_products_for_llm(
                products
            )
        )

        product_context = json.dumps(
            safe_products,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # =================================================
        # BUILD PROMPT
        # =================================================

        prompt = build_assistant_prompt(
            user_message=user_message,
            mode=mode,
            product_context=product_context,
            product_count=len(
                safe_products
            ),
            conversation_history=(
                conversation_history
            ),
            response_style=(
                response_style
            ),
        )

        # =================================================
        # GEMINI URL
        # =================================================

        url = (
            f"{settings.API_BASE_URL}/"
            f"{settings.GEMINI_MODEL}"
            ":generateContent"
        )

        # =================================================
        # GEMINI REQUEST BODY
        # =================================================

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
                    "thinkingLevel":
                        "MINIMAL"
                }
            },
        }

        # =================================================
        # GEMINI HEADERS
        #
        # API key stays server-side.
        # It is never returned to the frontend.
        # =================================================

        headers = {
            "Content-Type":
                "application/json",

            "x-goog-api-key":
                settings.GEMINI_API_KEY,
        }

        try:

            # =============================================
            # CALL GEMINI
            # =============================================

            response = await self.client.post(
                url,
                headers=headers,
                json=payload,
            )

            # =============================================
            # HTTP ERRORS
            #
            # Examples:
            # 400
            # 401
            # 403
            # 429
            # 500
            # =============================================

            response.raise_for_status()

            data = response.json()

            # =============================================
            # READ CANDIDATES
            # =============================================

            candidates = data.get(
                "candidates",
                [],
            )

            if not candidates:

                logger.error(
                    "Gemini returned no candidates."
                )

                raise LLMServiceError(
                    "Gemini returned no candidates."
                )

            candidate = candidates[0]

            parts = (
                candidate
                .get(
                    "content",
                    {},
                )
                .get(
                    "parts",
                    [],
                )
            )

            # =============================================
            # BUILD TEXT ANSWER
            # =============================================

            answer = "".join(
                part.get(
                    "text",
                    "",
                )

                for part in parts

                if part.get(
                    "text"
                )
            ).strip()

            if not answer:

                logger.error(
                    "Gemini returned empty output."
                )

                raise LLMServiceError(
                    "Gemini returned empty output."
                )

            return answer

        # =================================================
        # GEMINI TIMEOUT
        # =================================================

        except httpx.TimeoutException as exc:

            logger.error(
                "Gemini timed out after %s seconds.",
                settings.HARD_TIMEOUT_SECONDS,
            )

            raise LLMTimeoutError(
                "Gemini request timed out."
            ) from exc

        # =================================================
        # GEMINI HTTP ERROR
        # =================================================

        except httpx.HTTPStatusError as exc:

            logger.error(
                "Gemini HTTP error | status=%s",
                exc.response.status_code,
            )

            raise LLMServiceError(
                "Gemini returned an HTTP error."
            ) from exc

        # =================================================
        # OUR OWN EXPECTED SERVICE ERRORS
        #
        # Important:
        # do not convert these into another exception.
        # Just send them upward to chat.py.
        # =================================================

        except (
            LLMTimeoutError,
            LLMServiceError,
        ):

            raise

        # =================================================
        # UNKNOWN GEMINI ERROR
        # =================================================

        except Exception as exc:

            logger.exception(
                "Unexpected Gemini error | "
                "error_type=%s",
                type(exc).__name__,
            )

            raise LLMServiceError(
                "Unexpected Gemini failure."
            ) from exc

    # =====================================================
    # PRODUCT SAFETY FOR GEMINI
    # =====================================================

    @staticmethod
    def _prepare_products_for_llm(
        products: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        safe_products = []

        for product in products:

            safe = dict(
                product
            )

            confirmed = (
                product.get(
                    "availability_confirmed"
                )
                is True
            )

            if confirmed:

                try:

                    stock = int(
                        product.get(
                            "stock",
                            0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    stock = 0

                safe[
                    "availability_status"
                ] = (
                    "in_stock"
                    if stock > 0
                    else "out_of_stock"
                )

                safe[
                    "stock"
                ] = stock

            else:

                safe.pop(
                    "stock",
                    None,
                )

                safe[
                    "availability_status"
                ] = "unconfirmed"

            safe_products.append(
                safe
            )

        return safe_products


llm_service = LLMService()