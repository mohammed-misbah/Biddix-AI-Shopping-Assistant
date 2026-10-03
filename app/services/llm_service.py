import json
import logging
from typing import Any

from google import genai
from google.genai import errors

from app.core.config import settings


logger = logging.getLogger(__name__)


class LLMService:
    MODEL = "gemini-3.8-flash"

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={
                "api_version": "v1",
            },
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

        prompt = f"""
You are the Biddix AI Product Assistant.

Use ONLY the products provided by the backend.

Rules:
- Never invent products.
- Never invent prices or stock.
- Mention all supplied matching products.
- Keep the answer short.
- Use simple bullet points.
- If no products are supplied, say no matching products were found.

Products:
{product_context}

Customer:
{user_message}
""".strip()

        try:
            interaction = await self.client.aio.interactions.create(
                model=self.MODEL,
                input=prompt,

                # IMPORTANT:
                # do not make the customer wait forever
                timeout=10,
            )

            answer = interaction.output_text

            if not answer:
                return (
                    "I found matching products, but I couldn't generate "
                    "a detailed answer right now."
                )

            return answer.strip()

        except errors.APITimeoutError:
            logger.warning("Gemini timed out.")

            return (
                "I found matching products, but the AI assistant is taking "
                "too long to respond. Please try again."
            )

        except errors.APIError as exc:
            logger.exception("Gemini API error")

            return (
                "I found matching products, but the AI assistant is "
                "temporarily unavailable."
            )

        except Exception:
            logger.exception("Unexpected Gemini error")

            return (
                "I found matching products, but I couldn't generate "
                "a detailed answer right now."
            )


llm_service = LLMService()