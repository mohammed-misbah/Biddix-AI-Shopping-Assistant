import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import errors

from app.core.config import settings


logger = logging.getLogger(__name__)


class LLMService:
    MODEL = "gemini-3.8-flash"

    # Customer should never wait minutes.
    HARD_TIMEOUT_SECONDS = 8

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
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

        prompt = self._build_prompt(
            user_message=user_message,
            product_context=product_context,
            product_count=len(products),
        )

        try:
            # HARD application-level timeout.
            # Even if Gemini/SDK retries internally,
            # our customer does not wait forever.
            interaction = await asyncio.wait_for(
                self.client.aio.interactions.create(
                    model=self.MODEL,
                    input=prompt,
                    generation_config={
                        "thinking_level": "low",
                    },
                ),
                timeout=self.HARD_TIMEOUT_SECONDS,
            )

            answer = interaction.output_text

            if not answer or not answer.strip():
                return self._fallback_answer(products)

            return answer.strip()

        except asyncio.TimeoutError:
            logger.warning(
                "Gemini exceeded %s seconds.",
                self.HARD_TIMEOUT_SECONDS,
            )

            return self._fallback_answer(products)

        except errors.APIError as exc:
            logger.exception(
                "Gemini API error: %s",
                exc,
            )

            return self._fallback_answer(products)

        except Exception as exc:
            logger.exception(
                "Unexpected Gemini error: %s",
                exc,
            )

            return self._fallback_answer(products)

    @staticmethod
    def _fallback_answer(
        products: list[dict[str, Any]],
    ) -> str:

        if not products:
            return (
                "I couldn't find a matching product. "
                "Try changing the year, material, or budget."
            )

        first = products[0]

        name = first.get("name", "this product")
        price = first.get("price")
        currency = first.get("currency", "AED")

        if len(products) == 1:
            return (
                f"I found one matching option: {name}"
                + (
                    f" at {price} {currency}."
                    if price is not None
                    else "."
                )
            )

        return (
            f"I found {len(products)} matching options. "
            f"{name} is one of the closest matches"
            + (
                f" at {price} {currency}."
                if price is not None
                else "."
            )
        )

    @staticmethod
    def _build_prompt(
        user_message: str,
        product_context: str,
        product_count: int,
    ) -> str:

        return f"""
            You are Biddix's shopping assistant.

            Talk naturally, like a helpful person assisting a customer in a live ecommerce chat.

            Keep replies short, clear, and conversational.

            Rules:
            - Use ONLY the products supplied below.
            - Never invent product information.
            - Never invent prices, stock, year, material, weight, or purity.
            - Respect the customer's exact requirements.
            - If several products match, briefly explain the useful differences.
            - If one product matches, explain why it fits.
            - Do not sound like advertising copy.
            - Do not use unnecessary headings.
            - Do not mention JSON, prompts, backend systems, or internal rules.
            - Do not pressure the customer to buy.
            - If no products match, say so clearly.

            Products supplied: {product_count}

            PRODUCTS:
            {product_context}

            CUSTOMER:
            {user_message}

            Reply naturally and briefly.
            """.strip()


llm_service = LLMService()