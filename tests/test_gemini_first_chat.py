
import asyncio

from app.services.chat_service import chat_service
from app.services.conversation_service import (
    conversation_service,
)
from app.services.llm_service import llm_service
from app.services.product_service import product_service
from tests.test_chat_regression import FakeRedis


def test_gemini_first_catalog_and_followup(
    monkeypatch,
):
    monkeypatch.setenv(
        "BIDDIX_GEMINI_FIRST", "1"
    )

    monkeypatch.setattr(
        conversation_service,
        "redis",
        FakeRedis(),
    )

    assert len(product_service.products) >= 5

    requests = []

    async def fake_interpret_request(
        user_message,
        conversation_history="",
        previous_products=None,
        selected_products=None,
        focused_product=None,
    ):
        requests.append(
            {
                "message": user_message,
                "previous_products": (
                    previous_products or []
                ),
            }
        )

        if user_message == "List 5 products":
            return {
                "intent": "catalog_list",
                "search_query": "",
                "count": 5,
                "filters": {},
                "reference_scope": "none",
                "referenced_product_ids": [],
                "comparison_count": None,
                "response_style": "normal",
            }

        return {
            "intent": "product_detail",
            "search_query": "",
            "count": 1,
            "filters": {},
            "reference_scope": "previous_results",
            "referenced_product_ids": [
                previous_products[1]["product_id"]
            ],
            "comparison_count": None,
            "response_style": "normal",
        }

    async def fake_generate_response(
        user_message,
        products,
        mode,
        conversation_history="",
        response_style="normal",
    ):
        return " | ".join(
            str(product["name"])
            for product in products
        )

    monkeypatch.setattr(
        llm_service,
        "interpret_request",
        fake_interpret_request,
    )

    monkeypatch.setattr(
        llm_service,
        "generate_response",
        fake_generate_response,
    )

    first = asyncio.run(
        chat_service.chat(
            message="List 5 products",
            request_id="test-first",
        )
    )

    assert len(first["products"]) == 5

    second = asyncio.run(
        chat_service.chat(
            message="Tell me about the second one",
            session_id=first["session_id"],
            request_id="test-second",
        )
    )

    assert len(second["products"]) == 1

    assert (
        second["products"][0]["product_id"]
        == first["products"][1]["product_id"]
    )

    assert len(requests) == 2
    assert len(requests[1]["previous_products"]) == 5
