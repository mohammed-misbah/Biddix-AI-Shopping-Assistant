
import asyncio

from app.services.chat_service import chat_service
from app.services.conversation_service import (
    conversation_service,
)
from app.services.llm_service import llm_service
from app.services.product_service import product_service
from tests.test_chat_regression import FakeRedis


def test_complete_gemini_first_shopping_flow(
    monkeypatch,
):
    # Activate Gemini-first mode for this test only.
    monkeypatch.setenv("BIDDIX_GEMINI_FIRST", "1")

    # Use fake Redis so real customer sessions
    # are not changed by this test.
    monkeypatch.setattr(
        conversation_service,
        "redis",
        FakeRedis(),
    )

    catalog = product_service.products
    assert len(catalog) >= 5

    async def fake_interpret_request(
        user_message,
        conversation_history="",
        previous_products=None,
        selected_products=None,
        focused_product=None,
    ):
        text = user_message.lower()
        previous_products = previous_products or []

        result = {
            "intent": "catalog_list",
            "search_query": "",
            "count": 5,
            "filters": {},
            "reference_scope": "none",
            "referenced_product_ids": [],
            "comparison_count": None,
            "response_style": "normal",
        }

        if "second one" in text:
            result.update({
                "intent": "product_detail",
                "count": 1,
                "reference_scope": "previous_results",
                "referenced_product_ids": [
                    previous_products[1]["product_id"]
                ],
            })

        elif "silver" in text and "500" in text:
            result.update({
                "intent": "product_search",
                "search_query": "silver coins",
                "filters": {
                    "material": "silver",
                    "max_price": 499.99,
                },
            })

        return result

    async def fake_generate_response(
        user_message,
        products,
        mode,
        conversation_history="",
        response_style="normal",
    ):
        if not products:
            return "No matching products."

        return " | ".join(
            product["name"]
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

    # STEP 1: Customer requests five products.
    first = asyncio.run(
        chat_service.chat(
            message="List 5 products",
            request_id="final-test-1",
        )
    )

    assert len(first["products"]) == 5

    session_id = first["session_id"]

    # STEP 2: Customer changes the request.
    second = asyncio.run(
        chat_service.chat(
            message="Show me silver coins under AED 500",
            session_id=session_id,
            request_id="final-test-2",
        )
    )

    assert len(second["products"]) > 0

    for product in second["products"]:
        assert product["material"].lower() == "silver"
        assert float(product["price"]) < 500

    # STEP 3: Customer refers to a previous item.
    third = asyncio.run(
        chat_service.chat(
            message="Tell me about the second one",
            session_id=session_id,
            request_id="final-test-3",
        )
    )

    assert third["session_id"] == session_id
    assert len(third["products"]) == 1

    assert (
        third["products"][0]["product_id"]
        == second["products"][1]["product_id"]
    )

    # STEP 4: Check that Redis saved the session.
    assert conversation_service.session_exists(
        session_id
    )

    # STEP 5: Check saved product references.
    saved_ids = (
        conversation_service
        .get_last_result_product_ids(session_id)
    )

    assert (
        second["products"][1]["product_id"]
        in saved_ids
    )
