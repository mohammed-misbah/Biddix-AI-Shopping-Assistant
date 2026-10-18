from fastapi.testclient import TestClient

from app.main import app
from app.services.conversation_service import (
    conversation_service,
)
from app.services.llm_service import (
    llm_service,
)
from app.services.redis_service import (
    redis_service,
)
from app.services.retrieval_service import (
    retrieval_service,
)


TEST_PRODUCTS = [
    {
        "product_id": "integration-silver-1",
        "name": "Integration Silver Coin One",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "10 grams",
        "purity": "99.9%",
        "price": 200.0,
        "currency": "AED",
        "description": "Integration test product",
        "product_url": None,
    },
    {
        "product_id": "integration-silver-2",
        "name": "Integration Silver Coin Two",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "20 grams",
        "purity": "99.9%",
        "price": 300.0,
        "currency": "AED",
        "description": "Integration test product",
        "product_url": None,
    },
]


PRODUCT_MAP = {
    product["product_id"]: product
    for product in TEST_PRODUCTS
}


def test_fastapi_with_real_redis(
    monkeypatch,
):

    # =====================================================
    # 1. PROVE REAL CONVERSATION REDIS IS AVAILABLE
    # =====================================================

    assert conversation_service.ping() is True

    # =====================================================
    # 2. BYPASS RATE LIMITER ONLY
    #
    # WHY?
    #
    # This test is specifically testing:
    #
    # FastAPI
    #   +
    # REAL conversation/session Redis
    #
    # The rate limiter has its own Redis path and can stop
    # the request before the conversation service is reached.
    #
    # We do NOT mock conversation_service Redis.
    # =====================================================

    async def fake_check_rate_limit(
        identifier,
        max_requests,
        window_seconds,
    ):
        return (
            True,
            1,
            window_seconds,
        )

    monkeypatch.setattr(
        redis_service,
        "check_rate_limit",
        fake_check_rate_limit,
    )

    # =====================================================
    # 3. MOCK GEMINI
    #
    # We are not testing Google's API here.
    # =====================================================

    async def fake_generate_response(
        user_message,
        products,
        mode,
        conversation_history="",
        response_style=None,
    ):

        if products:
            return " | ".join(
                product["name"]
                for product in products
            )

        return "No products."

    monkeypatch.setattr(
        llm_service,
        "generate_response",
        fake_generate_response,
    )

    # =====================================================
    # 4. MOCK RETRIEVAL
    #
    # Real FAISS was already tested separately.
    #
    # This keeps this test focused on:
    #
    # FastAPI + real Redis session memory.
    # =====================================================

    def fake_retrieve(
        query,
        filters=None,
        limit=5,
        strict_match=False,
    ):

        return [
            dict(product)
            for product
            in TEST_PRODUCTS[:limit]
        ]

    monkeypatch.setattr(
        retrieval_service,
        "retrieve",
        fake_retrieve,
    )

    # =====================================================
    # 5. MOCK PRODUCT LOOKUP
    #
    # The second request reuses product IDs that were saved
    # in the REAL Redis conversation session.
    # =====================================================

    def fake_get_products_by_ids(
        product_ids,
    ):

        return [
            PRODUCT_MAP[product_id]
            for product_id
            in product_ids
            if product_id in PRODUCT_MAP
        ]

    monkeypatch.setattr(
        retrieval_service,
        "get_products_by_ids",
        fake_get_products_by_ids,
    )

    # =====================================================
    # 6. START REAL FASTAPI TEST CLIENT
    # =====================================================

    session_id = None

    try:

        with TestClient(app) as client:

            # =============================================
            # REQUEST 1
            #
            # Create a real conversation/session.
            # =============================================

            response1 = client.post(
                "/api/v1/chat",
                json={
                    "message": (
                        "Find silver under AED 500"
                    )
                },
            )

            assert response1.status_code == 200

            data1 = response1.json()

            session_id = data1["session_id"]

            assert session_id

            assert len(
                data1["products"]
            ) == 2

            # =============================================
            # CONFIRM SESSION REALLY EXISTS IN MEMURAI
            # =============================================

            assert (
                conversation_service
                .session_exists(
                    session_id
                )
                is True
            )

            # =============================================
            # CONFIRM REAL REDIS STORED PRODUCT IDS
            # =============================================

            stored_product_ids = (
                conversation_service
                .get_last_result_product_ids(
                    session_id
                )
            )

            assert set(
                stored_product_ids
            ) == {
                "integration-silver-1",
                "integration-silver-2",
            }

            # =============================================
            # REQUEST 2
            #
            # Use the SAME session.
            #
            # FastAPI must load previous data from real
            # Redis and remember the previous products.
            # =============================================

            response2 = client.post(
                "/api/v1/chat",
                json={
                    "message": (
                        "Which is cheapest among those?"
                    ),
                    "session_id": session_id,
                },
            )

            assert response2.status_code == 200

            data2 = response2.json()

            assert (
                data2["session_id"]
                == session_id
            )

            assert len(
                data2["products"]
            ) == 1

            assert (
                data2["products"][0][
                    "product_id"
                ]
                == "integration-silver-1"
            )

            # =============================================
            # CONFIRM FOCUS WAS SAVED BACK INTO REAL REDIS
            # =============================================

            focused_product_id = (
                conversation_service
                .get_focused_product_id(
                    session_id
                )
            )

            assert (
                focused_product_id
                == "integration-silver-1"
            )

    finally:

        # =================================================
        # CLEAN UP TEST SESSION FROM REAL REDIS
        # =================================================

        if session_id:

            conversation_service.delete_session(
                session_id
            )