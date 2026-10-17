import asyncio
import time

from app.services.chat_service import chat_service
from app.services.conversation_service import conversation_service
from app.services.llm_service import llm_service
from app.services.retrieval_service import retrieval_service


class FakeRedis:

    def __init__(self):
        self.data = {}
        self.expiry = {}

    def get(self, key):
        self._remove_if_expired(key)
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

        if ex is not None:
            self.expiry[key] = time.time() + ex
        else:
            self.expiry.pop(key, None)

        return True

    def delete(self, key):
        existed = key in self.data

        self.data.pop(key, None)
        self.expiry.pop(key, None)

        return 1 if existed else 0

    def exists(self, key):
        self._remove_if_expired(key)
        return 1 if key in self.data else 0

    def ttl(self, key):
        self._remove_if_expired(key)

        if key not in self.data:
            return -2

        if key not in self.expiry:
            return -1

        remaining = int(
            self.expiry[key] - time.time()
        )

        return max(remaining, 0)

    def ping(self):
        return True

    def close(self):
        return None

    def _remove_if_expired(self, key):
        expires_at = self.expiry.get(key)

        if (
            expires_at is not None
            and time.time() >= expires_at
        ):
            self.data.pop(key, None)
            self.expiry.pop(key, None)


SILVER_PRODUCTS = [
    {
        "product_id": "silver-1",
        "name": "Silver Coin One",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "10 grams",
        "purity": "99.9%",
        "price": 200.0,
        "currency": "AED",
        "description": "Silver test coin one",
        "product_url": None,
    },
    {
        "product_id": "silver-2",
        "name": "Silver Coin Two",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "20 grams",
        "purity": "99.9%",
        "price": 300.0,
        "currency": "AED",
        "description": "Silver test coin two",
        "product_url": None,
    },
]


GOLD_PRODUCTS = [
    {
        "product_id": "gold-1",
        "name": "Gold Coin One",
        "category": "coin",
        "year": 2024,
        "material": "Gold",
        "weight": "5 grams",
        "purity": "99.99%",
        "price": 1500.0,
        "currency": "AED",
        "description": "Gold test coin one",
        "product_url": None,
    },
    {
        "product_id": "gold-2",
        "name": "Gold Coin Two",
        "category": "coin",
        "year": 2024,
        "material": "Gold",
        "weight": "10 grams",
        "purity": "99.99%",
        "price": 2500.0,
        "currency": "AED",
        "description": "Gold test coin two",
        "product_url": None,
    },
]


ALL_PRODUCTS = {
    product["product_id"]: product
    for product in (
        SILVER_PRODUCTS
        + GOLD_PRODUCTS
    )
}


def run_chat(
    message,
    session_id=None,
):
    return asyncio.run(
        chat_service.chat(
            message=message,
            session_id=session_id,
            request_id="pytest-session-isolation",
        )
    )


def test_sessions_do_not_leak(
    monkeypatch,
):

    fake_redis = FakeRedis()

    monkeypatch.setattr(
        conversation_service,
        "redis",
        fake_redis,
    )

    def fake_retrieve(
        query,
        filters=None,
        limit=5,
        strict_match=False,
    ):

        text = query.lower()

        if "gold" in text:
            return [
                dict(product)
                for product
                in GOLD_PRODUCTS[:limit]
            ]

        return [
            dict(product)
            for product
            in SILVER_PRODUCTS[:limit]
        ]

    def fake_get_products_by_ids(
        product_ids,
    ):

        return [
            ALL_PRODUCTS[product_id]
            for product_id
            in product_ids
            if product_id in ALL_PRODUCTS
        ]

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
                for product
                in products
            )

        return "No products."

    monkeypatch.setattr(
        retrieval_service,
        "retrieve",
        fake_retrieve,
    )

    monkeypatch.setattr(
        retrieval_service,
        "get_products_by_ids",
        fake_get_products_by_ids,
    )

    monkeypatch.setattr(
        llm_service,
        "generate_response",
        fake_generate_response,
    )

    # -----------------------------------------
    # USER A
    # -----------------------------------------

    result_a1 = run_chat(
        "Find silver coins"
    )

    session_a = result_a1["session_id"]

    assert {
        product["product_id"]
        for product
        in result_a1["products"]
    } == {
        "silver-1",
        "silver-2",
    }

    # -----------------------------------------
    # USER B
    # -----------------------------------------

    result_b1 = run_chat(
        "Find gold coins"
    )

    session_b = result_b1["session_id"]

    assert session_a != session_b

    assert {
        product["product_id"]
        for product
        in result_b1["products"]
    } == {
        "gold-1",
        "gold-2",
    }

    # -----------------------------------------
    # CHECK STORED RESULTS ARE SEPARATE
    # -----------------------------------------

    a_last_results = (
        conversation_service
        .get_last_result_product_ids(
            session_a
        )
    )

    b_last_results = (
        conversation_service
        .get_last_result_product_ids(
            session_b
        )
    )

    assert set(
        a_last_results
    ) == {
        "silver-1",
        "silver-2",
    }

    assert set(
        b_last_results
    ) == {
        "gold-1",
        "gold-2",
    }

    # -----------------------------------------
    # USER A FOLLOWS UP
    # -----------------------------------------

    result_a2 = run_chat(
        "Which is cheapest among those?",
        session_a,
    )

    assert len(
        result_a2["products"]
    ) == 1

    assert (
        result_a2["products"][0][
            "product_id"
        ]
        == "silver-1"
    )

    # -----------------------------------------
    # USER B FOLLOWS UP
    # -----------------------------------------

    result_b2 = run_chat(
        "Which is cheapest among those?",
        session_b,
    )

    assert len(
        result_b2["products"]
    ) == 1

    assert (
        result_b2["products"][0][
            "product_id"
        ]
        == "gold-1"
    )

    # -----------------------------------------
    # CHECK FOCUS IS SEPARATE
    # -----------------------------------------

    a_focus = (
        conversation_service
        .get_focused_product_id(
            session_a
        )
    )

    b_focus = (
        conversation_service
        .get_focused_product_id(
            session_b
        )
    )

    assert a_focus == "silver-1"
    assert b_focus == "gold-1"

    # -----------------------------------------
    # CHECK HISTORY IS SEPARATE
    # -----------------------------------------

    a_history = (
        conversation_service
        .get_history(
            session_a
        )
    )

    b_history = (
        conversation_service
        .get_history(
            session_b
        )
    )

    a_text = " ".join(
        turn.content.lower()
        for turn in a_history
    )

    b_text = " ".join(
        turn.content.lower()
        for turn in b_history
    )

    assert "gold coins" not in a_text
    assert "silver coins" not in b_text