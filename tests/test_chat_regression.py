import asyncio
import json
import time

import pytest

from app.services.chat_service import chat_service
from app.services.conversation_service import conversation_service
from app.services.llm_service import llm_service
from app.services.retrieval_service import retrieval_service



# FAKE REDIS
#
# This lets the regression test use conversation memory
# without requiring real Memurai.


class FakeRedis:

    def __init__(self):
        self.data = {}
        self.expiry = {}

    def get(self, key):
        self._remove_if_expired(key)
        return self.data.get(key)

    def set(
        self,
        key,
        value,
        ex=None,
    ):
        self.data[key] = value

        if ex is not None:
            self.expiry[key] = (
                time.time()
                + ex
            )
        else:
            self.expiry.pop(
                key,
                None,
            )

        return True

    def delete(self, key):
        existed = (
            key in self.data
        )

        self.data.pop(
            key,
            None,
        )

        self.expiry.pop(
            key,
            None,
        )

        return 1 if existed else 0

    def exists(self, key):
        self._remove_if_expired(
            key
        )

        return (
            1
            if key in self.data
            else 0
        )

    def ttl(self, key):
        self._remove_if_expired(
            key
        )

        if key not in self.data:
            return -2

        if key not in self.expiry:
            return -1

        remaining = int(
            self.expiry[key]
            - time.time()
        )

        return max(
            remaining,
            0,
        )

    def ping(self):
        return True

    def close(self):
        return None

    def _remove_if_expired(
        self,
        key,
    ):
        expires_at = (
            self.expiry.get(
                key
            )
        )

        if (
            expires_at is not None
            and time.time()
            >= expires_at
        ):
            self.data.pop(
                key,
                None,
            )

            self.expiry.pop(
                key,
                None,
            )


# TEST PRODUCTS


ARABIAN_SANDS = {
    "product_id": "arabian-sands",
    "name": "2024 Arabian Sands Silver Coin",
    "category": "coin",
    "year": 2024,
    "material": "Silver",
    "weight": "10 grams",
    "purity": "99.9%",
    "price": 175.0,
    "currency": "AED",
    "description": "Arabian Sands test product",
    "product_url": None,
}


DESERT_MOON = {
    "product_id": "desert-moon",
    "name": "2023 Desert Moon Silver Coin",
    "category": "coin",
    "year": 2023,
    "material": "Silver",
    "weight": "15 grams",
    "purity": "99.9%",
    "price": 245.0,
    "currency": "AED",
    "description": "Desert Moon test product",
    "product_url": None,
}


RAMADAN_CRESCENT = {
    "product_id": "ramadan-crescent",
    "name": "2024 Ramadan Crescent Silver Coin",
    "category": "coin",
    "year": 2024,
    "material": "Silver",
    "weight": "18 grams",
    "purity": "99.9%",
    "price": 310.0,
    "currency": "AED",
    "description": "Ramadan Crescent test product",
    "product_url": None,
}


SILVER_EAGLE = {
    "product_id": "silver-eagle",
    "name": "2024 Silver Eagle Coin",
    "category": "coin",
    "year": 2024,
    "material": "Silver",
    "weight": "31.1 grams",
    "purity": "99.9%",
    "price": 350.0,
    "currency": "AED",
    "description": "Silver Eagle test product",
    "product_url": None,
}


UAE_HERITAGE = {
    "product_id": "uae-heritage",
    "name": "2023 UAE Heritage Silver Coin",
    "category": "coin",
    "year": 2023,
    "material": "Silver",
    "weight": "25 grams",
    "purity": "99.9%",
    "price": 425.0,
    "currency": "AED",
    "description": "UAE Heritage test product",
    "product_url": None,
}


LUNAR_DRAGON = {
    "product_id": "lunar-dragon",
    "name": "2024 Lunar Dragon Silver Coin",
    "category": "coin",
    "year": 2024,
    "material": "Silver",
    "weight": "31.1 grams",
    "purity": "99.9%",
    "price": 475.0,
    "currency": "AED",
    "description": "Lunar Dragon test product",
    "product_url": None,
}


ALL_PRODUCTS = {
    product["product_id"]: product

    for product in [
        ARABIAN_SANDS,
        DESERT_MOON,
        RAMADAN_CRESCENT,
        SILVER_EAGLE,
        UAE_HERITAGE,
        LUNAR_DRAGON,
    ]
}


FRESH_RESULTS = [
    DESERT_MOON,
    RAMADAN_CRESCENT,
    SILVER_EAGLE,
    UAE_HERITAGE,
    LUNAR_DRAGON,
]


REFINED_RESULTS = [
    ARABIAN_SANDS,
    SILVER_EAGLE,
    LUNAR_DRAGON,
    DESERT_MOON,
    RAMADAN_CRESCENT,
]


# HELPERS


def run_chat(
    message,
    session_id=None,
):
    return asyncio.run(
        chat_service.chat(
            message=message,
            session_id=session_id,
            request_id="pytest-regression",
        )
    )


# MAIN REGRESSION TEST


def test_complete_multi_turn_chat_flow(
    monkeypatch,
):

    # -----------------------------------------------------
    # ISOLATED REDIS
    # -----------------------------------------------------

    fake_redis = FakeRedis()

    monkeypatch.setattr(
        conversation_service,
        "redis",
        fake_redis,
    )

    # -----------------------------------------------------
    # FAKE PRODUCT LOOKUP
    #
    # This keeps FAISS/model behaviour outside this test.
    # We are testing chat state and routing here.
    # -----------------------------------------------------

    def fake_get_products_by_ids(
        product_ids,
    ):

        return [
            ALL_PRODUCTS[product_id]
            for product_id
            in product_ids
            if product_id
            in ALL_PRODUCTS
        ]

    monkeypatch.setattr(
        retrieval_service,
        "get_products_by_ids",
        fake_get_products_by_ids,
    )

    # -----------------------------------------------------
    # FAKE RETRIEVAL
    #
    # Fresh search:
    # returns the normal original five.
    #
    # Refinement:
    # includes Arabian Sands.
    # -----------------------------------------------------

    def fake_retrieve(
        query,
        filters=None,
        limit=5,
        strict_match=False,
    ):

        query_lower = (
            query.lower()
        )

        if (
            "only silver ones"
            in query_lower
        ):
            return [
                dict(product)
                for product
                in REFINED_RESULTS[:limit]
            ]

        return [
            dict(product)
            for product
            in FRESH_RESULTS[:limit]
        ]

    monkeypatch.setattr(
        retrieval_service,
        "retrieve",
        fake_retrieve,
    )

    # -----------------------------------------------------
    # FAKE GEMINI
    #
    # Gemini is NOT tested here.
    #
    # We only give deterministic answers so chat_service
    # can continue its real selection/state behaviour.
    # -----------------------------------------------------

    async def fake_generate_response(
        user_message,
        products,
        mode,
        conversation_history="",
        response_style=None,
    ):

        text = (
            user_message.lower()
        )

        if mode == "knowledge":

            if "shorter" in text:
                return (
                    "German coins have a long history."
                )

            return (
                "German coins include historical "
                "and modern collectible issues."
            )

        if (
            "choose two"
            in text
            and len(products) >= 2
        ):

            return (
                f"I choose {products[0]['name']} "
                f"and {products[1]['name']}."
            )

        if (
            "compare only those two"
            in text
            and len(products) >= 2
        ):

            return (
                f"{products[0]['name']} costs "
                f"AED {products[0]['price']}, while "
                f"{products[1]['name']} costs "
                f"AED {products[1]['price']}."
            )

        if products:

            names = [
                product["name"]
                for product in products
            ]

            return " | ".join(
                names
            )

        return "No products."

    monkeypatch.setattr(
        llm_service,
        "generate_response",
        fake_generate_response,
    )

    # =====================================================
    # 1. FRESH SEARCH
    # =====================================================

    result1 = run_chat(
        "Find silver under AED 500"
    )

    session_id = (
        result1["session_id"]
    )

    assert len(
        result1["products"]
    ) == 5

    assert {
        product["product_id"]
        for product
        in result1["products"]
    } == {
        product["product_id"]
        for product
        in FRESH_RESULTS
    }

    # =====================================================
    # 2. REFINE SEARCH
    # =====================================================

    result2 = run_chat(
        "Only silver ones",
        session_id,
    )

    assert len(
        result2["products"]
    ) == 5

    assert (
        "arabian-sands"
        in {
            product["product_id"]
            for product
            in result2["products"]
        }
    )

    # =====================================================
    # 3. CHEAPEST
    # =====================================================

    result3 = run_chat(
        "Which is the cheapest among those?",
        session_id,
    )

    assert len(
        result3["products"]
    ) == 1

    assert (
        result3["products"][0][
            "product_id"
        ]
        == "arabian-sands"
    )

    assert (
        result3["products"][0][
            "price"
        ]
        == 175.0
    )

    # =====================================================
    # 4. FOCUSED PRODUCT
    # =====================================================

    result4 = run_chat(
        "Tell me more about that coin",
        session_id,
    )

    assert len(
        result4["products"]
    ) == 1

    assert (
        result4["products"][0][
            "product_id"
        ]
        == "arabian-sands"
    )

    # =====================================================
    # 5. CHOOSE TWO FROM EARLIER RESULTS
    # =====================================================

    result5 = run_chat(
        (
            "Choose two from the products "
            "you showed me earlier"
        ),
        session_id,
    )

    assert len(
        result5["products"]
    ) == 2

    selected_ids = {
        product["product_id"]
        for product
        in result5["products"]
    }

    assert len(
        selected_ids
    ) == 2

    # =====================================================
    # 6. COMPARE ONLY SELECTED TWO
    # =====================================================

    result6 = run_chat(
        (
            "Compare only those two "
            "in one short paragraph"
        ),
        session_id,
    )

    assert {
        product["product_id"]
        for product
        in result6["products"]
    } == selected_ids

    # =====================================================
    # 7. CHEAPER SELECTED PRODUCT AVAILABILITY
    # =====================================================

    result7 = run_chat(
        "Is the cheaper one in stock?",
        session_id,
    )

    assert len(
        result7["products"]
    ) == 1

    selected_products = [
        ALL_PRODUCTS[
            product_id
        ]
        for product_id
        in selected_ids
    ]

    expected_cheapest = min(
        selected_products,
        key=lambda product:
            product["price"],
    )

    assert (
        result7["products"][0][
            "product_id"
        ]
        == expected_cheapest[
            "product_id"
        ]
    )

    # =====================================================
    # 8. KNOWLEDGE MODE
    # =====================================================

    result8 = run_chat(
        "I need to learn about German coins",
        session_id,
    )

    assert (
        result8["products"]
        == []
    )

    # =====================================================
    # 9. KNOWLEDGE FOLLOW-UP
    # =====================================================

    result9 = run_chat(
        "Make it shorter",
        session_id,
    )

    assert (
        result9["products"]
        == []
    )

    # =====================================================
    # 10. FRESH SEARCH AFTER KNOWLEDGE
    # =====================================================

    result10 = run_chat(
        "Find silver under AED 500",
        session_id,
    )

    assert len(
        result10["products"]
    ) == 5

    # =====================================================
    # 11. NAMED PRODUCT FROM PREVIOUS RESULTS
    # =====================================================

    result11 = run_chat(
        "Tell me more about Lunar Dragon",
        session_id,
    )

    assert len(
        result11["products"]
    ) == 1

    assert (
        result11["products"][0][
            "product_id"
        ]
        == "lunar-dragon"
    )

    # =====================================================
    # 12. FOLLOW-UP ON NAMED PRODUCT
    

    result12 = run_chat(
        "What's its purity?",
        session_id,
    )

    assert len(
        result12["products"]
    ) == 1

    assert (
        result12["products"][0][
            "product_id"
        ]
        == "lunar-dragon"
    )

    assert (
        result12["products"][0][
            "purity"
        ]
        == "99.9%"
    )