import asyncio
import time

from app.services.chat_service import chat_service
from app.services.conversation_service import conversation_service
from app.services.llm_service import llm_service
from app.services.retrieval_service import retrieval_service


class FakeRedis:

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def delete(self, key):
        existed = key in self.data
        self.data.pop(key, None)
        return 1 if existed else 0

    def exists(self, key):
        return 1 if key in self.data else 0

    def ttl(self, key):
        return 3600 if key in self.data else -2

    def ping(self):
        return True

    def close(self):
        return None


TEST_PRODUCTS = [
    {
        "product_id": "test-silver-1",
        "name": "Test Silver Coin 1",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "10 grams",
        "purity": "99.9%",
        "price": 200.0,
        "currency": "AED",
        "description": "Test silver coin",
        "product_url": None,
    },
    {
        "product_id": "test-silver-2",
        "name": "Test Silver Coin 2",
        "category": "coin",
        "year": 2024,
        "material": "Silver",
        "weight": "20 grams",
        "purity": "99.9%",
        "price": 300.0,
        "currency": "AED",
        "description": "Test silver coin",
        "product_url": None,
    },
]


ALL_PRODUCTS = {
    product["product_id"]: product
    for product in TEST_PRODUCTS
}


async def run_one_user(
    user_number: int,
):
    result = await chat_service.chat(
        message="Find silver under AED 500",
        session_id=None,
        request_id=f"load-test-{user_number}",
    )

    return result


def test_1000_simultaneous_users(
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
        return [
            dict(product)
            for product in TEST_PRODUCTS[:limit]
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
        await asyncio.sleep(0.05)

        return " | ".join(
            product["name"]
            for product in products
        )

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

    async def run_all_users():

        started = time.perf_counter()

        results = await asyncio.gather(
            *[
                run_one_user(user_number)
                for user_number
                in range(1000)
            ]
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        return results, elapsed

    results, elapsed = asyncio.run(
        run_all_users()
    )

    assert len(results) == 1000

    session_ids = [
        result["session_id"]
        for result in results
    ]

    assert len(
        set(session_ids)
    ) == 1000

    for result in results:

        assert len(
            result["products"]
        ) == 2

        assert {
            product["product_id"]
            for product
            in result["products"]
        } == {
            "test-silver-1",
            "test-silver-2",
        }

    print(
        f"\n1000 simultaneous users completed "
        f"in {elapsed:.3f} seconds"
    )