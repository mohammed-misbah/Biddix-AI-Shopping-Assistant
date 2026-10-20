
from app.services.grounding_service import grounding_service


PRODUCT = {
    "product_id": "test-silver-1",
    "name": "Test Silver Coin",
    "price": 350,
    "currency": "AED",
    "material": "Silver",
    "weight": "31.1 grams",
    "purity": "99.9%",
    "availability_confirmed": False,
    "product_url": "/products/test-silver-1",
}


def check(answer):
    return grounding_service.ground_response(
        answer=answer,
        products=[PRODUCT],
        mode="catalog",
        request_id="test-grounding",
    )


def test_correct_price_allowed():
    answer = "Test Silver Coin costs AED 350."
    assert check(answer) == answer


def test_invented_price_blocked():
    answer = "Test Silver Coin costs AED 100."
    result = check(answer)

    assert result != answer
    assert "AED 350" in result
    assert "AED 100" not in result


def test_unconfirmed_stock_not_invented():
    answer = "Test Silver Coin is in stock."
    result = check(answer)

    assert result != answer
    assert "Availability unconfirmed" in result


def test_fake_product_link_blocked():
    answer = (
        "View Test Silver Coin at "
        "/products/fake-coin."
    )

    result = check(answer)

    assert result != answer
    assert "/products/fake-coin" not in result


def test_correct_product_link_allowed():
    answer = (
        "View Test Silver Coin at "
        "/products/test-silver-1."
    )

    assert check(answer) == answer


def test_unconfirmed_wording_allowed():
    answer = (
        "Test Silver Coin costs AED 350. "
        "Its availability is unconfirmed."
    )

    assert check(answer) == answer


def test_knowledge_mode_unchanged():
    answer = (
        "Gold has been used in coinage "
        "for thousands of years."
    )

    result = grounding_service.ground_response(
        answer=answer,
        products=[],
        mode="knowledge",
    )

    assert result == answer
