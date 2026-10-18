
from app.services.product_service import product_service


def test_catalog_contains_products():
    assert len(product_service.products) >= 5


def test_list_five_products():
    products = product_service.list_products(limit=5)
    assert len(products) == 5


def test_lookup_existing_product():
    product = product_service.products[0]
    product_id = product["product_id"]

    result = product_service.get_product_by_id(
        product_id
    )

    assert result is not None
    assert result["product_id"] == product_id


def test_lookup_missing_product():
    result = product_service.get_product_by_id(
        "nonexistent-test-product-123"
    )

    assert result is None


def test_silver_under_500():
    products = product_service.filter_products(
        filters={
            "material": "silver",
            "max_price": 499.99,
        },
        limit=100,
    )

    assert products

    for product in products:
        assert product["material"].lower() == "silver"
        assert float(product["price"]) < 500


def test_price_filter_never_exceeds_budget():
    products = product_service.filter_products(
        filters={"max_price": 500},
        limit=100,
    )

    for product in products:
        assert float(product["price"]) <= 500


def test_search_interface_still_works():
    products = product_service.search_products(
        query="silver under AED 500",
        limit=5,
    )

    assert len(products) <= 5

    for product in products:
        assert product["material"].lower() == "silver"
        assert float(product["price"]) < 500
