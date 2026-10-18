
from app.services.conversation_service import CatalogFilters
from app.services.retrieval_service import retrieval_service


def test_general_catalog_listing():
    products = retrieval_service.retrieve(
        query="list 5 products",
        limit=5,
    )

    assert len(products) == 5
    assert all(
        product.get("product_id")
        for product in products
    )


def test_verified_price_and_material_filters():
    filters = CatalogFilters()
    filters.material = "silver"
    filters.max_price = 500.0
    filters.max_price_inclusive = False

    products = retrieval_service.list_products(
        limit=20,
        filters=filters,
    )

    assert products

    for product in products:
        assert (
            product["material"].lower()
            == "silver"
        )
        assert float(product["price"]) < 500


def test_catalog_listing_respects_requested_count():
    for count in (1, 3, 5):
        products = retrieval_service.list_products(
            limit=count,
        )

        assert len(products) == count


def test_nonexistent_product_id_returns_nothing():
    products = (
        retrieval_service.get_products_by_ids(
            ["this-product-id-does-not-exist"]
        )
    )

    assert products == []
