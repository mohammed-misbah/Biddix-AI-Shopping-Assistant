from app.services.retrieval_service import retrieval_service


def test_retrieval_returns_products():

    filters = retrieval_service.extract_filters(
        query="Find silver under AED 500"
    )

    products = retrieval_service.retrieve(
        query="Find silver under AED 500",
        filters=filters,
        limit=5,
        strict_match=False,
    )

    assert isinstance(
        products,
        list,
    )

    assert len(products) > 0

    assert len(products) <= 5

    for product in products:

        assert product.get(
            "product_id"
        )

        assert product.get(
            "name"
        )

        assert str(
            product.get(
                "material",
                "",
            )
        ).lower() == "silver"

        assert float(
            product.get(
                "price",
                0,
            )
        ) <= 500