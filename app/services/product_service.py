import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRODUCT_FILE = BASE_DIR / "data" / "products.json"


class ProductService:

    def __init__(self):
        self.products = self._load_products()

    def _load_products(self) -> list[dict[str, Any]]:
        if not PRODUCT_FILE.exists():
            raise FileNotFoundError(
                f"Products file not found: {PRODUCT_FILE}"
            )

        with open(PRODUCT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "products.json must contain a JSON array."
            )

        return data

    def get_product_by_id(
        self,
        product_id: str,
    ) -> dict[str, Any] | None:

        product_id = product_id.strip().lower()

        for product in self.products:
            current_id = str(
                product.get("product_id", "")
            ).strip().lower()

            if current_id == product_id:
                return product

        return None

    def search_products(
        self,
        query: str = "",
        *,
        material: str | None = None,
        category: str | None = None,
        year: int | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:

        products = list(self.products)

        # Structured filters

        if material:
            material = material.strip().lower()

            products = [
                product
                for product in products
                if str(
                    product.get("material", "")
                ).strip().lower() == material
            ]

        if category:
            category = category.strip().lower()

            products = [
                product
                for product in products
                if category
                in str(
                    product.get("category", "")
                ).strip().lower()
            ]

        if year is not None:
            products = [
                product
                for product in products
                if product.get("year") == year
            ]

        if min_price is not None:
            products = [
                product
                for product in products
                if self._get_price(product) >= min_price
            ]

        if max_price is not None:
            products = [
                product
                for product in products
                if self._get_price(product) <= max_price
            ]

        if in_stock_only:
            products = [
                product
                for product in products
                if self._get_stock(product) > 0
            ]

        # If there is no text query,
        # return the structured-filter results.
        query = query.strip().lower()

        if not query:
            return products[:limit]

        # Text relevance scoring

        query_words = [
            word
            for word in query.split()
            if len(word) > 1
        ]

        scored_products = []

        for product in products:

            name = str(
                product.get("name", "")
            ).lower()

            category_text = str(
                product.get("category", "")
            ).lower()

            material_text = str(
                product.get("material", "")
            ).lower()

            description = str(
                product.get("description", "")
            ).lower()

            year_text = str(
                product.get("year", "")
            ).lower()

            score = 0

            for word in query_words:

                if word in name:
                    score += 5

                if word in material_text:
                    score += 4

                if word in category_text:
                    score += 3

                if word in description:
                    score += 2

                if word in year_text:
                    score += 1

            if score > 0:
                scored_products.append(
                    (score, product)
                )

        scored_products.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            product
            for _, product
            in scored_products[:limit]
        ]

    @staticmethod
    def _get_price(product: dict[str, Any],) -> float:

        try:
            return float(
                product.get("price", 0)
            )
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _get_stock(product: dict[str, Any],) -> int:

        try:
            return int(
                product.get("stock", 0)
            )
        except (TypeError, ValueError):
            return 0


product_service = ProductService()