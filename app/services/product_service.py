import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRODUCT_FILE = BASE_DIR / "data" / "products.json"


class ProductService:

    def __init__(self):
        with open(
            PRODUCT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self.products: list[dict[str, Any]] = json.load(file)

    def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        query_lower = query.lower().strip()

        # -----------------------------------
        # 1. Understand important filters
        # -----------------------------------

        year = self._extract_year(query_lower)
        max_price = self._extract_max_price(query_lower)
        material = self._extract_material(query_lower)

        wants_uae = "uae" in query_lower

        wants_in_stock = (
            "in stock" in query_lower
            or "available" in query_lower
            or "availability" in query_lower
        )

        results = []

        # -----------------------------------
        # 2. Check every product
        # -----------------------------------

        for product in self.products:

            # YEAR
            if year is not None:
                if product.get("year") != year:
                    continue

            # MATERIAL
            if material is not None:
                product_material = str(
                    product.get("material", "")
                ).lower()

                if product_material != material:
                    continue

            # PRICE
            if max_price is not None:
                try:
                    price = float(
                        product.get("price", 0)
                    )
                except (TypeError, ValueError):
                    continue

                if price > max_price:
                    continue

            # STOCK
            if wants_in_stock:
                try:
                    stock = int(
                        product.get("stock", 0)
                    )
                except (TypeError, ValueError):
                    stock = 0

                if stock <= 0:
                    continue

            # UAE relevance
            if wants_uae:
                searchable_text = self._product_text(product)

                if "uae" not in searchable_text:
                    continue

            results.append(product)

        # -----------------------------------
        # 3. Rank remaining valid products
        # -----------------------------------

        scored_products = []

        query_words = self._useful_words(
            query_lower
        )

        for product in results:

            searchable_text = self._product_text(
                product
            )

            score = sum(
                1
                for word in query_words
                if word in searchable_text
            )

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
    def _extract_year(
        query: str,
    ) -> int | None:

        match = re.search(
            r"\b(19|20)\d{2}\b",
            query,
        )

        if not match:
            return None

        return int(match.group())

    @staticmethod
    def _extract_max_price(
        query: str,
    ) -> float | None:

        patterns = [
            r"budget(?:\s+is|\s+of)?\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"under\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"below\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"less than\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"up to\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"maximum\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
            r"max\s*(?:aed)?\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return float(
                    match.group(1)
                )

        # Handles:
        # "500 AED budget"
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*aed",
            query,
            re.IGNORECASE,
        )

        if match and (
            "budget" in query
            or "under" in query
            or "below" in query
            or "maximum" in query
        ):
            return float(
                match.group(1)
            )

        return None

    @staticmethod
    def _extract_material(
        query: str,
    ) -> str | None:

        materials = [
            "silver",
            "gold",
            "copper",
            "bronze",
            "platinum",
        ]

        for material in materials:
            if material in query:
                return material

        return None

    @staticmethod
    def _product_text(
        product: dict[str, Any],
    ) -> str:

        return " ".join([
            str(product.get("name", "")),
            str(product.get("category", "")),
            str(product.get("material", "")),
            str(product.get("year", "")),
            str(product.get("description", "")),
        ]).lower()

    @staticmethod
    def _useful_words(
        query: str,
    ) -> list[str]:

        ignored_words = {
            "i",
            "am",
            "a",
            "an",
            "the",
            "for",
            "from",
            "my",
            "is",
            "are",
            "show",
            "me",
            "all",
            "looking",
            "want",
            "and",
            "that",
            "them",
            "between",
            "explain",
            "available",
            "in",
            "stock",
            "aed",
        }

        return [
            word
            for word in re.findall(
                r"[a-z0-9]+",
                query,
            )
            if len(word) > 1
            and word not in ignored_words
        ]


product_service = ProductService()