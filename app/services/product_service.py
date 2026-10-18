
import json
import re
from typing import Any

from app.core.config import PRODUCT_FILE


class ProductService:
    """
    Read and search the real Biddix product catalog.

    Gemini understands the customer's language.
    Python checks actual catalog facts.
    """

    def __init__(self) -> None:
        with open(
            PRODUCT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Product catalog must contain a list."
            )

        self.products: list[dict[str, Any]] = data

        self.products_by_id = {
            str(product["product_id"]): product
            for product in self.products
            if isinstance(product, dict)
            and product.get("product_id") is not None
        }

    # =====================================================
    # DIRECT CATALOG OPERATIONS
    # =====================================================

    def get_product_by_id(
        self,
        product_id: str,
    ) -> dict[str, Any] | None:
        return self.products_by_id.get(
            str(product_id)
        )

    def get_products_by_ids(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:
        return [
            product
            for product_id in product_ids
            if (
                product := self.get_product_by_id(
                    product_id
                )
            ) is not None
        ]

    def list_products(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        return self.products[:limit]

    # =====================================================
    # GEMINI STRUCTURED-FILTER SEARCH
    # =====================================================

    def filter_products(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Apply structured filters to real catalog items.

        No customer keyword patterns are required.
        """

        if limit <= 0:
            return []

        filters = filters or {}

        if not isinstance(filters, dict):
            raise ValueError(
                "Product filters must be a dictionary."
            )

        material = filters.get("material")
        year = filters.get("year")
        min_price = filters.get("min_price")
        max_price = filters.get("max_price")
        in_stock = filters.get("in_stock")

        if isinstance(material, str):
            material = material.strip().lower()
        else:
            material = None

        results = []

        for product in self.products:
            if not isinstance(product, dict):
                continue

            if material:
                product_material = str(
                    product.get("material", "")
                ).strip().lower()

                if product_material != material:
                    continue

            if year is not None:
                try:
                    product_year = int(
                        product.get("year")
                    )
                except (ValueError, TypeError):
                    continue

                if product_year != year:
                    continue

            try:
                price = float(
                    product.get("price")
                )
            except (ValueError, TypeError):
                continue

            if min_price is not None:
                if price < float(min_price):
                    continue

            if max_price is not None:
                if price > float(max_price):
                    continue

            if in_stock is True:
                if (
                    product.get(
                        "availability_confirmed"
                    )
                    is not True
                ):
                    continue

                try:
                    stock = int(
                        product.get("stock", 0)
                    )
                except (TypeError, ValueError):
                    continue

                if stock <= 0:
                    continue

            results.append(product)

            if len(results) >= limit:
                break

        return results

    # =====================================================
    # EXISTING SEARCH INTERFACE
    #
    # Kept for older callers and regression tests.
    # =====================================================

    def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        text = query.lower().strip()

        filters = {
            "year": self._extract_year(text),
            "min_price": self._extract_min_price(text),
            "max_price": self._extract_max_price(text),
            "material": self._extract_material(text),
            "in_stock": (
                any(
                    phrase in text
                    for phrase in (
                        "in stock",
                        "available",
                        "availability",
                    )
                )
            ),
        }

        candidates = self.filter_products(
            filters=filters,
            limit=len(self.products),
        )

        query_words = self._useful_words(text)

        ranked = []

        for index, product in enumerate(candidates):
            product_text = self._product_text(
                product
            )

            score = sum(
                1
                for word in query_words
                if word in product_text
            )

            ranked.append(
                (score, index, product)
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            product
            for _, _, product
            in ranked[:limit]
        ]

    # =====================================================
    # COMPATIBLE LEGACY FILTER HELPERS
    # =====================================================

    @staticmethod
    def _extract_year(
        query: str,
    ) -> int | None:
        match = re.search(
            r"\b(?:19|20)\d{2}\b",
            query,
        )

        return (
            int(match.group())
            if match
            else None
        )

    @staticmethod
    def _extract_max_price(
        query: str,
    ) -> float | None:
        patterns = (
            r"\bbudget(?:\s+is|\s+of)?\s*"
            r"(?:aed\s*)?(\d+(?:\.\d+)?)",
            r"\bunder\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bbelow\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bless than\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bup to\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bmaximum\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bmax\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*aed\b",
            query,
            re.IGNORECASE,
        )

        if match and any(
            word in query
            for word in (
                "budget",
                "under",
                "below",
                "maximum",
                "max",
            )
        ):
            return float(match.group(1))

        return None

    @staticmethod
    def _extract_min_price(
        query: str,
    ) -> float | None:
        patterns = (
            r"\bgreater than\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bmore than\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bover\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\babove\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bminimum\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bat least\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
            r"\bstarting from\s*(?:aed\s*)?"
            r"(\d+(?:\.\d+)?)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        return None

    @staticmethod
    def _extract_material(
        query: str,
    ) -> str | None:
        # Legacy compatibility only.
        # Gemini-first requests use filter_products().
        materials = (
            "silver",
            "gold",
            "copper",
            "bronze",
            "platinum",
        )

        for material in materials:
            if re.search(
                rf"\b{re.escape(material)}\b",
                query,
            ):
                return material

        return None

    @staticmethod
    def _product_text(
        product: dict[str, Any],
    ) -> str:
        fields = (
            "name",
            "category",
            "country",
            "region",
            "theme",
            "era",
            "material",
            "year",
            "description",
        )

        return " ".join(
            str(product.get(field, ""))
            for field in fields
        ).lower()

    @staticmethod
    def _useful_words(
        query: str,
    ) -> list[str]:
        ignored_words = {
            "i", "am", "a", "an", "the",
            "for", "from", "my", "is",
            "are", "show", "me", "all",
            "looking", "want", "need",
            "and", "that", "them",
            "between", "explain",
            "available", "in", "stock",
            "aed", "coin", "coins",
            "product", "products",
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
