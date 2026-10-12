import json
import logging
import re
from collections import defaultdict
from typing import Any

import faiss

from app.core.config import (
    INDEX_FILE,
    METADATA_FILE,
)
from app.services.conversation_service import (
    CatalogFilters,
)
from app.services.embedding_service import (
    embedding_service,
)
from app.services.product_service import (
    product_service,
)


logger = logging.getLogger(__name__)


class RetrievalService:

    MIN_SEMANTIC_SCORE = 0.20

    STRICT_MIN_LEXICAL_SCORE = 0.03

    FACET_FIELDS = (
        "country",
        "region",
        "category",
        "theme",
        "era",
        "period",
        "denomination",
    )

    STOP_WORDS = {
        "i",
        "me",
        "my",
        "we",
        "you",
        "your",
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "for",
        "from",
        "of",
        "in",
        "on",
        "at",
        "and",
        "or",
        "with",
        "show",
        "find",
        "want",
        "need",
        "looking",
        "something",
        "please",
        "product",
        "products",
        "coin",
        "coins",
        "aed",
        "only",
        "ones",
    }

    def __init__(self) -> None:

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.products_by_id = {
            str(
                product["product_id"]
            ): product
            for product
            in product_service.products
        }

        if (
            self.index.ntotal
            != len(self.metadata)
        ):
            raise RuntimeError(
                "FAISS index and metadata are out of sync."
            )

        # Dynamically discover all catalog facet values.
        self.facet_values = (
            self._build_facet_values()
        )

        self.material_values = {
            str(
                product.get(
                    "material",
                    ""
                )
            ).strip().lower()

            for product
            in product_service.products

            if str(
                product.get(
                    "material",
                    ""
                )
            ).strip()
        }

    # =====================================================
    # Dynamic catalog facets
    # =====================================================

    def _build_facet_values(
        self,
    ) -> dict[
        str,
        set[str],
    ]:

        values: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for product in product_service.products:

            for field in self.FACET_FIELDS:

                value = product.get(field)

                if value is None:
                    continue

                if isinstance(
                    value,
                    (list, tuple, set),
                ):

                    for item in value:

                        text = str(
                            item
                        ).strip().lower()

                        if text:
                            values[
                                field
                            ].add(text)

                else:

                    text = str(
                        value
                    ).strip().lower()

                    if text:
                        values[
                            field
                        ].add(text)

        return dict(values)

    # =====================================================
    # Query filter extraction
    # =====================================================

    def extract_filters(
        self,
        query: str,
        base_filters: CatalogFilters | None = None,
    ) -> CatalogFilters:

        text = query.lower().strip()

        if base_filters is None:

            filters = CatalogFilters()

        else:

            filters = CatalogFilters(
                year=base_filters.year,

                min_price=(
                    base_filters.min_price
                ),

                min_price_inclusive=(
                    base_filters
                    .min_price_inclusive
                ),

                max_price=(
                    base_filters.max_price
                ),

                max_price_inclusive=(
                    base_filters
                    .max_price_inclusive
                ),

                material=(
                    base_filters.material
                ),

                in_stock=(
                    base_filters.in_stock
                ),

                facets={
                    field: set(values)

                    for field, values
                    in base_filters
                    .facets
                    .items()
                },
            )

        self._extract_year(
            text,
            filters,
        )

        self._extract_price_bounds(
            text,
            filters,
        )

        self._extract_material(
            text,
            filters,
        )

        self._extract_stock(
            text,
            filters,
        )

        self._extract_dynamic_facets(
            text,
            filters,
        )

        return filters

    @staticmethod
    def _extract_year(
        text: str,
        filters: CatalogFilters,
    ) -> None:

        match = re.search(
            r"\b(19|20)\d{2}\b",
            text,
        )

        if match:
            filters.year = int(
                match.group()
            )

    def _extract_material(
        self,
        text: str,
        filters: CatalogFilters,
    ) -> None:

        for material in sorted(
            self.material_values,
            key=len,
            reverse=True,
        ):

            if self._contains_phrase(
                text,
                material,
            ):

                filters.material = (
                    material
                )

                break

    @staticmethod
    def _extract_stock(
        text: str,
        filters: CatalogFilters,
    ) -> None:

        if any(
            phrase in text
            for phrase in (
                "in stock",
                "currently available",
                "available now",
            )
        ):
            filters.in_stock = True

    @staticmethod
    def _extract_price_bounds(
        text: str,
        filters: CatalogFilters,
    ) -> None:

        number = r"(?:aed\s*)?(\d+(?:\.\d+)?)"

        exclusive_min_patterns = (
            rf"\babove\s+{number}",
            rf"\bover\s+{number}",
            rf"\bgreater than\s+{number}",
            rf"\bmore than\s+{number}",
        )

        inclusive_min_patterns = (
            rf"\bat least\s+{number}",
            rf"\bminimum(?:\s+of)?\s+{number}",
            rf"\bmin(?:imum)?\s+{number}",
        )

        exclusive_max_patterns = (
            rf"\bunder\s+{number}",
            rf"\bbelow\s+{number}",
            rf"\bless than\s+{number}",
        )

        inclusive_max_patterns = (
            rf"\bup to\s+{number}",
            rf"\bmaximum(?:\s+of)?\s+{number}",
            rf"\bmax(?:imum)?\s+{number}",
            rf"\bbudget(?:\s+is|\s+of)?\s+{number}",
        )

        for pattern in exclusive_min_patterns:

            match = re.search(
                pattern,
                text,
            )

            if match:

                filters.min_price = float(
                    match.group(1)
                )

                filters.min_price_inclusive = False

                break

        for pattern in inclusive_min_patterns:

            match = re.search(
                pattern,
                text,
            )

            if match:

                filters.min_price = float(
                    match.group(1)
                )

                filters.min_price_inclusive = True

                break

        for pattern in exclusive_max_patterns:

            match = re.search(
                pattern,
                text,
            )

            if match:

                filters.max_price = float(
                    match.group(1)
                )

                filters.max_price_inclusive = False

                break

        for pattern in inclusive_max_patterns:

            match = re.search(
                pattern,
                text,
            )

            if match:

                filters.max_price = float(
                    match.group(1)
                )

                filters.max_price_inclusive = True

                break

        # "500 AED budget"
        budget_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*aed\s+budget\b",
            text,
        )

        if budget_match:

            filters.max_price = float(
                budget_match.group(1)
            )

            filters.max_price_inclusive = True

    def _extract_dynamic_facets(
        self,
        text: str,
        filters: CatalogFilters,
    ) -> None:

        for field, values in (
            self.facet_values.items()
        ):

            matches = set()

            for value in values:

                if self._contains_phrase(
                    text,
                    value,
                ):
                    matches.add(
                        value
                    )

            if matches:
                filters.facets[
                    field
                ] = matches

    # =====================================================
    # Product restoration
    # =====================================================

    def get_products_by_ids(
        self,
        product_ids: list[str],
    ) -> list[
        dict[str, Any]
    ]:

        products = []

        for product_id in product_ids:

            product = (
                self.products_by_id.get(
                    str(product_id)
                )
            )

            if product is not None:
                products.append(
                    product
                )

        return products

    # =====================================================
    # Retrieval
    # =====================================================

    def retrieve(
        self,
        query: str,
        filters: CatalogFilters | None = None,
        limit: int = 5,
        strict_match: bool = False,
    ) -> list[
        dict[str, Any]
    ]:

        query = query.strip()

        if not query or limit <= 0:
            return []

        if filters is None:
            filters = self.extract_filters(
                query
            )

        query_vector = (
            embedding_service
            .embed_text(
                query
            )
            .reshape(
                1,
                -1,
            )
        )

        scores, indexes = (
            self.index.search(
                query_vector,
                self.index.ntotal,
            )
        )

        query_words = (
            self._extract_query_words(
                query
            )
        )

        ranked = []

        for (
            semantic_score,
            index_position,
        ) in zip(
            scores[0],
            indexes[0],
        ):

            if index_position < 0:
                continue

            semantic_score = float(
                semantic_score
            )

            metadata_item = (
                self.metadata[
                    index_position
                ]
            )

            product_id = str(
                metadata_item.get(
                    "product_id",
                    "",
                )
            )

            product = (
                self.products_by_id.get(
                    product_id
                )
            )

            if product is None:
                continue

            if not self._passes_filters(
                product,
                filters,
            ):
                continue

            lexical_score = (
                self._calculate_lexical_score(
                    product,
                    query_words,
                )
            )

            has_filters = (
                self._has_hard_filters(
                    filters
                )
            )

            if (
                semantic_score
                < self.MIN_SEMANTIC_SCORE
                and not has_filters
            ):
                continue

            if (
                strict_match
                and query_words
                and lexical_score
                < self.STRICT_MIN_LEXICAL_SCORE
                and semantic_score < 0.45
            ):
                continue

            final_score = (
                semantic_score
                + lexical_score
            )

            ranked.append({
                "product": product,
                "semantic_score":
                    semantic_score,
                "lexical_score":
                    lexical_score,
                "final_score":
                    final_score,
            })

        ranked.sort(
            key=lambda item: (
                item["final_score"],
                item["semantic_score"],
            ),
            reverse=True,
        )

        results = ranked[
            :limit
        ]

        self._debug(
            query,
            filters,
            results,
        )

        return [
            item["product"]
            for item in results
        ]

    # =====================================================
    # Filter application
    # =====================================================

    def _passes_filters(
        self,
        product: dict[str, Any],
        filters: CatalogFilters,
    ) -> bool:

        # Year
        if filters.year is not None:

            try:

                product_year = int(
                    product.get(
                        "year"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                return False

            if product_year != filters.year:
                return False

        # Material
        if filters.material:

            product_material = str(
                product.get(
                    "material",
                    "",
                )
            ).strip().lower()

            if (
                product_material
                != filters.material.lower()
            ):
                return False

        # Price
        try:

            price = float(
                product.get(
                    "price"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

        if filters.min_price is not None:

            if (
                filters.min_price_inclusive
                and price < filters.min_price
            ):
                return False

            if (
                not filters.min_price_inclusive
                and price <= filters.min_price
            ):
                return False

        if filters.max_price is not None:

            if (
                filters.max_price_inclusive
                and price > filters.max_price
            ):
                return False

            if (
                not filters.max_price_inclusive
                and price >= filters.max_price
            ):
                return False

        # Availability
        if filters.in_stock:

            if (
                product.get(
                    "availability_confirmed"
                )
                is not True
            ):
                return False

            try:

                stock = int(
                    product.get(
                        "stock",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                return False

            if stock <= 0:
                return False

        # Dynamic facets
        for (
            field,
            requested_values,
        ) in filters.facets.items():

            product_value = (
                product.get(field)
            )

            if product_value is None:
                return False

            if isinstance(
                product_value,
                (list, tuple, set),
            ):

                actual_values = {
                    str(value)
                    .strip()
                    .lower()

                    for value in product_value
                }

            else:

                actual_values = {
                    str(product_value)
                    .strip()
                    .lower()
                }

            if not (
                actual_values
                & requested_values
            ):
                return False

        return True

    # =====================================================
    # Hybrid ranking
    # =====================================================

    def _calculate_lexical_score(
        self,
        product: dict[str, Any],
        query_words: set[str],
    ) -> float:

        weighted_fields = (
            ("name", 0.08),
            ("country", 0.07),
            ("region", 0.07),
            ("category", 0.06),
            ("theme", 0.06),
            ("era", 0.06),
            ("period", 0.06),
            ("denomination", 0.05),
            ("material", 0.05),
            ("description", 0.03),
        )

        score = 0.0

        for word in query_words:

            for (
                field,
                weight,
            ) in weighted_fields:

                value = product.get(
                    field
                )

                if value is None:
                    continue

                if isinstance(
                    value,
                    (list, tuple, set),
                ):

                    text = " ".join(
                        str(item)
                        for item in value
                    ).lower()

                else:

                    text = str(
                        value
                    ).lower()

                if word in text:

                    score += weight

                    break

        return min(
            score,
            0.35,
        )

    def _extract_query_words(
        self,
        query: str,
    ) -> set[str]:

        words = re.findall(
            r"[a-z0-9]+",
            query.lower(),
        )

        return {
            word

            for word in words

            if (
                len(word) > 2
                and word
                not in self.STOP_WORDS
            )
        }

    @staticmethod
    def _contains_phrase(
        text: str,
        value: str,
    ) -> bool:

        value = value.strip().lower()

        if not value:
            return False

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(value)
            + r"(?![a-z0-9])"
        )

        return (
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            is not None
        )

    @staticmethod
    def _has_hard_filters(
        filters: CatalogFilters,
    ) -> bool:

        return any([
            filters.year is not None,
            filters.min_price is not None,
            filters.max_price is not None,
            filters.material is not None,
            filters.in_stock,
            bool(filters.facets),
        ])

    # =====================================================
    # Debug
    # =====================================================

    @staticmethod
    def _debug(
        query: str,
        filters: CatalogFilters,
        results: list[dict],
    ) -> None:

        print(
            "\n--- RAG RESULTS ---"
        )

        print(
            f"Query: {query}"
        )

        print(
            "Filters:",
            {
                "year":
                    filters.year,

                "min_price":
                    filters.min_price,

                "max_price":
                    filters.max_price,

                "material":
                    filters.material,

                "in_stock":
                    filters.in_stock,

                "facets":
                    filters.facets,
            },
        )

        for item in results:

            print(
                item["product"].get(
                    "name",
                    "Unknown",
                ),
                "| semantic:",
                round(
                    item["semantic_score"],
                    3,
                ),
                "| lexical:",
                round(
                    item["lexical_score"],
                    3,
                ),
                "| final:",
                round(
                    item["final_score"],
                    3,
                ),
            )

        print(
            "-------------------\n"
        )


retrieval_service = RetrievalService()