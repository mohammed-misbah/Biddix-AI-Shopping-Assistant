import json
import logging
import re
from typing import Any

import faiss

from app.core.config import (
    INDEX_FILE,
    METADATA_FILE,
)
from app.services.embedding_service import (
    embedding_service,
)
from app.services.product_service import (
    product_service,
)


logger = logging.getLogger(__name__)


class RetrievalService:

    MIN_SEMANTIC_SCORE = 0.25

    STRICT_MIN_LEXICAL_SCORE = 0.03

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
        "have",
        "sell",
        "carry",
        "available",
        "stock",
    }

    def __init__(self) -> None:

        if not INDEX_FILE.exists():

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{INDEX_FILE}. "
                "Run: python -m scripts.index_products"
            )

        if not METADATA_FILE.exists():

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{METADATA_FILE}. "
                "Run: python -m scripts.index_products"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = (
                json.load(file)
            )

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
                "FAISS index and metadata "
                "are out of sync."
            )

    # =================================================
    # Restore exact previous product set
    # =================================================

    def get_products_by_ids(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:

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

    # =================================================
    # Main retrieval
    # =================================================

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int | None = None,
        strict_match: bool = False,
    ) -> list[dict[str, Any]]:

        query = query.strip()

        if not query or limit <= 0:
            return []

        query_lower = query.lower()

        # -------------------------------------------------
        # Structured filters
        # -------------------------------------------------

        year = (
            product_service
            ._extract_year(
                query_lower
            )
        )

        max_price = (
            product_service
            ._extract_max_price(
                query_lower
            )
        )

        min_price = (
            product_service
            ._extract_min_price(
                query_lower
            )
        )

        material = (
            product_service
            ._extract_material(
                query_lower
            )
        )

        wants_in_stock = any(
            phrase in query_lower
            for phrase in (
                "in stock",
                "available",
                "availability",
            )
        )

        # -------------------------------------------------
        # Embedding
        # -------------------------------------------------

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

        if candidate_limit is None:

            search_limit = (
                self.index.ntotal
            )

        else:

            search_limit = min(
                max(
                    candidate_limit,
                    limit,
                ),
                self.index.ntotal,
            )

        if search_limit <= 0:
            return []

        # -------------------------------------------------
        # FAISS
        # -------------------------------------------------

        scores, indexes = (
            self.index.search(
                query_vector,
                search_limit,
            )
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for (
            score,
            index_position,
        ) in zip(
            scores[0],
            indexes[0],
        ):

            if index_position < 0:
                continue

            semantic_score = float(
                score
            )

            if (
                semantic_score
                < self.MIN_SEMANTIC_SCORE
            ):
                continue

            metadata = self.metadata[
                index_position
            ]

            product_id = str(
                metadata.get(
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

            candidates.append({
                "product": product,
                "semantic_score":
                    semantic_score,
            })

        # -------------------------------------------------
        # Hard filters
        # -------------------------------------------------

        filtered = []

        for item in candidates:

            product = item[
                "product"
            ]

            # Year
            if year is not None:

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
                    continue

                if product_year != year:
                    continue

            # Material
            if material is not None:

                product_material = str(
                    product.get(
                        "material",
                        "",
                    )
                ).strip().lower()

                if (
                    product_material
                    != material.lower()
                ):
                    continue

            # Price
            try:

                price = float(
                    product.get(
                        "price",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                max_price is not None
                and price > max_price
            ):
                continue

            if (
                min_price is not None
                and price < min_price
            ):
                continue

            # -------------------------------------------------
            # Availability
            #
            # Numeric stock alone is not treated as confirmed
            # availability unless availability_confirmed=True.
            # -------------------------------------------------

            if wants_in_stock:

                if (
                    product.get(
                        "availability_confirmed"
                    )
                    is not True
                ):
                    continue

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
                    stock = 0

                if stock <= 0:
                    continue

            filtered.append(
                item
            )

        # -------------------------------------------------
        # Hybrid ranking
        # -------------------------------------------------

        query_words = (
            self._extract_query_words(
                query_lower
            )
        )

        ranked = []

        for item in filtered:

            lexical_score = (
                self._calculate_lexical_score(
                    product=item[
                        "product"
                    ],
                    query_words=(
                        query_words
                    ),
                )
            )

            item[
                "lexical_score"
            ] = lexical_score

            item[
                "final_score"
            ] = (
                item[
                    "semantic_score"
                ]
                + lexical_score
            )

            if (
                strict_match
                and query_words
                and lexical_score
                < self.STRICT_MIN_LEXICAL_SCORE
            ):
                continue

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda item: (
                item[
                    "final_score"
                ],
                item[
                    "semantic_score"
                ],
            ),
            reverse=True,
        )

        results = ranked[
            :limit
        ]

        # -------------------------------------------------
        # Development diagnostics
        # -------------------------------------------------

        print(
            "\n--- RAG RESULTS ---"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Strict match: "
            f"{strict_match}"
        )

        print(
            f"Candidates: "
            f"{len(candidates)}"
        )

        print(
            f"After filters: "
            f"{len(ranked)}"
        )

        for item in results:

            print(
                item[
                    "product"
                ].get(
                    "name",
                    "Unknown",
                ),
                "| semantic:",
                round(
                    item[
                        "semantic_score"
                    ],
                    3,
                ),
                "| lexical:",
                round(
                    item[
                        "lexical_score"
                    ],
                    3,
                ),
                "| final:",
                round(
                    item[
                        "final_score"
                    ],
                    3,
                ),
            )

        print(
            "-------------------\n"
        )

        logger.info(
            "RAG query=%r "
            "strict=%s "
            "candidates=%d "
            "returned=%d",
            query,
            strict_match,
            len(candidates),
            len(results),
        )

        return [
            item[
                "product"
            ]
            for item in results
        ]

    # =================================================
    # Lexical scoring
    # =================================================

    def _calculate_lexical_score(
        self,
        product: dict[str, Any],
        query_words: set[str],
    ) -> float:

        fields = (
            (
                str(
                    product.get(
                        "name",
                        "",
                    )
                ).lower(),
                0.08,
            ),
            (
                str(
                    product.get(
                        "country",
                        "",
                    )
                ).lower(),
                0.07,
            ),
            (
                str(
                    product.get(
                        "region",
                        "",
                    )
                ).lower(),
                0.07,
            ),
            (
                str(
                    product.get(
                        "category",
                        "",
                    )
                ).lower(),
                0.06,
            ),
            (
                str(
                    product.get(
                        "theme",
                        "",
                    )
                ).lower(),
                0.06,
            ),
            (
                str(
                    product.get(
                        "era",
                        "",
                    )
                ).lower(),
                0.06,
            ),
            (
                str(
                    product.get(
                        "material",
                        "",
                    )
                ).lower(),
                0.05,
            ),
            (
                str(
                    product.get(
                        "description",
                        "",
                    )
                ).lower(),
                0.03,
            ),
        )

        score = 0.0

        for word in query_words:

            for (
                field_text,
                weight,
            ) in fields:

                if word in field_text:

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


retrieval_service = RetrievalService()