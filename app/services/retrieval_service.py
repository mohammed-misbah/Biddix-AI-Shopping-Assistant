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

    # Very weak semantic matches are ignored.
    #
    # Do not make this too high initially.
    # Tune it later using real customer queries.
    MIN_SEMANTIC_SCORE = 0.25

    # Words that usually provide little ranking value.
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
        "aed",
    }

    def __init__(self) -> None:

        # -----------------------------------------
        # Validate vector files
        # -----------------------------------------

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}. "
                "Run: python -m scripts.index_products"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}. "
                "Run: python -m scripts.index_products"
            )

        # -----------------------------------------
        # Load FAISS
        # -----------------------------------------

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # -----------------------------------------
        # Load metadata
        # -----------------------------------------

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # -----------------------------------------
        # Full product lookup
        # -----------------------------------------

        self.products_by_id = {
            str(product["product_id"]): product
            for product in product_service.products
        }

        # -----------------------------------------
        # Safety checks
        # -----------------------------------------

        if self.index.ntotal != len(
            self.metadata
        ):
            raise RuntimeError(
                "FAISS index and metadata are out of sync. "
                "Run the indexing script again."
            )

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int | None = None,
    ) -> list[dict[str, Any]]:

        query = query.strip()

        if not query:
            return []

        if limit <= 0:
            return []

        query_lower = query.lower()

        # =================================================
        # 1. Extract structured / hard requirements
        # =================================================

        year = product_service._extract_year(
            query_lower
        )

        max_price = (
            product_service._extract_max_price(
                query_lower
            )
        )

        material = (
            product_service._extract_material(
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

        # =================================================
        # 2. Embed customer query
        # =================================================

        query_vector = (
            embedding_service.embed_text(
                query
            )
        )

        query_vector = query_vector.reshape(
            1,
            -1,
        )

        # =================================================
        # 3. Decide how many FAISS candidates to search
        # =================================================
        #
        # With your current ~40 products, search everything.
        #
        # This prevents a valid hard-filter match from being
        # lost before filtering.
        #
        # Later with thousands of products you can use an
        # over-fetch value instead.
        # =================================================

        if candidate_limit is None:
            search_limit = self.index.ntotal
        else:
            search_limit = min(
                max(candidate_limit, limit),
                self.index.ntotal,
            )

        if search_limit <= 0:
            return []

        # =================================================
        # 4. Semantic FAISS search
        # =================================================

        scores, indexes = self.index.search(
            query_vector,
            search_limit,
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for score, index_position in zip(
            scores[0],
            indexes[0],
        ):

            if index_position < 0:
                continue

            semantic_score = float(
                score
            )

            # Remove obviously weak matches.
            if (
                semantic_score
                < self.MIN_SEMANTIC_SCORE
            ):
                continue

            metadata_item = self.metadata[
                index_position
            ]

            product_id = str(
                metadata_item.get(
                    "product_id",
                    "",
                )
            )

            product = self.products_by_id.get(
                product_id
            )

            if not product:
                continue

            candidates.append({
                "product": product,
                "semantic_score": semantic_score,
            })

        # =================================================
        # 5. Apply HARD filters
        # =================================================

        filtered_candidates: list[
            dict[str, Any]
        ] = []

        for item in candidates:

            product = item["product"]

            # -------------------------
            # Year
            # -------------------------

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

            # -------------------------
            # Material
            # -------------------------

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

            # -------------------------
            # Maximum price
            # -------------------------

            if max_price is not None:

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

                if price > max_price:
                    continue

            # -------------------------
            # Stock
            # -------------------------

            if wants_in_stock:

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

            filtered_candidates.append(
                item
            )

        # =================================================
        # 6. Extract useful query terms
        # =================================================

        query_words = (
            self._extract_query_words(
                query_lower
            )
        )

        # =================================================
        # 7. Hybrid ranking
        # =================================================
        #
        # Semantic similarity remains the main score.
        #
        # Exact lexical matches provide smaller bonuses.
        #
        # This works generically for:
        #
        # UAE
        # Abu Dhabi
        # Dubai
        # German
        # Greek
        # Roman
        # Islamic
        # heritage
        # wildlife
        # historic
        # commemorative
        # etc.
        # =================================================

        for item in filtered_candidates:

            product = item["product"]

            lexical_score = (
                self._calculate_lexical_score(
                    product=product,
                    query_words=query_words,
                )
            )

            item[
                "lexical_score"
            ] = lexical_score

            item[
                "final_score"
            ] = (
                item["semantic_score"]
                + lexical_score
            )

        # =================================================
        # 8. Sort strongest matches first
        # =================================================

        filtered_candidates.sort(
            key=lambda item: (
                item["final_score"],
                item["semantic_score"],
            ),
            reverse=True,
        )

        results = filtered_candidates[
            :limit
        ]

        # =================================================
        # 9. Development debug output
        # =================================================
        #
        # Keep this while developing.
        # Later replace/remove print() in production.
        # =================================================

        print("\n--- RAG RESULTS ---")

        print(
            f"Query: {query}"
        )

        print(
            f"Semantic candidates: "
            f"{len(candidates)}"
        )

        print(
            f"After hard filters: "
            f"{len(filtered_candidates)}"
        )

        print()

        for item in results:

            product = item["product"]

            print(
                product.get(
                    "name",
                    "Unknown Product",
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

        print("-------------------\n")

        logger.info(
            "RAG retrieval query=%r "
            "semantic_candidates=%d "
            "filtered=%d returned=%d",
            query,
            len(candidates),
            len(filtered_candidates),
            len(results),
        )

        # =================================================
        # 10. Return only actual product objects
        # =================================================

        return [
            item["product"]
            for item in results
        ]

    def _calculate_lexical_score(
        self,
        product: dict[str, Any],
        query_words: set[str],
    ) -> float:
        """
        Generic keyword relevance bonus.

        It does NOT contain special rules for UAE,
        Roman, Greek, Islamic, German, etc.

        Exact matches in important product fields
        receive additional ranking weight.
        """

        name = str(
            product.get(
                "name",
                "",
            )
        ).lower()

        category = str(
            product.get(
                "category",
                "",
            )
        ).lower()

        country = str(
            product.get(
                "country",
                "",
            )
        ).lower()

        region = str(
            product.get(
                "region",
                "",
            )
        ).lower()

        theme = str(
            product.get(
                "theme",
                "",
            )
        ).lower()

        era = str(
            product.get(
                "era",
                "",
            )
        ).lower()

        material = str(
            product.get(
                "material",
                "",
            )
        ).lower()

        description = str(
            product.get(
                "description",
                "",
            )
        ).lower()

        score = 0.0

        for word in query_words:

            # Product name is strongest.
            if word in name:
                score += 0.08

            # Country / region are strong signals.
            elif (
                word in country
                or word in region
            ):
                score += 0.07

            # Category/theme/era are also important.
            elif (
                word in category
                or word in theme
                or word in era
            ):
                score += 0.06

            # Material.
            elif word in material:
                score += 0.05

            # Description is softer.
            elif word in description:
                score += 0.03

        # Prevent exact keywords from overpowering
        # semantic meaning completely.
        return min(
            score,
            0.35,
        )

    def _extract_query_words(
        self,
        query: str,
    ) -> set[str]:
        """
        Extract useful terms from customer query.
        """

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