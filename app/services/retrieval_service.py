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

    # For explicit inventory questions such as:
    # "Do you sell German coins?"
    #
    # We need stronger evidence than semantic similarity alone.
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
                f"FAISS index not found: {INDEX_FILE}. "
                "Run: python -m scripts.index_products"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}. "
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
            self.metadata = json.load(file)

        self.products_by_id = {
            str(product["product_id"]): product
            for product in product_service.products
        }

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "FAISS index and metadata are out of sync. "
                "Run the indexing script again."
            )

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

        # =================================================
        # 1. Hard filters
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
        # 2. Query embedding
        # =================================================

        query_vector = (
            embedding_service.embed_text(
                query
            )
            .reshape(1, -1)
        )

        # Small catalog:
        # search all products.
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
        # 3. FAISS semantic search
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

            semantic_score = float(score)

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
        # 4. Apply hard filters
        # =================================================

        filtered_candidates: list[
            dict[str, Any]
        ] = []

        for item in candidates:

            product = item["product"]

            # Year
            if year is not None:

                try:
                    product_year = int(
                        product.get("year")
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

            # Stock
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
        # 5. Query keywords
        # =================================================

        query_words = (
            self._extract_query_words(
                query_lower
            )
        )

        # =================================================
        # 6. Hybrid ranking
        # =================================================

        ranked_candidates: list[
            dict[str, Any]
        ] = []

        for item in filtered_candidates:

            product = item["product"]

            lexical_score = (
                self._calculate_lexical_score(
                    product=product,
                    query_words=query_words,
                )
            )

            item["lexical_score"] = lexical_score

            item["final_score"] = (
                item["semantic_score"]
                + lexical_score
            )

            # IMPORTANT:
            #
            # Explicit catalog existence question:
            #
            # "Do you sell German coins?"
            #
            # If "German" does not match any catalog
            # fields, do not return unrelated products.
            if (
                strict_match
                and query_words
                and lexical_score
                < self.STRICT_MIN_LEXICAL_SCORE
            ):
                continue

            ranked_candidates.append(
                item
            )

        # =================================================
        # 7. Sort
        # =================================================

        ranked_candidates.sort(
            key=lambda item: (
                item["final_score"],
                item["semantic_score"],
            ),
            reverse=True,
        )

        results = ranked_candidates[
            :limit
        ]

        # =================================================
        # 8. Development output
        # =================================================

        print("\n--- RAG RESULTS ---")
        print(f"Query: {query}")
        print(f"Strict match: {strict_match}")
        print(
            f"Semantic candidates: "
            f"{len(candidates)}"
        )
        print(
            f"After filters/ranking: "
            f"{len(ranked_candidates)}"
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

        print("-------------------\n")

        logger.info(
            "RAG query=%r strict=%s "
            "candidates=%d returned=%d",
            query,
            strict_match,
            len(candidates),
            len(results),
        )

        return [
            item["product"]
            for item in results
        ]

    def _calculate_lexical_score(
        self,
        product: dict[str, Any],
        query_words: set[str],
    ) -> float:

        name = str(
            product.get("name", "")
        ).lower()

        category = str(
            product.get("category", "")
        ).lower()

        country = str(
            product.get("country", "")
        ).lower()

        region = str(
            product.get("region", "")
        ).lower()

        theme = str(
            product.get("theme", "")
        ).lower()

        era = str(
            product.get("era", "")
        ).lower()

        material = str(
            product.get("material", "")
        ).lower()

        description = str(
            product.get(
                "description",
                "",
            )
        ).lower()

        score = 0.0

        for word in query_words:

            if word in name:
                score += 0.08

            elif (
                word in country
                or word in region
            ):
                score += 0.07

            elif (
                word in category
                or word in theme
                or word in era
            ):
                score += 0.06

            elif word in material:
                score += 0.05

            elif word in description:
                score += 0.03

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