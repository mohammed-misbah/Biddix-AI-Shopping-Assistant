import json
from typing import Any

import faiss

from app.core.config import (
    INDEX_FILE,
    METADATA_FILE,
)
from app.services.embedding_service import embedding_service
from app.services.product_service import product_service


class RetrievalService:

    def __init__(self):

        # Load FAISS index
        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # Load metadata
        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # Quick lookup for full product details
        self.products_by_id = {
            product["product_id"]: product
            for product in product_service.products
        }

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> list[dict[str, Any]]:

        query = query.strip()

        if not query:
            return []

        # 1. Convert query into vector

        query_vector = embedding_service.embed_text(
            query
        )

        query_vector = query_vector.reshape(
            1,
            -1,
        )

        # 2. Search similar products

        scores, indexes = self.index.search(
            query_vector,
            candidate_limit,
        )

        candidates = []

        for score, index_position in zip(
            scores[0],
            indexes[0],
        ):

            if index_position < 0:
                continue

            metadata_item = self.metadata[
                index_position
            ]

            product_id = metadata_item[
                "product_id"
            ]

            product = self.products_by_id.get(
                product_id
            )

            if not product:
                continue

            candidates.append({
                "product": product,
                "semantic_score": float(score),
            })

        # 3. Extract filters

        query_lower = query.lower()

        year = product_service._extract_year(
            query_lower
        )

        max_price = product_service._extract_max_price(
            query_lower
        )

        material = product_service._extract_material(
            query_lower
        )

        wants_in_stock = (
            "in stock" in query_lower
            or "available" in query_lower
            or "availability" in query_lower
        )

        # 4. Apply hard filters

        filtered_candidates = []

        for item in candidates:

            product = item["product"]

            # Year
            if year is not None:
                if product.get("year") != year:
                    continue

            # Material
            if material is not None:
                product_material = str(
                    product.get(
                        "material",
                        "",
                    )
                ).lower()

                if product_material != material:
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
                except (TypeError, ValueError):
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
                except (TypeError, ValueError):
                    stock = 0

                if stock <= 0:
                    continue

            filtered_candidates.append(item)

        # 5. Sort by semantic score

        filtered_candidates.sort(
            key=lambda item: item[
                "semantic_score"
            ],
            reverse=True,
        )

        # 6. Return top products

        return [
            item["product"]
            for item in filtered_candidates[
                :limit
            ]
        ]


retrieval_service = RetrievalService()