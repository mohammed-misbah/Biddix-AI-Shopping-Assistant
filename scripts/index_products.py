import json

import faiss

from app.core.config import (
    INDEX_FILE,
    METADATA_FILE,
    PRODUCT_FILE,
    VECTOR_DIR,
)
from app.services.embedding_service import embedding_service


def build_product_text(product: dict) -> str:
    """
    Build semantic text used for embeddings.

    Keep stable descriptive information here.
    Do NOT use frequently changing values such as stock
    as primary semantic information.
    """

    semantic_fields = [
        ("Name", "name"),
        ("Category", "category"),
        ("Country", "country"),
        ("Region", "region"),
        ("Theme", "theme"),
        ("Era", "era"),
        ("Period", "period"),
        ("Year", "year"),
        ("Material", "material"),
        ("Weight", "weight"),
        ("Purity", "purity"),
        ("Denomination", "denomination"),
        ("Description", "description"),
    ]

    parts = []

    for label, field_name in semantic_fields:
        value = product.get(field_name)

        if value is not None and str(value).strip():
            parts.append(
                f"{label}: {value}."
            )

    return " ".join(parts)


def main() -> None:

    # -----------------------------------------
    # 1. Create vector-store directory
    # -----------------------------------------

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # 2. Load products
    # -----------------------------------------

    with open(
        PRODUCT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        products = json.load(file)

    if not isinstance(products, list):
        raise ValueError(
            "products.json must contain a JSON array."
        )

    if not products:
        raise ValueError(
            "products.json contains no products."
        )

    # -----------------------------------------
    # 3. Validate product IDs
    # -----------------------------------------

    product_ids = []

    for product in products:

        product_id = product.get(
            "product_id"
        )

        if not product_id:
            raise ValueError(
                "Every product must have a product_id."
            )

        product_ids.append(
            str(product_id)
        )

    if len(product_ids) != len(set(product_ids)):
        raise ValueError(
            "Duplicate product_id values found."
        )

    # -----------------------------------------
    # 4. Build semantic product text
    # -----------------------------------------

    searchable_texts = [
        build_product_text(product)
        for product in products
    ]

    # -----------------------------------------
    # 5. Create embeddings
    # -----------------------------------------

    embeddings = embedding_service.embed_texts(
        searchable_texts
    )

    if len(embeddings) != len(products):
        raise RuntimeError(
            "Embedding count does not match product count."
        )

    dimension = embeddings.shape[1]

    # -----------------------------------------
    # 6. Build FAISS cosine-similarity index
    # -----------------------------------------
    #
    # Your embedding service already uses:
    #
    # normalize_embeddings=True
    #
    # Therefore IndexFlatIP acts like cosine similarity.
    # -----------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # -----------------------------------------
    # 7. Save FAISS index
    # -----------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    # -----------------------------------------
    # 8. Save metadata
    # -----------------------------------------

    metadata = [
        {
            "product_id": str(
                product["product_id"]
            ),
            "search_text": text,
        }
        for product, text in zip(
            products,
            searchable_texts,
        )
    ]

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------
    # 9. Final checks
    # -----------------------------------------

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata counts do not match."
        )

    print(
        f"Successfully indexed {len(products)} products."
    )

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"FAISS index saved to: {INDEX_FILE}"
    )

    print(
        f"Metadata saved to: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()