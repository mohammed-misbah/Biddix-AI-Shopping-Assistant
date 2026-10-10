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
    return " ".join([
        f"Name: {product.get('name', '')}.",
        f"Category: {product.get('category', '')}.",
        f"Year: {product.get('year', '')}.",
        f"Material: {product.get('material', '')}.",
        f"Weight: {product.get('weight', '')}.",
        f"Purity: {product.get('purity', '')}.",
        f"Description: {product.get('description', '')}.",
    ])


def main():

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PRODUCT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        products = json.load(file)

    searchable_texts = [
        build_product_text(product)
        for product in products
    ]

    embeddings = embedding_service.embed_texts(
        searchable_texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    metadata = [
        {
            "product_id": product["product_id"],
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

    print(
        f"Successfully indexed {len(products)} products."
    )


if __name__ == "__main__":
    main()