import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            settings.MODEL_NAME
        )

    def embed_text(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return np.asarray(
            vector,
            dtype="float32",
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print(f"Embedding {len(texts)} texts into vectors of dimension {vectors.shape[1]}.")

        return np.asarray(
            vectors,
            dtype="float32",
        )


embedding_service = EmbeddingService()