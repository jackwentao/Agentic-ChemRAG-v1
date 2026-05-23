from math import sqrt

from backend.app.domain.documents import Chunk, Evidence
from backend.app.providers.embeddings import EmbeddingProvider


class InMemoryVectorRepository:
    expects_text_query = True

    def __init__(self) -> None:
        self.rows: list[tuple[Chunk, list[float]]] = []

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.rows.extend(zip(chunks, embeddings, strict=True))

    def search(self, query: str, embedding_provider: EmbeddingProvider, top_k: int = 5) -> list[Evidence]:
        query_vector = embedding_provider.embed_query(query)
        scored = []
        for chunk, vector in self.rows:
            if chunk.chunk_type != "child":
                continue
            scored.append((self._cosine(query_vector, vector), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Evidence(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                score=score,
            )
            for score, chunk in scored[:top_k]
        ]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sqrt(sum(a * a for a in left)) or 1.0
        right_norm = sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)
