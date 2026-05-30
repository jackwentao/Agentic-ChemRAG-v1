import json

from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.domain.documents import Chunk, DocumentProfile, Evidence, EvidenceImage, StoredImage


class PgVectorRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def healthcheck(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("SELECT 1"))

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self.engine.begin() as conn:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                conn.execute(
                    text(
                        """
                        INSERT INTO chunks (
                            chunk_id, document_id, parent_id, text, page_start, page_end,
                            section_title, section_path, chunk_type, embedding
                        )
                        VALUES (
                            :chunk_id, :document_id, :parent_id, :text, :page_start, :page_end,
                            :section_title, :section_path, :chunk_type, :embedding
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "parent_id": chunk.parent_id,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "section_title": chunk.section_title,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "embedding": self._vector_literal(embedding),
                    },
                )

    def add_document(
        self,
        document_id: str,
        filename: str,
        owner_id: str = "demo",
        file_hash: str | None = None,
        file_size: int | None = None,
        original_filename: str | None = None,
        indexing_status: str | None = None,
        pdf_object_key: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO documents (
                        document_id, owner_id, filename, file_hash, file_size,
                        original_filename, indexing_status, pdf_object_key
                    )
                    VALUES (
                        :document_id, :owner_id, :filename, :file_hash, :file_size,
                        :original_filename, COALESCE(:indexing_status, 'pending'), :pdf_object_key
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        filename = EXCLUDED.filename,
                        file_hash = COALESCE(EXCLUDED.file_hash, documents.file_hash),
                        file_size = COALESCE(EXCLUDED.file_size, documents.file_size),
                        original_filename = COALESCE(EXCLUDED.original_filename, documents.original_filename),
                        pdf_object_key = COALESCE(EXCLUDED.pdf_object_key, documents.pdf_object_key),
                        indexing_status = COALESCE(:indexing_status, documents.indexing_status),
                        indexing_updated_at = CASE
                            WHEN :indexing_status IS NULL THEN documents.indexing_updated_at
                            ELSE now()
                        END
                    """
                ),
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "filename": filename,
                    "file_hash": file_hash,
                    "file_size": file_size,
                    "original_filename": original_filename or filename,
                    "indexing_status": indexing_status,
                    "pdf_object_key": pdf_object_key,
                },
            )

    def update_document_indexing_status(
        self,
        document_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE documents
                    SET indexing_status = :status,
                        indexing_error = :error,
                        indexing_updated_at = now(),
                        indexed_at = CASE WHEN :status = 'succeeded' THEN now() ELSE indexed_at END
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id, "status": status, "error": error},
            )

    def mark_stale_indexing_documents_failed(
        self,
        owner_id: str = "demo",
        stale_after_minutes: int = 60,
    ) -> list[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    UPDATE documents
                    SET indexing_status = 'failed',
                        indexing_error = :error,
                        indexing_updated_at = now()
                    WHERE owner_id = :owner_id
                      AND indexing_status IN ('pending', 'running')
                      AND COALESCE(indexing_updated_at, created_at) < now() - (:stale_after_minutes * interval '1 minute')
                    RETURNING document_id
                    """
                ),
                {
                    "owner_id": owner_id,
                    "stale_after_minutes": stale_after_minutes,
                    "error": f"Indexing task timed out after {stale_after_minutes} minutes without a terminal status.",
                },
            ).scalars().all()
        return [str(row) for row in rows]

    def find_document_by_hash(self, owner_id: str, file_hash: str) -> dict[str, object] | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT documents.document_id,
                           COALESCE(documents.original_filename, documents.filename) AS filename,
                           documents.created_at,
                           documents.file_hash,
                           documents.indexing_status,
                           documents.indexing_error,
                           COUNT(chunks.chunk_id) AS chunk_count,
                           documents.indexing_status = 'succeeded' AND COUNT(chunks.chunk_id) > 0 AS is_indexed
                    FROM documents
                    LEFT JOIN chunks ON chunks.document_id = documents.document_id
                    WHERE documents.owner_id = :owner_id
                      AND documents.file_hash = :file_hash
                    GROUP BY documents.document_id, documents.original_filename, documents.filename,
                             documents.created_at, documents.file_hash
                             , documents.indexing_status, documents.indexing_error
                    LIMIT 1
                    """
                ),
                {"owner_id": owner_id, "file_hash": file_hash},
            ).mappings().first()
        return dict(row) if row else None

    def update_document_hash(
        self,
        document_id: str,
        file_hash: str,
        file_size: int,
        original_filename: str | None = None,
        pdf_object_key: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE documents
                    SET file_hash = :file_hash,
                        file_size = :file_size,
                        original_filename = COALESCE(:original_filename, documents.original_filename),
                        pdf_object_key = COALESCE(:pdf_object_key, documents.pdf_object_key)
                    WHERE document_id = :document_id
                    """
                ),
                {
                    "document_id": document_id,
                    "file_hash": file_hash,
                    "file_size": file_size,
                    "original_filename": original_filename,
                    "pdf_object_key": pdf_object_key,
                },
            )

    def get_document_asset(self, document_id: str) -> dict[str, object] | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT document_id,
                           COALESCE(original_filename, filename) AS filename,
                           pdf_object_key
                    FROM documents
                    WHERE document_id = :document_id
                    LIMIT 1
                    """
                ),
                {"document_id": document_id},
            ).mappings().first()
        return dict(row) if row else None

    def get_image_asset(self, image_id: str) -> dict[str, object] | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT image_id, document_id, page, path, extension
                    FROM images
                    WHERE image_id = :image_id
                    LIMIT 1
                    """
                ),
                {"image_id": image_id},
            ).mappings().first()
        return dict(row) if row else None

    def add_images(self, images: list[StoredImage]) -> None:
        if not images:
            return
        with self.engine.begin() as conn:
            for image in images:
                conn.execute(
                    text(
                        """
                        INSERT INTO images (
                            image_id, document_id, page, path, extension, width, height,
                            label, caption, nearby_text, bbox, source_type
                        )
                        VALUES (
                            :image_id, :document_id, :page, :path, :extension, :width, :height,
                            :label, :caption, :nearby_text, CAST(:bbox AS jsonb), :source_type
                        )
                        ON CONFLICT (image_id) DO UPDATE SET
                            path = EXCLUDED.path,
                            extension = EXCLUDED.extension,
                            width = EXCLUDED.width,
                            height = EXCLUDED.height,
                            label = EXCLUDED.label,
                            caption = EXCLUDED.caption,
                            nearby_text = EXCLUDED.nearby_text,
                            bbox = EXCLUDED.bbox,
                            source_type = EXCLUDED.source_type
                        """
                    ),
                    {
                        "image_id": image.image_id,
                        "document_id": image.document_id,
                        "page": image.page,
                        "path": image.path,
                        "extension": image.extension,
                        "width": image.width,
                        "height": image.height,
                        "label": image.label,
                        "caption": image.caption,
                        "nearby_text": image.nearby_text,
                        "bbox": json.dumps(image.bbox) if image.bbox else None,
                        "source_type": image.source_type,
                    },
                )

    def add_figures(self, images: list[StoredImage], embeddings: list[list[float]]) -> None:
        figure_images = [image for image in images if image.caption or image.nearby_text]
        if not figure_images:
            return
        with self.engine.begin() as conn:
            for image, embedding in zip(figure_images, embeddings, strict=True):
                conn.execute(
                    text(
                        """
                        INSERT INTO figures (
                            figure_id, image_id, document_id, page, label, caption, nearby_text,
                            bbox, path, extension, width, height, source_type, embedding
                        )
                        VALUES (
                            :figure_id, :image_id, :document_id, :page, :label, :caption, :nearby_text,
                            CAST(:bbox AS jsonb), :path, :extension, :width, :height, :source_type, :embedding
                        )
                        ON CONFLICT (figure_id) DO UPDATE SET
                            image_id = EXCLUDED.image_id,
                            document_id = EXCLUDED.document_id,
                            page = EXCLUDED.page,
                            label = EXCLUDED.label,
                            caption = EXCLUDED.caption,
                            nearby_text = EXCLUDED.nearby_text,
                            bbox = EXCLUDED.bbox,
                            path = EXCLUDED.path,
                            extension = EXCLUDED.extension,
                            width = EXCLUDED.width,
                            height = EXCLUDED.height,
                            source_type = EXCLUDED.source_type,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "figure_id": image.image_id,
                        "image_id": image.image_id,
                        "document_id": image.document_id,
                        "page": image.page,
                        "label": image.label,
                        "caption": image.caption,
                        "nearby_text": image.nearby_text,
                        "bbox": json.dumps(image.bbox) if image.bbox else None,
                        "path": image.path,
                        "extension": image.extension,
                        "width": image.width,
                        "height": image.height,
                        "source_type": image.source_type,
                        "embedding": self._vector_literal(embedding),
                    },
                )

    def list_documents(self, owner_id: str = "demo") -> list[dict[str, object]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT documents.document_id, documents.owner_id,
                           COALESCE(documents.original_filename, documents.filename) AS filename,
                           documents.created_at,
                           documents.file_hash,
                           documents.indexing_status,
                           documents.indexing_error,
                           COUNT(chunks.chunk_id) AS chunk_count,
                           documents.indexing_status = 'succeeded' AND COUNT(chunks.chunk_id) > 0 AS is_indexed
                    FROM documents
                    LEFT JOIN chunks ON chunks.document_id = documents.document_id
                    WHERE documents.owner_id = :owner_id
                    GROUP BY documents.document_id, documents.owner_id, documents.original_filename,
                             documents.filename, documents.created_at, documents.file_hash,
                             documents.indexing_status, documents.indexing_error
                    ORDER BY documents.created_at DESC
                    """
                ),
                {"owner_id": owner_id},
            ).mappings().all()
        return [dict(row) for row in rows]

    def list_indexed_document_ids(self, owner_id: str = "demo") -> list[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT documents.document_id
                    FROM documents
                    JOIN chunks ON chunks.document_id = documents.document_id
                    WHERE documents.owner_id = :owner_id
                      AND documents.indexing_status = 'succeeded'
                    GROUP BY documents.document_id
                    HAVING COUNT(chunks.chunk_id) > 0
                    ORDER BY MAX(documents.created_at) DESC
                    """
                ),
                {"owner_id": owner_id},
            ).scalars().all()
        return [str(row) for row in rows]

    def search(self, query_embedding: list[float], top_k: int = 5, document_ids: list[str] | None = None) -> list[Evidence]:
        filters = "AND document_id = ANY(:document_ids)" if document_ids else ""
        sql = text(
            f"""
            WITH ranked AS (
                SELECT chunk_id, document_id, text, page_start, page_end, section_title,
                       1 - (embedding <=> :query_embedding) AS score
                FROM chunks
                WHERE chunk_type = 'child' {filters}
                ORDER BY embedding <=> :query_embedding
                LIMIT :top_k
            )
            SELECT ranked.*,
                   COALESCE(image_candidates.image_ids, ARRAY[]::text[]) AS image_ids,
                   COALESCE(image_candidates.images, '[]'::jsonb) AS images
            FROM ranked
            LEFT JOIN LATERAL (
                SELECT array_agg(limited.image_id ORDER BY limited.page, limited.image_id) AS image_ids,
                       jsonb_agg(
                           jsonb_build_object(
                               'image_id', limited.image_id,
                               'page', limited.page,
                               'width', limited.width,
                               'height', limited.height,
                               'label', limited.label,
                               'caption', limited.caption,
                               'nearby_text', limited.nearby_text
                           )
                           ORDER BY limited.page, limited.image_id
                       ) AS images
                FROM (
                    SELECT image_id, page, width, height, label, caption, nearby_text
                    FROM images
                    WHERE images.document_id = ranked.document_id
                      AND images.page BETWEEN ranked.page_start AND ranked.page_end
                    ORDER BY images.page, images.image_id
                    LIMIT 5
                ) AS limited
            ) AS image_candidates ON TRUE
            ORDER BY ranked.score DESC
            LIMIT :top_k
            """
        )
        params = {"query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        if document_ids:
            params["document_ids"] = document_ids
        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()
        evidence = []
        for row in rows:
            data = dict(row)
            images = data.pop("images") or []
            data["images"] = [EvidenceImage(**dict(image)) for image in images]
            evidence.append(Evidence(**data))
        return evidence

    def search_figures(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        document_ids: list[str] | None = None,
    ) -> list[Evidence]:
        filters = "AND document_id = ANY(:document_ids)" if document_ids else ""
        sql = text(
            f"""
            SELECT figure_id, image_id, document_id, page, label, caption, nearby_text,
                   width, height,
                   1 - (embedding <=> :query_embedding) AS score
            FROM figures
            WHERE embedding IS NOT NULL {filters}
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
            """
        )
        params = {"query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        if document_ids:
            params["document_ids"] = document_ids
        with self.engine.begin() as conn:
            conn.execute(text("SET LOCAL enable_indexscan = off"))
            conn.execute(text("SET LOCAL enable_bitmapscan = off"))
            rows = conn.execute(sql, params).mappings().all()

        evidence: list[Evidence] = []
        for row in rows:
            figure = dict(row)
            caption = str(figure.get("caption") or figure.get("nearby_text") or "")
            label = str(figure.get("label") or "")
            text_value = f"{label} {caption}".strip()
            page = int(figure["page"])
            image = EvidenceImage(
                image_id=str(figure["image_id"]),
                page=page,
                width=figure.get("width"),
                height=figure.get("height"),
                label=figure.get("label"),
                caption=figure.get("caption"),
                nearby_text=figure.get("nearby_text"),
            )
            evidence.append(
                Evidence(
                    chunk_id=f"figure:{figure['figure_id']}",
                    document_id=str(figure["document_id"]),
                    text=text_value,
                    page_start=page,
                    page_end=page,
                    section_title="Figure Index",
                    score=float(figure["score"] or 0),
                    image_ids=[image.image_id],
                    images=[image],
                )
            )
        return evidence

    def upsert_document_profile(self, profile: DocumentProfile, embedding: list[float]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO document_profiles (
                        document_id, title, abstract, summary_text, topics, entities,
                        figure_summary, embedding, updated_at
                    )
                    VALUES (
                        :document_id, :title, :abstract, :summary_text,
                        CAST(:topics AS jsonb), CAST(:entities AS jsonb),
                        :figure_summary, :embedding, now()
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        summary_text = EXCLUDED.summary_text,
                        topics = EXCLUDED.topics,
                        entities = EXCLUDED.entities,
                        figure_summary = EXCLUDED.figure_summary,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ),
                {
                    "document_id": profile.document_id,
                    "title": profile.title,
                    "abstract": profile.abstract,
                    "summary_text": profile.summary_text,
                    "topics": json.dumps(profile.topics, ensure_ascii=False),
                    "entities": json.dumps(profile.entities, ensure_ascii=False),
                    "figure_summary": profile.figure_summary,
                    "embedding": self._vector_literal(embedding),
                },
            )

    def search_document_profiles(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[Evidence]:
        filters = "AND document_id = ANY(:document_ids)" if document_ids else ""
        sql = text(
            f"""
            SELECT document_id, title, abstract, summary_text, topics, entities, figure_summary,
                   1 - (embedding <=> :query_embedding) AS score
            FROM document_profiles
            WHERE embedding IS NOT NULL {filters}
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
            """
        )
        params = {"query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        if document_ids:
            params["document_ids"] = document_ids
        with self.engine.begin() as conn:
            conn.execute(text("SET LOCAL enable_indexscan = off"))
            conn.execute(text("SET LOCAL enable_bitmapscan = off"))
            rows = conn.execute(sql, params).mappings().all()

        evidence: list[Evidence] = []
        for row in rows:
            profile = dict(row)
            text_value = str(profile.get("summary_text") or "")
            title = str(profile.get("title") or "Document Profile")
            evidence.append(
                Evidence(
                    chunk_id=f"profile:{profile['document_id']}",
                    document_id=str(profile["document_id"]),
                    text=text_value,
                    page_start=1,
                    page_end=1,
                    section_title=title,
                    score=float(profile["score"] or 0),
                    image_ids=[],
                    images=[],
                )
            )
        return evidence

    def _vector_literal(self, vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
