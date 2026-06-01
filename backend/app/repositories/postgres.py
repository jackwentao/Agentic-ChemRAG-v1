import json
from dataclasses import asdict

from sqlalchemy.exc import DBAPIError
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.domain.documents import Chunk, DocumentProfile, DocumentProfileField, Evidence, EvidenceImage, SourceRef, StoredImage


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
                            chunk_id, document_id, owner_id, parent_id, text, page_start, page_end,
                            section_title, section_path, chunk_type, embedding
                        )
                        VALUES (
                            :chunk_id, :document_id,
                            COALESCE(:owner_id, (SELECT owner_id FROM documents WHERE document_id = :document_id), 'demo'),
                            :parent_id, :text, :page_start, :page_end,
                            :section_title, :section_path, :chunk_type, :embedding
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            owner_id = EXCLUDED.owner_id,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "owner_id": chunk.owner_id,
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

    def count_searchable_chunks(
        self,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
    ) -> int:
        filters, params = _ownership_filters(document_ids=document_ids, owner_id=owner_id)
        where = " AND ".join(["chunk_type = 'child'", *filters])
        with self.engine.begin() as conn:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM chunks WHERE {where}"),
                params,
            ).scalar_one()
        return int(count)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
        search_strategy: str = "hnsw",
        exact_timeout_ms: int | None = None,
    ) -> list[Evidence]:
        if search_strategy == "exact_then_hnsw":
            try:
                return self._search_chunks(
                    query_embedding,
                    top_k=top_k,
                    document_ids=document_ids,
                    owner_id=owner_id,
                    search_strategy="exact",
                    exact_timeout_ms=exact_timeout_ms,
                )
            except DBAPIError:
                return self._search_chunks(
                    query_embedding,
                    top_k=top_k,
                    document_ids=document_ids,
                    owner_id=owner_id,
                    search_strategy="hnsw",
                    exact_timeout_ms=None,
                )
        return self._search_chunks(
            query_embedding,
            top_k=top_k,
            document_ids=document_ids,
            owner_id=owner_id,
            search_strategy=search_strategy,
            exact_timeout_ms=exact_timeout_ms,
        )

    def _search_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None,
        owner_id: str | None,
        search_strategy: str,
        exact_timeout_ms: int | None,
    ) -> list[Evidence]:
        filter_parts, params = _ownership_filters(document_ids=document_ids, owner_id=owner_id)
        filters = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""
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
        params = {**params, "query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        with self.engine.begin() as conn:
            if search_strategy == "exact":
                if exact_timeout_ms:
                    conn.execute(text("SET LOCAL statement_timeout = :timeout"), {"timeout": f"{int(exact_timeout_ms)}ms"})
                conn.execute(text("SET LOCAL enable_indexscan = off"))
                conn.execute(text("SET LOCAL enable_bitmapscan = off"))
                conn.execute(text("SET LOCAL enable_indexonlyscan = off"))
            elif search_strategy == "hnsw":
                conn.execute(text("SET LOCAL enable_seqscan = off"))
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
        owner_id: str | None = None,
    ) -> list[Evidence]:
        filter_parts = []
        params: dict[str, object] = {}
        if document_ids:
            filter_parts.append("document_id = ANY(:document_ids)")
            params["document_ids"] = document_ids
        if owner_id:
            filter_parts.append("document_id IN (SELECT document_id FROM documents WHERE owner_id = :owner_id)")
            params["owner_id"] = owner_id
        filters = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""
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
        params = {**params, "query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        with self.engine.begin() as conn:
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
                    evidence_type="figure",
                    source_scope="page_span",
                    source_refs=[
                        SourceRef(
                            source_type="figure_caption",
                            page_start=page,
                            page_end=page,
                            section_title="Figure Index",
                            field_name="caption",
                            text=text_value[:500],
                        )
                    ],
                )
            )
        return evidence

    def upsert_document_profile(self, profile: DocumentProfile, embedding: list[float]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO document_profiles (
                        document_id, owner_id, title, abstract, conclusion, summary_text, topics, entities,
                        profile_json, outline_json, source_refs_json, figure_summary, embedding, updated_at
                    )
                    VALUES (
                        :document_id,
                        COALESCE(:owner_id, (SELECT owner_id FROM documents WHERE document_id = :document_id), 'demo'),
                        :title, :abstract, :conclusion, :summary_text,
                        CAST(:topics AS jsonb), CAST(:entities AS jsonb),
                        CAST(:profile_json AS jsonb), CAST(:outline_json AS jsonb),
                        CAST(:source_refs_json AS jsonb),
                        :figure_summary, :embedding, now()
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        conclusion = EXCLUDED.conclusion,
                        summary_text = EXCLUDED.summary_text,
                        topics = EXCLUDED.topics,
                        entities = EXCLUDED.entities,
                        profile_json = EXCLUDED.profile_json,
                        outline_json = EXCLUDED.outline_json,
                        source_refs_json = EXCLUDED.source_refs_json,
                        figure_summary = EXCLUDED.figure_summary,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ),
                {
                    "document_id": profile.document_id,
                    "owner_id": profile.owner_id,
                    "title": profile.title,
                    "abstract": profile.abstract,
                    "conclusion": profile.conclusion,
                    "summary_text": profile.summary_text,
                    "topics": json.dumps(profile.topics, ensure_ascii=False),
                    "entities": json.dumps(profile.entities, ensure_ascii=False),
                    "profile_json": json.dumps(_profile_payload(profile), ensure_ascii=False),
                    "outline_json": json.dumps([asdict(section) for section in profile.outline], ensure_ascii=False),
                    "source_refs_json": json.dumps([asdict(ref) for ref in profile.source_refs], ensure_ascii=False),
                    "figure_summary": profile.figure_summary,
                    "embedding": self._vector_literal(embedding),
                },
            )

    def upsert_document_profile_fields(
        self,
        document_id: str,
        fields: list[DocumentProfileField],
        embeddings: list[list[float]],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM document_profile_embeddings WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
            for field, embedding in zip(fields, embeddings, strict=True):
                conn.execute(
                    text(
                        """
                        INSERT INTO document_profile_embeddings (
                            profile_field_id, document_id, owner_id, field_name, field_text,
                            source_refs_json, metadata_json, embedding, updated_at
                        )
                        VALUES (
                            :profile_field_id, :document_id,
                            COALESCE(:owner_id, (SELECT owner_id FROM documents WHERE document_id = :document_id), 'demo'),
                            :field_name, :field_text,
                            CAST(:source_refs_json AS jsonb), CAST(:metadata_json AS jsonb),
                            :embedding, now()
                        )
                        ON CONFLICT (profile_field_id) DO UPDATE SET
                            owner_id = EXCLUDED.owner_id,
                            field_text = EXCLUDED.field_text,
                            source_refs_json = EXCLUDED.source_refs_json,
                            metadata_json = EXCLUDED.metadata_json,
                            embedding = EXCLUDED.embedding,
                            updated_at = now()
                        """
                    ),
                    {
                        "profile_field_id": _profile_field_id(field.document_id, field.field_name),
                        "document_id": field.document_id,
                        "owner_id": field.owner_id,
                        "field_name": field.field_name,
                        "field_text": field.text,
                        "source_refs_json": json.dumps([asdict(ref) for ref in field.source_refs], ensure_ascii=False),
                        "metadata_json": json.dumps(field.metadata, ensure_ascii=False),
                        "embedding": self._vector_literal(embedding),
                    },
                )

    def search_document_profiles(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
    ) -> list[Evidence]:
        field_evidence = self.search_document_profile_fields(
            query_embedding,
            top_k=top_k,
            document_ids=document_ids,
            owner_id=owner_id,
        )
        if field_evidence:
            return field_evidence
        filter_parts, params = _ownership_filters(document_ids=document_ids, owner_id=owner_id)
        filters = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""
        sql = text(
            f"""
            SELECT document_id, title, abstract, conclusion, summary_text, topics, entities,
                   profile_json, outline_json, source_refs_json, figure_summary,
                   1 - (embedding <=> :query_embedding) AS score
            FROM document_profiles
            WHERE embedding IS NOT NULL {filters}
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
            """
        )
        params = {**params, "query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        evidence: list[Evidence] = []
        for row in rows:
            profile = dict(row)
            text_value = str(profile.get("summary_text") or "")
            title = str(profile.get("title") or "Document Profile")
            source_refs = _source_refs_from_profile_row(profile)
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
                    evidence_type="document_profile",
                    source_scope="document_level",
                    source_refs=source_refs,
                    metadata={
                        "title": profile.get("title"),
                        "abstract": profile.get("abstract"),
                        "conclusion": profile.get("conclusion"),
                        "topics": _json_value(profile.get("topics"), []),
                        "entities": _json_value(profile.get("entities"), []),
                        "figure_summary": profile.get("figure_summary"),
                        "outline": _json_value(profile.get("outline_json"), []),
                        "profile": _json_value(profile.get("profile_json"), {}),
                    },
                )
            )
        return evidence

    def search_document_profile_fields(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
        field_names: list[str] | None = None,
        owner_id: str | None = None,
    ) -> list[Evidence]:
        filters, params = _ownership_filters(document_ids=document_ids, owner_id=owner_id)
        params = {**params, "query_embedding": self._vector_literal(query_embedding), "top_k": top_k}
        if field_names:
            filters.append("field_name = ANY(:field_names)")
            params["field_names"] = field_names
        where = " AND ".join(["embedding IS NOT NULL", *filters])
        sql = text(
            f"""
            SELECT profile_field_id, document_id, field_name, field_text,
                   source_refs_json, metadata_json,
                   1 - (embedding <=> :query_embedding) AS score
            FROM document_profile_embeddings
            WHERE {where}
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        evidence: list[Evidence] = []
        for row in rows:
            field = dict(row)
            field_name = str(field.get("field_name") or "profile_field")
            metadata = _json_value(field.get("metadata_json"), {})
            evidence.append(
                Evidence(
                    chunk_id=f"profile:{field['document_id']}:{field_name}",
                    document_id=str(field["document_id"]),
                    text=str(field.get("field_text") or ""),
                    page_start=1,
                    page_end=1,
                    section_title=f"Document Profile / {field_name}",
                    score=float(field["score"] or 0),
                    evidence_type="document_profile",
                    source_scope="document_level",
                    source_refs=_source_refs_from_json(field.get("source_refs_json"), fallback_text=str(field.get("field_text") or "")),
                    metadata={
                        **(metadata if isinstance(metadata, dict) else {}),
                        "profile_field": field_name,
                    },
                )
            )
        return evidence

    def _vector_literal(self, vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def _ownership_filters(
    *,
    document_ids: list[str] | None,
    owner_id: str | None,
) -> tuple[list[str], dict[str, object]]:
    filters: list[str] = []
    params: dict[str, object] = {}
    if document_ids:
        filters.append("document_id = ANY(:document_ids)")
        params["document_ids"] = document_ids
    if owner_id:
        filters.append("owner_id = :owner_id")
        params["owner_id"] = owner_id
    return filters, params


def _profile_payload(profile: DocumentProfile) -> dict[str, object]:
    return {
        "document_id": profile.document_id,
        "owner_id": profile.owner_id,
        "title": profile.title,
        "abstract": profile.abstract,
        "conclusion": profile.conclusion,
        "topics": profile.topics,
        "entities": profile.entities,
        "typed_entities": [asdict(entity) for entity in profile.typed_entities],
        "key_findings": profile.key_findings,
        "methods": profile.methods,
        "properties": profile.properties,
        "applications": profile.applications,
        "limitations": profile.limitations,
        "extraction_method": profile.extraction_method,
        "extraction_warnings": profile.extraction_warnings,
    }


def _profile_field_id(document_id: str, field_name: str) -> str:
    return f"{document_id}:{field_name}"


def _source_refs_from_profile_row(profile: dict) -> list[SourceRef]:
    return _source_refs_from_json(profile.get("source_refs_json"), fallback_text=str(profile.get("summary_text") or ""))


def _source_refs_from_json(value, fallback_text: str) -> list[SourceRef]:
    raw_refs = _json_value(value, [])
    refs: list[SourceRef] = []
    if isinstance(raw_refs, list):
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict):
                continue
            refs.append(
                SourceRef(
                    source_type=str(raw_ref.get("source_type") or "profile_field"),
                    page_start=_optional_int(raw_ref.get("page_start")),
                    page_end=_optional_int(raw_ref.get("page_end")),
                    section_title=_optional_str(raw_ref.get("section_title")),
                    field_name=_optional_str(raw_ref.get("field_name")),
                    text=_optional_str(raw_ref.get("text")),
                )
            )
    if refs:
        return refs
    return [
        SourceRef(
            source_type="generated_profile",
            field_name="summary_text",
            text=fallback_text[:500],
        )
    ]


def _json_value(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _optional_str(value) -> str | None:
    return None if value is None else str(value)


def _optional_int(value) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
