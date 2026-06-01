from backend.app.domain.documents import DocumentProfile, DocumentProfileField, StoredImage
from backend.app.database import prepare_database
from backend.app.repositories.postgres import PgVectorRepository


class FakeMappings:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    def mappings(self):
        return FakeMappings(self.rows)


class FakeConnection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.engine.calls.append((sql, params or {}))
        if "WITH ranked AS" in sql:
            return FakeResult(self.engine.search_rows)
        if "FROM figures" in sql and "ORDER BY embedding" in sql:
            return FakeResult(self.engine.figure_rows)
        if "FROM document_profiles" in sql and "ORDER BY embedding" in sql:
            return FakeResult(self.engine.profile_rows)
        if "FROM document_profile_embeddings" in sql and "ORDER BY embedding" in sql:
            return FakeResult(self.engine.profile_field_rows)
        return FakeResult()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.search_rows = []
        self.figure_rows = []
        self.profile_rows = []
        self.profile_field_rows = []

    def begin(self):
        return FakeConnection(self)


def test_prepare_database_adds_image_metadata_columns():
    engine = FakeEngine()

    prepare_database(engine)

    statements = "\n".join(sql for sql, _params in engine.calls)
    assert "ALTER TABLE images ADD COLUMN IF NOT EXISTS caption TEXT" in statements
    assert "ALTER TABLE images ADD COLUMN IF NOT EXISTS nearby_text TEXT" in statements
    assert "CREATE TABLE IF NOT EXISTS figures" in statements
    assert "CREATE TABLE IF NOT EXISTS document_profiles" in statements
    assert "CREATE TABLE IF NOT EXISTS document_profile_embeddings" in statements
    assert "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS owner_id TEXT" in statements
    assert "ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS owner_id TEXT" in statements
    assert "ALTER TABLE document_profile_embeddings ADD COLUMN IF NOT EXISTS owner_id TEXT" in statements
    assert "CREATE INDEX IF NOT EXISTS chunks_owner_document_idx" in statements
    assert "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx" in statements
    assert "CREATE INDEX IF NOT EXISTS figures_embedding_idx" in statements
    assert "CREATE INDEX IF NOT EXISTS document_profiles_embedding_idx" not in statements
    assert "DROP INDEX IF EXISTS document_profiles_embedding_idx" in statements


def test_pgvector_repository_add_images_persists_caption_metadata():
    engine = FakeEngine()
    repository = PgVectorRepository(engine)

    repository.add_images(
        [
            StoredImage(
                image_id="img1",
                document_id="doc1",
                page=2,
                path="/tmp/img1.png",
                extension="png",
                caption="Figure 1. PL spectra.",
                nearby_text="Nearby PL discussion.",
                label="Figure 1",
                bbox=(1.0, 2.0, 3.0, 4.0),
                source_type="page_crop",
            )
        ]
    )

    _sql, params = engine.calls[0]
    assert params["caption"] == "Figure 1. PL spectra."
    assert params["nearby_text"] == "Nearby PL discussion."
    assert params["label"] == "Figure 1"
    assert params["bbox"] == "[1.0, 2.0, 3.0, 4.0]"
    assert params["source_type"] == "page_crop"


def test_pgvector_repository_search_returns_image_caption_metadata():
    engine = FakeEngine()
    engine.search_rows = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "Photoluminescence increases after doping.",
            "page_start": 2,
            "page_end": 2,
            "section_title": "Results",
            "score": 0.91,
            "image_ids": ["img1"],
            "images": [
                {
                    "image_id": "img1",
                    "page": 2,
                    "width": 640,
                    "height": 480,
                    "label": "Figure 1",
                    "caption": "Figure 1. PL spectra.",
                    "nearby_text": "Nearby PL discussion.",
                }
            ],
        }
    ]
    repository = PgVectorRepository(engine)

    evidence = repository.search([0.1, 0.2], top_k=1)

    assert evidence[0].images[0].caption == "Figure 1. PL spectra."
    assert evidence[0].images[0].nearby_text == "Nearby PL discussion."
    assert evidence[0].images[0].label == "Figure 1"


def test_pgvector_repository_chunk_search_filters_by_owner_and_can_use_exact_timeout():
    engine = FakeEngine()
    engine.search_rows = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "Photoluminescence increases after doping.",
            "page_start": 2,
            "page_end": 2,
            "section_title": "Results",
            "score": 0.91,
            "image_ids": [],
            "images": [],
        }
    ]
    repository = PgVectorRepository(engine)

    evidence = repository.search(
        [0.1, 0.2],
        top_k=1,
        document_ids=["doc1"],
        owner_id="tenant-a",
        search_strategy="exact_then_hnsw",
        exact_timeout_ms=250,
    )

    statements = "\n".join(sql for sql, _params in engine.calls)
    search_params = engine.calls[-1][1]
    assert evidence[0].chunk_id == "c1"
    assert "SET LOCAL statement_timeout" in statements
    assert "SET LOCAL enable_indexscan = off" in statements
    assert "owner_id = :owner_id" in statements
    assert search_params["owner_id"] == "tenant-a"
    assert search_params["document_ids"] == ["doc1"]


def test_pgvector_repository_add_figures_and_search_returns_figure_evidence():
    engine = FakeEngine()
    engine.figure_rows = [
        {
            "figure_id": "img1",
            "image_id": "img1",
            "document_id": "doc1",
            "page": 3,
            "label": "Figure 2",
            "caption": "Figure 2. PL spectra.",
            "nearby_text": None,
            "width": 640,
            "height": 480,
            "score": 0.88,
        }
    ]
    repository = PgVectorRepository(engine)

    repository.add_figures(
        [
            StoredImage(
                image_id="img1",
                document_id="doc1",
                page=3,
                path="images/img1.png",
                extension="png",
                label="Figure 2",
                caption="Figure 2. PL spectra.",
            )
        ],
        [[0.1, 0.2]],
    )
    evidence = repository.search_figures([0.1, 0.2], top_k=1)

    assert engine.calls[0][1]["figure_id"] == "img1"
    assert evidence[0].chunk_id == "figure:img1"
    assert evidence[0].images[0].label == "Figure 2"


def test_pgvector_repository_vector_searches_do_not_disable_index_scans():
    engine = FakeEngine()
    engine.figure_rows = [
        {
            "figure_id": "fig1",
            "image_id": "img1",
            "document_id": "doc1",
            "page": 1,
            "label": "Figure 1",
            "caption": "Band gap trend.",
            "nearby_text": None,
            "width": 320,
            "height": 240,
            "score": 0.91,
        }
    ]
    engine.profile_rows = [
        {
            "document_id": "doc1",
            "title": "Photocatalyst profile",
            "abstract": None,
            "conclusion": None,
            "summary_text": "Summary for photocatalyst document.",
            "topics": [],
            "entities": [],
            "profile_json": {},
            "outline_json": [],
            "source_refs_json": [],
            "figure_summary": None,
            "score": 0.87,
        }
    ]
    repository = PgVectorRepository(engine)

    figure_evidence = repository.search_figures([0.1, 0.2], top_k=1)
    profile_evidence = repository.search_document_profiles([0.1, 0.2], top_k=1)

    statements = "\n".join(sql for sql, _params in engine.calls)
    assert "SET LOCAL enable_indexscan = off" not in statements
    assert "SET LOCAL enable_bitmapscan = off" not in statements
    assert figure_evidence[0].chunk_id == "figure:fig1"
    assert profile_evidence[0].chunk_id == "profile:doc1"


def test_pgvector_repository_upserts_and_searches_document_profile():
    engine = FakeEngine()
    engine.profile_rows = [
        {
            "document_id": "doc1",
            "title": "Photocatalyst profile",
            "abstract": "Abstract text.",
            "conclusion": "Conclusion text.",
            "summary_text": "Document level summary.",
            "topics": ["photocatalysis"],
            "entities": ["TiO2"],
            "profile_json": {"conclusion": "Conclusion text."},
            "outline_json": [],
            "source_refs_json": [],
            "figure_summary": "Figure summary.",
            "score": 0.82,
        }
    ]
    repository = PgVectorRepository(engine)

    repository.upsert_document_profile(
        DocumentProfile(
            document_id="doc1",
            title="Photocatalyst profile",
            abstract="Abstract text.",
            conclusion="Conclusion text.",
            summary_text="Document level summary.",
            topics=["photocatalysis"],
            entities=["TiO2"],
            figure_summary="Figure summary.",
        ),
        [0.1, 0.2],
    )
    evidence = repository.search_document_profiles([0.1, 0.2], top_k=1)

    assert engine.calls[0][1]["document_id"] == "doc1"
    assert engine.calls[0][1]["conclusion"] == "Conclusion text."
    assert engine.calls[0][1]["topics"] == '["photocatalysis"]'
    assert engine.calls[0][1]["entities"] == '["TiO2"]'
    assert "outline_json" in engine.calls[0][1]
    assert evidence[0].chunk_id == "profile:doc1"
    assert evidence[0].section_title == "Photocatalyst profile"
    assert evidence[0].evidence_type == "document_profile"
    assert evidence[0].source_scope == "document_level"
    assert evidence[0].metadata["conclusion"] == "Conclusion text."


def test_pgvector_repository_upserts_and_searches_document_profile_fields():
    engine = FakeEngine()
    engine.profile_field_rows = [
        {
            "profile_field_id": "doc1:properties",
            "document_id": "doc1",
            "field_name": "properties",
            "field_text": "photoluminescence emission intensity",
            "source_refs_json": [],
            "metadata_json": {"profile_title": "Photocatalyst profile"},
            "score": 0.91,
        }
    ]
    repository = PgVectorRepository(engine)

    repository.upsert_document_profile_fields(
        "doc1",
        [
            DocumentProfileField(
                document_id="doc1",
                field_name="properties",
                text="photoluminescence emission intensity",
                metadata={"profile_title": "Photocatalyst profile"},
            )
        ],
        [[0.1, 0.2]],
    )
    evidence = repository.search_document_profiles([0.1, 0.2], top_k=1)

    statements = "\n".join(sql for sql, _params in engine.calls)
    assert "DELETE FROM document_profile_embeddings" in statements
    assert "INSERT INTO document_profile_embeddings" in statements
    assert evidence[0].chunk_id == "profile:doc1:properties"
    assert evidence[0].evidence_type == "document_profile"
    assert evidence[0].source_scope == "document_level"
    assert evidence[0].metadata["profile_field"] == "properties"
