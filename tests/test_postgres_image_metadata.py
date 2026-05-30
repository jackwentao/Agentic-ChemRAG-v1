from backend.app.domain.documents import StoredImage
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
        return FakeResult()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.search_rows = []
        self.figure_rows = []
        self.profile_rows = []

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
