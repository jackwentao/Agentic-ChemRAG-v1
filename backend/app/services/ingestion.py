from pathlib import Path

from backend.app.domain.documents import ParsedDocument
from backend.app.parsing.chunker import ChunkingStrategy
from backend.app.parsing.parser import DocumentParser
from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.services.document_profile import DocumentProfileBuilder
from backend.app.services.image_store import LocalImageStore


class IngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
        repository,
        image_store: LocalImageStore | None = None,
        profile_builder: DocumentProfileBuilder | None = None,
        max_indexed_figures_per_document: int = 120,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.image_store = image_store
        self.profile_builder = profile_builder or DocumentProfileBuilder()
        self.max_indexed_figures_per_document = max_indexed_figures_per_document

    def ingest(
        self,
        document_id: str,
        file_path: Path,
        filename: str | None = None,
        file_hash: str | None = None,
        file_size: int | None = None,
    ) -> int:
        parsed = self.parser.parse(document_id, file_path)
        self._save_document(parsed, filename=filename, file_hash=file_hash, file_size=file_size)
        chunks = self.chunker.chunk(parsed)
        images = self.image_store.save_images(parsed) if self.image_store else []
        if hasattr(self.repository, "add_images"):
            self.repository.add_images(images)
        figure_images = [image for image in images if image.caption or image.nearby_text][: self.max_indexed_figures_per_document]
        if figure_images and hasattr(self.repository, "add_figures"):
            figure_embeddings = self.embedding_provider.embed_documents([self._figure_text(image) for image in figure_images])
            self.repository.add_figures(figure_images, figure_embeddings)
        if hasattr(self.repository, "upsert_document_profile"):
            profile = self.profile_builder.build(parsed)
            profile_embedding = self.embedding_provider.embed_query(profile.summary_text)
            self.repository.upsert_document_profile(profile, profile_embedding)
            profile_fields = self.profile_builder.field_texts(profile)
            if profile_fields and hasattr(self.repository, "upsert_document_profile_fields"):
                field_embeddings = self.embedding_provider.embed_documents([field.text for field in profile_fields])
                self.repository.upsert_document_profile_fields(profile.document_id, profile_fields, field_embeddings)
        indexable_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in indexable_chunks])
        self.repository.add_chunks(indexable_chunks, embeddings)
        return len(indexable_chunks)

    def _save_document(
        self,
        parsed: ParsedDocument,
        filename: str | None = None,
        file_hash: str | None = None,
        file_size: int | None = None,
    ) -> None:
        if hasattr(self.repository, "add_document"):
            self.repository.add_document(
                parsed.document_id,
                filename or parsed.filename,
                owner_id="demo",
                file_hash=file_hash,
                file_size=file_size,
                original_filename=filename or parsed.filename,
            )

    def _figure_text(self, image) -> str:
        return " ".join(
            value
            for value in [image.label, image.caption, image.nearby_text]
            if value
        )
