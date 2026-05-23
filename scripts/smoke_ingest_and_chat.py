import argparse
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.config import get_settings
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.providers.llm import build_llm_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.image_store import LocalImageStore
from backend.app.services.ingestion import IngestionService
from backend.app.services.rag import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real ingestion and chat smoke test.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--question", default="What are the main photoluminescence findings in this paper?")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    settings = get_settings()
    document_id = uuid4().hex
    repository = PgVectorRepository(settings.database_url)
    embedding_provider = build_embedding_provider(settings)

    ingestion = IngestionService(
        parser=PyMuPDFParser(),
        chunker=ParentChildChunker(),
        embedding_provider=embedding_provider,
        repository=repository,
        image_store=LocalImageStore(settings.image_dir),
    )
    chunk_count = ingestion.ingest(document_id, args.pdf)
    print(f"document_id: {document_id}")
    print(f"chunk_count: {chunk_count}")

    graph = MultiAgentRagGraph(
        retrieval_service=RetrievalService(repository, embedding_provider),
        llm_provider=build_llm_provider(settings),
    )
    state = graph.invoke(args.question)
    print("answer:")
    print(state.get("answer", ""))
    print("citations:")
    for item in state.get("verified_evidence", []):
        print(f"- page {item.page_start}-{item.page_end}, score={item.score:.4f}, chunk={item.chunk_id}")


if __name__ == "__main__":
    main()
