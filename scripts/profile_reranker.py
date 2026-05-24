import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.providers.rerankers import RerankCandidate, build_reranker_provider


def main() -> None:
    args = parse_args()
    settings = get_settings()
    reranker = build_reranker_provider(settings)
    if reranker is None:
        raise RuntimeError("当前配置未启用重排器，无法执行性能剖析。")

    base_text = args.base_text or (
        "ZnO photoluminescence solvent-dependent nonlinear optical response with spectra and transmittance results. "
    )

    warmup_started_at = perf_counter()
    reranker.warmup()
    print(json.dumps({"stage": "warmup", "latency_ms": round((perf_counter() - warmup_started_at) * 1000, 2)}, ensure_ascii=False))

    for count in args.candidate_counts:
        candidates = [
            RerankCandidate(candidate_id=str(index), text=base_text * (args.repeat_base + (index % max(1, args.repeat_jitter))))
            for index in range(count)
        ]
        started_at = perf_counter()
        reranker.rerank(args.query, candidates, top_k=min(args.top_k, count))
        latency_ms = (perf_counter() - started_at) * 1000
        total_chars = sum(len(candidate.text) for candidate in candidates)
        print(
            json.dumps(
                {
                    "stage": "rerank",
                    "candidate_count": count,
                    "latency_ms": round(latency_ms, 2),
                    "latency_per_candidate_ms": round(latency_ms / max(1, count), 2),
                    "avg_candidate_chars": round(total_chars / max(1, count), 1),
                    "device": settings.reranker_device,
                    "model": settings.reranker_model,
                },
                ensure_ascii=False,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile reranker latency across candidate counts.")
    parser.add_argument("--query", default="Which PL spectrum figures are shown in this paper?")
    parser.add_argument("--candidate-counts", type=_parse_counts, default=[1, 5, 10, 20, 30])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repeat-base", type=int, default=8)
    parser.add_argument("--repeat-jitter", type=int, default=5)
    parser.add_argument("--base-text", default=None)
    return parser.parse_args()


def _parse_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("candidate counts cannot be empty")
    return counts


if __name__ == "__main__":
    main()
