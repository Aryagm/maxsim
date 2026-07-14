from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import maxsim


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal maxsim SDK example for precomputed RAG or multimodal embeddings.")
    parser.add_argument("--embeddings", type=Path, required=True, help="NPZ with doc_embeddings, doc_offsets, query_embeddings, and doc_ids.")
    parser.add_argument("--corpus", type=Path, default=Path("benchmark-results/example-corpus.maxsim.npz"))
    parser.add_argument(
        "--mode",
        choices=["auto", "binary", "binary_q40", "int4", "int4_per_token", "int4_residual"],
        default="auto",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rescore-candidates", type=int, default=None)
    args = parser.parse_args()

    with np.load(args.embeddings, allow_pickle=False) as data:
        doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"]))
        docs = np.asarray(data["doc_embeddings"], dtype=np.float32)
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        query = _load_query(data, args.query_index)

    if args.corpus.exists():
        reranker = maxsim.Reranker.load(args.corpus, device=args.device)
    else:
        corpus = maxsim.Corpus.from_embeddings(
            doc_ids=doc_ids,
            embeddings=docs,
            offsets=doc_offsets,
            mode=args.mode,
            metadata={"source": str(args.embeddings), "mode": args.mode},
        )
        corpus.save(args.corpus)
        reranker = maxsim.Reranker.from_corpus(corpus, device=args.device)

    for result in reranker.search(
        query,
        k=args.k,
        rescore_candidates=args.rescore_candidates,
    ):
        print(f"{result.rank}\t{result.doc_id}\t{result.score:.4f}")


def _load_query(data, query_index: int) -> np.ndarray:
    queries = np.asarray(data["query_embeddings"], dtype=np.float32)
    if queries.ndim == 3:
        return np.ascontiguousarray(queries[int(query_index)], dtype=np.float32)
    offsets = np.asarray(data["query_offsets"], dtype=np.int64)
    start = int(offsets[int(query_index)])
    end = int(offsets[int(query_index) + 1])
    return np.ascontiguousarray(queries[start:end], dtype=np.float32)


if __name__ == "__main__":
    main()
