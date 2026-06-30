from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def build_docscale_stress(*, input_path: Path, output_path: Path, target_docs: int, query_limit: int | None = None, dataset_name: str | None = None) -> Path:
    if target_docs < 1:
        raise ValueError("target_docs must be >= 1")
    source = _load_source(input_path, query_limit=query_limit)
    selected = _selected_doc_indices(source["qrels"], source_doc_count=len(source["doc_ids"]), target_docs=int(target_docs))
    doc_embeddings, doc_offsets = _selected_doc_embeddings(source["doc_embeddings"], source["doc_offsets"], selected)
    qrels = _selected_qrels(source["qrels"], selected)
    doc_ids = _selected_doc_ids(source["doc_ids"], selected)
    name = dataset_name or f"{source['dataset_name']}:stress:{int(target_docs)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        doc_embeddings=doc_embeddings,
        doc_offsets=doc_offsets,
        query_embeddings=source["query_embeddings"],
        query_offsets=source["query_offsets"],
        qrels=qrels,
        doc_ids=np.asarray(doc_ids),
        query_ids=np.asarray(source["query_ids"]),
        dataset_name=np.asarray(name),
        source_dataset=np.asarray(source["dataset_name"]),
        source_path=np.asarray(str(input_path)),
        model_name=np.asarray(source["model_name"]),
        stress_target_docs=np.asarray(int(target_docs)),
        stress_query_limit=np.asarray(int(len(source["query_ids"]))),
    )
    return output_path


def _load_source(path: Path, *, query_limit: int | None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        doc_embeddings = np.asarray(data["doc_embeddings"], dtype=np.float32)
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        query_embeddings, query_offsets = _query_embeddings_and_offsets(data)
        query_count = int(query_offsets.shape[0] - 1)
        if query_limit is not None:
            query_count = min(query_count, int(query_limit))
            query_end = int(query_offsets[query_count])
            query_embeddings = query_embeddings[:query_end]
            query_offsets = query_offsets[: query_count + 1]
        doc_count = int(doc_offsets.shape[0] - 1)
        qrels = np.asarray(data["qrels"], dtype=np.float32)[:query_count, :doc_count]
        doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"])) if "doc_ids" in data.files else tuple(f"doc-{idx}" for idx in range(doc_count))
        query_ids = tuple(str(value) for value in np.asarray(data["query_ids"])[:query_count]) if "query_ids" in data.files else tuple(f"query-{idx}" for idx in range(query_count))
        dataset_name = str(np.asarray(data["dataset_name"]).item()) if "dataset_name" in data.files else path.stem
        model_name = str(np.asarray(data["model_name"]).item()) if "model_name" in data.files else ""
    return {
        "doc_embeddings": np.ascontiguousarray(doc_embeddings, dtype=np.float32),
        "doc_offsets": doc_offsets,
        "query_embeddings": np.ascontiguousarray(query_embeddings, dtype=np.float32),
        "query_offsets": query_offsets,
        "qrels": qrels,
        "doc_ids": doc_ids,
        "query_ids": query_ids,
        "dataset_name": dataset_name,
        "model_name": model_name,
    }


def _query_embeddings_and_offsets(data) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(data["query_embeddings"], dtype=np.float32)
    if queries.ndim == 2:
        if "query_offsets" not in data.files:
            raise ValueError("2D query_embeddings require query_offsets")
        return np.ascontiguousarray(queries, dtype=np.float32), np.asarray(data["query_offsets"], dtype=np.int64)
    if queries.ndim == 3:
        query_count, query_tokens, dim = queries.shape
        flat = queries.reshape(query_count * query_tokens, dim)
        offsets = np.arange(0, (query_count + 1) * query_tokens, query_tokens, dtype=np.int64)
        return np.ascontiguousarray(flat, dtype=np.float32), offsets
    raise ValueError("query_embeddings must have shape [tokens, dim] or [queries, tokens, dim]")


def _selected_doc_indices(qrels: np.ndarray, *, source_doc_count: int, target_docs: int) -> list[int]:
    positive_set = set(int(idx) for idx in np.flatnonzero(np.any(qrels > 0, axis=0)))
    if len(positive_set) > target_docs:
        raise ValueError("target_docs is smaller than the number of positive docs for the selected queries")

    if target_docs >= source_doc_count:
        selected = list(range(source_doc_count))
    else:
        selected = [idx for idx in range(source_doc_count) if idx in positive_set]
        for idx in range(source_doc_count):
            if len(selected) >= target_docs:
                break
            if idx not in positive_set:
                selected.append(idx)

    nonpositive = [idx for idx in range(source_doc_count) if idx not in positive_set]
    if len(selected) < target_docs:
        if not nonpositive:
            raise ValueError("cannot expand docscale stress corpus without non-positive distractor docs")
        cursor = 0
        while len(selected) < target_docs:
            selected.append(nonpositive[cursor % len(nonpositive)])
            cursor += 1
    return selected


def _selected_doc_embeddings(doc_embeddings: np.ndarray, doc_offsets: np.ndarray, selected: list[int]) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    offsets = [0]
    for doc_idx in selected:
        start = int(doc_offsets[doc_idx])
        end = int(doc_offsets[doc_idx + 1])
        chunk = np.ascontiguousarray(doc_embeddings[start:end], dtype=np.float32)
        chunks.append(chunk)
        offsets.append(offsets[-1] + int(chunk.shape[0]))
    return np.concatenate(chunks, axis=0), np.asarray(offsets, dtype=np.int64)


def _selected_qrels(qrels: np.ndarray, selected: list[int]) -> np.ndarray:
    source_counts: dict[int, int] = {}
    columns = []
    for doc_idx in selected:
        count = source_counts.get(doc_idx, 0)
        source_counts[doc_idx] = count + 1
        if count == 0:
            columns.append(qrels[:, doc_idx])
        else:
            columns.append(np.zeros((qrels.shape[0],), dtype=np.float32))
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def _selected_doc_ids(doc_ids: tuple[str, ...], selected: list[int]) -> list[str]:
    source_counts: dict[int, int] = {}
    output = []
    for out_idx, doc_idx in enumerate(selected):
        count = source_counts.get(doc_idx, 0)
        source_counts[doc_idx] = count + 1
        base = doc_ids[doc_idx]
        output.append(base if count == 0 else f"{base}:stress-{out_idx}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-query document-count stress corpora from existing embeddings.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-docs", type=int, required=True)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()
    print(
        build_docscale_stress(
            input_path=args.input,
            output_path=args.output,
            target_docs=args.target_docs,
            query_limit=args.query_limit,
            dataset_name=args.dataset_name,
        )
    )


if __name__ == "__main__":
    main()
