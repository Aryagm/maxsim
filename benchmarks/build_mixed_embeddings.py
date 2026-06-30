from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def build_mixed_embeddings(*, output_path: Path, input_paths: tuple[Path, ...], dataset_name: str | None = None) -> Path:
    if not input_paths:
        raise ValueError("at least one input embedding file is required")

    parts = [_load_part(path) for path in input_paths]
    dim = int(parts[0]["doc_embeddings"].shape[1])
    for part in parts:
        if int(part["doc_embeddings"].shape[1]) != dim or int(part["query_embeddings"].shape[1]) != dim:
            raise ValueError("all mixed embedding files must have the same embedding dimension")

    doc_embeddings = np.concatenate([part["doc_embeddings"] for part in parts], axis=0)
    query_embeddings = np.concatenate([part["query_embeddings"] for part in parts], axis=0)
    doc_offsets = _concat_offsets([part["doc_offsets"] for part in parts])
    query_offsets = _concat_offsets([part["query_offsets"] for part in parts])
    qrels = _block_diagonal_qrels(parts)

    doc_ids = []
    query_ids = []
    source_datasets = []
    for part in parts:
        source = str(part["dataset_name"])
        source_datasets.append(source)
        doc_ids.extend(f"{source}:{doc_id}" for doc_id in part["doc_ids"])
        query_ids.extend(f"{source}:{query_id}" for query_id in part["query_ids"])

    name = dataset_name or f"mixed:{sum(len(part['doc_ids']) for part in parts)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        doc_embeddings=np.ascontiguousarray(doc_embeddings, dtype=np.float32),
        doc_offsets=doc_offsets,
        query_embeddings=np.ascontiguousarray(query_embeddings, dtype=np.float32),
        query_offsets=query_offsets,
        qrels=qrels,
        doc_ids=np.asarray(doc_ids),
        query_ids=np.asarray(query_ids),
        dataset_name=np.asarray(name),
        source_datasets=np.asarray(source_datasets),
        source_paths=np.asarray([str(path) for path in input_paths]),
        model_name=np.asarray(_model_name(parts)),
    )
    return output_path


def _load_part(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        doc_embeddings = np.asarray(data["doc_embeddings"], dtype=np.float32)
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        query_embeddings, query_offsets = _query_embeddings_and_offsets(data)
        doc_count = int(doc_offsets.shape[0] - 1)
        query_count = int(query_offsets.shape[0] - 1)
        qrels = np.asarray(data["qrels"], dtype=np.float32)
        if qrels.shape != (query_count, doc_count):
            raise ValueError(f"{path} qrels shape {qrels.shape} does not match queries/docs {(query_count, doc_count)}")
        dataset_name = str(np.asarray(data["dataset_name"]).item()) if "dataset_name" in data.files else path.stem
        doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"])) if "doc_ids" in data.files else tuple(f"doc-{idx}" for idx in range(doc_count))
        query_ids = tuple(str(value) for value in np.asarray(data["query_ids"])) if "query_ids" in data.files else tuple(f"query-{idx}" for idx in range(query_count))
        model_name = str(np.asarray(data["model_name"]).item()) if "model_name" in data.files else ""
        return {
            "doc_embeddings": np.ascontiguousarray(doc_embeddings, dtype=np.float32),
            "doc_offsets": doc_offsets,
            "query_embeddings": query_embeddings,
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


def _concat_offsets(offsets: list[np.ndarray]) -> np.ndarray:
    output = [0]
    total = 0
    for item in offsets:
        lengths = np.diff(np.asarray(item, dtype=np.int64))
        for length in lengths:
            total += int(length)
            output.append(total)
    return np.asarray(output, dtype=np.int64)


def _block_diagonal_qrels(parts: list[dict[str, Any]]) -> np.ndarray:
    total_queries = sum(int(part["qrels"].shape[0]) for part in parts)
    total_docs = sum(int(part["qrels"].shape[1]) for part in parts)
    output = np.zeros((total_queries, total_docs), dtype=np.float32)
    query_offset = 0
    doc_offset = 0
    for part in parts:
        qrels = part["qrels"]
        query_end = query_offset + int(qrels.shape[0])
        doc_end = doc_offset + int(qrels.shape[1])
        output[query_offset:query_end, doc_offset:doc_end] = qrels
        query_offset = query_end
        doc_offset = doc_end
    return output


def _model_name(parts: list[dict[str, Any]]) -> str:
    names = {str(part["model_name"]) for part in parts if str(part["model_name"])}
    if len(names) == 1:
        return next(iter(names))
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine compatible benchmark embedding .npz files into one mixed corpus.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()
    print(build_mixed_embeddings(output_path=args.output, input_paths=tuple(args.inputs), dataset_name=args.dataset_name))


if __name__ == "__main__":
    main()
