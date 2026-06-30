from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import bitmax

try:
    import torch as _torch
except ImportError:  # pragma: no cover - depends on optional local install
    _torch = None


@dataclass(frozen=True)
class RetrievalEmbeddings:
    name: str
    query_embeddings: tuple[np.ndarray, ...]
    doc_embeddings: np.ndarray
    doc_offsets: np.ndarray
    qrels: np.ndarray
    query_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]

    @property
    def num_queries(self) -> int:
        return len(self.query_embeddings)

    @property
    def num_docs(self) -> int:
        return int(self.doc_offsets.shape[0] - 1)

    @property
    def dim(self) -> int:
        return int(self.doc_embeddings.shape[1])


def run_stage(
    stage: str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    gate_path: Path | str | None = None,
    top_k: int = 10,
    repeat: int | None = None,
    scale: str | float | None = None,
) -> dict[str, Any]:
    _check_stage_gate(stage, gate_path)
    dataset = _load_stage_dataset(stage, input_path)
    actual_repeat = repeat if repeat is not None else _stage_repeat(stage)
    native_device, baseline_device = _stage_devices(stage)
    output = Path(output_path) if output_path is not None else Path("benchmark-results") / f"{stage}.json"

    dense_scores, dense_latency = _time_call(
        lambda: _dense_fp16_scores(dataset, device=baseline_device),
        repeat=actual_repeat,
    )
    packed = bitmax.pack_signs(dataset.doc_embeddings, dataset.doc_offsets, scale=scale)
    scoring_packed = _prepare_bitmax_packed(packed, native_device)
    bitmax_name = "bitmax_cuda" if native_device == "cuda" else "bitmax_native"
    bitmax_scores, bitmax_latency = _time_call(
        lambda: _bitmax_scores(dataset, scoring_packed, device=native_device),
        repeat=actual_repeat,
    )

    dense_row = _result_row(
        stage,
        dataset,
        "dense_fp16_baseline",
        dense_latency,
        dense_scores,
        dataset.qrels,
        top_k=top_k,
        doc_storage_bytes=_dense_doc_bytes(dataset, 2),
        dense_reference_scores=dense_scores,
        metadata={"baseline_device": baseline_device, "formula": "dense_fp16_maxsim"},
    )
    bitmax_row = _result_row(
        stage,
        dataset,
        bitmax_name,
        bitmax_latency,
        bitmax_scores,
        dataset.qrels,
        top_k=top_k,
        doc_storage_bytes=_packed_doc_bytes(dataset),
        dense_reference_scores=dense_scores,
        baseline_row=dense_row,
        metadata={"requested_device": native_device, "scale": "none" if scale is None else scale},
    )

    rows = [dense_row, bitmax_row]
    result = {
        "schema_version": 1,
        "benchmark": "retrieval",
        "stage": stage,
        "dataset": _dataset_metadata(dataset),
        "top_k": int(top_k),
        "repeat": int(actual_repeat),
        "baselines": ["dense_fp16_baseline"],
        "gate_passed": _gate_passed(rows),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _check_stage_gate(stage: str, gate_path: Path | str | None) -> None:
    required_stage = {"embeddings-cuda-smoke": "embeddings-smoke"}.get(stage)
    if required_stage is None:
        return
    if gate_path is None:
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")
    gate = Path(gate_path)
    if not gate.exists():
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")
    data = json.loads(gate.read_text())
    if data.get("stage") != required_stage or data.get("gate_passed") is not True:
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")


def _load_stage_dataset(stage: str, input_path: Path | str | None) -> RetrievalEmbeddings:
    if stage == "fixture-smoke":
        return _fixture_dataset()
    if stage in {"embeddings-smoke", "embeddings-cuda-smoke"}:
        if input_path is None:
            raise RuntimeError(f"{stage} requires --input with a retrieval embedding .npz file")
        return _load_embedding_file(Path(input_path))
    raise ValueError(f"unknown retrieval benchmark stage: {stage}")


def _stage_repeat(stage: str) -> int:
    if stage == "fixture-smoke":
        return 3
    if stage == "embeddings-smoke":
        return 3
    if stage == "embeddings-cuda-smoke":
        return 5
    raise ValueError(f"unknown retrieval benchmark stage: {stage}")


def _stage_devices(stage: str) -> tuple[str, str]:
    if stage == "embeddings-cuda-smoke":
        return "cuda", "cuda"
    return "auto", "cpu"


def _fixture_dataset() -> RetrievalEmbeddings:
    doc_embeddings = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query_embeddings = (
        np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32),
        np.array([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.float32),
    )
    qrels = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    return RetrievalEmbeddings(
        name="fixture-smoke",
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        qrels=qrels,
        query_ids=("query-0", "query-1"),
        doc_ids=("doc-0", "doc-1", "doc-2"),
    )


def _load_embedding_file(path: Path) -> RetrievalEmbeddings:
    with np.load(path, allow_pickle=False) as data:
        required = {"doc_embeddings", "doc_offsets", "query_embeddings"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"embedding file is missing required arrays: {sorted(missing)}")

        doc_embeddings = np.asarray(data["doc_embeddings"], dtype=np.float32)
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        query_embeddings = _load_queries(data)
        qrels = _load_qrels(data, len(query_embeddings), int(doc_offsets.shape[0] - 1))
        dataset_name = _load_scalar_string(data, "dataset_name", path.stem)
        query_ids = _load_ids(data, "query_ids", len(query_embeddings), "query")
        doc_ids = _load_ids(data, "doc_ids", int(doc_offsets.shape[0] - 1), "doc")

    _validate_dataset_arrays(doc_embeddings, doc_offsets, query_embeddings, qrels)
    return RetrievalEmbeddings(
        name=dataset_name,
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        doc_offsets=doc_offsets,
        qrels=qrels,
        query_ids=query_ids,
        doc_ids=doc_ids,
    )


def _load_queries(data) -> tuple[np.ndarray, ...]:
    queries = np.asarray(data["query_embeddings"], dtype=np.float32)
    if queries.ndim == 3:
        return tuple(np.ascontiguousarray(query, dtype=np.float32) for query in queries)
    if queries.ndim != 2 or "query_offsets" not in data.files:
        raise ValueError("query_embeddings must be 3-D, or 2-D with query_offsets")
    offsets = np.asarray(data["query_offsets"], dtype=np.int64)
    if offsets.ndim != 1 or offsets.shape[0] < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != queries.shape[0]:
        raise ValueError("query_offsets must span flattened query_embeddings")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("query_offsets must be monotonically nondecreasing")
    return tuple(np.ascontiguousarray(queries[int(start) : int(end)], dtype=np.float32) for start, end in zip(offsets[:-1], offsets[1:]))


def _load_qrels(data, num_queries: int, num_docs: int) -> np.ndarray:
    if "qrels" in data.files:
        qrels = np.asarray(data["qrels"], dtype=np.float32)
    elif "relevant_doc_ids" in data.files:
        relevant_doc_ids = np.asarray(data["relevant_doc_ids"], dtype=np.int64)
        if relevant_doc_ids.ndim != 1 or relevant_doc_ids.shape[0] != num_queries:
            raise ValueError("relevant_doc_ids must have shape [num_queries]")
        qrels = np.zeros((num_queries, num_docs), dtype=np.float32)
        for query_idx, doc_idx in enumerate(relevant_doc_ids):
            if int(doc_idx) < 0 or int(doc_idx) >= num_docs:
                raise ValueError("relevant_doc_ids contains a doc id outside the corpus")
            qrels[query_idx, int(doc_idx)] = 1.0
    else:
        raise ValueError("embedding file must contain qrels or relevant_doc_ids")
    if qrels.shape != (num_queries, num_docs):
        raise ValueError(f"qrels must have shape [{num_queries}, {num_docs}]")
    return qrels


def _load_scalar_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    raise ValueError(f"{key} must be a scalar string")


def _load_ids(data, key: str, count: int, prefix: str) -> tuple[str, ...]:
    if key not in data.files:
        return tuple(f"{prefix}-{idx}" for idx in range(count))
    values = np.asarray(data[key])
    if values.ndim != 1 or values.shape[0] != count:
        raise ValueError(f"{key} must have shape [{count}]")
    return tuple(str(value) for value in values)


def _validate_dataset_arrays(
    doc_embeddings: np.ndarray,
    doc_offsets: np.ndarray,
    query_embeddings: tuple[np.ndarray, ...],
    qrels: np.ndarray,
) -> None:
    if doc_embeddings.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")
    if doc_embeddings.shape[1] % 8 != 0:
        raise ValueError("embedding dim must be divisible by 8")
    if doc_offsets.ndim != 1 or doc_offsets.shape[0] < 2:
        raise ValueError("doc_offsets must have shape [num_docs + 1]")
    if int(doc_offsets[0]) != 0 or int(doc_offsets[-1]) != doc_embeddings.shape[0]:
        raise ValueError("doc_offsets must span doc_embeddings")
    if np.any(doc_offsets[1:] < doc_offsets[:-1]):
        raise ValueError("doc_offsets must be monotonically nondecreasing")
    if not query_embeddings:
        raise ValueError("at least one query embedding is required")
    for query in query_embeddings:
        if query.ndim != 2 or query.shape[1] != doc_embeddings.shape[1]:
            raise ValueError("each query embedding must have shape [query_tokens, dim]")
    if qrels.shape != (len(query_embeddings), doc_offsets.shape[0] - 1):
        raise ValueError("qrels shape does not match queries/docs")


def _time_call(fn, *, repeat: int):
    best_latency = float("inf")
    best_value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        latency_ms = (time.perf_counter() - start) * 1_000.0
        if latency_ms < best_latency:
            best_latency = latency_ms
            best_value = value
    return best_value, best_latency


def _dense_fp16_scores(dataset: RetrievalEmbeddings, *, device: str) -> np.ndarray:
    torch_device = _resolve_torch_device(device)
    if torch_device is not None and str(torch_device) == "cuda":
        return _torch_dense_fp16_scores(dataset, torch_device)
    return _numpy_dense_fp16_scores(dataset)


def _resolve_torch_device(requested: str):
    if _torch is None or requested != "cuda":
        return None
    if _torch.cuda.is_available() and _torch_cuda_supports_current_device():
        return _torch.device("cuda")
    return None


def _torch_cuda_supports_current_device() -> bool:
    try:
        major, minor = _torch.cuda.get_device_capability()
        current_arch = f"sm_{major}{minor}"
        supported_arches = set(_torch.cuda.get_arch_list())
    except Exception:
        return False
    return not supported_arches or current_arch in supported_arches


def _numpy_dense_fp16_scores(dataset: RetrievalEmbeddings) -> np.ndarray:
    docs = dataset.doc_embeddings.astype(np.float16).astype(np.float32)
    scores = np.empty((dataset.num_queries, dataset.num_docs), dtype=np.float32)
    uniform_doc_tokens = _uniform_doc_tokens(dataset.doc_offsets)
    for query_idx, query in enumerate(dataset.query_embeddings):
        query_float = query.astype(np.float16).astype(np.float32)
        if uniform_doc_tokens is not None:
            dots = query_float @ docs.T
            scores[query_idx] = dots.reshape(query.shape[0], dataset.num_docs, uniform_doc_tokens).max(axis=2).sum(axis=0, dtype=np.float32)
            continue
        for doc_idx in range(dataset.num_docs):
            start = int(dataset.doc_offsets[doc_idx])
            end = int(dataset.doc_offsets[doc_idx + 1])
            doc = docs[start:end]
            scores[query_idx, doc_idx] = np.max(query_float @ doc.T, axis=1).sum(dtype=np.float32) if doc.shape[0] else 0.0
    return scores


def _torch_dense_fp16_scores(dataset: RetrievalEmbeddings, device) -> np.ndarray:
    docs = _torch.as_tensor(dataset.doc_embeddings, dtype=_torch.float16, device=device).to(_torch.float32)
    scores = []
    uniform_doc_tokens = _uniform_doc_tokens(dataset.doc_offsets)
    for query in dataset.query_embeddings:
        query_tensor = _torch.as_tensor(query, dtype=_torch.float16, device=device).to(_torch.float32)
        if uniform_doc_tokens is not None:
            dots = query_tensor @ docs.T
            row = dots.reshape(query_tensor.shape[0], dataset.num_docs, uniform_doc_tokens).max(dim=2).values.sum(dim=0)
        else:
            per_doc = []
            for doc_idx in range(dataset.num_docs):
                start = int(dataset.doc_offsets[doc_idx])
                end = int(dataset.doc_offsets[doc_idx + 1])
                doc = docs[start:end]
                per_doc.append((query_tensor @ doc.T).max(dim=1).values.sum() if doc.shape[0] else _torch.zeros((), device=device))
            row = _torch.stack(per_doc)
        scores.append(row)
    _torch.cuda.synchronize()
    return _torch.stack(scores).detach().cpu().numpy().astype(np.float32, copy=False)


def _bitmax_scores(dataset: RetrievalEmbeddings, packed: bitmax.PackedDocs, *, device: str) -> np.ndarray:
    if device == "cuda":
        return bitmax.maxsim(_padded_query_batch(dataset.query_embeddings), packed, device=device).astype(np.float32, copy=False)
    rows = [bitmax.maxsim(query, packed, device=device) for query in dataset.query_embeddings]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _prepare_bitmax_packed(packed: bitmax.PackedDocs, device: str):
    return bitmax.to_device(packed, "cuda") if device == "cuda" else packed


def _padded_query_batch(query_embeddings: tuple[np.ndarray, ...]) -> np.ndarray:
    if not query_embeddings:
        raise ValueError("at least one query embedding is required")
    batch = len(query_embeddings)
    max_tokens = max(int(query.shape[0]) for query in query_embeddings)
    dim = int(query_embeddings[0].shape[1])
    padded = np.zeros((batch, max_tokens, dim), dtype=np.float32)
    for query_idx, query in enumerate(query_embeddings):
        if query.ndim != 2 or int(query.shape[1]) != dim:
            raise ValueError("each query embedding must have shape [query_tokens, dim]")
        padded[query_idx, : query.shape[0], :] = query.astype(np.float32, copy=False)
    return padded


def _uniform_doc_tokens(doc_offsets: np.ndarray) -> int | None:
    lengths = np.diff(doc_offsets)
    if lengths.size == 0 or np.any(lengths != lengths[0]):
        return None
    return int(lengths[0])


def _result_row(
    stage: str,
    dataset: RetrievalEmbeddings,
    implementation: str,
    latency_ms: float,
    scores: np.ndarray,
    qrels: np.ndarray,
    *,
    top_k: int,
    doc_storage_bytes: int,
    dense_reference_scores: np.ndarray,
    baseline_row: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_top_k = int(top_k)
    effective_top_k = min(requested_top_k, dataset.num_docs)
    metrics = _ranking_metrics(scores, qrels, k=effective_top_k)
    dense_metrics = _ranking_metrics(dense_reference_scores, qrels, k=effective_top_k)
    row = {
        "stage": stage,
        "implementation": implementation,
        "gate_blocking": True,
        "latency_ms": float(latency_ms),
        "queries_per_second": float(dataset.num_queries / max(latency_ms / 1_000.0, 1e-12)),
        "query_count": int(dataset.num_queries),
        "docs": int(dataset.num_docs),
        "dim": int(dataset.dim),
        "top_k": requested_top_k,
        "effective_top_k": int(effective_top_k),
        "doc_storage_bytes": int(doc_storage_bytes),
        "bytes_read": int(doc_storage_bytes),
        "score_checksum": float(np.sum(scores, dtype=np.float64)),
        "recall_at_1": float(_ranking_metrics(scores, qrels, k=1)["recall_at_k"]),
        f"recall_at_{requested_top_k}": float(metrics["recall_at_k"]),
        f"mrr_at_{requested_top_k}": float(metrics["mrr_at_k"]),
        f"ndcg_at_{requested_top_k}": float(metrics["ndcg_at_k"]),
        f"topk_agreement_vs_dense_at_{requested_top_k}": float(_topk_agreement(scores, dense_reference_scores, effective_top_k)),
        f"quality_delta_vs_dense_ndcg_at_{requested_top_k}": float(metrics["ndcg_at_k"] - dense_metrics["ndcg_at_k"]),
    }
    if baseline_row is not None:
        row["baseline_latency_ms"] = {"dense_fp16_baseline": baseline_row["latency_ms"]}
        row["speedup_vs_dense_fp16"] = float(baseline_row["latency_ms"] / max(latency_ms, 1e-12))
        row["doc_memory_compression_vs_fp16"] = float(_dense_doc_bytes(dataset, 2) / max(doc_storage_bytes, 1))
        row["doc_memory_compression_vs_fp32"] = float(_dense_doc_bytes(dataset, 4) / max(doc_storage_bytes, 1))
    if metadata is not None:
        row.update(metadata)
    return row


def _ranking_metrics(scores: np.ndarray, qrels: np.ndarray, *, k: int) -> dict[str, float]:
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    for query_scores, query_relevance in zip(scores, qrels):
        relevant_total = float(np.sum(query_relevance > 0))
        if relevant_total == 0:
            continue
        order = _rank_indices(query_scores, k)
        hits = query_relevance[order] > 0
        recalls.append(float(np.sum(hits)) / relevant_total)
        hit_positions = np.flatnonzero(hits)
        reciprocal_ranks.append(0.0 if hit_positions.size == 0 else 1.0 / float(hit_positions[0] + 1))
        ndcgs.append(_ndcg(query_relevance, order, k))
    if not recalls:
        return {"recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}
    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr_at_k": float(np.mean(reciprocal_ranks)),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }


def _rank_indices(scores: np.ndarray, k: int) -> np.ndarray:
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((doc_ids, -scores))
    return order[:k].astype(np.int64, copy=False)


def _ndcg(relevance: np.ndarray, order: np.ndarray, k: int) -> float:
    gains = np.power(2.0, relevance[order].astype(np.float64)) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, gains.shape[0] + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(relevance.astype(np.float64))[::-1][:k]
    ideal_gains = np.power(2.0, ideal) - 1.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.shape[0] + 2, dtype=np.float64))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _topk_agreement(scores: np.ndarray, reference_scores: np.ndarray, k: int) -> float:
    agreements = []
    for row, reference_row in zip(scores, reference_scores):
        actual = set(int(idx) for idx in _rank_indices(row, k))
        expected = set(int(idx) for idx in _rank_indices(reference_row, k))
        agreements.append(len(actual & expected) / float(k))
    return float(np.mean(agreements)) if agreements else 0.0


def _dataset_metadata(dataset: RetrievalEmbeddings) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "queries": int(dataset.num_queries),
        "docs": int(dataset.num_docs),
        "dim": int(dataset.dim),
        "doc_tokens": int(dataset.doc_embeddings.shape[0]),
        "qrels_nonzero": int(np.sum(dataset.qrels > 0)),
    }


def _dense_doc_bytes(dataset: RetrievalEmbeddings, bytes_per_value: int) -> int:
    return int(dataset.doc_embeddings.shape[0]) * int(dataset.dim) * int(bytes_per_value)


def _packed_doc_bytes(dataset: RetrievalEmbeddings) -> int:
    return int(dataset.doc_embeddings.shape[0]) * (int(dataset.dim) // 8)


def _gate_passed(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if not np.isfinite(float(row["latency_ms"])):
            return False
        metric_values = [value for key, value in row.items() if key.startswith(("recall_at_", "mrr_at_", "ndcg_at_"))]
        if any(not np.isfinite(float(value)) for value in metric_values):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-level bitmax benchmarks from multi-vector embeddings.")
    parser.add_argument("--stage", required=True, choices=["fixture-smoke", "embeddings-smoke", "embeddings-cuda-smoke"])
    parser.add_argument("--input", type=Path, default=None, help="Input .npz with doc/query embeddings and qrels.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gate", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--scale", default=None, help="Optional bitmax scale value, or 'global'.")
    args = parser.parse_args()

    scale = None if args.scale in {None, "none"} else args.scale
    result = run_stage(
        args.stage,
        input_path=args.input,
        output_path=args.output,
        gate_path=args.gate,
        top_k=args.top_k,
        repeat=args.repeat,
        scale=scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
