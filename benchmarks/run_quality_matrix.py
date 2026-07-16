"""Run the paper quality matrix over an existing multi-vector cache.

This runner evaluates exact stored-format reconstructions rather than ad-hoc
quantization approximations. It checkpoints per-query NDCG@10 and Recall@10
atomically, so an interrupted full-corpus run resumes at the last checkpoint.

Example:
    python -m benchmarks.run_quality_matrix \
        caches-full/beir/beir-fiqa-jina-colbert-v2.npz \
        --device cuda \
        --output benchmark-results/quality-fiqa-jina-colbert-v2.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import maxsim
from benchmarks.run_retrieval import RetrievalEmbeddings, _load_embedding_file
from maxsim.experimental import pack_int4_symmetric


FORMAT_NAMES = (
    "dense_fp16",
    "binary",
    "binary_token_scale_u4",
    "pool3_binary",
    "int8_per_token",
    "int4_per_tensor",
    "int4_per_token",
)


@dataclass(frozen=True)
class StoredFormat:
    docs: np.ndarray
    offsets: np.ndarray
    storage_bytes: int
    query_fp16: bool
    metadata: dict[str, Any]


def _decode_signs(packed: maxsim.PackedDocs) -> np.ndarray:
    data = np.ascontiguousarray(packed.data, dtype=np.uint8)
    bits = np.unpackbits(data, axis=1, bitorder="little")[:, : packed.dim]
    return bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)


def _materialize_format(dataset: RetrievalEmbeddings, name: str) -> StoredFormat:
    docs = dataset.doc_embeddings
    offsets = dataset.doc_offsets
    common = {
        "format_semantics": "decoded_from_stored_codes",
        "common_doc_offsets_bytes": int(offsets.nbytes),
    }

    if name == "dense_fp16":
        stored = docs.astype(np.float16)
        decoded = stored.astype(np.float32)
        return StoredFormat(
            docs=decoded,
            offsets=offsets,
            storage_bytes=int(stored.nbytes),
            query_fp16=True,
            metadata={**common, "codes": "fp16", "query_precision": "fp16"},
        )

    if name == "binary":
        packed = maxsim.pack_signs(docs, offsets)
        return StoredFormat(
            docs=_decode_signs(packed),
            offsets=offsets,
            storage_bytes=int(np.asarray(packed.data).nbytes),
            query_fp16=False,
            metadata={**common, "codes": "packed_sign_bits", "bits_per_dimension": 1},
        )

    if name == "binary_token_scale_u4":
        packed = maxsim.pack_signs(docs, offsets, token_scale="mean_abs_u4")
        decoded = _decode_signs(packed)
        decoded *= packed.token_scale[:, np.newaxis]
        scale_code_bytes = (docs.shape[0] + 1) // 2
        return StoredFormat(
            docs=decoded,
            offsets=offsets,
            storage_bytes=int(np.asarray(packed.data).nbytes + scale_code_bytes + 16),
            query_fp16=False,
            metadata={
                **common,
                "codes": "packed_sign_bits",
                "token_scale": "mean_abs_log_u4",
                "token_scale_code_bytes": int(scale_code_bytes),
                "token_scale_metadata_bytes": 16,
            },
        )

    if name == "pool3_binary":
        from maxsim.pooling import pool_doc_tokens

        pooled_docs, pooled_offsets = pool_doc_tokens(docs, offsets, 3)
        packed = maxsim.pack_signs(pooled_docs, pooled_offsets)
        return StoredFormat(
            docs=_decode_signs(packed),
            offsets=pooled_offsets,
            storage_bytes=int(np.asarray(packed.data).nbytes),
            query_fp16=False,
            metadata={
                **common,
                "codes": "packed_sign_bits",
                "pool_factor": 3,
                "common_doc_offsets_bytes": int(pooled_offsets.nbytes),
                "stored_doc_tokens": int(pooled_offsets[-1]),
                "realized_token_reduction": float(docs.shape[0] / max(int(pooled_offsets[-1]), 1)),
            },
        )

    if name == "int8_per_token":
        token_scale = np.max(np.abs(docs), axis=1).astype(np.float32) / np.float32(127.0)
        token_scale = np.where(token_scale > 0.0, token_scale, np.float32(1.0)).astype(
            np.float32,
            copy=False,
        )
        values = np.clip(
            np.rint(docs / token_scale[:, np.newaxis]),
            -127,
            127,
        ).astype(np.int8)
        decoded = values.astype(np.float32) * token_scale[:, np.newaxis]
        return StoredFormat(
            docs=decoded,
            offsets=offsets,
            storage_bytes=int(values.nbytes + token_scale.nbytes),
            query_fp16=False,
            metadata={
                **common,
                "codes": "symmetric_int8",
                "code_dtype": "int8",
                "quantization_range": [-127, 127],
                "scale_granularity": "token",
                "scale_storage": "fp32",
                "bytes_per_token": int(docs.shape[1] + np.dtype(np.float32).itemsize),
                "scoring_path": "decoded_generic",
                "native_cuda_kernel": False,
            },
        )

    if name == "int4_per_tensor":
        packed = pack_int4_symmetric(docs, offsets, scale_granularity="tensor")
        decoded = packed.values.astype(np.float32) * np.float32(packed.scale)
        return StoredFormat(
            docs=decoded,
            offsets=offsets,
            storage_bytes=int(packed.data.nbytes + 4),
            query_fp16=False,
            metadata={
                **common,
                "codes": "packed_symmetric_int4",
                "scale_granularity": "tensor",
                "scale_storage": "fp32",
                "scale": float(packed.scale),
            },
        )

    if name == "int4_per_token":
        packed = pack_int4_symmetric(docs, offsets, scale_granularity="token")
        decoded = packed.values.astype(np.float32)
        decoded *= packed.token_scale[:, np.newaxis]
        return StoredFormat(
            docs=decoded,
            offsets=offsets,
            storage_bytes=int(packed.data.nbytes + packed.token_scale.nbytes),
            query_fp16=False,
            metadata={
                **common,
                "codes": "packed_symmetric_int4",
                "scale_granularity": "token",
                "scale_storage": "fp32",
            },
        )

    raise ValueError(f"unknown quality-matrix format: {name}")


class _NumpyScorer:
    def __init__(self, docs: np.ndarray, offsets: np.ndarray):
        self.docs = np.ascontiguousarray(docs, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def score(self, query: np.ndarray) -> np.ndarray:
        num_docs = self.offsets.shape[0] - 1
        if self.docs.shape[0] == 0:
            return np.zeros(num_docs, dtype=np.float32)
        dots = self.docs @ np.ascontiguousarray(query, dtype=np.float32).T
        starts = self.offsets[:-1]
        empty = starts == self.offsets[1:]
        safe_starts = np.minimum(starts, self.docs.shape[0] - 1)
        per_doc_max = np.maximum.reduceat(dots, safe_starts, axis=0)
        per_doc_max[empty] = 0.0
        return per_doc_max.sum(axis=1, dtype=np.float32)

    def close(self) -> None:
        return None


def _doc_chunks(offsets: np.ndarray, max_tokens: int) -> tuple[tuple[int, int, int, int], ...]:
    chunks = []
    num_docs = offsets.shape[0] - 1
    doc_start = 0
    while doc_start < num_docs:
        token_start = int(offsets[doc_start])
        doc_end = doc_start
        while doc_end < num_docs:
            candidate_end = doc_end + 1
            candidate_tokens = int(offsets[candidate_end]) - token_start
            if doc_end > doc_start and candidate_tokens > max_tokens:
                break
            doc_end = candidate_end
            if candidate_tokens >= max_tokens:
                break
        chunks.append((doc_start, doc_end, token_start, int(offsets[doc_end])))
        doc_start = doc_end
    return tuple(chunks)


class _TorchCudaScorer:
    def __init__(self, docs: np.ndarray, offsets: np.ndarray, max_chunk_tokens: int):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requires an available CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        self.torch = torch
        self.docs = torch.as_tensor(np.ascontiguousarray(docs), dtype=torch.float32, device="cuda")
        self.offsets = np.asarray(offsets, dtype=np.int64)
        lengths = np.diff(self.offsets)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long, device="cuda")
        self.token_doc_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long, device="cuda"),
            self.lengths,
        )
        self.chunks = _doc_chunks(self.offsets, max_chunk_tokens)

    def score(self, query: np.ndarray) -> np.ndarray:
        torch = self.torch
        query_tensor = torch.as_tensor(np.ascontiguousarray(query), dtype=torch.float32, device="cuda")
        output = np.empty(self.offsets.shape[0] - 1, dtype=np.float32)
        for doc_start, doc_end, token_start, token_end in self.chunks:
            if token_start == token_end:
                output[doc_start:doc_end] = 0.0
                continue
            dots = self.docs[token_start:token_end] @ query_tensor.T
            local = torch.full(
                (doc_end - doc_start, query_tensor.shape[0]),
                -torch.inf,
                dtype=torch.float32,
                device="cuda",
            )
            token_doc_ids = self.token_doc_ids[token_start:token_end] - doc_start
            local.scatter_reduce_(
                0,
                token_doc_ids[:, np.newaxis].expand(-1, query_tensor.shape[0]),
                dots,
                reduce="amax",
                include_self=True,
            )
            empty = self.lengths[doc_start:doc_end] == 0
            local[empty] = 0.0
            output[doc_start:doc_end] = local.sum(dim=1).cpu().numpy()
        return output

    def close(self) -> None:
        del self.docs
        del self.lengths
        del self.token_doc_ids
        self.torch.cuda.empty_cache()


def _resolve_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("--device cuda requires PyTorch")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("--device cuda requires an available CUDA device")
    return "cpu"


def _make_scorer(stored: StoredFormat, device: str, max_chunk_tokens: int):
    if device == "cuda":
        return _TorchCudaScorer(stored.docs, stored.offsets, max_chunk_tokens)
    return _NumpyScorer(stored.docs, stored.offsets)


def _rank_indices(scores: np.ndarray, k: int) -> np.ndarray:
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((doc_ids, -scores))[:k].astype(np.int64, copy=False)


def _query_metrics(scores: np.ndarray, relevance: np.ndarray, k: int) -> tuple[float | None, float | None]:
    relevant_total = int(np.sum(relevance > 0))
    if relevant_total == 0:
        return None, None
    order = _rank_indices(scores, min(k, scores.shape[0]))
    recall = float(np.sum(relevance[order] > 0)) / relevant_total
    gains = np.power(2.0, relevance[order].astype(np.float64)) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, gains.shape[0] + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(relevance.astype(np.float64))[::-1][: min(k, relevance.shape[0])]
    ideal_gains = np.power(2.0, ideal) - 1.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.shape[0] + 2, dtype=np.float64))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    ndcg = 0.0 if idcg == 0.0 else dcg / idcg
    return ndcg, recall


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _cache_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    metadata = {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "builder_schema_version",
            "dataset_id",
            "dataset_name",
            "model_name",
            "model_requested",
            "model_resolved",
            "model_profile",
            "model_options_json",
            "model_fallback_used",
            "qrels_semantics",
            "source_dataset",
            "source_config",
            "source_split",
        ):
            if key not in data.files:
                continue
            value = np.asarray(data[key])
            metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def _normalize_formats(formats: str | Iterable[str] | None) -> tuple[str, ...]:
    if formats is None:
        return FORMAT_NAMES
    values = tuple(part.strip() for part in formats.split(",")) if isinstance(formats, str) else tuple(formats)
    if not values:
        raise ValueError("at least one format is required")
    unknown = set(values) - set(FORMAT_NAMES)
    if unknown:
        raise ValueError(f"unknown formats: {sorted(unknown)}")
    if len(values) != len(set(values)):
        raise ValueError("formats must not contain duplicates")
    return values


def _mean_present(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else 0.0


def run_quality_matrix(
    cache_path: Path | str,
    output_path: Path | str,
    *,
    formats: str | Iterable[str] | None = None,
    top_k: int = 10,
    device: str = "auto",
    checkpoint_every: int = 8,
    max_chunk_tokens: int = 1_000_000,
    limit_queries: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache = Path(cache_path)
    output = Path(output_path)
    selected_formats = _normalize_formats(formats)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_chunk_tokens < 1:
        raise ValueError("max_chunk_tokens must be >= 1")
    if limit_queries < 0:
        raise ValueError("limit_queries must be >= 0")

    resolved_device = _resolve_device(device)
    provenance = _cache_provenance(cache)
    dataset = _load_embedding_file(cache)
    query_count = dataset.num_queries if limit_queries == 0 else min(limit_queries, dataset.num_queries)
    config = {
        "formats": list(selected_formats),
        "top_k": int(top_k),
        "query_count": int(query_count),
        "device": resolved_device,
        "max_chunk_tokens": int(max_chunk_tokens),
    }

    if output.exists() and not overwrite:
        payload = json.loads(output.read_text())
        if payload.get("schema_version") != 1 or payload.get("benchmark") != "stored-format-quality-matrix":
            raise RuntimeError(f"cannot resume incompatible output: {output}")
        if payload.get("cache", {}).get("sha256") != provenance["sha256"]:
            raise RuntimeError(f"cannot resume {output}: input cache fingerprint changed")
        if payload.get("config") != config:
            raise RuntimeError(f"cannot resume {output}: benchmark configuration changed")
    else:
        payload = {
            "schema_version": 1,
            "benchmark": "stored-format-quality-matrix",
            "status": "in_progress",
            "cache": provenance,
            "dataset": {
                "name": dataset.name,
                "docs": dataset.num_docs,
                "queries_available": dataset.num_queries,
                "queries_evaluated": query_count,
                "dim": dataset.dim,
                "doc_tokens": int(dataset.doc_embeddings.shape[0]),
                "qrels_nonzero": int(np.sum(dataset.qrels[:query_count] > 0)),
            },
            "config": config,
            "query_ids": list(dataset.query_ids[:query_count]),
            "results": [],
        }
        _atomic_write_json(output, payload)

    rows = {row["format"]: row for row in payload["results"]}
    for format_name in selected_formats:
        row = rows.get(format_name)
        if row is not None and row.get("status") == "complete":
            print(f"{format_name}: already complete", flush=True)
            continue
        if row is None:
            row = {
                "format": format_name,
                "status": "in_progress",
                "completed_queries": 0,
                f"per_query_ndcg_at_{top_k}": [],
                f"per_query_recall_at_{top_k}": [],
            }
            payload["results"].append(row)
            rows[format_name] = row
            _atomic_write_json(output, payload)

        ndcg_values = row[f"per_query_ndcg_at_{top_k}"]
        recall_values = row[f"per_query_recall_at_{top_k}"]
        completed = int(row.get("completed_queries", 0))
        if len(ndcg_values) != completed or len(recall_values) != completed or completed > query_count:
            raise RuntimeError(f"cannot resume {format_name}: invalid per-query checkpoint lengths")

        print(f"{format_name}: materializing stored representation", flush=True)
        stored = _materialize_format(dataset, format_name)
        row["doc_storage_bytes"] = stored.storage_bytes
        row["compression_vs_dense_fp16"] = float(
            (dataset.doc_embeddings.shape[0] * dataset.dim * 2) / max(stored.storage_bytes, 1)
        )
        row["stored_doc_tokens"] = int(stored.offsets[-1])
        row["representation"] = stored.metadata
        scorer = _make_scorer(stored, resolved_device, max_chunk_tokens)
        try:
            for query_idx in range(completed, query_count):
                query = dataset.query_embeddings[query_idx]
                if stored.query_fp16:
                    query = query.astype(np.float16).astype(np.float32)
                scores = scorer.score(query)
                ndcg, recall = _query_metrics(scores, dataset.qrels[query_idx], top_k)
                ndcg_values.append(None if ndcg is None else round(ndcg, 8))
                recall_values.append(None if recall is None else round(recall, 8))
                row["completed_queries"] = query_idx + 1
                if (query_idx + 1) % checkpoint_every == 0 or query_idx + 1 == query_count:
                    _atomic_write_json(output, payload)
                    print(f"{format_name}: {query_idx + 1}/{query_count} queries", flush=True)
        finally:
            scorer.close()
            del scorer
            del stored
            gc.collect()

        row[f"ndcg_at_{top_k}"] = _mean_present(ndcg_values)
        row[f"recall_at_{top_k}"] = _mean_present(recall_values)
        row["status"] = "complete"
        _atomic_write_json(output, payload)

    payload["status"] = "complete"
    _atomic_write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--formats", default=",".join(FORMAT_NAMES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--max-chunk-tokens", type=int, default=1_000_000)
    parser.add_argument("--limit-queries", type=int, default=0, help="0 = all queries")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output or Path("benchmark-results") / f"quality-matrix-{args.cache.stem}.json"
    run_quality_matrix(
        args.cache,
        output,
        formats=args.formats,
        top_k=args.top_k,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        max_chunk_tokens=args.max_chunk_tokens,
        limit_queries=args.limit_queries,
        overwrite=args.overwrite,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
