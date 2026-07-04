from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import maxsim
from maxsim.experimental import (
    fit_dim_centroid_calibration,
    pack_dim_centroid_signs,
    topk_dim_centroid_maxsim,
    transform_query_dim_centroids,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional CUDA benchmark dependency
    torch = None


def run_benchmark(stage: str, *, output_path: Path | None = None) -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA top-k benchmarks require torch with CUDA")
    if stage == "lut-sweep":
        result = _run_lut_sweep()
    elif stage == "blog-shape":
        result = _run_blog_shape()
    else:
        raise ValueError(f"unknown CUDA top-k benchmark stage: {stage}")

    output = output_path or Path("benchmark-results") / f"cuda-topk-{stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _run_lut_sweep() -> dict[str, Any]:
    cases = [
        ("docvqa_like_64", dict(batch=64, query_tokens=23, docs=64, doc_tokens=750, seed=11, repeat=20)),
        ("rerank_512", dict(batch=32, query_tokens=32, docs=512, doc_tokens=128, seed=12, repeat=20)),
        ("rerank_4096", dict(batch=16, query_tokens=32, docs=4096, doc_tokens=32, seed=13, repeat=20)),
        ("blog_single_query", dict(batch=1, query_tokens=33, docs=1000, doc_tokens=786, seed=14, repeat=30)),
        ("docscale_blog_single_query", dict(batch=1, query_tokens=33, docs=1000, doc_tokens=786, seed=15, repeat=30, scale=True)),
    ]
    rows = [_lut_case(name, **config) for name, config in cases]
    return {
        "schema_version": 1,
        "benchmark": "dim128_lut_topk",
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
    }


def _lut_case(
    name: str,
    *,
    batch: int,
    query_tokens: int,
    docs: int,
    doc_tokens: int,
    seed: int,
    repeat: int,
    dim: int = 128,
    k: int = 10,
    scale: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    query_int8 = rng.integers(-8, 9, size=(batch, query_tokens, dim), dtype=np.int8)
    query_float = query_int8.astype(np.float32)
    doc_embeddings = rng.normal(size=(docs * doc_tokens, dim)).astype(np.float32)
    offsets = np.arange(docs + 1, dtype=np.int64) * doc_tokens
    packed = maxsim.to_device(maxsim.pack_signs(doc_embeddings, offsets, scale="doc" if scale else None), "cuda")

    (base_scores, base_indices), base_best, base_median = _time_call(
        lambda: packed.data.topk_batch(query_float, k, 1.0, scale),
        repeat=repeat,
    )
    (lut_scores, lut_indices), lut_best, lut_median = _time_call(
        lambda: packed.data.topk_lut_batch(query_float, k, 1.0, scale),
        repeat=repeat,
    )
    (_api_scores, _api_indices), api_best, api_median = _time_call(
        lambda: maxsim.topk_maxsim(query_int8, packed, k),
        repeat=repeat,
    )
    return {
        "name": name,
        "batch": batch,
        "query_tokens": query_tokens,
        "docs": docs,
        "doc_tokens": doc_tokens,
        "dim": dim,
        "k": k,
        "use_doc_scale": scale,
        "resident_topk_best_ms": base_best,
        "resident_topk_median_ms": base_median,
        "lut_topk_best_ms": lut_best,
        "lut_topk_median_ms": lut_median,
        "api_topk_best_ms": api_best,
        "api_topk_median_ms": api_median,
        "lut_speedup_median": base_median / lut_median,
        "api_speedup_vs_resident_median": base_median / api_median,
        "score_delta": float(np.max(np.abs(np.asarray(base_scores) - np.asarray(lut_scores)))),
        "indices_match": bool(np.array_equal(np.asarray(base_indices), np.asarray(lut_indices))),
    }


def _run_blog_shape() -> dict[str, Any]:
    batch = 1
    query_tokens = 33
    docs = 1000
    doc_tokens = 786
    dim = 128
    k = 10
    repeat = 50
    rng = np.random.default_rng(20260630)
    query = rng.normal(0.0, 2.0, size=(batch, query_tokens, dim)).astype(np.float32)
    doc_embeddings = rng.normal(0.0, 2.0, size=(docs * doc_tokens, dim)).astype(np.float32)
    offsets = np.arange(docs + 1, dtype=np.int64) * doc_tokens
    query_int8 = np.clip(np.rint(query), -127, 127).astype(np.int8)

    fp32_values, fp32_indices, fp32_best, fp32_median = _dense_torch_topk(query, doc_embeddings, docs, doc_tokens, k, torch.float32, repeat=repeat)
    fp16_values, fp16_indices, fp16_best, fp16_median = _dense_torch_topk(query, doc_embeddings, docs, doc_tokens, k, torch.float16, repeat=repeat)
    binary_values, binary_indices, binary_best, binary_median = _bitmax_topk(query_int8, doc_embeddings, offsets, k, repeat=repeat)
    doc_values, doc_indices, doc_best, doc_median = _bitmax_topk(query_int8, doc_embeddings, offsets, k, scale="doc", repeat=repeat)
    (
        centroid_values,
        centroid_indices,
        centroid_host_best,
        centroid_host_median,
        centroid_fused_best,
        centroid_fused_median,
        centroid_storage,
    ) = _bitmax_centroid_topk(query_int8, doc_embeddings, offsets, k, repeat=repeat)

    return {
        "schema_version": 1,
        "benchmark": "blog_shape_gpu_topk",
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"batch": batch, "query_tokens": query_tokens, "docs": docs, "doc_tokens": doc_tokens, "dim": dim, "k": k},
        "rows": [
            {
                "implementation": "torch_fp32_docs_fp32_query_topk",
                "latency_best_ms": fp32_best,
                "latency_median_ms": fp32_median,
                "doc_storage_bytes_per_doc": doc_tokens * dim * 4,
                "speedup_vs_torch_fp32_median": 1.0,
                "topk_agreement_vs_fp32": 1.0,
                "max_abs_topk_score_delta_vs_fp32": 0.0,
            },
            {
                "implementation": "torch_fp16_docs_fp16_query_topk",
                "latency_best_ms": fp16_best,
                "latency_median_ms": fp16_median,
                "doc_storage_bytes_per_doc": doc_tokens * dim * 2,
                "speedup_vs_torch_fp32_median": fp32_median / fp16_median,
                "topk_agreement_vs_fp32": _topk_agreement(fp32_indices, fp16_indices, k),
                "max_abs_topk_score_delta_vs_fp32": float(np.max(np.abs(fp32_values - fp16_values))),
            },
            {
                "implementation": "bitmax_cuda_int8_query_binary_docs_lut_topk",
                "latency_best_ms": binary_best,
                "latency_median_ms": binary_median,
                "doc_storage_bytes_per_doc": doc_tokens * dim // 8,
                "speedup_vs_torch_fp32_median": fp32_median / binary_median,
                "speedup_vs_blog_int8_binary_latency_3_71ms": 3.71 / binary_median,
                "topk_agreement_vs_fp32": _topk_agreement(fp32_indices, binary_indices, k),
                "max_abs_topk_score_delta_vs_fp32": float(np.max(np.abs(fp32_values - binary_values))),
            },
            {
                "implementation": "bitmax_cuda_int8_query_binary_docs_doc_scale_lut_topk",
                "latency_best_ms": doc_best,
                "latency_median_ms": doc_median,
                "doc_storage_bytes_per_doc": doc_tokens * dim // 8 + 4,
                "speedup_vs_torch_fp32_median": fp32_median / doc_median,
                "speedup_vs_blog_int8_binary_latency_3_71ms": 3.71 / doc_median,
                "topk_agreement_vs_fp32": _topk_agreement(fp32_indices, doc_indices, k),
                "max_abs_topk_score_delta_vs_fp32": float(np.max(np.abs(fp32_values - doc_values))),
            },
            {
                "implementation": "bitmax_cuda_int8_query_binary_docs_centroid_host_transform_topk",
                "latency_best_ms": centroid_host_best,
                "latency_median_ms": centroid_host_median,
                "doc_storage_bytes_per_doc": centroid_storage,
                "speedup_vs_torch_fp32_median": fp32_median / centroid_host_median,
                "speedup_vs_blog_int8_binary_latency_3_71ms": 3.71 / centroid_host_median,
                "topk_agreement_vs_fp32": _topk_agreement(fp32_indices, centroid_indices, k),
                "max_abs_topk_score_delta_vs_fp32": float(np.max(np.abs(fp32_values - centroid_values))),
            },
            {
                "implementation": "bitmax_cuda_int8_query_binary_docs_centroid_fused_topk",
                "latency_best_ms": centroid_fused_best,
                "latency_median_ms": centroid_fused_median,
                "doc_storage_bytes_per_doc": centroid_storage,
                "speedup_vs_torch_fp32_median": fp32_median / centroid_fused_median,
                "speedup_vs_blog_int8_binary_latency_3_71ms": 3.71 / centroid_fused_median,
                "speedup_vs_centroid_host_transform_median": centroid_host_median / centroid_fused_median,
                "topk_agreement_vs_fp32": _topk_agreement(fp32_indices, centroid_indices, k),
                "max_abs_topk_score_delta_vs_fp32": float(np.max(np.abs(fp32_values - centroid_values))),
            },
        ],
    }


def _dense_torch_topk(query: np.ndarray, docs: np.ndarray, num_docs: int, doc_tokens: int, k: int, dtype, *, repeat: int):
    query_tensor = torch.as_tensor(query, device="cuda", dtype=dtype)
    docs_tensor = torch.as_tensor(docs.reshape(num_docs, doc_tokens, query.shape[-1]), device="cuda", dtype=dtype)

    def fn():
        scores = torch.einsum("bqd,ntd->bqnt", query_tensor, docs_tensor).amax(dim=3).sum(dim=1)
        return torch.topk(scores, k, dim=1)

    result, best, median = _time_call(fn, repeat=repeat, warmup=10)
    return result.values.detach().cpu().numpy(), result.indices.detach().cpu().numpy(), best, median


def _bitmax_topk(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray, k: int, *, scale=None, repeat: int):
    packed = maxsim.to_device(maxsim.pack_signs(docs, offsets, scale=scale), "cuda")
    result, best, median = _time_call(lambda: maxsim.topk_maxsim(query, packed, k), repeat=repeat, warmup=10)
    return np.asarray(result[0]), np.asarray(result[1]), best, median


def _bitmax_centroid_topk(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray, k: int, *, repeat: int):
    calibration = fit_dim_centroid_calibration(docs)
    packed, calibration = pack_dim_centroid_signs(docs, offsets, calibration=calibration)
    cuda_packed = maxsim.to_device(packed, "cuda")

    def host_transform():
        transformed = transform_query_dim_centroids(query, calibration)
        scores, indices = maxsim.topk_maxsim(transformed, cuda_packed, k)
        return _restore_centroid_top_scores(scores, query, calibration), indices

    host_result, host_best, host_median = _time_call(host_transform, repeat=repeat, warmup=10)
    fused_result, fused_best, fused_median = _time_call(
        lambda: topk_dim_centroid_maxsim(query, cuda_packed, calibration, k),
        repeat=repeat,
        warmup=10,
    )
    np.testing.assert_allclose(np.asarray(fused_result[0]), np.asarray(host_result[0]), rtol=0, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(fused_result[1]), np.asarray(host_result[1]))
    doc_count = int(offsets.shape[0] - 1)
    storage_per_doc = (docs.shape[0] * docs.shape[1] // 8 + calibration.metadata_bytes) / float(doc_count)
    return (
        np.asarray(fused_result[0]),
        np.asarray(fused_result[1]),
        host_best,
        host_median,
        fused_best,
        fused_median,
        storage_per_doc,
    )


def _restore_centroid_top_scores(scores: np.ndarray, query: np.ndarray, calibration) -> np.ndarray:
    score_values = np.asarray(scores, dtype=np.float32)
    query_float = np.asarray(query, dtype=np.float32)
    centroid_sum = calibration.positive_centroids + calibration.negative_centroids
    if query_float.ndim == 2:
        constant = np.float32(0.5 * np.sum(query_float * centroid_sum[np.newaxis, :], dtype=np.float64))
        return np.asarray(score_values * 0.5 + constant, dtype=np.float32)
    constants = 0.5 * np.sum(query_float * centroid_sum[np.newaxis, np.newaxis, :], axis=(1, 2), dtype=np.float64)
    return np.asarray(score_values * 0.5 + constants[:, np.newaxis].astype(np.float32), dtype=np.float32)


def _time_call(fn, *, repeat: int, warmup: int = 5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    last = None
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    values.sort()
    return last, values[0], values[len(values) // 2]


def _topk_agreement(reference_indices: np.ndarray, indices: np.ndarray, k: int) -> float:
    agreements = []
    for reference_row, row in zip(reference_indices, indices):
        agreements.append(len(set(reference_row.tolist()) & set(row.tolist())) / float(k))
    return float(np.mean(agreements)) if agreements else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-resident top-k benchmarks.")
    parser.add_argument("--stage", required=True, choices=["lut-sweep", "blog-shape"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.stage, output_path=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
