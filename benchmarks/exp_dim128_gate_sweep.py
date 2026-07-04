"""A/B the dim128 unrolled-kernel routing gate with warmup+median timing.

Compares the generic and unrolled scoring kernels on realistic shapes
(long-doc retrieval corpora plus the documented short-doc regression shapes)
by flipping the runtime gate, and verifies score parity between both paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import maxsim
from maxsim import _maxsim_cuda

SHAPES = [
    {"name": "docvqa_like_256", "docs": 256, "tokens_per_doc": 750, "batch": 64, "query_tokens": 23},
    {"name": "docs_1000_long", "docs": 1000, "tokens_per_doc": 750, "batch": 8, "query_tokens": 32},
    {"name": "docs_5000_long", "docs": 5000, "tokens_per_doc": 750, "batch": 8, "query_tokens": 32},
    {"name": "rerank_512_short", "docs": 512, "tokens_per_doc": 128, "batch": 32, "query_tokens": 32},
    {"name": "rerank_4096_short", "docs": 4096, "tokens_per_doc": 32, "batch": 16, "query_tokens": 32},
]


def _median_ms(fn, *, warmup: int = 5, repeat: int = 20) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return float(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(97)
    rows = []
    for shape in SHAPES:
        docs = rng.standard_normal((shape["docs"] * shape["tokens_per_doc"], 128)).astype(np.float32)
        offsets = np.arange(0, shape["docs"] + 1, dtype=np.int64) * shape["tokens_per_doc"]
        query = rng.standard_normal((shape["batch"], shape["query_tokens"], 128)).astype(np.float32)
        packed = maxsim.pack_signs(docs, offsets)
        cuda_packed = maxsim.to_device(packed, "cuda")
        handle = cuda_packed.data
        k = min(10, shape["docs"])

        variants = {}
        scores = {}
        for label, threshold in (("generic", 1 << 30), ("unrolled", 1)):
            _maxsim_cuda.set_dim128_unrolled_min_avg_tokens(threshold)
            assert handle.maxsim_kernel_variant == ("generic" if label == "generic" else "dim128_unrolled") or shape["docs"] <= 128
            variants[label] = _median_ms(lambda: handle.topk_batch(query, k), repeat=args.repeat)
            scores[label] = handle.topk_batch(query, k)

        np.testing.assert_array_equal(scores["generic"][1], scores["unrolled"][1])
        np.testing.assert_allclose(scores["generic"][0], scores["unrolled"][0], rtol=0, atol=1e-3)

        rows.append(
            {
                **shape,
                "generic_median_ms": variants["generic"],
                "unrolled_median_ms": variants["unrolled"],
                "unrolled_speedup": variants["generic"] / max(variants["unrolled"], 1e-9),
                "parity": "exact_indices_tol_scores",
            }
        )
        print(
            f"{shape['name']:20s} generic={variants['generic']:8.3f}ms "
            f"unrolled={variants['unrolled']:8.3f}ms speedup={rows[-1]['unrolled_speedup']:.2f}x"
        )

    _maxsim_cuda.set_dim128_unrolled_min_avg_tokens(1 << 30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
