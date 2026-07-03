"""A/B the q-tiled one-pass kernel against the unrolled kernel at scale.

Synthesizes packed corpora directly (latency-only; packing quality is
irrelevant here), flips the runtime routing threshold, and reports
warmup+median timings plus score/index parity. The interesting shapes are the
ones whose packed corpus exceeds the L2 cache (72MB on RTX 4090).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import bitmax
from bitmax import _bitmax_cuda

SHAPES = [
    {"name": "docs_1000_long_12MB", "docs": 1000, "tokens_per_doc": 750, "batch": 8, "query_tokens": 24},
    {"name": "docs_5000_long_60MB", "docs": 5000, "tokens_per_doc": 750, "batch": 8, "query_tokens": 24},
    {"name": "docs_10000_long_120MB", "docs": 10000, "tokens_per_doc": 750, "batch": 8, "query_tokens": 24},
    {"name": "docs_25000_long_300MB", "docs": 25000, "tokens_per_doc": 750, "batch": 4, "query_tokens": 24},
]


def _median_ms(fn, *, warmup: int = 3, repeat: int = 10) -> float:
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
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--token-scale", action="store_true", help="Also measure with a resident token scale vector.")
    args = parser.parse_args()

    rng = np.random.default_rng(131)
    rows = []
    for shape in SHAPES:
        tokens = shape["docs"] * shape["tokens_per_doc"]
        packed_data = rng.integers(0, 256, size=(tokens, 16), dtype=np.uint8)
        offsets = np.arange(0, shape["docs"] + 1, dtype=np.int64) * shape["tokens_per_doc"]
        query = rng.standard_normal((shape["batch"], shape["query_tokens"], 128)).astype(np.float32)
        token_scale = rng.uniform(0.05, 0.4, size=tokens).astype(np.float32) if args.token_scale else None
        packed = bitmax.PackedDocs(
            data=packed_data,
            doc_offsets=offsets,
            dim=128,
            num_docs=shape["docs"],
            token_scale=token_scale,
        )
        cuda_packed = bitmax.to_device(packed, "cuda")
        handle = cuda_packed.data
        k = 10
        use_token_scale = token_scale is not None

        results = {}
        scores = {}
        for label, threshold in (("unrolled", 1 << 62), ("qtile", 1)):
            _bitmax_cuda.set_dim128_qtile_min_packed_bytes(threshold)
            expected = "dim128_qtile" if label == "qtile" else "dim128_unrolled"
            assert handle.maxsim_kernel_variant == expected, (label, handle.maxsim_kernel_variant)
            results[label] = _median_ms(
                lambda: handle.topk_batch(query, k, 1.0, False, use_token_scale), repeat=args.repeat
            )
            scores[label] = handle.topk_batch(query, k, 1.0, False, use_token_scale)

        np.testing.assert_array_equal(scores["unrolled"][1], scores["qtile"][1])
        np.testing.assert_allclose(scores["unrolled"][0], scores["qtile"][0], rtol=1e-4, atol=1e-2)

        row = {
            **shape,
            "packed_mb": tokens * 16 / 1e6,
            "token_scale": use_token_scale,
            "unrolled_median_ms": results["unrolled"],
            "qtile_median_ms": results["qtile"],
            "qtile_speedup": results["unrolled"] / max(results["qtile"], 1e-9),
        }
        rows.append(row)
        print(
            f"{shape['name']:24s} unrolled={results['unrolled']:9.3f}ms qtile={results['qtile']:9.3f}ms "
            f"speedup={row['qtile_speedup']:.2f}x"
        )
        del handle, cuda_packed, packed, packed_data

    _bitmax_cuda.set_dim128_qtile_min_packed_bytes(48 << 20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
