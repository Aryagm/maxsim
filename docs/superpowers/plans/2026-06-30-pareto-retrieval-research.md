# Pareto Retrieval Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve bitmax against the blog-style late-interaction baseline across accuracy, latency, and storage bytes per document.

**Architecture:** Keep `bitmax` a kernel library, not a search engine. Add benchmark and experimental backend surfaces that can compare fp32/fp16/int8 dense baselines, int8-query x binary-doc scoring, true fused CUDA top-k, ternary docs, and magnitude-restoration variants on the same fixtures and retrieval slices.

**Tech Stack:** Python, NumPy, pytest, C++/CUDA via pybind11/scikit-build-core, VAST for CUDA validation.

---

## File Structure

- `benchmarks/blog_baseline.py`: new benchmark runner for the blog-style 33 query tokens x 1000 docs x 786 doc tokens x 128 dim shape and smaller smoke shapes.
- `tests/test_blog_baseline.py`: correctness and JSON-schema tests for the new benchmark runner.
- `src/bitmax/_api.py`: public experimental format routing only when a kernel graduates beyond benchmark-only code.
- `cpp/bitmax/cuda_extension.cu`: CUDA kernels for true fused MaxSim+top-k and later ternary scoring.
- `tests/test_cuda_extension.py`: CUDA correctness tests for fused streaming top-k and ternary scoring.
- `docs/gpu_optimization.md`: measured results only, with artifact paths.
- `docs/benchmarks.md`: commands and gate rules for local and VAST runs.
- `benchmark-results/*.json`: ignored measured artifacts fetched from local/VAST runs.

## Task 1: Blog-Style Int8 Query x Binary Doc Benchmark

**Files:**
- Create: `benchmarks/blog_baseline.py`
- Create: `tests/test_blog_baseline.py`
- Modify: `docs/benchmarks.md`

- [x] **Step 1: Write the failing schema/correctness test**

```python
def test_blog_binary_benchmark_emits_blog_shape_storage_and_correct_scores(tmp_path):
    from benchmarks.blog_baseline import run_benchmark

    out = tmp_path / "blog.json"
    result = run_benchmark(stage="smoke", output_path=out, repeat=1)

    assert result["benchmark"] == "blog_baseline"
    rows = {row["implementation"]: row for row in result["results"]}
    assert rows["int8_query_binary_docs"]["doc_storage_bytes_per_doc"] == 4
    assert rows["int8_query_binary_docs"]["max_abs_delta_vs_reference"] == 0.0
```

- [x] **Step 2: Verify the test fails**

Run: `.venv/bin/python -m pytest -q tests/test_blog_baseline.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.blog_baseline'`.

- [x] **Step 3: Implement the benchmark runner**

Create `benchmarks/blog_baseline.py` with:

```python
def run_benchmark(stage="smoke", output_path=None, repeat=None):
    specs = _stage_specs(stage)
    rows = []
    for spec in specs:
        query, docs, offsets = _make_fixture(spec)
        packed = bitmax.pack_signs(docs, offsets)
        ref_scores, ref_ms = _time_call(lambda: _dense_fp32_maxsim(query, docs, offsets), repeat=repeat or spec["repeat"])
        binary_scores, binary_ms = _time_call(lambda: bitmax.maxsim(query.astype(np.int8), packed), repeat=repeat or spec["repeat"])
        rows.extend([...])
    result = {"schema_version": 1, "benchmark": "blog_baseline", "stage": stage, "results": rows}
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
```

The actual implementation must include concrete row dictionaries with `latency_ms`, `doc_storage_bytes_per_doc`, `speedup_vs_fp32`, and `max_abs_delta_vs_reference`.

- [x] **Step 4: Verify the focused test passes**

Run: `.venv/bin/python -m pytest -q tests/test_blog_baseline.py`

Expected: PASS.

- [x] **Step 5: Run local smoke benchmark**

Run: `.venv/bin/python -m benchmarks.blog_baseline --stage smoke --output benchmark-results/blog-baseline-smoke.json`

Expected: JSON artifact with fp32, int8 x int8, int8 x binary, and binary x binary rows.

- [x] **Step 6: Commit**

```bash
git add benchmarks/blog_baseline.py tests/test_blog_baseline.py docs/benchmarks.md
git commit -m "bench: add blog-style quantization baseline"
```

## Task 2: True Fused CUDA MaxSim + Top-K

**Files:**
- Modify: `cpp/bitmax/cuda_extension.cu`
- Modify: `src/bitmax/_api.py`
- Modify: `tests/test_cuda_extension.py`
- Modify: `docs/gpu_optimization.md`

- [x] **Step 1: Write failing CUDA correctness test**

Add a CUDA-marked test that compares `cuda_packed.data.streaming_topk_batch(query, k, 1.0, False)` with `bitmax.topk_maxsim(query, cuda_packed, k)` for ragged docs, including deterministic lower-doc-id tie-breaking.

- [ ] **Step 2: Verify the test fails on VAST**

Run remotely: `python -m pytest -q -m cuda tests/test_cuda_extension.py::test_cuda_streaming_topk_matches_resident_topk`

Expected: FAIL with missing `streaming_topk_batch`.

- [x] **Step 3: Implement streaming top-k kernel**

Add a CUDA kernel that assigns one block to each `(batch, doc)` score, computes the document score, and updates a compact per-batch top-k buffer without writing the full `[batch, docs]` matrix. Keep the existing full-score fused top-k as a fallback until benchmarks prove the streaming kernel wins.

- [x] **Step 4: Verify CUDA correctness**

Done on project-owned VAST instance `43248165`.

Run remotely: `python -m pytest -q -m cuda tests/test_cuda_extension.py`

Expected: all CUDA tests pass.

- [x] **Step 5: Benchmark against existing fused top-k**

Run a VAST timing script for `docs=64`, `512`, and `4096`, reporting latency, score delta, index equality, and host bytes returned.

- [x] **Step 6: Gate routing**

Use streaming top-k only for shapes where same-host VAST timing shows lower latency with exact indices.

Result: streaming top-k was exact but slower, so it is not routed.

## Task 3: Experimental Ternary Document Backend

**Files:**
- Modify: `src/bitmax/_api.py`
- Modify: `cpp/bitmax/cuda_extension.cu`
- Create: `tests/test_ternary_reference.py`
- Modify: `tests/test_cuda_extension.py`
- Modify: `benchmarks/blog_baseline.py`

- [x] **Step 1: Write failing reference tests**

Add tests for `pack_ternary(docs, threshold=...)` where values with `abs(x) <= threshold` score as zero, positive values score as `+1`, and negative values score as `-1`.

- [x] **Step 2: Implement Python reference pack/scoring**

Add an experimental `TernaryPackedDocs` dataclass and `pack_ternary` behind `bitmax.experimental` or a private benchmark module so the v0.1 public API does not expand prematurely.

- [ ] **Step 3: Add CUDA ternary smoke test**

Compare CUDA ternary MaxSim scores to Python reference for small and dim128 fixtures.

- [ ] **Step 4: Benchmark storage/latency/quality**

Compare binary 1-bit docs to ternary 2-bit docs on smoke, blog-shape synthetic, and ViDoRe slices. Report storage as 32x vs 16x fp32.

Partial: smoke, blog-shape synthetic, and local targeted retrieval fixtures are
measured. The ViDoRe rerun is blocked because the embedding `.npz` is not
present locally and would need to be regenerated.

## Task 4: Magnitude Restoration Variants

**Files:**
- Modify: `benchmarks/blog_baseline.py`
- Modify: `benchmarks/run_retrieval.py`
- Modify: `src/bitmax/_api.py` only for variants that pass benchmark gates.
- Modify: `docs/gpu_optimization.md`

- [x] **Step 1: Add per-token scale benchmark variant**

Compute one `mean(abs(token))` scale per doc token and apply it in the reference scorer. Gate public/kernel work on retrieval quality improvement.

- [x] **Step 2: Add grouped scale benchmark variant**

Compute one scale per 16 or 32 dimensions and compare storage, latency, and NDCG.

- [x] **Step 3: Add calibrated threshold benchmark variant**

Estimate per-dimension sign thresholds on a calibration split and apply the thresholds to held-out docs.

Current implementation is a median-threshold reference probe over the benchmark
docs. A true train/held-out calibration split remains future work before
promotion.

- [x] **Step 4: Promote only measured wins**

Only move a variant into `src/bitmax/_api.py` or CUDA if it improves recall/NDCG enough to justify added storage/latency.

## Task 5: Retrieval Pareto Report

**Files:**
- Modify: `docs/gpu_optimization.md`
- Modify: `docs/benchmarks.md`

- [x] **Step 1: Run local smoke**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m benchmarks.blog_baseline --stage smoke`

- [x] **Step 2: Run VAST CUDA gate**

Run CUDA tests and benchmark scripts on one ledger-owned VAST instance. Destroy only the instance ID recorded in `.vast/bitmax-instances.jsonl` with a live `bitmax-v0-` label.

Done on project-owned VAST instance `43248165`. The worker is intentionally left
running for follow-up GPU experiments.

- [x] **Step 3: Report Pareto rows**

Document each variant with latency, storage bytes/doc, recall@1, recall@10, MRR@10, NDCG@10, score delta, and artifact path.

## Self-Review

- Spec coverage: the plan covers blog reproduction, true fused top-k, ternary docs, magnitude restoration, retrieval metrics, default-gating, and VAST cleanup.
- Placeholder scan: no task relies on `TBD` or unbounded "implement later" language.
- Type consistency: planned names are stable: `blog_baseline`, `streaming_topk_batch`, `pack_ternary`, and `TernaryPackedDocs`.
