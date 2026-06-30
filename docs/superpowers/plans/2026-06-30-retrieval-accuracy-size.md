# Retrieval Accuracy And Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve bitmax retrieval accuracy and size tradeoffs on a real ViDoRe/ColQwen-style slice while preserving GPU-first benchmarking.

**Architecture:** First build a real embedding artifact on the persistent VAST RTX 4090 worker, then measure existing binary, doc-scale, threshold, scale, and ternary reference variants on the same slice. Promote the smallest winning variant: thresholded binary packing first because it preserves one-bit document storage and reuses existing CUDA scoring; per-token scale only if the measured quality gain justifies added storage.

**Tech Stack:** Python, NumPy, PyTorch CUDA, transformers/datasets for embedding generation, C++/CUDA pybind kernels, pytest, VAST.

**Execution Outcome:** The same-size median/quantile threshold variants did not win on the real slice. The promoted variant is instead zero-threshold per-dimension centroid calibration under `bitmax.experimental`, which keeps one-bit document storage and reuses the existing CUDA binary kernel through query preprocessing. Per-token/group scale and ternary were not promoted because they were worse on the measured accuracy/storage Pareto frontier.

---

## File Structure

- `benchmark-results/*.npz`: ignored ViDoRe embedding artifacts generated on VAST and optionally fetched locally.
- `benchmark-results/*.json`: ignored retrieval and CUDA benchmark result artifacts.
- `benchmarks/run_retrieval.py`: add real-slice variant support only when a variant needs benchmark wiring.
- `src/bitmax/experimental.py`: add thresholded binary packing helpers if same-size thresholding wins.
- `tests/test_ternary_reference.py` or a new `tests/test_experimental_thresholds.py`: test any experimental packing behavior before implementation.
- `cpp/bitmax/cuda_extension.cu`: modify only if per-token scale is promoted to GPU.
- `tests/test_cuda_extension.py`: add CUDA correctness tests for any new GPU scoring path.
- `docs/gpu_optimization.md`: record measured Pareto rows and artifact names.

## Task 1: Build Real Retrieval Slice On VAST

- [ ] **Step 1: Validate the persistent worker**

Run:

```bash
.venv/bin/vastai show instance 43248165
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai 'nvidia-smi && python --version'
```

Expected: instance label starts with `bitmax-v0-`, GPU is RTX 4090, SSH succeeds.

- [ ] **Step 2: Install retrieval dependencies on the worker**

Run:

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m pip install ".[vision-retrieval]"'
```

Expected: dependencies install without replacing the CUDA-enabled torch base package.

- [ ] **Step 3: Build a limit-64 embedding artifact**

Run:

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m benchmarks.build_vidore_embeddings \
    --dataset vidore/docvqa_test_subsampled \
    --split test \
    --limit 64 \
    --model vidore/colqwen2-v1.0-hf \
    --batch-size 1 \
    --output benchmark-results/vidore-docvqa-colqwen2-limit64.npz'
```

Expected: `.npz` exists with `doc_embeddings`, `doc_offsets`, `query_embeddings`, and `qrels`.

## Task 2: Measure Existing Variants On The Same Slice

- [ ] **Step 1: Run dense and default CUDA retrieval**

Run:

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m benchmarks.run_retrieval \
    --stage embeddings-cuda-smoke \
    --input benchmark-results/vidore-docvqa-colqwen2-limit64.npz \
    --gate benchmark-results/retrieval-docvqa-colqwen2-limit64-cpu-gate.json \
    --output benchmark-results/retrieval-docvqa-colqwen2-limit64-cuda-default.json'
```

If the gate file is missing, first run `embeddings-smoke` on the same input and use that JSON as the gate.

- [ ] **Step 2: Run all current Pareto variants**

Run:

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m benchmarks.run_retrieval \
    --stage embeddings-cuda-smoke \
    --input benchmark-results/vidore-docvqa-colqwen2-limit64.npz \
    --gate benchmark-results/retrieval-docvqa-colqwen2-limit64-cpu-gate.json \
    --variants all \
    --output benchmark-results/retrieval-docvqa-colqwen2-limit64-pareto.json'
```

Expected: JSON rows include dense fp16, raw binary, doc-scale binary, ternary threshold, token scale, grouped scale, and calibrated threshold.

- [ ] **Step 3: Fetch and summarize artifacts**

Run:

```bash
scp -i ~/.ssh/id_ed25519 -P 18164 root@ssh2.vast.ai:/workspace/bitmax/benchmark-results/retrieval-docvqa-colqwen2-limit64-pareto.json benchmark-results/
jq '.results[] | {implementation, latency_ms, doc_storage_bytes, speedup_vs_dense_fp16, recall_at_1, recall_at_10, mrr_at_10, ndcg_at_10, doc_memory_compression_vs_fp32}' benchmark-results/retrieval-docvqa-colqwen2-limit64-pareto.json
```

Expected: one table identifies whether same-size calibrated binary or a scale variant improves NDCG/recall.

## Task 3: Promote Thresholded Binary If It Wins

- [ ] **Step 1: Write a failing threshold packing test**

Create `tests/test_experimental_thresholds.py`:

```python
import numpy as np
import bitmax


def test_pack_threshold_signs_preserves_one_bit_storage_and_uses_per_dim_thresholds():
    from bitmax.experimental import pack_threshold_signs

    docs = np.array(
        [
            [0.1, 2.0, -1.0, 4.0, 0.0, -0.5, 3.0, -4.0],
            [0.3, 0.5, -3.0, 1.0, -0.1, -0.4, 2.0, -2.0],
        ],
        dtype=np.float32,
    )
    thresholds = np.array([0.2, 1.0, -2.0, 2.0, 0.0, -0.45, 2.5, -3.0], dtype=np.float32)

    packed = pack_threshold_signs(docs, thresholds=thresholds)

    assert packed.dim == 8
    assert packed.num_docs == 2
    assert packed.data.shape == (2, 1)
    expected_signs = np.where(docs >= thresholds[np.newaxis, :], 1.0, -1.0)
    np.testing.assert_allclose(bitmax.maxsim(np.eye(8, dtype=np.float32), packed), bitmax.maxsim(np.eye(8, dtype=np.float32), bitmax.pack_signs(expected_signs)))
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_experimental_thresholds.py
```

Expected: fails because `pack_threshold_signs` is missing.

- [ ] **Step 3: Implement `pack_threshold_signs`**

Add to `src/bitmax/experimental.py`:

```python
def pack_threshold_signs(doc_embeddings, doc_offsets=None, *, thresholds) -> PackedDocs:
    docs = _as_numpy(doc_embeddings)
    threshold_values = _as_numpy(thresholds).astype(np.float32, copy=False)
    if docs.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")
    if threshold_values.ndim != 1 or threshold_values.shape[0] != docs.shape[1]:
        raise ValueError("thresholds must have shape [dim]")
    thresholded = np.where(docs >= threshold_values[np.newaxis, :], 1.0, -1.0).astype(np.float32)
    return pack_signs(thresholded, doc_offsets)
```

Also import `PackedDocs` and `pack_signs` from `bitmax._api`.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_experimental_thresholds.py tests/test_ternary_reference.py
```

Expected: all pass.

## Task 4: Promote Per-Token Scale Only If Needed

- [ ] **Step 1: Gate decision from Task 2**

Proceed only if `binary_token_scale` or grouped scale improves NDCG@10 or recall@1 meaningfully over raw binary and thresholded binary on the real slice.

- [ ] **Step 2: Write a CUDA correctness test before kernel work**

Add a CUDA test to `tests/test_cuda_extension.py` that compares token-scaled CUDA MaxSim against a dense reference where each binary document token is multiplied by its token scale before MaxSim.

- [ ] **Step 3: Implement minimal CUDA token-scale scoring**

Add a resident token-scale vector to `CudaPackedDocs`, apply `token_scale[token] * dot` before per-query-token max, and expose a benchmark-only Python path.

- [ ] **Step 4: Benchmark storage and quality**

Run the same ViDoRe slice and report bytes/doc, latency, recall@1, recall@10, MRR@10, and NDCG@10.

## Task 5: Document And Commit

- [ ] **Step 1: Update `docs/gpu_optimization.md`**

Record only measured rows and artifact names. Include the decision about whether a variant is promoted, deferred, or rejected.

- [ ] **Step 2: Run final verification**

Run:

```bash
.venv/bin/python -m pytest -q
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 18164 root@ssh2.vast.ai 'cd /workspace/bitmax && . .venv/bin/activate && python -m pytest -q'
```

Expected: local and remote tests pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs benchmarks src tests cpp
git commit -m "exp: measure retrieval accuracy size tradeoffs"
git branch -f main HEAD
```

## Self-Review

- Spec coverage: covers VAST usage, real retrieval measurement, same-size thresholding, scale escalation, docs, and tests.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: names are stable: `pack_threshold_signs`, `retrieval-docvqa-colqwen2-limit64-pareto.json`, and `cuda-dim128-lut-topk-4090.json`.
