# Binary-First Accuracy And Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve bitmax usability and accuracy while keeping 32x binary document compression as the primary path and int4 as an optional experimental backend.

**Architecture:** Add persistence for CPU packed docs and centroid calibration, then fuse centroid query weighting into the CUDA top-k path. Validate on larger VAST retrieval slices before making broader accuracy claims. Add int4 as a separate experimental format with its own tests and metrics.

**Tech Stack:** Python, NumPy, PyTorch CUDA, pybind11/CUDA extension, pytest, VAST.

---

## File Structure

- `src/bitmax/io.py`: save/load helpers and `PackedBundle` dataclass.
- `src/bitmax/__init__.py`: expose `save_packed`, `load_packed`, and `PackedBundle`.
- `src/bitmax/experimental.py`: centroid helpers plus experimental int4 dataclasses/functions.
- `cpp/bitmax/cuda_extension.cu`: CUDA resident centroid top-k query-weighting method.
- `tests/test_io.py`: persistence round-trip tests.
- `tests/test_experimental_dim_centroids.py`: fused centroid top-k behavior tests.
- `tests/test_int4_reference.py`: int4 reference packing/scoring tests.
- `tests/test_cuda_extension.py`: CUDA centroid top-k correctness test.
- `benchmarks/run_retrieval.py`: add int4 variant and larger-slice-friendly reporting.
- `docs/api.md`, `docs/benchmarks.md`, `docs/gpu_optimization.md`, `README.md`: user-facing usage and measured results.

## Task 1: Persistence For Packed Docs

- [ ] **Step 1: Write failing persistence tests**

Create `tests/test_io.py` with tests that:

```python
import numpy as np

import bitmax
from bitmax.experimental import fit_dim_centroid_calibration, pack_dim_centroid_signs


def test_save_load_packed_docs_roundtrips_scores_and_metadata(tmp_path):
    docs = np.array([[1, -2, 3, -4, 5, -6, 7, -8], [-1, 2, -3, 4, -5, 6, -7, 8]], dtype=np.float32)
    query = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    packed = bitmax.pack_signs(docs, scale="doc")

    bitmax.save_packed(tmp_path / "docs.npz", packed, metadata={"name": "tiny"})
    bundle = bitmax.load_packed(tmp_path / "docs.npz")

    assert bundle.metadata == {"name": "tiny"}
    np.testing.assert_allclose(bitmax.maxsim(query, bundle.packed), bitmax.maxsim(query, packed))


def test_save_load_centroid_bundle_roundtrips_calibration(tmp_path):
    docs = np.array([[1, -2, 3, -4, 5, -6, 7, -8], [-1, 2, -3, 4, -5, 6, -7, 8]], dtype=np.float32)
    calibration = fit_dim_centroid_calibration(docs)
    packed, calibration = pack_dim_centroid_signs(docs, calibration=calibration)

    bitmax.save_packed(tmp_path / "centroid.npz", packed, calibration=calibration)
    bundle = bitmax.load_packed(tmp_path / "centroid.npz")

    assert bundle.centroid_calibration is not None
    np.testing.assert_allclose(bundle.centroid_calibration.positive_centroids, calibration.positive_centroids)
```

- [ ] **Step 2: Verify persistence tests fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_io.py
```

Expected: import or attribute failure for missing `save_packed`/`load_packed`.

- [ ] **Step 3: Implement persistence**

Add `src/bitmax/io.py` with `PackedBundle`, `save_packed`, and `load_packed`.
Use `np.savez_compressed`, JSON metadata, explicit `scale_kind`, and optional
centroid calibration arrays. Reject CUDA packed docs with `ValueError`.

- [ ] **Step 4: Expose persistence and verify**

Update `src/bitmax/__init__.py`. Run:

```bash
.venv/bin/python -m pytest -q tests/test_io.py
```

Expected: tests pass.

## Task 2: CUDA Fused Centroid Top-K

- [ ] **Step 1: Write failing API/CUDA tests**

Add to `tests/test_experimental_dim_centroids.py` a monkeypatch test proving
`topk_dim_centroid_maxsim` calls `packed.data.topk_centroid_batch` when present.
Add to `tests/test_cuda_extension.py` a CUDA test comparing
`topk_dim_centroid_maxsim(query, cuda_packed, calibration, k)` to the CPU
centroid top-k reference.

- [ ] **Step 2: Verify tests fail**

Run locally:

```bash
.venv/bin/python -m pytest -q tests/test_experimental_dim_centroids.py
```

Expected: monkeypatch test fails because the fused method is not used.

Run on VAST:

```bash
ssh -i ~/.ssh/id_ed25519 -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m pytest -q -m cuda tests/test_cuda_extension.py'
```

Expected: CUDA test fails until the pybind method exists.

- [ ] **Step 3: Implement Python routing**

Update `topk_dim_centroid_maxsim` to route CUDA `PackedDocs` with
`topk_centroid_batch` through that method. Continue restoring top-k scores in
Python with `_restore_dim_centroid_scores`.

- [ ] **Step 4: Implement CUDA method**

In `cpp/bitmax/cuda_extension.cu`, add an in-place query-weighting kernel and
`CudaPackedDocs::topk_centroid_batch(query, weights, k, scale, use_scale_vector)`.
It copies raw query and weights to device, weights query dimensions on device,
then calls the existing resident maxsim/top-k path.

- [ ] **Step 5: Verify local and VAST tests**

Run local focused tests and remote CUDA tests:

```bash
.venv/bin/python -m pytest -q tests/test_experimental_dim_centroids.py
ssh -i ~/.ssh/id_ed25519 -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && BITMAX_BUILD_CUDA=1 python -m pip install -e ".[dev]" && python -m pytest -q -m cuda tests/test_cuda_extension.py'
```

Expected: tests pass.

## Task 3: Larger VAST Retrieval Eval

- [ ] **Step 1: Build or reuse limit-256 embeddings**

On the persistent VAST worker, run `benchmarks.build_vidore_embeddings` with
`--limit 256` if the artifact does not exist.

- [ ] **Step 2: Run CPU gate and CUDA Pareto benchmark**

Run `benchmarks.run_retrieval` for `embeddings-smoke` and
`embeddings-cuda-smoke --variants all`, outputting limit-256 artifacts.

- [ ] **Step 3: Run centroid threshold/calibration sweep**

Run the same zero/mean/median/quantile sweep used for limit-64 on limit-256 and
record whether any same-storage calibration beats zero-threshold centroids.

- [ ] **Step 4: Fetch artifacts**

Fetch JSON artifacts to local `benchmark-results/`.

## Task 4: Experimental Int4 Backend

- [ ] **Step 1: Write failing int4 tests**

Create `tests/test_int4_reference.py` with tests for:

- symmetric per-tensor int4 packing into nibbles,
- score equivalence to dequantized dense int4 docs,
- top-k deterministic tie-breaking,
- reported storage bytes.

- [ ] **Step 2: Verify int4 tests fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_int4_reference.py
```

Expected: missing int4 API failure.

- [ ] **Step 3: Implement reference int4**

Add `Int4PackedDocs`, `pack_int4_symmetric`, `int4_maxsim`, and
`topk_int4_maxsim` under `bitmax.experimental`.

- [ ] **Step 4: Add retrieval benchmark variant**

Add `int4_symmetric_per_tensor` to `benchmarks/run_retrieval.py --variants all`
with storage bytes and retrieval metrics.

- [ ] **Step 5: Verify and run VAST int4 metric**

Run local int4 tests, then run the VAST retrieval benchmark on the largest
available slice.

## Task 5: Documentation, Verification, Commit

- [ ] **Step 1: Update docs**

Update README/API/benchmark docs with persistence, fused centroid CUDA top-k,
limit-256 results, and int4 status.

- [ ] **Step 2: Final verification**

Run:

```bash
.venv/bin/python -m pytest -q
ssh -i ~/.ssh/id_ed25519 -p 18164 root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m pytest -q'
```

Expected: local and remote pass.

- [ ] **Step 3: Commit**

Commit implementation with:

```bash
git add README.md docs src tests benchmarks cpp
git commit -m "feat: improve binary accuracy usability backends"
```

## Self-Review

- Spec coverage: persistence, fused centroid CUDA top-k, larger VAST evals,
  int4 backend, documentation, and verification are covered.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: names are stable: `save_packed`, `load_packed`,
  `PackedBundle`, `topk_centroid_batch`, `Int4PackedDocs`,
  `pack_int4_symmetric`, and `int4_maxsim`.
