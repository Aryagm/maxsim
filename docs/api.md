# bitmax API

The production SDK path is `Corpus` plus `Reranker`:

```python
corpus = bitmax.Corpus.from_embeddings(
    doc_ids=doc_ids,
    embeddings=doc_embeddings,
    offsets=doc_offsets,
    mode="binary_q40",
)
corpus.save("docs.bitmax.npz")

reranker = bitmax.Reranker.load("docs.bitmax.npz", device="cuda")
results = reranker.search(query_embeddings, k=10)
reranked = reranker.rerank(query_embeddings, candidate_ids=["doc-a", "doc-b"], k=2)
```

Each result is a `SearchResult(doc_id: str, score: float, rank: int)`.

## SDK Corpus

`Corpus.from_embeddings(doc_ids, embeddings, offsets, mode="binary")` builds a
portable packed corpus from multi-vector document embeddings.

Inputs:

- `doc_ids`: one stable external ID per document.
- `embeddings`: flattened `[total_doc_tokens, dim]` float array.
- `offsets`: `[num_docs + 1]` offsets into `embeddings`.

Modes:

- `binary`: fastest, 32x fp32 document compression on large corpora.
- `binary_q40`: experimental 32x-ish q40 centroid calibration.
- `int4`: experimental accuracy-first mode, 8x fp32 document compression.

## SDK Reranker

`Reranker.search(query_embeddings, k=10)` scores the whole corpus and returns the
top documents. `Reranker.rerank(query_embeddings, candidate_ids, k=10)` returns
only the requested candidates sorted by compressed MaxSim score. Candidate IDs
can come from Qdrant, LanceDB, Elasticsearch, Vespa, pgvector, or a custom
retriever.

Use `device="cuda"` for production benchmark claims. CPU is available for
correctness checks and small local experiments.

Both `search` and `rerank` accept `reducer`, `query_weights`, and `temperature`
with the same meanings as the low-level reducer API below. `rerank` passes the
resolved candidate positions to the candidate-only scorer instead of scoring
the full corpus. The calibrated `binary_q40` mode currently remains MaxSim-only.

## Low-Level API

The lower-level kernel API exposes direct packed scoring:

```python
packed = bitmax.pack_signs(doc_embeddings, doc_offsets=None, scale=None)
scores = bitmax.maxsim(query_tokens, packed)
scores, indices = bitmax.topk_maxsim(query_tokens, packed, k=10)
bitmax.save_packed("docs.bitmax.npz", packed)
bundle = bitmax.load_packed("docs.bitmax.npz")
```

If `doc_offsets` is omitted, every input row is treated as a single-token
document. For ragged multi-vector documents, pass offsets shaped
`[num_docs + 1]`, starting at `0` and ending at `num_doc_tokens`.

`scale="global"` stores `mean(abs(doc_embeddings))` in `PackedDocs.scale`.
`scale="doc"` stores one `mean(abs(doc_tokens))` scale per document. `maxsim`
applies stored scales by default. CUDA-resident `PackedDocs` upload stored
per-document scales with the packed document bits, so resident `maxsim` and
fused `topk_maxsim` can apply those scales on device. Host-packed CUDA and CPU
paths still apply vector scales after native scoring.

## CUDA reducer selection

`maxsim.score` selects the token-to-document reduction without changing the
packed binary or int4 representation. The default remains ordinary MaxSim, so
existing `maxsim.maxsim(query_tokens, packed)` calls keep their behavior and
their tuned legacy CUDA path.

```python
scores = maxsim.score(query_tokens, cuda_packed, reducer="maxsim")
weighted = maxsim.score(
    query_tokens,
    cuda_packed,
    reducer="weighted_maxsim",
    query_weights=query_weights,
)
top4 = maxsim.score(query_tokens, cuda_packed, reducer="topk4")
smooth = maxsim.score(
    query_tokens,
    cuda_packed,
    reducer="smoothsim",
    temperature=0.25,
)

# Score only these documents, returning columns in this exact order.
candidate_scores = maxsim.score(
    query_tokens,
    cuda_packed,
    reducer="topk2",
    candidate_indices=np.array([19, 3, 11], dtype=np.int64),
)
```

For query token `i`, document token `j`, similarity `s[i,j]`, and a document
with `m` tokens, the supported reducers are:

- `maxsim`: `sum_i max_j s[i,j]`.
- `weighted_maxsim`: `sum_i w[i] max_j s[i,j]`. `query_weights` has shape
  `[query_tokens]` or `[batch, query_tokens]`.
- `topk2` and `topk4`: `sum_i mean(top_k_j s[i,j])`, with `k=2` or `k=4`.
  A document shorter than `k` averages its available `min(k, m)` tokens.
- `smoothsim`: `sum_i temperature * logsumexp_j(s[i,j] / temperature)`.
  `temperature` must be finite and greater than zero. The CUDA policy uses a
  max-shifted log-sum-exp so large logits remain finite.

Empty documents score zero for every reducer. `candidate_indices` is an
optional one-dimensional integer array of document positions. Candidate-only
scoring reads only those documents and returns `[batch, candidates]` (or
`[candidates]` for one unbatched query) in caller-provided order; it does not
sort or deduplicate the positions.

The same reducer arguments work with CUDA-resident `Int4PackedDocs` through
`maxsim.score` or `maxsim.experimental.int4_maxsim`. Binary token scales and
document scales, and the intrinsic int4 quantization scale, are applied in the
similarity units expected by each reducer. In particular, SmoothSim applies
reconstruction scale before log-sum-exp because that reduction is nonlinear.

On a CUDA host, run `python benchmarks/run_cuda_reducers.py` to record PyTorch
reference deltas plus best and median end-to-end latency for both formats, all
reducers, and full-corpus versus candidate-only scoring. The default JSON output
is `benchmark-results/cuda-reducers.json`.

## Experimental Centroid Binary

`bitmax.experimental` includes a one-bit per-dimension centroid calibration path.
It fits positive/negative centroids per dimension, packs thresholded document
signs, and transforms query vectors before calling the same binary MaxSim
kernel.

```python
from bitmax.experimental import (
    dim_centroid_maxsim,
    fit_dim_centroid_calibration,
    pack_dim_centroid_signs,
    topk_dim_centroid_maxsim,
)

calibration = fit_dim_centroid_calibration(doc_embeddings)
packed, calibration = pack_dim_centroid_signs(
    doc_embeddings,
    doc_offsets,
    calibration=calibration,
)
scores = dim_centroid_maxsim(query_tokens, packed, calibration)
top_scores, top_indices = topk_dim_centroid_maxsim(
    query_tokens,
    packed,
    calibration,
    k=10,
)
```

This API is experimental. It is useful when one-bit storage is required but raw
sign scoring loses too much dimension-magnitude signal.

For measured q40 threshold calibration, pass a per-dimension threshold vector
before fitting:

```python
thresholds = np.percentile(doc_embeddings, 40.0, axis=0).astype("float32")
calibration = fit_dim_centroid_calibration(doc_embeddings, thresholds=thresholds)
packed, calibration = pack_dim_centroid_signs(
    doc_embeddings,
    doc_offsets,
    calibration=calibration,
)
```

## Experimental Int4

`bitmax.experimental` also includes symmetric signed-int4 document packing. It
stores two 4-bit signed values per byte plus one tensor scale. This is the
current high-accuracy compression option when 8x fp32 document reduction is
acceptable.

```python
from bitmax.experimental import (
    int4_maxsim,
    int4_to_device,
    pack_int4_symmetric,
    topk_int4_maxsim,
)

packed = pack_int4_symmetric(doc_embeddings, doc_offsets)
scores = int4_maxsim(query_tokens, packed)

cuda_packed = int4_to_device(packed)
top_scores, top_indices = topk_int4_maxsim(query_tokens, cuda_packed, k=10)
```

The int4 API is experimental and may change. It exists to measure the
accuracy/storage/speed frontier before promoting a stable public kernel.
