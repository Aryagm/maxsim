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
