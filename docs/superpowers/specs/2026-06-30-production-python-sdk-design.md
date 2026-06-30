# Production Python SDK Design

## Goal

Turn `bitmax` from a kernel/benchmark library into a polished Python SDK that
people can use in local search, multimodal RAG, legal/document retrieval, and
bring-your-own-retriever pipelines.

The SDK should make compressed multi-vector retrieval feel like a normal Python
workflow: build a corpus from embeddings, save it, load it on CPU or CUDA, run
full local search for small corpora, or rerank candidates from an existing
retriever.

## Product Positioning

`bitmax` is not a vector database and should not try to own ingestion,
filtering, sharding, auth, or serving in this SDK phase. It is the compressed
late-interaction scoring layer for modern multi-vector embeddings.

The user-facing pitch:

> Use ColBERT/ColPali/ColQwen-style multi-vector retrieval with 8-32x smaller
> document embeddings and fast local or GPU MaxSim search/reranking.

## CUDA-First Scope

This phase is strictly CUDA-first for proof, benchmarks, and release claims.
CPU paths can exist for cheap API correctness tests and developer convenience,
but they are not the product proof. MLX, Metal, WebGPU, ONNX Runtime, and other
local accelerator kernels are explicitly deferred.

All performance claims must come from CUDA runs against CUDA baselines:

- dense fp16 MaxSim on CUDA;
- torch-style dense top-k/reranking on CUDA when practical;
- `bitmax` binary, `binary_q40`, and int4 on CUDA;
- measured storage bytes, latency, speedup, recall@1, recall@10, MRR@10, and
  NDCG@10 on the same embedding slices.

## Target Users

- Engineers building local/private document search over PDFs, screenshots,
  scanned pages, slides, or technical docs.
- RAG builders who already use Qdrant, LanceDB, Elasticsearch, Vespa, pgvector,
  or a custom retriever for first-stage candidate generation.
- Legal, finance, medical, enterprise, and developer-tool teams that care about
  privacy, quality, storage, and latency.
- Researchers who want a clean API to compare dense fp16, binary, centroid
  binary, and int4 late-interaction tradeoffs.

## Supported Embedding Shape

The SDK works with multi-vector embeddings:

- `embeddings`: a flattened `[total_doc_tokens, dim]` float array.
- `offsets`: `[num_docs + 1]` int64 offsets into `embeddings`.
- `doc_ids`: one stable external ID per document.

Query embeddings are `[query_tokens, dim]` or `[batch, query_tokens, dim]`.

Single-vector embeddings can technically be represented as one token per doc,
but the SDK should document that its main value is for late-interaction
multi-vector models.

## Public SDK API

### Corpus

Add `bitmax.Corpus`, a high-level persisted corpus object.

```python
corpus = bitmax.Corpus.from_embeddings(
    doc_ids=["doc-a", "doc-b"],
    embeddings=doc_embeddings,
    offsets=doc_offsets,
    mode="binary_q40",
    metadata={"model": "vidore/colqwen2-v1.0-hf"},
)

corpus.save("docs.bitmax.npz")
loaded = bitmax.Corpus.load("docs.bitmax.npz")
```

Supported modes:

- `binary`: raw one-bit signs. Fastest, stable default, 32x fp32 document
  compression.
- `binary_q40`: per-dimension q40 centroid calibration. Experimental 32x-ish
  accuracy mode; stores centroid calibration metadata.
- `int4`: symmetric signed int4 docs. Experimental accuracy-first mode, 8x
  fp32 document compression.

`Corpus` properties:

- `doc_ids: tuple[str, ...]`
- `num_docs: int`
- `dim: int`
- `mode: str`
- `storage_bytes: int`
- `metadata: dict[str, Any]`

### Reranker

Add `bitmax.Reranker`, a runtime scoring object. The SDK should support
`device="cpu"` for correctness and small local experimentation, but README and
benchmark examples should use `device="cuda"` as the production path.

```python
reranker = bitmax.Reranker.from_corpus(corpus, device="cuda")
reranker = bitmax.Reranker.load("docs.bitmax.npz", device="cuda")
```

Methods:

```python
results = reranker.search(query_embeddings, k=10)

results = reranker.rerank(
    query_embeddings,
    candidate_ids=["doc-b", "doc-a"],
    k=2,
)
```

Return `SearchResult` objects:

```python
SearchResult(doc_id="doc-b", score=123.4, rank=1)
```

For batch queries, return a list per query:

```python
batch_results = reranker.search(batch_query_embeddings, k=10)
assert isinstance(batch_results[0][0], bitmax.SearchResult)
```

### Candidate Reranking Semantics

`rerank(query_embeddings, candidate_ids, k)` returns only the requested
candidate documents, sorted by score descending with deterministic lower input
order tie-breaking. Duplicate candidate IDs are allowed and are collapsed to the
first occurrence before ranking, because returning the same document twice is
not useful in a retrieval result.

Initial implementation can compute full-corpus scores internally and then
filter/sort the requested candidates. That is acceptable for SDK usability and
correctness because the public behavior is candidate-only output. A later kernel
can add candidate-only scoring for high-document-count workloads.

The method must reject unknown candidate IDs with a clear `KeyError`.

## Persistence Format

Reuse the existing `.npz` persistence path but extend it for high-level SDK
metadata:

- `doc_ids`: string array with one entry per document.
- `mode`: string.
- `metadata_json`: existing JSON metadata.
- Existing packed data arrays for binary and centroid binary.
- Int4 arrays for `int4` mode.
- Optional centroid calibration arrays for `binary_q40`.

Backward compatibility:

- Existing `save_packed` / `load_packed` stay supported for low-level users.
- `Corpus.load` should load only corpus files written by `Corpus.save`; it does
  not need to infer doc IDs from low-level packed files.

## Demo

Add `examples/local_multivector_search.py`.

The demo consumes an existing retrieval embedding `.npz` file, such as the
ViDoRe/ColQwen2 benchmark artifact, and prints a compact comparison table:

- dense fp16 CUDA baseline document storage and latency.
- `binary` storage, latency, speedup, recall@1, recall@10, MRR@10, NDCG@10.
- `binary_q40` storage, latency, speedup, recall@1, recall@10, MRR@10,
  NDCG@10.
- `int4` storage, latency, speedup, recall@1, recall@10, MRR@10, NDCG@10.

Command:

```bash
python examples/local_multivector_search.py \
  --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
  --device cuda \
  --output benchmark-results/sdk-demo-local-search.json
```

The demo should also support `--limit-queries` for fast local smoke runs and
`--modes binary,binary_q40,int4` for targeted comparisons.

## Testing

Local tests:

- `Corpus.from_embeddings` validates doc IDs, offsets, dim, duplicate IDs, and
  mode.
- `Corpus.save` / `Corpus.load` roundtrip IDs, metadata, mode, and scores.
- `Reranker.search` returns sorted `SearchResult` objects with stable ranks.
- `Reranker.search` ties by lower corpus order.
- `Reranker.rerank` handles candidate IDs, unknown IDs, duplicate candidate IDs,
  and `k` larger than candidate count.
- `binary_q40` and `int4` modes can be searched through the SDK.
- Demo smoke test runs on a tiny `.npz` fixture and emits JSON with size,
  latency, and ranking metrics.

Local tests are correctness-only. They do not establish performance claims.

CUDA tests:

- `Reranker.load(..., device="cuda")` works for binary and int4 on a CUDA build.
- CUDA SDK search/rerank scores match CPU SDK results on small fixtures.
- The SDK demo runs against the VAST limit-256 embedding artifact on CUDA and
  emits rows comparing dense fp16 CUDA, binary CUDA, q40 centroid CUDA, and int4
  CUDA.
- The demo artifact is the source of README benchmark claims for this SDK phase.

## Documentation

Update:

- `README.md`: replace low-level-only examples with the SDK path first, then
  link to lower-level kernels.
- `docs/api.md`: document `Corpus`, `Reranker`, `SearchResult`, modes, and
  persistence.
- `docs/benchmarks.md`: document the SDK demo command and output fields.

## Non-Goals

- No embedding model wrapper in this phase. Users bring embeddings.
- No web server, auth, billing, hosted API, or vector database features.
- No first-stage ANN index.
- No production candidate-only CUDA kernel yet; candidate filtering can be
  implemented in Python after full-corpus scoring for the first SDK release.
- No MLX, Metal, WebGPU, or local accelerator kernels in this phase.

## Release Bar

The SDK is ready to call an alpha when:

- The public SDK tests pass locally.
- CUDA SDK tests pass on the persistent VAST worker.
- The CUDA local search demo produces a JSON artifact and a readable table on
  the persistent VAST worker.
- README shows a complete end-to-end SDK example.
- Existing low-level APIs and benchmarks remain compatible.
