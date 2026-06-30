# bitmax API

`bitmax` exposes three public functions:

```python
packed = bitmax.pack_signs(doc_embeddings, doc_offsets=None, scale=None)
scores = bitmax.maxsim(query_tokens, packed)
scores, indices = bitmax.topk_maxsim(query_tokens, packed, k=10)
```

If `doc_offsets` is omitted, every input row is treated as a single-token
document. For ragged multi-vector documents, pass offsets shaped
`[num_docs + 1]`, starting at `0` and ending at `num_doc_tokens`.

`scale="global"` stores `mean(abs(doc_embeddings))` in `PackedDocs.scale`.
`scale="doc"` stores one `mean(abs(doc_tokens))` scale per document. `maxsim`
applies stored scales by default. Per-document scales are applied after native
scoring in v0.1; CUDA fused top-k falls back to full-score host top-k for that
mode until a resident vector-scale kernel is added.
