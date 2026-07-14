from dataclasses import replace

import numpy as np
import pytest

import maxsim


def _tiny_multivector_docs():
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    return docs, offsets


def test_corpus_from_embeddings_exposes_metadata_and_storage():
    docs, offsets = _tiny_multivector_docs()

    corpus = maxsim.Corpus.from_embeddings(
        doc_ids=["positive", "negative", "mixed"],
        embeddings=docs,
        offsets=offsets,
        mode="binary",
        metadata={"model": "tiny"},
    )

    assert corpus.doc_ids == ("positive", "negative", "mixed")
    assert corpus.num_docs == 3
    assert corpus.dim == 8
    assert corpus.mode == "binary"
    assert corpus.metadata == {"model": "tiny"}
    assert corpus.storage_bytes == 3


def test_reranker_search_returns_ranked_results_for_single_query():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.search(query, k=2)

    assert [result.doc_id for result in results] == ["positive", "mixed"]
    assert [result.rank for result in results] == [1, 2]
    assert all(isinstance(result.score, float) for result in results)


def test_reranker_search_returns_batch_results_for_batched_queries():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    query = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )

    results = reranker.search(query, k=1)

    assert [[result.doc_id for result in row] for row in results] == [["positive"], ["negative"]]
    assert [[result.rank for result in row] for row in results] == [[1], [1]]


def test_reranker_rerank_returns_only_candidates_and_collapses_duplicates():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.rerank(query, candidate_ids=["negative", "mixed", "mixed"], k=3)

    assert [result.doc_id for result in results] == ["mixed", "negative"]
    assert [result.rank for result in results] == [1, 2]


def test_reranker_rejects_unknown_candidate_id():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    with pytest.raises(KeyError, match="missing"):
        reranker.rerank(np.ones((1, 8), dtype=np.float32), candidate_ids=["missing"], k=1)


def test_corpus_rejects_invalid_doc_ids_offsets_and_mode():
    docs, offsets = _tiny_multivector_docs()

    with pytest.raises(ValueError, match="unique"):
        maxsim.Corpus.from_embeddings(["dup", "dup", "mixed"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="one more entry"):
        maxsim.Corpus.from_embeddings(["one", "two"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="mode"):
        maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="bad")


def test_binary_q40_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary_q40")
    reranker = maxsim.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "binary_q40"
    assert corpus.storage_bytes > 3
    assert len(results) == 2
    assert results[0].doc_id == "positive"


def test_int4_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    reranker = maxsim.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "int4"
    assert corpus.storage_bytes == 3 * 8 // 2 + 4
    assert results[0].doc_id == "positive"


def test_corpus_save_load_roundtrips_binary_q40_scores_and_metadata(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="binary_q40",
        metadata={"source": "tiny"},
    )
    query = np.ones((1, 8), dtype=np.float32)
    expected = maxsim.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny.maxsim.npz")
    loaded = maxsim.Corpus.load(tmp_path / "tiny.maxsim.npz")
    actual = maxsim.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.doc_ids == corpus.doc_ids
    assert loaded.mode == "binary_q40"
    assert loaded.metadata == {"source": "tiny"}
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_reranker_load_constructs_runtime_from_saved_corpus(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    corpus.save(tmp_path / "tiny.maxsim.npz")

    reranker = maxsim.Reranker.load(tmp_path / "tiny.maxsim.npz")
    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=1)

    assert results[0].doc_id == "positive"


def test_reranker_load_preserves_requested_int4_query_policy(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    path = tmp_path / "tiny-int4-query.maxsim.npz"
    maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode="int4_per_token"
    ).save(path)

    reranker = maxsim.Reranker.load(path, int4_query="int8")

    assert reranker.int4_query == "int8"


def test_corpus_save_load_roundtrips_int4_scores(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    query = np.ones((1, 8), dtype=np.float32)
    expected = maxsim.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny-int4.maxsim.npz")
    loaded = maxsim.Corpus.load(tmp_path / "tiny-int4.maxsim.npz")
    actual = maxsim.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.mode == "int4"
    assert loaded.storage_bytes == corpus.storage_bytes
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_corpus_save_load_roundtrips_per_token_int4_scores_and_scales(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode="int4_per_token"
    )
    query = np.ones((1, 8), dtype=np.float32)
    expected = maxsim.Reranker.from_corpus(corpus).search(query, k=3)

    path = tmp_path / "tiny-int4-per-token.maxsim.npz"
    corpus.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert int(np.asarray(data["schema_version"]).item()) == 2
        assert data["int4_token_scale"].dtype == np.float32
    loaded = maxsim.Corpus.load(path)
    actual = maxsim.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.mode == "int4_per_token"
    np.testing.assert_array_equal(loaded.int4_packed.token_scale, corpus.int4_packed.token_scale)
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_residual_int4_mode_searches_reranks_and_runs_one_call_cascade():
    from maxsim.cascade import cascade_topk, residual_score

    rng = np.random.default_rng(197)
    lengths = np.array([2, 4, 1, 3, 5], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    docs = rng.normal(size=(int(offsets[-1]), 8)).astype(np.float32)
    query = rng.normal(size=(3, 8)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(len(lengths))]
    corpus = maxsim.Index.from_embeddings(doc_ids, docs, offsets, mode="int4_residual")
    reranker = maxsim.Reranker.from_corpus(corpus)

    cascade_scores, cascade_indices = cascade_topk(
        query, corpus.int4_packed, 2, candidates=4
    )
    results = reranker.search(query, k=2, rescore_candidates=4)
    reranked = reranker.rerank(query, ["doc-4", "doc-1", "doc-3"], k=3)
    candidate_indices = np.array([4, 1, 3], dtype=np.int64)
    expected_candidate_scores = residual_score(
        query, corpus.int4_packed, candidate_indices=candidate_indices
    )
    expected_candidate_order = np.lexsort(
        (np.arange(3, dtype=np.int64), -expected_candidate_scores)
    )

    assert corpus.mode == "int4_residual"
    assert [result.doc_id for result in results] == [doc_ids[idx] for idx in cascade_indices]
    np.testing.assert_allclose(
        [result.score for result in results], cascade_scores, rtol=0, atol=1e-5
    )
    assert [result.doc_id for result in reranked] == [
        ["doc-4", "doc-1", "doc-3"][idx] for idx in expected_candidate_order
    ]


def test_residual_search_without_candidate_budget_uses_full_fused_scores():
    from maxsim.cascade import prefix_score, residual_score

    rng = np.random.default_rng(4)
    offsets = np.arange(0, 16, 3, dtype=np.int64)
    docs = rng.normal(size=(15, 8)).astype(np.float32)
    query = rng.normal(size=(2, 8)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(5)]
    corpus = maxsim.Index.from_embeddings(
        doc_ids,
        docs,
        offsets,
        mode="int4_residual",
    )

    results = maxsim.Reranker.from_corpus(corpus).search(query, k=3)
    fused = residual_score(query, corpus.int4_packed)
    prefix = prefix_score(query, corpus.int4_packed)
    expected = np.lexsort((np.arange(5, dtype=np.int64), -fused))[:3]
    prefix_order = np.lexsort((np.arange(5, dtype=np.int64), -prefix))[:3]

    assert not np.array_equal(expected, prefix_order)
    assert [result.doc_id for result in results] == [doc_ids[idx] for idx in expected]
    np.testing.assert_allclose(
        [result.score for result in results],
        fused[expected],
        rtol=0,
        atol=1e-5,
    )


def test_corpus_save_load_roundtrips_residual_int4_cascade(tmp_path):
    rng = np.random.default_rng(199)
    docs = rng.normal(size=(12, 8)).astype(np.float32)
    offsets = np.array([0, 3, 5, 9, 12], dtype=np.int64)
    doc_ids = ["a", "b", "c", "d"]
    query = rng.normal(size=(2, 8)).astype(np.float32)
    corpus = maxsim.Index.from_embeddings(doc_ids, docs, offsets, mode="int4_residual")
    expected = maxsim.Reranker.from_corpus(corpus).search(
        query, k=2, rescore_candidates=3
    )

    path = tmp_path / "residual.maxsim.npz"
    corpus.save(path)
    loaded = maxsim.Index.load(path)
    actual = maxsim.Reranker.from_corpus(loaded).search(
        query, k=2, rescore_candidates=3
    )

    assert loaded.mode == "int4_residual"
    assert loaded.storage_bytes == corpus.storage_bytes
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose(
        [result.score for result in actual],
        [result.score for result in expected],
        rtol=0,
        atol=1e-5,
    )


def test_rescore_candidates_requires_residual_mode():
    docs, offsets = _tiny_multivector_docs()
    reranker = maxsim.Reranker.from_corpus(
        maxsim.Index.from_embeddings(
            ["positive", "negative", "mixed"], docs, offsets, mode="int4_per_token"
        )
    )

    with pytest.raises(ValueError, match="int4_residual"):
        reranker.search(np.ones((1, 8), dtype=np.float32), k=2, rescore_candidates=3)


def test_auto_and_max_quality_use_per_token_int4():

    rng = np.random.default_rng(211)
    docs = rng.standard_normal((12, 16)).astype(np.float32)
    offsets = np.arange(0, 13, 2, dtype=np.int64)
    doc_ids = [f"d{i}" for i in range(6)]

    automatic = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets)
    quality = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="max_quality")
    tensor_control = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="int4")

    assert automatic.mode == "int4_per_token"
    assert quality.mode == "int4_per_token"
    assert automatic.int4_packed.token_scale is not None
    assert quality.int4_packed.token_scale is not None
    assert tensor_control.mode == "int4"
    assert tensor_control.int4_packed.token_scale is None


def test_index_is_the_product_name_for_corpus():
    import maxsim

    assert maxsim.Index is maxsim.Corpus


def test_auto_mode_is_stable_for_unit_normalized_embeddings():

    rng = np.random.default_rng(227)
    docs = rng.standard_normal((24, 16)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    offsets = np.arange(0, 25, 2, dtype=np.int64)
    corpus = maxsim.Index.from_embeddings([f"d{i}" for i in range(12)], docs, offsets, mode="auto")
    assert corpus.mode == "int4_per_token"


def test_max_compression_defaults_to_pool3_and_honors_overrides():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(229)
    docs = rng.standard_normal((18, 16)).astype(np.float32)
    offsets = np.array([0, 6, 12, 18], dtype=np.int64)
    doc_ids = ["a", "b", "c"]

    preset = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="max_compression")
    overridden = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="max_compression", pool_factor=2)
    direct = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="pooled_binary")

    assert preset.metadata["pool_factor"] == 3
    assert overridden.metadata["pool_factor"] == 2
    assert direct.metadata["pool_factor"] == 2


def test_pool_factor_rejects_non_integer_values():
    pytest.importorskip("scipy")
    docs, offsets = _tiny_multivector_docs()

    with pytest.raises(ValueError, match="integer"):
        maxsim.Corpus.from_embeddings(
            ["positive", "negative", "mixed"], docs, offsets, mode="pooled_binary", pool_factor=2.5
        )


def test_memory_report_separates_encoded_host_and_serialized_bytes(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode="int4_per_token"
    )

    report = corpus.memory_report()
    expected_host = sum(
        array.nbytes
        for array in (
            corpus.int4_packed.data,
            corpus.int4_packed.values,
            corpus.int4_packed.doc_offsets,
            corpus.int4_packed.token_scale,
        )
    )
    assert report.encoded_bytes == corpus.storage_bytes
    assert report.host_array_bytes == expected_host
    assert report.device_index_bytes == 0
    assert report.device_workspace_bytes == 0
    assert report.serialized_bytes is None

    path = tmp_path / "memory.maxsim.npz"
    corpus.save(path)
    assert corpus.memory_report().serialized_bytes == path.stat().st_size
    assert corpus.memory_report(path).serialized_bytes == path.stat().st_size
    assert maxsim.Corpus.load(path).memory_report().serialized_bytes == path.stat().st_size


def test_corpus_save_preserves_a_path_without_npz_suffix(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="int4_per_token",
    )
    path = tmp_path / "corpus.index"

    corpus.save(path)

    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".npz").exists()
    assert corpus.memory_report().serialized_bytes == path.stat().st_size
    assert maxsim.Corpus.load(path).mode == "int4_per_token"


def test_memory_report_uses_native_cuda_allocation_accessors():
    class FakeCudaHandle:
        dim = 8
        num_docs = 3
        packed_size = 3
        resident_bytes = 91
        workspace_bytes = 37

    packed = maxsim.PackedDocs(
        data=FakeCudaHandle(),
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        dim=8,
        num_docs=3,
        device="cuda",
    )
    corpus = maxsim.Corpus(doc_ids=("a", "b", "c"), mode="binary", packed=packed)

    report = corpus.memory_report()

    assert report.host_array_bytes == packed.doc_offsets.nbytes
    assert report.device_index_bytes == 91
    assert report.device_workspace_bytes == 37


def test_encoded_bytes_include_binary_reconstruction_scales():
    docs, offsets = _tiny_multivector_docs()
    packed = maxsim.pack_signs(docs, offsets, scale="doc")
    corpus = maxsim.Corpus(
        doc_ids=("positive", "negative", "mixed"),
        mode="binary",
        packed=packed,
    )

    assert corpus.encoded_bytes == packed.data.nbytes + packed.scale.nbytes


def test_load_accepts_legacy_schema_v1_global_int4(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    packed = maxsim.experimental.pack_int4_symmetric(docs, offsets)
    path = tmp_path / "legacy-int4-v1.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        format=np.array("maxsim_corpus"),
        mode=np.array("int4"),
        doc_ids=np.array(["positive", "negative", "mixed"]),
        metadata_json=np.array("{}"),
        int4_data=packed.data,
        doc_offsets=offsets,
        dim=np.array(8, dtype=np.int64),
        int4_scale=np.array(packed.scale, dtype=np.float32),
    )

    loaded = maxsim.Corpus.load(path)

    assert loaded.mode == "int4"
    np.testing.assert_array_equal(loaded.int4_packed.values, packed.values)


def test_load_accepts_legacy_schema_v1_binary_token_scale(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    packed = maxsim.pack_signs(docs, offsets, token_scale="mean_abs_fp16")
    path = tmp_path / "legacy-binary-v1.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        format=np.array("maxsim_corpus"),
        mode=np.array("binary_token_scale"),
        doc_ids=np.array(["positive", "negative", "mixed"]),
        metadata_json=np.array("{}"),
        packed_data=packed.data,
        doc_offsets=offsets,
        dim=np.array(8, dtype=np.int64),
        scale_kind=np.array("none"),
        scale_values=np.empty((0,), dtype=np.float32),
        token_scale_fp16=packed.token_scale.astype(np.float16),
    )

    loaded = maxsim.Corpus.load(path)

    assert loaded.mode == "binary_token_scale"
    np.testing.assert_allclose(
        loaded.packed.token_scale,
        packed.token_scale.astype(np.float16).astype(np.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("bad_offsets", "message"),
    [
        (np.array([1, 1, 2, 3], dtype=np.int64), "start at 0"),
        (np.array([0, 2, 1, 3], dtype=np.int64), "monotonically"),
        (np.array([0, 1, 2, 4], dtype=np.int64), "end at num_doc_tokens"),
    ],
)
def test_corpus_load_rejects_corrupt_offset_values(tmp_path, bad_offsets, message):
    docs, offsets = _tiny_multivector_docs()
    path = tmp_path / "valid.maxsim.npz"
    maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode="binary"
    ).save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["doc_offsets"] = bad_offsets
    corrupt = tmp_path / "corrupt.maxsim.npz"
    np.savez_compressed(corrupt, **payload)

    with pytest.raises(ValueError, match=message):
        maxsim.Corpus.load(corrupt)


@pytest.mark.parametrize("field", ["packed_data", "int4_data"])
def test_corpus_load_rejects_non_uint8_packed_payload(tmp_path, field):
    docs, offsets = _tiny_multivector_docs()
    mode = "binary" if field == "packed_data" else "int4_per_token"
    path = tmp_path / f"valid-{field}.maxsim.npz"
    maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode=mode
    ).save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload[field] = payload[field].astype(np.float32)
    corrupt = tmp_path / f"corrupt-{field}.maxsim.npz"
    np.savez_compressed(corrupt, **payload)

    with pytest.raises(ValueError, match="dtype uint8"):
        maxsim.Corpus.load(corrupt)


def test_corpus_load_rejects_non_float32_int4_token_scales(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    path = tmp_path / "valid-int4-scales.maxsim.npz"
    maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="int4_per_token",
    ).save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["int4_token_scale"] = payload["int4_token_scale"].astype(np.float64)
    corrupt = tmp_path / "corrupt-int4-scales.maxsim.npz"
    np.savez_compressed(corrupt, **payload)

    with pytest.raises(ValueError, match="dtype float32"):
        maxsim.Corpus.load(corrupt)


def test_corpus_load_rejects_negative_eight_in_symmetric_int4(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    path = tmp_path / "valid-int4.maxsim.npz"
    maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"], docs, offsets, mode="int4"
    ).save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["int4_data"] = payload["int4_data"].copy()
    payload["int4_data"][0, 0] = (payload["int4_data"][0, 0] & 0xF0) | 0x08
    corrupt = tmp_path / "corrupt-negative-eight.maxsim.npz"
    np.savez_compressed(corrupt, **payload)

    with pytest.raises(ValueError, match="symmetric int4"):
        maxsim.Corpus.load(corrupt)


def test_corpus_save_rejects_mismatched_int4_data_and_values(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="int4_per_token",
    )
    corrupt_data = corpus.int4_packed.data.copy()
    corrupt_data[0, 0] ^= np.uint8(1)
    corrupt = replace(
        corpus,
        int4_packed=replace(corpus.int4_packed, data=corrupt_data),
    )

    with pytest.raises(ValueError, match="does not encode"):
        corrupt.save(tmp_path / "corrupt.maxsim.npz")


def test_corpus_save_rejects_manually_constructed_invalid_offsets(tmp_path):
    docs, _ = _tiny_multivector_docs()
    packed = maxsim.pack_signs(docs)
    invalid = maxsim.PackedDocs(
        data=packed.data,
        doc_offsets=np.array([0, 2, 1, 3], dtype=np.int64),
        dim=packed.dim,
        num_docs=packed.num_docs,
    )
    corpus = maxsim.Corpus(doc_ids=("a", "b", "c"), mode="binary", packed=invalid)

    with pytest.raises(ValueError, match="monotonically"):
        corpus.save(tmp_path / "invalid.maxsim.npz")


def test_corpus_rejects_non_integer_offsets():
    docs, _ = _tiny_multivector_docs()

    with pytest.raises(ValueError, match="integers"):
        maxsim.Corpus.from_embeddings(
            ["positive", "negative", "mixed"], docs, np.array([0.0, 1.0, 2.0, 3.0]), mode="binary"
        )
