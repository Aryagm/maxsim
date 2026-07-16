import json

import numpy as np
import pytest

import maxsim
from benchmarks.build_beir_colbert_embeddings import (
    CANONICAL_COLBERT_V2,
    JINA_COLBERT_V2,
    _load_colbert_model,
)
from benchmarks.run_quality_matrix import (
    FORMAT_NAMES,
    _materialize_format,
    _normalize_formats,
    run_quality_matrix,
)
from benchmarks.run_retrieval import _load_embedding_file


def _write_cache(path):
    docs = np.array(
        [
            [0.9, 0.4, 0.2, -0.1, -0.2, -0.4, -0.8, -1.0],
            [0.7, 0.2, -0.2, -0.4, 0.8, 0.1, -0.5, -0.9],
            [0.6, -0.1, -0.3, -0.5, 0.9, 0.3, -0.4, -0.7],
            [-0.8, -0.4, -0.2, 0.1, 0.2, 0.4, 0.8, 1.0],
            [-0.7, -0.2, 0.2, 0.4, -0.8, -0.1, 0.5, 0.9],
            [-0.6, 0.1, 0.3, 0.5, -0.9, -0.3, 0.4, 0.7],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [[1.0, 0.5, 0.2, -0.2, -0.3, -0.4, -0.8, -1.0]],
            [[-1.0, -0.5, -0.2, 0.2, 0.3, 0.4, 0.8, 1.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        builder_schema_version=np.array(2),
        dataset_name=np.array("tiny-text"),
        dataset_id=np.array("BeIR/tiny"),
        model_name=np.array("vidore/colpali-v1.3-hf"),
        model_requested=np.array(JINA_COLBERT_V2),
        model_resolved=np.array(JINA_COLBERT_V2),
        model_profile=np.array("jina-colbert-v2"),
        model_options_json=np.array("{}"),
        model_fallback_used=np.array(False),
        qrels_semantics=np.array("binary_positive"),
        source_dataset=np.array("vidore/docvqa_test_subsampled"),
        source_config=np.array("default"),
        source_split=np.array("test"),
        doc_embeddings=docs,
        doc_offsets=np.array([0, 3, 6], dtype=np.int64),
        query_embeddings=queries,
        qrels=np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=np.array(["q0", "q1"]),
        doc_ids=np.array(["d0", "d1"]),
    )


def test_jina_profile_uses_official_pylate_options_and_canonical_fallback():
    calls = []

    class Models:
        @staticmethod
        def ColBERT(**kwargs):
            calls.append(kwargs)
            if kwargs["model_name_or_path"] == JINA_COLBERT_V2:
                raise OSError("remote checkpoint unavailable")
            return "fallback-model"

    model, resolved, profile, options, failures = _load_colbert_model(
        Models,
        JINA_COLBERT_V2,
        device="cuda",
        fallback_model=CANONICAL_COLBERT_V2,
    )

    assert model == "fallback-model"
    assert resolved == CANONICAL_COLBERT_V2
    assert profile == "canonical-colbert-v2"
    assert options == {}
    assert len(failures) == 1
    assert calls[0] == {
        "model_name_or_path": JINA_COLBERT_V2,
        "device": "cuda",
        "query_prefix": "[QueryMarker]",
        "document_prefix": "[DocumentMarker]",
        "attend_to_expansion_tokens": True,
        "trust_remote_code": True,
    }
    assert calls[1] == {"model_name_or_path": CANONICAL_COLBERT_V2, "device": "cuda"}


def test_stored_format_materialization_matches_packed_codes(tmp_path):
    cache = tmp_path / "tiny.npz"
    _write_cache(cache)
    dataset = _load_embedding_file(cache)

    binary = _materialize_format(dataset, "binary")
    packed_binary = maxsim.pack_signs(dataset.doc_embeddings, dataset.doc_offsets)
    expected_bits = np.unpackbits(packed_binary.data, axis=1, bitorder="little")[:, : dataset.dim]
    np.testing.assert_array_equal(binary.docs, expected_bits.astype(np.float32) * 2.0 - 1.0)

    compact = _materialize_format(dataset, "binary_token_scale_u4")
    packed_compact = maxsim.pack_signs(
        dataset.doc_embeddings,
        dataset.doc_offsets,
        token_scale="mean_abs_u4",
    )
    np.testing.assert_allclose(
        compact.docs,
        binary.docs * packed_compact.token_scale[:, np.newaxis],
        rtol=0.0,
        atol=0.0,
    )
    assert compact.storage_bytes == packed_compact.data.nbytes + 3 + 16


def test_int8_per_token_decodes_stored_values_and_accounts_for_132_bytes(tmp_path):
    cache = tmp_path / "tiny.npz"
    _write_cache(cache)
    dataset = _load_embedding_file(cache)
    docs = np.pad(dataset.doc_embeddings, ((0, 0), (0, 120)))
    dataset = type(dataset)(
        name=dataset.name,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=docs,
        doc_offsets=dataset.doc_offsets,
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )

    stored = _materialize_format(dataset, "int8_per_token")
    scales = np.max(np.abs(docs), axis=1).astype(np.float32) / np.float32(127.0)
    scales = np.where(scales > 0.0, scales, np.float32(1.0)).astype(np.float32)
    values = np.clip(np.rint(docs / scales[:, np.newaxis]), -127, 127).astype(np.int8)

    np.testing.assert_array_equal(
        stored.docs,
        values.astype(np.float32) * scales[:, np.newaxis],
    )
    assert stored.storage_bytes == docs.shape[0] * 132
    assert stored.query_fp16 is False
    assert stored.metadata == {
        "format_semantics": "decoded_from_stored_codes",
        "common_doc_offsets_bytes": dataset.doc_offsets.nbytes,
        "codes": "symmetric_int8",
        "code_dtype": "int8",
        "quantization_range": [-127, 127],
        "scale_granularity": "token",
        "scale_storage": "fp32",
        "bytes_per_token": 132,
        "scoring_path": "decoded_generic",
        "native_cuda_kernel": False,
    }


def test_int8_per_token_is_a_normalized_default_format():
    assert FORMAT_NAMES.count("int8_per_token") == 1
    assert _normalize_formats(None) == FORMAT_NAMES
    assert _normalize_formats("int8_per_token") == ("int8_per_token",)


@pytest.mark.benchmark_smoke
def test_quality_matrix_emits_per_query_metrics_and_resumes(tmp_path, monkeypatch):
    cache = tmp_path / "tiny.npz"
    output = tmp_path / "matrix.json"
    _write_cache(cache)

    result = run_quality_matrix(
        cache,
        output,
        device="cpu",
        checkpoint_every=1,
        max_chunk_tokens=3,
    )

    assert result["status"] == "complete"
    assert result["cache"]["model_profile"] == "jina-colbert-v2"
    assert result["cache"]["model_name"] == "vidore/colpali-v1.3-hf"
    assert result["cache"]["source_dataset"] == "vidore/docvqa_test_subsampled"
    assert result["cache"]["source_config"] == "default"
    assert result["cache"]["source_split"] == "test"
    assert {row["format"] for row in result["results"]} == set(FORMAT_NAMES)
    for row in result["results"]:
        assert row["status"] == "complete"
        assert row["completed_queries"] == 2
        assert len(row["per_query_ndcg_at_10"]) == 2
        assert len(row["per_query_recall_at_10"]) == 2
        assert row["doc_storage_bytes"] > 0

    assert not list(tmp_path.glob(".matrix.json.*.tmp"))
    from_disk = json.loads(output.read_text())
    assert from_disk == result

    monkeypatch.setattr(
        "benchmarks.run_quality_matrix._materialize_format",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("completed rows must be skipped")),
    )
    resumed = run_quality_matrix(
        cache,
        output,
        device="cpu",
        checkpoint_every=1,
        max_chunk_tokens=3,
    )
    assert resumed == result


@pytest.mark.benchmark_smoke
def test_quality_matrix_resumes_partial_query_vector(tmp_path):
    cache = tmp_path / "tiny.npz"
    output = tmp_path / "matrix.json"
    _write_cache(cache)
    result = run_quality_matrix(cache, output, formats="binary", device="cpu", checkpoint_every=1)

    row = result["results"][0]
    row["status"] = "in_progress"
    row["completed_queries"] = 1
    row["per_query_ndcg_at_10"] = row["per_query_ndcg_at_10"][:1]
    row["per_query_recall_at_10"] = row["per_query_recall_at_10"][:1]
    result["status"] = "in_progress"
    output.write_text(json.dumps(result))

    resumed = run_quality_matrix(cache, output, formats="binary", device="cpu", checkpoint_every=1)
    resumed_row = resumed["results"][0]
    assert resumed_row["status"] == "complete"
    assert resumed_row["completed_queries"] == 2
    assert len(resumed_row["per_query_ndcg_at_10"]) == 2
