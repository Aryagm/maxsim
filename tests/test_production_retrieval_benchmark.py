import json

import numpy as np
import pytest

from benchmarks.run_production_retrieval import (
    _build_parser,
    _document_chunks,
    deterministic_query_indices,
    load_retrieval_cache,
    run_production_benchmark,
)


def _write_ragged_cache(path):
    rng = np.random.default_rng(91)
    docs = rng.normal(size=(10, 8)).astype(np.float32)
    doc_offsets = np.array([0, 2, 5, 6, 10], dtype=np.int64)
    queries = np.concatenate(
        [
            docs[0:1],
            docs[2:4],
            docs[5:6],
            docs[6:9],
            docs[1:2],
        ],
        axis=0,
    )
    query_offsets = np.array([0, 1, 3, 4, 7, 8], dtype=np.int64)
    qrels = np.zeros((5, 4), dtype=np.float32)
    qrels[np.arange(4), np.arange(4)] = 1.0
    qrels[4, 0] = 1.0
    np.savez_compressed(
        path,
        dataset_name=np.array("tiny-ragged"),
        doc_embeddings=docs,
        doc_offsets=doc_offsets,
        query_embeddings=queries,
        query_offsets=query_offsets,
        qrels=qrels,
        doc_ids=np.array(["d0", "d1", "d2", "d3"]),
        query_ids=np.array(["q0", "q1", "q2", "q3", "q4"]),
    )


def test_load_cache_preserves_ragged_queries_and_deterministic_subset(tmp_path):
    source = tmp_path / "ragged.npz"
    _write_ragged_cache(source)

    cache = load_retrieval_cache(source)
    first = deterministic_query_indices(cache.num_queries, 3, 17)
    second = deterministic_query_indices(cache.num_queries, 3, 17)

    assert cache.name == "tiny-ragged"
    assert [cache.query(index).shape[0] for index in range(cache.num_queries)] == [1, 2, 1, 3, 1]
    np.testing.assert_array_equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    np.testing.assert_array_equal(deterministic_query_indices(5, None, 1), np.arange(5))


@pytest.mark.benchmark_smoke
def test_production_runner_emits_all_sdk_modes_metrics_and_checkpoints(tmp_path):
    source = tmp_path / "ragged.npz"
    output = tmp_path / "production.json"
    _write_ragged_cache(source)

    result = run_production_benchmark(
        source,
        output,
        device="cpu",
        top_k=2,
        query_limit=3,
        seed=7,
        warmup=0,
        latency_repeats=2,
        throughput_batch_size=3,
        cascade_candidates=(3,),
    )

    expected = {
        "dense",
        "binary",
        "binary_token_scale_u4",
        "pooled_binary",
        "int4",
        "int4_per_token",
        "int4_residual",
        "int4_residual_cascade_m3",
    }
    assert result["status"] == "complete"
    assert {row["case_id"] for row in result["cases"]} == expected
    assert output.is_file()
    assert json.loads(output.read_text())["run_signature"] == result["run_signature"]
    assert len(result["selected_queries"]["indices"]) == 3

    checkpoint_dir = tmp_path / "production.checkpoints"
    assert {path.stem for path in checkpoint_dir.glob("*.json")} == expected
    for row in result["cases"]:
        assert len(row["quality"]["ndcg_at_10_per_query"]) == 3
        assert len(row["quality"]["recall_at_10_per_query"]) == 3
        assert len(row["latency"]["samples_ms"]) == 6
        assert row["latency"]["p50_ms"] >= 0.0
        assert row["latency"]["p95_ms"] >= row["latency"]["p50_ms"]
        assert row["throughput"]["batch_size"] == 3
        assert row["throughput"]["queries_per_second"] > 0.0
        assert row["storage"]["encoded_bytes"] > 0
        assert len(row["rankings"]["doc_indices"]) == 3

    for mode in (
        "binary",
        "binary_token_scale_u4",
        "pooled_binary",
        "int4",
        "int4_per_token",
        "int4_residual",
    ):
        row = next(value for value in result["cases"] if value["case_id"] == mode)
        assert row["storage"]["packing_ms"] >= 0.0
        assert row["storage"]["serialization_ms"] >= 0.0
        assert row["storage"]["load_ms"] >= 0.0
        assert row["storage"]["serialized_bytes"] > 0
        assert (tmp_path / "production.indexes" / f"{mode}.maxsim.npz").is_file()

    pooled = next(value for value in result["cases"] if value["case_id"] == "pooled_binary")
    assert result["config"]["pooled_binary_pool_factor"] == 3
    assert pooled["storage"]["index_metadata"]["pool_factor"] == 3
    assert pooled["storage"]["index_metadata"]["original_tokens"] == 10
    assert pooled["storage"]["index_metadata"]["pooled_tokens"] < 10


def test_completed_cases_resume_without_repacking(tmp_path, monkeypatch):
    source = tmp_path / "ragged.npz"
    output = tmp_path / "resume.json"
    _write_ragged_cache(source)
    first = run_production_benchmark(
        source,
        output,
        device="cpu",
        modes="binary_token_scale_u4,pooled_binary",
        top_k=2,
        query_limit=2,
        warmup=0,
        latency_repeats=1,
        throughput_batch_size=2,
    )

    def fail_repack(*args, **kwargs):
        raise AssertionError("completed checkpoints should bypass packing")

    monkeypatch.setattr("benchmarks.run_production_retrieval.maxsim.Index.from_embeddings", fail_repack)
    resumed = run_production_benchmark(
        source,
        output,
        device="cpu",
        modes="binary_token_scale_u4,pooled_binary",
        top_k=2,
        query_limit=2,
        warmup=0,
        latency_repeats=1,
        throughput_batch_size=2,
    )

    assert resumed["run_signature"] == first["run_signature"]
    assert resumed["resumed_cases"] == ["binary_token_scale_u4", "pooled_binary"]


def test_parser_and_validation_reject_invalid_cascade_budget(tmp_path):
    args = _build_parser().parse_args(["--input", "in.npz", "--output", "out.json"])
    assert args.device == "cuda"
    assert args.query_limit == 0
    assert args.serialize_indexes is True
    assert args.pooled_binary_pool_factor == 3

    source = tmp_path / "ragged.npz"
    _write_ragged_cache(source)
    with pytest.raises(ValueError, match="cascade candidates"):
        run_production_benchmark(
            source,
            tmp_path / "bad.json",
            device="cpu",
            modes="int4_residual",
            top_k=2,
            cascade_candidates=(1,),
        )

    with pytest.raises(ValueError, match="pooled_binary_pool_factor"):
        run_production_benchmark(
            source,
            tmp_path / "bad-pool.json",
            device="cpu",
            modes="pooled_binary",
            top_k=2,
            pooled_binary_pool_factor=0,
        )


def test_document_chunks_bound_tokens_without_splitting_documents():
    offsets = np.array([0, 2, 5, 6, 10], dtype=np.int64)

    assert list(_document_chunks(offsets, 4)) == [(0, 1), (1, 3), (3, 4)]
