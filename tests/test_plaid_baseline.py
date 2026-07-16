import json
from pathlib import Path

import numpy as np

from benchmarks.run_plaid_baseline import _scores_from_results, run_benchmark
from benchmarks.run_production_retrieval import deterministic_query_indices


def _write_fixture(path: Path) -> None:
    np.savez(
        path,
        doc_embeddings=np.eye(3, 4, dtype=np.float32),
        doc_offsets=np.arange(4, dtype=np.int64),
        query_embeddings=np.eye(3, 4, dtype=np.float32)[:, None, :],
        qrels=np.eye(3, dtype=np.float32),
        doc_ids=np.asarray(["a", "b", "c"]),
        query_ids=np.asarray(["qa", "qb", "qc"]),
        dataset_name=np.asarray("plaid-fixture"),
    )


class _ResultObject:
    def __init__(self, result_id, score):
        self.id = result_id
        self.score = score


def test_scores_from_results_accepts_direct_and_pylate_shapes():
    raw = [
        [(0, 3.0), (1, 2.0)],
        [{"id": "b", "score": 4.0}],
        [_ResultObject("c", 5.0)],
    ]
    scores = _scores_from_results(raw, ("a", "b", "c"), query_count=3)
    assert scores[0, 0] == 3.0
    assert scores[1, 1] == 4.0
    assert scores[2, 2] == 5.0
    assert np.isneginf(scores[1, 0])


def test_plaid_runner_records_build_raw_timings_quality_and_checkpoints(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "plaid.json"
    search_batch_sizes = []
    _write_fixture(source)

    class FakeAdapter:
        def __init__(self, *, index_dir, dataset, **_):
            self.index_dir = index_dir
            self.dataset = dataset

        def build(self):
            self.index_dir.mkdir(parents=True, exist_ok=True)
            (self.index_dir / "index.bin").write_bytes(b"x" * 32)

        def sync(self):
            return None

        def prepare_queries(self):
            return None

        def search(self, *, k, query_indices=None):
            rows = []
            indices = (
                range(len(self.dataset["query_embeddings"]))
                if query_indices is None
                else query_indices
            )
            indices = tuple(indices)
            search_batch_sizes.append(len(indices))
            for query_idx in indices:
                relevant_idx = int(np.argmax(self.dataset["qrels"][query_idx]))
                order = [relevant_idx] + [idx for idx in range(3) if idx != relevant_idx]
                rows.append(
                    [
                        {"id": self.dataset["doc_ids"][doc_idx], "score": float(3 - rank)}
                        for rank, doc_idx in enumerate(order[:k])
                    ]
                )
            return rows

    artifact = run_benchmark(
        input_path=source,
        output_path=output,
        backend="fast-plaid",
        device="cpu",
        k=1,
        metric_ks=(1, 3),
        repeat=3,
        warmup=1,
        limit_queries=2,
        query_seed=20260715,
        adapter_factory=lambda **kwargs: FakeAdapter(**kwargs),
    )

    assert artifact["status"] == "ok"
    assert artifact["build"]["index_bytes"] == 32
    batch1 = artifact["search"]["batch1"]
    throughput = artifact["search"]["throughput"]
    assert batch1["batch_size"] == 1
    assert batch1["completed_queries"] == 2
    assert batch1["repeats_per_query"] == 3
    assert len(batch1["latency_samples_ms_by_query"]) == 2
    assert all(len(samples) == 3 for samples in batch1["latency_samples_ms_by_query"])
    assert len(batch1["latency_samples_ms"]) == 6
    assert batch1["latency_p95_ms"] >= batch1["latency_p50_ms"]
    assert throughput["batch_queries"] == 2
    assert len(throughput["latency_samples_ms"]) == 3
    assert throughput["latency_p95_ms"] >= throughput["latency_p50_ms"]
    assert throughput["deterministic_topk_across_repeats"] is True
    assert search_batch_sizes.count(1) == 7  # one warmup plus 2 queries x 3 repeats
    assert search_batch_sizes.count(2) == 4  # one warmup plus 3 throughput repeats
    assert artifact["config"]["search_depth"] == 3
    assert artifact["quality"]["ndcg_at_3"] == 1.0
    expected_indices = deterministic_query_indices(3, 2, 20260715).tolist()
    assert artifact["selected_queries"]["indices"] == expected_indices
    assert artifact["selected_queries"]["ids"] == [f"q{'abc'[idx]}" for idx in expected_indices]
    assert len(artifact["per_query_quality"]) == 2
    assert len(artifact["rankings"]) == 2
    assert artifact["rankings"][0]["hits"][0]["rank"] == 1
    assert [stage["stage"] for stage in artifact["stages"]] == [
        "load_dataset",
        "prepare_index",
        "build_index",
        "prepare_queries",
        "warmup_search",
        "timed_search",
        "quality",
    ]
    saved = json.loads(output.read_text())
    assert saved["status"] == "ok"
    assert saved["search"]["batch1"]["completed_queries"] == 2
    assert saved["search"]["throughput"]["completed_repeats"] == 3


def test_plaid_runner_writes_explicit_dependency_failure(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "plaid.json"
    _write_fixture(source)

    def missing_factory(**_):
        try:
            raise ModuleNotFoundError("no module named pylate")
        except ModuleNotFoundError as cause:
            raise RuntimeError("pylate-plaid backend requires pylate") from cause

    artifact = run_benchmark(
        input_path=source,
        output_path=output,
        backend="pylate-plaid",
        device="cpu",
        adapter_factory=missing_factory,
    )

    assert artifact["status"] == "unavailable"
    assert artifact["failure"]["stage"] == "prepare_index"
    assert artifact["failure"]["classification"] == "dependency_unavailable"
    assert artifact["failure"]["cause_type"] == "ModuleNotFoundError"
    saved = json.loads(output.read_text())
    assert saved["failure"]["message"] == "pylate-plaid backend requires pylate"


def test_plaid_runner_never_deletes_existing_index_without_overwrite(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "plaid.json"
    index_dir = tmp_path / "existing-index"
    index_dir.mkdir()
    marker = index_dir / "keep.txt"
    marker.write_text("owned by user")
    _write_fixture(source)

    artifact = run_benchmark(
        input_path=source,
        output_path=output,
        backend="fast-plaid",
        device="cpu",
        index_dir=index_dir,
        adapter_factory=lambda **_: None,
    )

    assert artifact["status"] == "failed"
    assert artifact["failure"]["type"] == "FileExistsError"
    assert marker.read_text() == "owned by user"
    assert artifact["index"]["cleaned_up"] is False


def test_plaid_runner_refuses_to_overwrite_input_cache(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    original_size = source.stat().st_size

    try:
        run_benchmark(input_path=source, output_path=source, backend="fast-plaid", device="cpu")
    except ValueError as exc:
        assert "must not overwrite" in str(exc)
    else:
        raise AssertionError("expected input/output collision to be rejected")
    assert source.stat().st_size == original_size


def test_plaid_runner_never_removes_index_directory_containing_input(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    source = index_dir / "source.npz"
    output = tmp_path / "result.json"
    _write_fixture(source)

    try:
        run_benchmark(
            input_path=source,
            output_path=output,
            backend="fast-plaid",
            device="cpu",
            index_dir=index_dir,
            overwrite_index=True,
        )
    except ValueError as exc:
        assert "input cache must not be inside" in str(exc)
    else:
        raise AssertionError("expected unsafe index path to be rejected")
    assert source.exists()
