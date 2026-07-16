import json
from pathlib import Path

import numpy as np

import benchmarks.compare_open_source as compare_open_source
from benchmarks.run_nested_scale import _checkpoint_suffix, create_plan, execute_plan, materialize_run
from benchmarks.run_production_retrieval import deterministic_query_indices


def _write_fixture(path: Path) -> None:
    doc_lengths = [1, 2, 1, 3, 1, 2]
    doc_offsets = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(doc_lengths, out=doc_offsets[1:])
    doc_embeddings = np.arange(int(doc_offsets[-1]) * 4, dtype=np.float32).reshape(-1, 4)
    query_lengths = [1, 2, 1, 3]
    query_offsets = np.zeros(len(query_lengths) + 1, dtype=np.int64)
    np.cumsum(query_lengths, out=query_offsets[1:])
    query_embeddings = -np.arange(int(query_offsets[-1]) * 4, dtype=np.float32).reshape(-1, 4)
    qrels = np.zeros((4, 6), dtype=np.float32)
    qrels[0, 0] = 1
    qrels[1, 1] = 2
    qrels[2, 0] = 1
    qrels[3, 1] = 1
    np.savez(
        path,
        doc_embeddings=doc_embeddings,
        doc_offsets=doc_offsets,
        query_embeddings=query_embeddings,
        query_offsets=query_offsets,
        qrels=qrels,
        doc_ids=np.asarray([f"doc-{idx}" for idx in range(6)]),
        query_ids=np.asarray([f"query-{idx}" for idx in range(4)]),
        dataset_name=np.asarray("fixture"),
        model_name=np.asarray("fixture-model"),
        split=np.asarray("test"),
        source_paths=np.asarray(["first.npz", "second.npz"]),
    )


def test_plan_is_nested_reproducible_and_keeps_relevant_docs(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    first = create_plan(
        input_path=source,
        output_dir=tmp_path / "runs-a",
        sizes=(3, 5, 8),
        seeds=(17, 29),
        query_limit=2,
        query_seed=7,
    )
    second = create_plan(
        input_path=source,
        output_dir=tmp_path / "runs-b",
        sizes=(3, 5, 8),
        seeds=(17, 29),
        query_limit=2,
        query_seed=7,
    )

    assert first["fixed_query_ids"] == second["fixed_query_ids"]
    assert first["query_indices"] == second["query_indices"]
    assert first["query_indices"] == deterministic_query_indices(4, 2, 7).tolist()
    assert first["relevant_source_doc_indices"] == [0, 1]
    for seed in (17, 29):
        runs = [row for row in first["runs"] if row["seed"] == seed]
        reruns = [row for row in second["runs"] if row["seed"] == seed]
        assert runs[0]["membership_source_doc_indices"] == runs[1]["membership_source_doc_indices"][:3]
        assert runs[1]["membership_source_doc_indices"] == runs[2]["membership_source_doc_indices"][:5]
        assert runs[0]["source_doc_indices"] == runs[1]["source_doc_indices"][:3]
        assert runs[1]["source_doc_indices"] == runs[2]["source_doc_indices"][:5]
        assert runs[2]["source_doc_indices"] == reruns[2]["source_doc_indices"]
        assert set(first["relevant_source_doc_indices"]).issubset(runs[0]["source_doc_indices"])
        assert any(
            value not in first["relevant_source_doc_indices"]
            for value in runs[0]["source_doc_indices"][: len(first["relevant_source_doc_indices"])]
        )
        assert len(runs[2]["source_doc_indices"]) == 8
    assert (
        [row for row in first["runs"] if row["seed"] == 17][-1]["selection_sha256"]
        != [row for row in first["runs"] if row["seed"] == 29][-1]["selection_sha256"]
    )


def test_materialize_preserves_queries_qrels_ids_and_scalar_metadata(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    plan = create_plan(
        input_path=source,
        output_dir=tmp_path / "runs",
        sizes=(3, 8),
        seeds=(17,),
        query_limit=2,
        query_seed=7,
    )
    output = materialize_run(plan, seed=17, doc_count=8)

    with np.load(source, allow_pickle=False) as original, np.load(output, allow_pickle=False) as data:
        assert data["query_ids"].tolist() == plan["fixed_query_ids"]
        assert str(data["model_name"].item()) == "fixture-model"
        assert str(data["split"].item()) == "test"
        assert data["source_paths"].tolist() == ["first.npz", "second.npz"]
        assert int(data["nested_scale_doc_count"].item()) == 8
        assert int(data["nested_scale_seed"].item()) == 17
        assert data["nested_scale_source_doc_indices"].shape == (8,)
        assert np.count_nonzero(data["nested_scale_duplicate_ordinals"]) == 2
        duplicate_ids = [value for value in data["doc_ids"].tolist() if "nested-scale-copy" in value]
        assert len(duplicate_ids) == 2

        selected_source_qrels = original["qrels"][np.asarray(plan["query_indices"], dtype=np.int64)]
        np.testing.assert_array_equal(data["qrels"].sum(axis=1), selected_source_qrels.sum(axis=1))
        for query_out_idx, query_source_idx in enumerate(plan["query_indices"]):
            source_start = int(original["query_offsets"][query_source_idx])
            source_end = int(original["query_offsets"][query_source_idx + 1])
            output_start = int(data["query_offsets"][query_out_idx])
            output_end = int(data["query_offsets"][query_out_idx + 1])
            np.testing.assert_array_equal(
                data["query_embeddings"][output_start:output_end],
                original["query_embeddings"][source_start:source_end],
            )


def test_execute_plan_checkpoints_and_removes_transient_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    plan_dir = tmp_path / "runs"
    plan = create_plan(
        input_path=source,
        output_dir=plan_dir,
        sizes=(3,),
        seeds=(17,),
        query_limit=2,
        implementations="dense_fp16",
        device="cpu",
        repeat=1,
    )
    calls = []

    def fake_compare(argv):
        calls.append(argv)
        input_path = Path(argv[argv.index("--input") + 1])
        output_path = Path(argv[argv.index("--output") + 1])
        assert input_path.exists()
        output_path.write_text(json.dumps({"status": "ok"}))

    monkeypatch.setattr(compare_open_source, "main", fake_compare)
    checkpoint = execute_plan(
        plan,
        plan_path=plan_dir / "nested-scale-plan.json",
        keep_caches=False,
    )

    assert checkpoint["status"] == "ok"
    assert checkpoint["runs"][0]["status"] == "ok"
    assert len(calls) == 1
    assert not Path(plan["runs"][0]["cache_path"]).exists()
    saved = json.loads((plan_dir / "nested-scale-checkpoint.json").read_text())
    assert saved["status"] == "ok"


def test_execute_plan_does_not_delete_colliding_cache_and_does_not_skip_stale_result(tmp_path, monkeypatch):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    plan_dir = tmp_path / "runs"
    plan = create_plan(
        input_path=source,
        output_dir=plan_dir,
        sizes=(3,),
        seeds=(17,),
        query_limit=2,
        implementations="dense_fp16",
        device="cpu",
        repeat=1,
    )
    cache_path = Path(plan["runs"][0]["cache_path"])
    result_path = Path(plan["runs"][0]["result_path"])
    cache_path.write_text("unrelated")
    result_path.write_text("{}")
    monkeypatch.setattr(compare_open_source, "main", lambda _: (_ for _ in ()).throw(AssertionError("not reached")))

    try:
        execute_plan(plan, plan_path=plan_dir / "nested-scale-plan.json")
    except FileExistsError as exc:
        assert "unrelated cache" in str(exc)
    else:
        raise AssertionError("expected colliding cache to be rejected")
    assert cache_path.read_text() == "unrelated"


def test_materialize_refuses_to_overwrite_source(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    plan = create_plan(
        input_path=source,
        output_dir=tmp_path / "runs",
        sizes=(3,),
        seeds=(17,),
        query_limit=2,
    )
    original_size = source.stat().st_size
    try:
        materialize_run(plan, seed=17, doc_count=3, output_path=source)
    except ValueError as exc:
        assert "must not overwrite" in str(exc)
    else:
        raise AssertionError("expected source/output collision to be rejected")
    assert source.stat().st_size == original_size


def test_plan_rejects_scale_too_small_for_positive_documents(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    try:
        create_plan(
            input_path=source,
            output_dir=tmp_path / "runs",
            sizes=(1,),
            seeds=(17,),
        )
    except ValueError as exc:
        assert "cannot contain all" in str(exc)
    else:
        raise AssertionError("expected a too-small scale to be rejected")


def test_parallel_shards_use_distinct_checkpoint_names():
    assert _checkpoint_suffix(seeds={17}, sizes={1000}) != _checkpoint_suffix(
        seeds={29}, sizes={1000}
    )


def test_execute_plan_reports_filters_that_match_no_runs(tmp_path):
    source = tmp_path / "source.npz"
    _write_fixture(source)
    plan_dir = tmp_path / "runs"
    plan = create_plan(
        input_path=source,
        output_dir=plan_dir,
        sizes=(3,),
        seeds=(17,),
        query_limit=2,
    )
    checkpoint = execute_plan(
        plan,
        plan_path=plan_dir / "nested-scale-plan.json",
        seeds={999},
    )
    assert checkpoint["status"] == "failed"
    assert checkpoint["failure"]["type"] == "NoMatchingRuns"
