import json

from benchmarks import reproduce


def test_reproduce_plan_includes_environment_and_refreshable_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    plan = reproduce.build_plan("smoke", output_path=tmp_path / "plan.json")

    assert plan["suite"] == "smoke"
    assert plan["steps"]
    assert plan["environment"]["cwd"] == str(tmp_path)
    assert any("pytest" in " ".join(step["command"]) for step in plan["steps"])
    assert plan["ledger_refresh_command"][-2:] == ["benchmark-results/", "docs/benchmark_results/raw/"]


def test_reproduce_dry_run_writes_plan_without_running_commands(tmp_path, capsys):
    output = tmp_path / "plan.json"

    result = reproduce.main(["--suite", "smoke", "--output", str(output)])

    captured = capsys.readouterr()
    assert result["executed"] is False
    assert "DRY RUN" in captured.out
    assert json.loads(output.read_text())["suite"] == "smoke"


def test_unique_5k_plan_records_cuda_comparison_artifact(tmp_path):
    plan = reproduce.build_plan("cuda-unique-5k", output_path=tmp_path / "plan.json")

    assert plan["suite"] == "cuda-unique-5k"
    assert len(plan["steps"]) == 1
    step = plan["steps"][0]
    assert step["requires_cuda"] is True
    assert "benchmarks.compare_open_source" in " ".join(step["command"])
    assert any("unique-mixed-syntheticdocqa-5k-rich" in artifact for artifact in step["artifacts"])
