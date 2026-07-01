from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkStep:
    name: str
    command: list[str]
    artifacts: list[str]
    requires_cuda: bool = False
    expensive: bool = False
    notes: str = ""


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Plan or run reproducible bitmax benchmark suites.")
    parser.add_argument(
        "--suite",
        choices=["smoke", "build-unique-caches", "cuda-unique-5k", "cuda-unique-10k", "cuda-docscale", "ledger-refresh"],
        default="smoke",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of writing a dry-run plan.")
    parser.add_argument("--refresh-ledger", action="store_true", help="Mirror benchmark-results JSON/PNG artifacts into docs/benchmark_results/raw.")
    args = parser.parse_args(argv)

    output = args.output or Path("benchmark-results") / f"reproduce-{args.suite}.json"
    plan = build_plan(args.suite, output_path=output)
    plan["executed"] = bool(args.execute)
    if args.execute:
        plan["step_results"] = [_run_step(step) for step in plan["steps"]]
        if args.refresh_ledger or args.suite == "ledger-refresh":
            plan["ledger_refresh_result"] = _run_command(plan["ledger_refresh_command"])
    else:
        print(f"DRY RUN: {args.suite}")
        for step in plan["steps"]:
            print(" ".join(step["command"]))
        print("ledger:", " ".join(plan["ledger_refresh_command"]))
    _write_json(output, plan)
    return plan


def build_plan(suite: str, *, output_path: Path) -> dict[str, Any]:
    if suite == "smoke":
        steps = _smoke_steps()
    elif suite == "build-unique-caches":
        steps = _build_unique_cache_steps()
    elif suite == "cuda-unique-5k":
        steps = _cuda_unique_5k_steps()
    elif suite == "cuda-unique-10k":
        steps = _cuda_unique_10k_steps()
    elif suite == "cuda-docscale":
        steps = _cuda_docscale_steps()
    elif suite == "ledger-refresh":
        steps = []
    else:
        raise ValueError(f"unknown suite: {suite}")
    return {
        "schema_version": 1,
        "suite": suite,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "environment": _environment(),
        "steps": [asdict(step) for step in steps],
        "ledger_refresh_command": _ledger_refresh_command(),
    }


def _smoke_steps() -> list[BenchmarkStep]:
    return [
        BenchmarkStep(
            name="unit-tests",
            command=[sys.executable, "-m", "pytest", "-q"],
            artifacts=[],
            notes="Runs local unit tests; CUDA tests are skipped automatically when unavailable.",
        ),
        BenchmarkStep(
            name="stage0-synthetic",
            command=[sys.executable, "-m", "benchmarks.run_synthetic", "--stage", "stage0", "--output", "benchmark-results/stage0.json"],
            artifacts=["benchmark-results/stage0.json"],
        ),
    ]


def _build_unique_cache_steps() -> list[BenchmarkStep]:
    sources = [
        ("docvqa-test", "vidore/docvqa_test_subsampled", "test", 500, False),
        ("infovqa-test", "vidore/infovqa_test_subsampled", "test", 500, False),
        ("arxivqa-test", "vidore/arxivqa_test_subsampled", "test", 500, False),
        ("tatdqa-test", "vidore/tatdqa_test", "test", 1663, False),
        ("syntheticdocqa-ai", "vidore/syntheticDocQA_artificial_intelligence_test", "test", 1000, False),
        ("syntheticdocqa-energy", "vidore/syntheticDocQA_energy_test", "test", 1000, False),
        ("syntheticdocqa-government", "vidore/syntheticDocQA_government_reports_test", "test", 1000, False),
        ("syntheticdocqa-healthcare", "vidore/syntheticDocQA_healthcare_industry_test", "test", 1000, False),
        ("syntheticdocqa-shift", "vidore/shiftproject_test", "test", 1000, False),
        ("docvqa-train", "vidore/docvqa_train", "train", 3000, True),
        ("infovqa-train", "vidore/infovqa_train", "train", 1200, True),
        ("arxivqa-train", "vidore/arxivqa_train", "train", 1200, True),
        ("tatdqa-train", "vidore/tatdqa_train", "train", 1200, True),
        ("syntheticdocqa-energy-train", "vidore/syntheticDocQA_energy_train", "train", 3000, True),
    ]
    steps: list[BenchmarkStep] = []
    source_artifacts: list[str] = []
    for slug, dataset, split, limit, streaming in sources:
        artifact = f"benchmark-results/vidore-{slug}-colqwen2-limit{limit}.npz"
        source_artifacts.append(artifact)
        command = [
            sys.executable,
            "-m",
            "benchmarks.build_vidore_embeddings",
            "--dataset",
            dataset,
            "--split",
            split,
            "--limit",
            str(limit),
            "--model",
            "vidore/colqwen2-v1.0-hf",
            "--output",
            artifact,
        ]
        if streaming:
            command.append("--streaming")
        steps.append(
            BenchmarkStep(
                name=f"embed-{slug}",
                command=command,
                artifacts=[artifact],
                requires_cuda=True,
                expensive=True,
                notes="Builds a unique public ViDoRe/ColQwen2 embedding cache from raw dataset rows.",
            )
        )
    steps.append(
        BenchmarkStep(
            name="mix-unique-10k",
            command=[
                sys.executable,
                "-m",
                "benchmarks.build_mixed_embeddings",
                "--inputs",
                *source_artifacts,
                "--dataset-name",
                "vidore/mixed_public_unique:test:10000",
                "--output",
                "benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz",
            ],
            artifacts=["benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz"],
            requires_cuda=False,
            expensive=True,
            notes="Combines unique source caches. Actual doc count is recorded in the output metadata/results.",
        )
    )
    return steps


def _cuda_unique_5k_steps() -> list[BenchmarkStep]:
    return [
        BenchmarkStep(
            name="unique-mixed-5k-rich",
            command=[
                sys.executable,
                "-m",
                "benchmarks.compare_open_source",
                "--input",
                "benchmark-results/vidore-mixed-syntheticdocqa-colqwen2-limit5000.npz",
                "--output",
                "benchmark-results/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json",
                "--device",
                "cuda",
                "--implementations",
                "dense_fp16,faiss_pooled,cuvs_pooled,fast_plaid,bitmax_binary,bitmax_binary_q40,bitmax_int4",
                "--limit-queries",
                "256",
                "--repeat",
                "3",
                "--metric-ks",
                "1,5,10",
                "--allow-unavailable",
            ],
            artifacts=[
                "benchmark-results/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json"
            ],
            requires_cuda=True,
            expensive=True,
            notes="Requires the 5k unique mixed ColQwen2 embedding cache.",
        )
    ]


def _cuda_unique_10k_steps() -> list[BenchmarkStep]:
    return [
        BenchmarkStep(
            name="unique-public-10k-rich",
            command=[
                sys.executable,
                "-m",
                "benchmarks.compare_open_source",
                "--input",
                "benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz",
                "--output",
                "benchmark-results/unique-public-10k-rich/vidore-mixed-public-unique-colqwen2-limit10000-comparison.json",
                "--device",
                "cuda",
                "--implementations",
                "dense_fp16,faiss_pooled,cuvs_pooled,fast_plaid,bitmax_binary,bitmax_binary_q40,bitmax_int4",
                "--limit-queries",
                "256",
                "--repeat",
                "3",
                "--metric-ks",
                "1,5,10",
                "--allow-unavailable",
            ],
            artifacts=["benchmark-results/unique-public-10k-rich/vidore-mixed-public-unique-colqwen2-limit10000-comparison.json"],
            requires_cuda=True,
            expensive=True,
            notes="Requires the mixed public unique cache built by the build-unique-caches suite.",
        )
    ]


def _cuda_docscale_steps() -> list[BenchmarkStep]:
    return [
        BenchmarkStep(
            name="mixed-3k-docscale",
            command=[
                sys.executable,
                "-m",
                "benchmarks.compare_open_source",
                "--input",
                "benchmark-results/vidore-mixed-syntheticdocqa-colqwen2-limit3000.npz",
                "--output",
                "benchmark-results/mixed-syntheticdocqa-docscale-rich/vidore-mixed-syntheticdocqa-colqwen2-limit3000-comparison.json",
                "--device",
                "cuda",
                "--implementations",
                "dense_fp16,faiss_pooled,cuvs_pooled,fast_plaid,bitmax_binary,bitmax_binary_q40,bitmax_int4",
                "--limit-queries",
                "256",
                "--repeat",
                "3",
                "--metric-ks",
                "1,5,10",
                "--allow-unavailable",
            ],
            artifacts=["benchmark-results/mixed-syntheticdocqa-docscale-rich/vidore-mixed-syntheticdocqa-colqwen2-limit3000-comparison.json"],
            requires_cuda=True,
            expensive=True,
            notes="Requires prebuilt mixed ColQwen2 embedding cache.",
        ),
        BenchmarkStep(
            name="docscale-stress-25k-core",
            command=[
                sys.executable,
                "-m",
                "benchmarks.compare_open_source",
                "--input",
                "benchmark-results/vidore-mixed-syntheticdocqa-docscale-stress-colqwen2-limit25000.npz",
                "--output",
                "benchmark-results/docscale-stress-5k-10k-25k-rich/vidore-mixed-syntheticdocqa-docscale-stress-colqwen2-limit25000-comparison.json",
                "--device",
                "cuda",
                "--implementations",
                "dense_fp16,bitmax_binary,bitmax_binary_q40,bitmax_int4",
                "--repeat",
                "3",
                "--metric-ks",
                "1,5,10",
                "--allow-unavailable",
            ],
            artifacts=["benchmark-results/docscale-stress-5k-10k-25k-rich/vidore-mixed-syntheticdocqa-docscale-stress-colqwen2-limit25000-comparison.json"],
            requires_cuda=True,
            expensive=True,
            notes="25k dense fp16 exact MaxSim is intentionally slow.",
        ),
    ]


def _ledger_refresh_command() -> list[str]:
    return [
        "rsync",
        "-av",
        "--include=*/",
        "--include=*.json",
        "--include=*.png",
        "--exclude=*",
        "benchmark-results/",
        "docs/benchmark_results/raw/",
    ]


def _environment() -> dict[str, Any]:
    return {
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _command_text(["git", "rev-parse", "HEAD"]),
        "git_branch": _command_text(["git", "branch", "--show-current"]),
        "git_dirty": bool(_command_text(["git", "status", "--porcelain"])),
        "gpu": _gpu_info(),
    }


def _gpu_info() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    return _command_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])


def _run_step(step: dict[str, Any]) -> dict[str, Any]:
    result = _run_command(step["command"])
    result["name"] = step["name"]
    result["artifacts"] = {artifact: Path(artifact).exists() for artifact in step.get("artifacts", [])}
    return result


def _run_command(command: list[str]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, text=True, capture_output=True)
    finished = datetime.now(timezone.utc)
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _command_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError:
        return ""
    return completed.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
