from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.build_vidore_embeddings import build_vidore_embeddings
from benchmarks.compare_open_source import main as compare_open_source_main


DEFAULT_MODEL = "vidore/colqwen2-v1.0-hf"
DEFAULT_IMPLEMENTATIONS = "dense_fp16,faiss_pooled,cuvs_pooled,fast_plaid,bitmax_binary,bitmax_binary_q40,bitmax_int4"
SLOW_IMPLEMENTATIONS = "faiss_token_dense_rerank,qdrant_multivector"


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    config: str = "default"
    split: str = "test"


DATASET_ALIASES = {
    "docvqa": DatasetSpec("vidore/docvqa_test_subsampled"),
    "infovqa": DatasetSpec("vidore/infovqa_test_subsampled"),
    "arxivqa": DatasetSpec("vidore/arxivqa_test_subsampled"),
    "tabfquad": DatasetSpec("vidore/tabfquad_test_subsampled"),
}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the bitmax CUDA comparison across multiple cached embedding datasets.")
    parser.add_argument("--datasets", default="docvqa,infovqa,arxivqa,tabfquad")
    parser.add_argument("--embedding-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/multidataset"))
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--limits", default=None, help="Comma-separated corpus limits for scaling sweeps; defaults to --limit.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--implementations", default=DEFAULT_IMPLEMENTATIONS)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--build-missing", action="store_true")
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--faiss-token-topn", type=int, default=512)
    parser.add_argument("--metric-ks", default="1,5,10")
    args = parser.parse_args(argv)

    specs = _parse_dataset_specs(args.datasets)
    limits = _parse_limits(args.limits, default_limit=args.limit)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    implementations = _comparison_implementations(args.implementations, include_slow=args.include_slow)
    comparison_paths = []
    embedding_paths = []

    for limit in limits:
        for spec in specs:
            embedding_path = args.embedding_dir / _embedding_filename(spec, args.model, limit)
            if not embedding_path.exists():
                if not args.build_missing:
                    raise RuntimeError(f"missing embedding cache: {embedding_path}; rerun with --build-missing to create it")
                build_vidore_embeddings(
                    output_path=embedding_path,
                    dataset_name=spec.dataset,
                    config=spec.config,
                    split=spec.split,
                    limit=limit,
                    model_name=args.model,
                    batch_size=args.batch_size,
                )
            embedding_paths.append(str(embedding_path))

            comparison_path = args.output_dir / f"{embedding_path.stem}-comparison.json"
            compare_args = [
                "--input",
                str(embedding_path),
                "--output",
                str(comparison_path),
                "--device",
                args.device,
                "--implementations",
                implementations,
                "--repeat",
                str(args.repeat),
                "--faiss-token-topn",
                str(args.faiss_token_topn),
                "--metric-ks",
                args.metric_ks,
            ]
            if args.allow_unavailable:
                compare_args.append("--allow-unavailable")
            compare_open_source_main(compare_args)
            comparison_paths.append(comparison_path)

    summary = _aggregate_comparison_results(tuple(comparison_paths))
    summary.update(
        {
            "schema_version": 1,
            "benchmark": "multidataset_open_source_comparison",
            "model": args.model,
            "limit": int(limits[0]) if len(limits) == 1 else None,
            "limits": [int(limit) for limit in limits],
            "device": args.device,
            "repeat": int(args.repeat),
            "metric_ks": [int(value) for value in args.metric_ks.split(",") if value.strip()],
            "implementations": implementations.split(","),
            "embedding_paths": embedding_paths,
            "comparison_paths": [str(path) for path in comparison_paths],
            "uses_existing_embedding_caches": not args.build_missing,
            "embedding_cache_policy": "build_missing_with_frozen_public_model" if args.build_missing else "require_existing_npz",
        }
    )
    output = args.summary_output or (args.output_dir / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _print_summary(summary)
    return summary


def _parse_dataset_specs(value: str) -> tuple[DatasetSpec, ...]:
    specs = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in DATASET_ALIASES:
            specs.append(DATASET_ALIASES[item])
        elif "/" in item:
            specs.append(DatasetSpec(item))
        else:
            raise ValueError(f"unknown dataset alias: {item}")
    if not specs:
        raise ValueError("at least one dataset is required")
    return tuple(specs)


def _parse_limits(value: str | None, *, default_limit: int) -> tuple[int, ...]:
    raw_values = value.split(",") if value else [str(default_limit)]
    limits = []
    for raw in raw_values:
        item = raw.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 1:
            raise ValueError("limits must be >= 1")
        if parsed not in limits:
            limits.append(parsed)
    if not limits:
        raise ValueError("at least one limit is required")
    return tuple(limits)


def _embedding_filename(spec: DatasetSpec, model_name: str, limit: int) -> str:
    return f"vidore-{_dataset_short_name(spec.dataset)}-{_model_short_name(model_name)}-limit{int(limit)}.npz"


def _dataset_short_name(dataset: str) -> str:
    value = dataset.split("/")[-1]
    for suffix in ("_test_subsampled", "_train_subsampled", "_val_subsampled"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return _slug(value)


def _model_short_name(model_name: str) -> str:
    name = model_name.split("/")[-1].lower()
    if "colqwen2" in name:
        return "colqwen2"
    if "colpali" in name:
        return "colpali"
    return _slug(name)


def _slug(value: str) -> str:
    chars = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def _comparison_implementations(implementations: str, *, include_slow: bool) -> str:
    values = [item.strip() for item in implementations.split(",") if item.strip()]
    if include_slow:
        existing = set(values)
        for item in SLOW_IMPLEMENTATIONS.split(","):
            if item not in existing:
                values.append(item)
    return ",".join(values)


def _aggregate_comparison_results(paths: tuple[Path, ...]) -> dict[str, Any]:
    datasets = []
    per_dataset = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    unavailable: dict[str, int] = {}
    per_dataset_deltas: dict[str, dict[str, dict[str, float]]] = {}
    scaling_grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        dataset_name = str(data.get("dataset", {}).get("name", Path(path).stem))
        datasets.append(dataset_name)
        per_dataset[dataset_name] = data
        scale_key = _scale_key(data, path)
        dense_row = _dense_result_row(data)
        per_dataset_deltas[dataset_name] = {}
        for row in data.get("results", []):
            name = str(row["implementation"])
            if _is_quality_row(row):
                row_with_meta = dict(row)
                row_with_meta["_dataset_name"] = dataset_name
                row_with_meta["_dataset_scale"] = scale_key
                grouped.setdefault(name, []).append(row_with_meta)
                scaling_grouped.setdefault(scale_key, {}).setdefault(name, []).append(row_with_meta)
                if dense_row is not None:
                    per_dataset_deltas[dataset_name][name] = _quality_deltas(row, dense_row)
            else:
                unavailable[name] = unavailable.get(name, 0) + 1

    aggregate = _aggregate_grouped_results(grouped, unavailable=unavailable)
    aggregate.sort(key=lambda row: (-(row.get("mean_ndcg_at_10", -1.0)), -(row.get("geomean_speedup_vs_dense_fp16", 0.0))))
    scaling_results = {
        scale: {
            row["implementation"]: row
            for row in _aggregate_grouped_results(grouped_at_scale, unavailable={})
            if row["datasets_ok"] > 0
        }
        for scale, grouped_at_scale in sorted(scaling_grouped.items(), key=lambda item: _scale_sort_key(item[0]))
    }
    return {
        "datasets": datasets,
        "per_dataset": per_dataset,
        "per_dataset_deltas": per_dataset_deltas,
        "scaling_results": scaling_results,
        "aggregate_results": aggregate,
    }


def _is_quality_row(row: dict[str, Any]) -> bool:
    return row.get("status", "ok") == "ok" and "latency_ms" in row and any(key.startswith("ndcg_at_") for key in row)


def _dense_result_row(data: dict[str, Any]) -> dict[str, Any] | None:
    for row in data.get("results", []):
        if row.get("implementation") == "dense_fp16_baseline" and _is_quality_row(row):
            return row
    return None


def _quality_deltas(row: dict[str, Any], dense_row: dict[str, Any]) -> dict[str, float]:
    deltas = {}
    for prefix in ("recall_at_", "mrr_at_", "ndcg_at_"):
        for key, value in row.items():
            if not key.startswith(prefix) or key not in dense_row:
                continue
            suffix = key.removeprefix(prefix)
            label = prefix.removesuffix("_at_")
            deltas[f"{label}_delta_at_{suffix}"] = float(value) - float(dense_row[key])
    return deltas


def _aggregate_grouped_results(grouped: dict[str, list[dict[str, Any]]], *, unavailable: dict[str, int]) -> list[dict[str, Any]]:
    aggregate = []
    for name in sorted(set(grouped) | set(unavailable)):
        rows = grouped.get(name, [])
        item = {
            "implementation": name,
            "datasets_ok": len(rows),
            "datasets_unavailable": int(unavailable.get(name, 0)),
        }
        if rows:
            item.update(
                {
                    "mean_latency_ms": _mean(row["latency_ms"] for row in rows),
                    "median_latency_ms": _median(row["latency_ms"] for row in rows),
                    "mean_doc_memory_compression_vs_fp32": _mean(row["doc_memory_compression_vs_fp32"] for row in rows),
                    "geomean_speedup_vs_dense_fp16": _geomean(row.get("speedup_vs_dense_fp16", 1.0) for row in rows),
                }
            )
            for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
                if any(key in row for row in rows):
                    item[f"mean_{key}"] = _mean(row[key] for row in rows if key in row)
            for key in ("docs", "query_count"):
                if any(key in row for row in rows):
                    item[f"mean_{key}"] = _mean(row[key] for row in rows if key in row)
            for key in _metric_keys(rows):
                item[f"mean_{key}"] = _mean(row[key] for row in rows if key in row)
            item.update(_aggregate_token_buckets(rows))
        aggregate.append(item)
    return aggregate


def _metric_keys(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if (
                key.startswith(("recall_at_", "mrr_at_", "ndcg_at_", "quality_delta_vs_dense_ndcg_at_"))
                and isinstance(value, int | float)
            ):
                keys.add(key)
    return tuple(sorted(keys, key=_metric_sort_key))


def _aggregate_token_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    keys = sorted({key for row in rows for key in row if key.startswith("token_bucket_quality_at_")}, key=_metric_sort_key)
    for key in keys:
        bucket_names = sorted({bucket for row in rows for bucket in row.get(key, {})})
        output[key] = {}
        for bucket in bucket_names:
            bucket_rows = [
                row[key][bucket]
                for row in rows
                if key in row and bucket in row[key] and int(row[key][bucket].get("queries") or 0) > 0
            ]
            if not bucket_rows:
                continue
            item = {
                "datasets": int(len(bucket_rows)),
                "queries": int(sum(int(bucket_row["queries"]) for bucket_row in bucket_rows)),
            }
            numeric_keys = sorted(
                {
                    metric_key
                    for bucket_row in bucket_rows
                    for metric_key, value in bucket_row.items()
                    if metric_key != "queries" and isinstance(value, int | float)
                },
                key=_metric_sort_key,
            )
            for metric_key in numeric_keys:
                item[f"mean_{metric_key}"] = _mean(bucket_row[metric_key] for bucket_row in bucket_rows if metric_key in bucket_row)
            output[key][bucket] = item
    return output


def _scale_key(data: dict[str, Any], path: Path) -> str:
    dataset_name = str(data.get("dataset", {}).get("name", ""))
    tail = dataset_name.rsplit(":", 1)[-1]
    if tail.isdigit():
        return tail
    stem = Path(path).stem
    marker = "limit"
    if marker in stem:
        tail = stem.rsplit(marker, 1)[-1].split("-", 1)[0]
        if tail.isdigit():
            return tail
    docs = data.get("dataset", {}).get("docs")
    return str(docs) if docs is not None else "unknown"


def _scale_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**12, value)


def _metric_sort_key(value: str) -> tuple[int, str]:
    tail = value.rsplit("_at_", 1)[-1]
    return (int(tail), value) if tail.isdigit() else (10**12, value)


def _mean(values) -> float:
    data = [float(value) for value in values]
    return float(sum(data) / len(data))


def _median(values) -> float:
    data = sorted(float(value) for value in values)
    middle = len(data) // 2
    if len(data) % 2:
        return float(data[middle])
    return float((data[middle - 1] + data[middle]) * 0.5)


def _geomean(values) -> float:
    data = [float(value) for value in values if float(value) > 0.0]
    if not data:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in data) / len(data)))


def _print_summary(summary: dict[str, Any]) -> None:
    print("implementation, datasets_ok, mean_latency_ms, fp32_reduction, recall@10, ndcg@10, geomean_speedup")
    for row in summary["aggregate_results"]:
        if row["datasets_ok"] == 0:
            print(f"{row['implementation']}, 0, n/a, n/a, n/a, n/a, n/a")
            continue
        print(
            f"{row['implementation']}, {row['datasets_ok']}, {row['mean_latency_ms']:.3f}, "
            f"{row['mean_doc_memory_compression_vs_fp32']:.2f}x, {row['mean_recall_at_10']:.3f}, "
            f"{row['mean_ndcg_at_10']:.3f}, {row['geomean_speedup_vs_dense_fp16']:.2f}x"
        )


if __name__ == "__main__":
    main()
