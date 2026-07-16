"""Build and execute reproducible nested-corpus scale benchmarks.

The plan is intentionally lightweight: it stores source document indices and
materializes only the cache currently being measured. Query IDs are fixed
across all runs, every relevant document is present at every scale, and each
distractor seed defines one deterministic prefix ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.run_production_retrieval import deterministic_query_indices


CORE_ARRAY_KEYS = {
    "doc_embeddings",
    "doc_offsets",
    "query_embeddings",
    "query_offsets",
    "qrels",
    "doc_ids",
    "query_ids",
}
DEFAULT_IMPLEMENTATIONS = (
    "dense_fp16_vectorized,bitmax_binary,bitmax_binary_token_scale_u4,"
    "bitmax_pooled_binary3,bitmax_int4,bitmax_int4_per_token,bitmax_int4_residual"
)


def create_plan(
    *,
    input_path: Path,
    output_dir: Path,
    sizes: Iterable[int],
    seeds: Iterable[int],
    query_limit: int | None = None,
    query_seed: int = 20260715,
    implementations: str = DEFAULT_IMPLEMENTATIONS,
    device: str = "cuda",
    repeat: int = 5,
    k: int = 10,
    metric_ks: str = "1,5,10",
) -> dict[str, Any]:
    sizes = tuple(sorted(set(int(value) for value in sizes)))
    seeds = tuple(dict.fromkeys(int(value) for value in seeds))
    if not sizes or sizes[0] < 1:
        raise ValueError("sizes must contain positive document counts")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if repeat < 1 or k < 1:
        raise ValueError("repeat and k must be positive")

    source = _load_plan_source(input_path)
    query_indices = _select_query_indices(
        source["query_ids"], query_limit=query_limit, query_seed=int(query_seed)
    )
    selected_qrels = source["qrels"][query_indices]
    positive_indices = np.flatnonzero(np.any(selected_qrels > 0, axis=0)).astype(np.int64).tolist()
    if len(positive_indices) > sizes[0]:
        raise ValueError(
            f"smallest size {sizes[0]} cannot contain all {len(positive_indices)} relevant documents"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        membership_sequence = _nested_doc_sequence(
            doc_ids=source["doc_ids"],
            positive_indices=positive_indices,
            seed=seed,
            minimum_docs=sizes[0],
            target_docs=sizes[-1],
        )
        for size in sizes:
            membership = membership_sequence[:size]
            selection = membership
            slug = f"seed{seed}-n{size}"
            runs.append(
                {
                    "seed": seed,
                    "doc_count": size,
                    "membership_source_doc_indices": membership,
                    "source_doc_indices": selection,
                    "membership_sha256": _int_sequence_digest(membership),
                    "selection_sha256": _int_sequence_digest(selection),
                    "cache_path": str(output_dir / f"nested-scale-{slug}.npz"),
                    "result_path": str(output_dir / f"nested-scale-{slug}.json"),
                }
            )

    fixed_query_ids = [source["query_ids"][idx] for idx in query_indices]
    stat = input_path.stat()
    plan = {
        "schema_version": 1,
        "benchmark": "nested_corpus_scale",
        "created_at": _utc_now(),
        "source_path": str(input_path),
        "source_path_resolved": str(input_path.resolve()),
        "source_file_size": int(stat.st_size),
        "source_sha256": _file_sha256(input_path),
        "source_dataset_name": source["dataset_name"],
        "source_model_name": source["model_name"],
        "source_docs": len(source["doc_ids"]),
        "source_queries": len(source["query_ids"]),
        "source_scalar_metadata": source["scalar_metadata"],
        "sizes": list(sizes),
        "distractor_seeds": list(seeds),
        "query_seed": int(query_seed),
        "query_indices": query_indices,
        "fixed_query_ids": fixed_query_ids,
        "fixed_query_ids_sha256": _string_sequence_digest(fixed_query_ids),
        "relevant_source_doc_indices": positive_indices,
        "relevant_source_docs": len(positive_indices),
        "benchmark_config": {
            "implementations": implementations,
            "device": device,
            "repeat": int(repeat),
            "k": int(k),
            "metric_ks": metric_ks,
        },
        "runs": runs,
    }
    plan["benchmark_config_sha256"] = _json_digest(plan["benchmark_config"])
    _atomic_write_json(output_dir / "nested-scale-plan.json", plan)
    return plan


def materialize_run(
    plan: dict[str, Any],
    *,
    seed: int,
    doc_count: int,
    output_path: Path | None = None,
    source_path: Path | None = None,
    compressed: bool = False,
    overwrite: bool = False,
    verify_source: bool = True,
) -> Path:
    spec = _find_run(plan, seed=seed, doc_count=doc_count)
    source_path = source_path or _existing_source_path(plan)
    output_path = output_path or Path(spec["cache_path"])
    if output_path.resolve() == source_path.resolve():
        raise ValueError("nested-scale output must not overwrite the source cache")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing cache: {output_path}")
    if verify_source:
        _verify_plan_source(plan, source_path)
    query_indices = [int(value) for value in plan["query_indices"]]
    selected = [int(value) for value in spec["source_doc_indices"]]

    with np.load(source_path, allow_pickle=False) as data:
        doc_embeddings = np.asarray(data["doc_embeddings"], dtype=np.float32)
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        source_doc_count = int(doc_offsets.shape[0] - 1)
        if any(value < 0 or value >= source_doc_count for value in selected):
            raise ValueError("plan contains a source document index outside the source cache")
        source_doc_ids = _ids(data, "doc_ids", source_doc_count, "doc")
        source_query_count = _query_count(data)
        source_query_ids = _ids(data, "query_ids", source_query_count, "query")
        source_qrels = np.asarray(data["qrels"], dtype=np.float32)
        metadata_arrays = _metadata_arrays(data)
        queries, query_offsets = _selected_query_embeddings(data, query_indices)

        chunks: list[np.ndarray] = []
        offsets = [0]
        doc_ids: list[str] = []
        duplicate_ordinals: list[int] = []
        seen: dict[int, int] = {}
        qrels = np.zeros((len(query_indices), len(selected)), dtype=np.float32)
        selected_source_qrels = source_qrels[np.asarray(query_indices, dtype=np.int64)]
        for output_idx, source_idx in enumerate(selected):
            start, end = int(doc_offsets[source_idx]), int(doc_offsets[source_idx + 1])
            chunk = np.ascontiguousarray(doc_embeddings[start:end], dtype=np.float32)
            chunks.append(chunk)
            offsets.append(offsets[-1] + int(chunk.shape[0]))
            occurrence = seen.get(source_idx, 0)
            seen[source_idx] = occurrence + 1
            duplicate_ordinals.append(occurrence)
            source_id = source_doc_ids[source_idx]
            doc_ids.append(source_id if occurrence == 0 else f"{source_id}::nested-scale-copy-{occurrence}")
            if occurrence == 0:
                qrels[:, output_idx] = selected_source_qrels[:, source_idx]

        if _int_sequence_digest(selected) != spec["selection_sha256"]:
            raise ValueError("plan selection digest mismatch")
        expected_query_ids = [str(value) for value in plan["fixed_query_ids"]]
        query_ids = [source_query_ids[idx] for idx in query_indices]
        if query_ids != expected_query_ids:
            raise ValueError("source query IDs no longer match the plan")

        source_dataset = _scalar_string(data, "dataset_name", source_path.stem)
        payload: dict[str, Any] = metadata_arrays
        payload.update(
            {
                "doc_embeddings": np.concatenate(chunks, axis=0),
                "doc_offsets": np.asarray(offsets, dtype=np.int64),
                "query_embeddings": queries,
                "query_offsets": query_offsets,
                "qrels": qrels,
                "doc_ids": np.asarray(doc_ids),
                "query_ids": np.asarray(query_ids),
                "dataset_name": np.asarray(f"{source_dataset}:nested-scale:n{doc_count}:seed{seed}"),
                "nested_scale_source_dataset": np.asarray(source_dataset),
                "nested_scale_source_path": np.asarray(str(source_path)),
                "nested_scale_seed": np.asarray(int(seed)),
                "nested_scale_doc_count": np.asarray(int(doc_count)),
                "nested_scale_query_seed": np.asarray(int(plan["query_seed"])),
                "nested_scale_source_doc_indices": np.asarray(selected, dtype=np.int64),
                "nested_scale_duplicate_ordinals": np.asarray(duplicate_ordinals, dtype=np.int32),
                "nested_scale_selection_sha256": np.asarray(spec["selection_sha256"]),
                "nested_scale_membership_sha256": np.asarray(spec["membership_sha256"]),
                "nested_scale_source_sha256": np.asarray(plan["source_sha256"]),
                "nested_scale_fixed_query_ids_sha256": np.asarray(plan["fixed_query_ids_sha256"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    writer = np.savez_compressed if compressed else np.savez
    try:
        writer(temporary, **payload)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def execute_plan(
    plan: dict[str, Any],
    *,
    plan_path: Path,
    seeds: set[int] | None = None,
    sizes: set[int] | None = None,
    keep_caches: bool = False,
    compressed_caches: bool = False,
    allow_unavailable: bool = False,
    resume: bool = True,
    continue_on_error: bool = False,
    source_path: Path | None = None,
) -> dict[str, Any]:
    from benchmarks.compare_open_source import main as compare_open_source

    config = plan["benchmark_config"]
    checkpoint_path = plan_path.with_name(
        f"nested-scale-checkpoint{_checkpoint_suffix(seeds=seeds, sizes=sizes)}.json"
    )
    verified_source_path = source_path or _existing_source_path(plan)
    _verify_plan_source(plan, verified_source_path)
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "nested_corpus_scale_execution",
        "plan_path": str(plan_path),
        "started_at": _utc_now(),
        "status": "running",
        "runs": [],
    }
    _atomic_write_json(checkpoint_path, checkpoint)
    failures = 0
    matched_runs = 0
    for spec in plan["runs"]:
        seed = int(spec["seed"])
        doc_count = int(spec["doc_count"])
        if seeds is not None and seed not in seeds:
            continue
        if sizes is not None and doc_count not in sizes:
            continue
        matched_runs += 1
        cache_path = Path(spec["cache_path"])
        result_path = Path(spec["result_path"])
        if resume and _result_matches_spec(result_path, spec=spec, plan=plan):
            checkpoint["runs"].append(
                {"seed": seed, "doc_count": doc_count, "status": "skipped_existing", "result_path": str(result_path)}
            )
            _atomic_write_json(checkpoint_path, checkpoint)
            continue

        record: dict[str, Any] = {
            "seed": seed,
            "doc_count": doc_count,
            "status": "materializing",
            "started_at": _utc_now(),
            "cache_path": str(cache_path),
            "result_path": str(result_path),
        }
        checkpoint["runs"].append(record)
        _atomic_write_json(checkpoint_path, checkpoint)
        cache_owned = False
        try:
            if cache_path.exists():
                if not _cache_matches_spec(cache_path, spec=spec, plan=plan):
                    raise FileExistsError(f"refusing to replace unrelated cache: {cache_path}")
                record["cache_reused"] = True
            else:
                materialize_run(
                    plan,
                    seed=seed,
                    doc_count=doc_count,
                    output_path=cache_path,
                    source_path=verified_source_path,
                    compressed=compressed_caches,
                    verify_source=False,
                )
                cache_owned = True
                record["cache_reused"] = False
            record["status"] = "benchmarking"
            _atomic_write_json(checkpoint_path, checkpoint)
            argv = [
                "--input", str(cache_path),
                "--output", str(result_path),
                "--device", str(config["device"]),
                "--implementations", str(config["implementations"]),
                "--repeat", str(config["repeat"]),
                "--k", str(config["k"]),
                "--metric-ks", str(config["metric_ks"]),
            ]
            if allow_unavailable:
                argv.append("--allow-unavailable")
            result = compare_open_source(argv)
            if result is None:
                result = json.loads(result_path.read_text())
            result["nested_scale_provenance"] = _run_provenance(spec=spec, plan=plan)
            _atomic_write_json(result_path, result)
            record.update({"status": "ok", "completed_at": _utc_now()})
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "status": "failed",
                    "completed_at": _utc_now(),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            if not continue_on_error:
                checkpoint.update({"status": "failed", "completed_at": _utc_now()})
                _atomic_write_json(checkpoint_path, checkpoint)
                raise
        finally:
            if not keep_caches and cache_owned:
                cache_path.unlink(missing_ok=True)
            _atomic_write_json(checkpoint_path, checkpoint)

    if matched_runs == 0:
        checkpoint.update(
            {
                "status": "failed",
                "failure_count": 1,
                "failure": {"type": "NoMatchingRuns", "message": "filters matched no planned runs"},
                "completed_at": _utc_now(),
            }
        )
    else:
        checkpoint.update(
            {
                "status": "ok" if failures == 0 else "completed_with_failures",
                "failure_count": failures,
                "completed_at": _utc_now(),
            }
        )
    _atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint


def _load_plan_source(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        doc_count = int(offsets.shape[0] - 1)
        query_count = _query_count(data)
        qrels = np.asarray(data["qrels"], dtype=np.float32)
        if qrels.shape != (query_count, doc_count):
            raise ValueError(f"qrels must have shape {(query_count, doc_count)}, got {qrels.shape}")
        doc_ids = _ids(data, "doc_ids", doc_count, "doc")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("source doc_ids must be unique")
        query_ids = _ids(data, "query_ids", query_count, "query")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("source query_ids must be unique")
        return {
            "qrels": qrels,
            "doc_ids": doc_ids,
            "query_ids": query_ids,
            "dataset_name": _scalar_string(data, "dataset_name", path.stem),
            "model_name": _scalar_string(data, "model_name", ""),
            "scalar_metadata": _scalar_metadata(data),
        }


def _query_count(data) -> int:
    queries = np.asarray(data["query_embeddings"])
    if queries.ndim == 3:
        return int(queries.shape[0])
    if queries.ndim == 2 and "query_offsets" in data.files:
        return int(np.asarray(data["query_offsets"]).shape[0] - 1)
    raise ValueError("query_embeddings must be 3D or 2D with query_offsets")


def _select_query_indices(query_ids: list[str], *, query_limit: int | None, query_seed: int) -> list[int]:
    return deterministic_query_indices(len(query_ids), query_limit, query_seed).tolist()


def _nested_doc_sequence(
    *, doc_ids: list[str], positive_indices: list[int], seed: int, minimum_docs: int, target_docs: int
) -> list[int]:
    positives = sorted(set(int(value) for value in positive_indices))
    positive_set = set(positives)
    distractors = [idx for idx in range(len(doc_ids)) if idx not in positive_set]
    if target_docs > len(positives) and not distractors:
        raise ValueError("cannot add distractors because every source document is relevant")
    first_order = sorted(
        distractors,
        key=lambda idx: _stable_key("distractor", seed, 0, doc_ids[idx], idx),
    )
    mandatory_distractors = first_order[: max(0, minimum_docs - len(positives))]
    mandatory = positives + mandatory_distractors
    mandatory.sort(key=lambda idx: _stable_key("mandatory-cohort", seed, doc_ids[idx], idx))
    sequence = mandatory + first_order[len(mandatory_distractors) :]
    copy_round = 1
    while len(sequence) < target_docs:
        copied_order = sorted(
            distractors,
            key=lambda idx: _stable_key("distractor", seed, copy_round, doc_ids[idx], idx),
        )
        sequence.extend(copied_order)
        copy_round += 1
    return sequence[:target_docs]


def _selected_query_embeddings(data, query_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(data["query_embeddings"], dtype=np.float32)
    if queries.ndim == 3:
        chunks = [np.ascontiguousarray(queries[idx], dtype=np.float32) for idx in query_indices]
    elif queries.ndim == 2:
        offsets = np.asarray(data["query_offsets"], dtype=np.int64)
        chunks = [
            np.ascontiguousarray(queries[int(offsets[idx]) : int(offsets[idx + 1])], dtype=np.float32)
            for idx in query_indices
        ]
    else:
        raise ValueError("query_embeddings must be 2D or 3D")
    selected_offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum([chunk.shape[0] for chunk in chunks], out=selected_offsets[1:])
    return np.concatenate(chunks, axis=0), selected_offsets


def _scalar_metadata(data) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in data.files:
        if key in CORE_ARRAY_KEYS:
            continue
        value = np.asarray(data[key])
        if value.ndim != 0 or value.dtype.kind not in "biufUS":
            continue
        item = value.item()
        if isinstance(item, np.generic):
            item = item.item()
        metadata[key] = item
    return metadata


def _metadata_arrays(data) -> dict[str, np.ndarray]:
    """Copy every non-schema array so cache provenance survives subsetting."""
    return {
        key: np.array(data[key], copy=True)
        for key in data.files
        if key not in CORE_ARRAY_KEYS
    }


def _ids(data, key: str, count: int, prefix: str) -> list[str]:
    if key not in data.files:
        return [f"{prefix}-{idx}" for idx in range(count)]
    values = [str(value) for value in np.asarray(data[key])]
    if len(values) != count:
        raise ValueError(f"{key} must contain {count} values")
    return values


def _scalar_string(data, key: str, fallback: str) -> str:
    return str(np.asarray(data[key]).item()) if key in data.files else fallback


def _stable_key(*parts: Any) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def _int_sequence_digest(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _string_sequence_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_plan_source(plan: dict[str, Any], path: Path) -> None:
    if int(path.stat().st_size) != int(plan["source_file_size"]):
        raise ValueError("source cache size does not match the planned source")
    if _file_sha256(path) != plan["source_sha256"]:
        raise ValueError("source cache SHA-256 does not match the planned source")


def _find_run(plan: dict[str, Any], *, seed: int, doc_count: int) -> dict[str, Any]:
    for spec in plan["runs"]:
        if int(spec["seed"]) == int(seed) and int(spec["doc_count"]) == int(doc_count):
            return spec
    raise ValueError(f"plan has no run for seed={seed}, doc_count={doc_count}")


def _run_provenance(*, spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha256": plan["source_sha256"],
        "selection_sha256": spec["selection_sha256"],
        "membership_sha256": spec["membership_sha256"],
        "fixed_query_ids_sha256": plan["fixed_query_ids_sha256"],
        "benchmark_config_sha256": plan["benchmark_config_sha256"],
        "seed": int(spec["seed"]),
        "doc_count": int(spec["doc_count"]),
    }


def _cache_matches_spec(path: Path, *, spec: dict[str, Any], plan: dict[str, Any]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(np.asarray(data["nested_scale_seed"]).item()) == int(spec["seed"])
                and int(np.asarray(data["nested_scale_doc_count"]).item()) == int(spec["doc_count"])
                and str(np.asarray(data["nested_scale_selection_sha256"]).item()) == spec["selection_sha256"]
                and str(np.asarray(data["nested_scale_membership_sha256"]).item()) == spec["membership_sha256"]
                and str(np.asarray(data["nested_scale_source_sha256"]).item()) == plan["source_sha256"]
                and str(np.asarray(data["nested_scale_fixed_query_ids_sha256"]).item())
                == plan["fixed_query_ids_sha256"]
            )
    except (OSError, ValueError, KeyError):
        return False


def _result_matches_spec(path: Path, *, spec: dict[str, Any], plan: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        data.get("benchmark") == "open_source_comparison"
        and data.get("nested_scale_provenance") == _run_provenance(spec=spec, plan=plan)
        and int(data.get("dataset", {}).get("docs", -1)) == int(spec["doc_count"])
        and int(data.get("dataset", {}).get("queries", -1)) == len(plan["fixed_query_ids"])
        and bool(data.get("results"))
    )


def _checkpoint_suffix(*, seeds: set[int] | None, sizes: set[int] | None) -> str:
    if seeds is None and sizes is None:
        return ""
    seed_part = "all" if seeds is None else "-".join(str(value) for value in sorted(seeds))
    size_part = "all" if sizes is None else "-".join(str(value) for value in sorted(sizes))
    return f"-seeds{seed_part}-sizes{size_part}"


def _existing_source_path(plan: dict[str, Any]) -> Path:
    candidates = [Path(plan["source_path"]), Path(plan["source_path_resolved"])]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"source cache not found at any planned path: {candidates}")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return parsed


def _load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Write a lightweight deterministic scale plan.")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--sizes", type=_parse_ints, default=(1000, 5000, 10000, 25000))
    plan.add_argument("--seeds", type=_parse_ints, default=(17, 29, 41))
    plan.add_argument(
        "--query-limit",
        type=int,
        default=256,
        help="Fixed deterministic query sample (default: 256; 0 selects all queries).",
    )
    plan.add_argument("--query-seed", type=int, default=20260715)
    plan.add_argument("--implementations", default=DEFAULT_IMPLEMENTATIONS)
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    plan.add_argument("--repeat", type=int, default=5)
    plan.add_argument("--k", type=int, default=10)
    plan.add_argument("--metric-ks", default="1,5,10")

    materialize = commands.add_parser("materialize", help="Materialize one cache from a scale plan.")
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--seed", type=int, required=True)
    materialize.add_argument("--size", type=int, required=True)
    materialize.add_argument("--output", type=Path, default=None)
    materialize.add_argument("--source", type=Path, default=None)
    materialize.add_argument("--compressed", action="store_true")
    materialize.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run", help="Materialize and benchmark planned scales sequentially.")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--seeds", type=_parse_ints, default=None)
    run.add_argument("--sizes", type=_parse_ints, default=None)
    run.add_argument("--keep-caches", action="store_true")
    run.add_argument("--compressed-caches", action="store_true")
    run.add_argument("--allow-unavailable", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--source", type=Path, default=None, help="Override the source cache path stored in the plan.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = create_plan(
            input_path=args.input,
            output_dir=args.output_dir,
            sizes=args.sizes,
            seeds=args.seeds,
            query_limit=args.query_limit,
            query_seed=args.query_seed,
            implementations=args.implementations,
            device=args.device,
            repeat=args.repeat,
            k=args.k,
            metric_ks=args.metric_ks,
        )
        print(args.output_dir / "nested-scale-plan.json")
        print(f"planned {len(result['runs'])} runs with {len(result['fixed_query_ids'])} fixed queries")
        return 0
    if args.command == "materialize":
        path = materialize_run(
            _load_plan(args.plan),
            seed=args.seed,
            doc_count=args.size,
            output_path=args.output,
            source_path=args.source,
            compressed=args.compressed,
            overwrite=args.overwrite,
        )
        print(path)
        return 0
    if args.command == "run":
        checkpoint = execute_plan(
            _load_plan(args.plan),
            plan_path=args.plan,
            seeds=None if args.seeds is None else set(args.seeds),
            sizes=None if args.sizes is None else set(args.sizes),
            keep_caches=args.keep_caches,
            compressed_caches=args.compressed_caches,
            allow_unavailable=args.allow_unavailable,
            resume=not args.no_resume,
            continue_on_error=args.continue_on_error,
            source_path=args.source,
        )
        return 0 if checkpoint["status"] == "ok" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
