"""Checkpointed closest-prior PLAID benchmark over cached embeddings.

Both backends consume the same precomputed document/query token embeddings:

* ``fast-plaid`` uses the direct FastPlaid API used by the legacy comparison.
* ``pylate-plaid`` uses PyLate's public ``indexes.PLAID`` interface.

The output is updated atomically after load, initialization, build, and every
search repeat so an unavailable dependency or interrupted long run remains an
explicit result rather than a missing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.compare_open_source import _directory_size, _latency_stats
from benchmarks.run_production_retrieval import deterministic_query_indices
from examples.local_multivector_search import _load_demo_dataset, _ranking_metrics


AdapterFactory = Callable[..., Any]


class DependencyUnavailableError(RuntimeError):
    pass


class _DirectFastPlaidAdapter:
    def __init__(self, *, index_dir: Path, device: str, config: dict[str, Any], dataset) -> None:
        try:
            import torch
            from fast_plaid.search import FastPlaid
        except ImportError as exc:  # pragma: no cover - optional benchmark dependency
            raise RuntimeError("fast-plaid backend requires torch and fast-plaid") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("fast-plaid CUDA benchmark requested but torch CUDA is unavailable")
        self.torch = torch
        self.device = "cuda:0" if device == "cuda" else "cpu"
        self.config = config
        self.dataset = dataset
        self.searcher = FastPlaid(
            str(index_dir), device=self.device, low_memory=bool(config["low_memory"])
        )
        self.documents = None
        self.queries = None

    def build(self) -> None:
        self.documents = [
            self.torch.as_tensor(
                np.ascontiguousarray(
                    self.dataset["doc_embeddings"][int(start) : int(end)], dtype=np.float32
                ),
                device=self.device,
            )
            for start, end in zip(self.dataset["doc_offsets"][:-1], self.dataset["doc_offsets"][1:])
        ]
        self.searcher.create(
            self.documents,
            nbits=int(self.config["nbits"]),
            metadata=[{"doc_id": doc_id} for doc_id in self.dataset["doc_ids"]],
            start_from_scratch=0,
            seed=int(self.config["seed"]),
            use_triton_kmeans=self.config["use_triton"],
        )
        self.sync()

    def prepare_queries(self) -> None:
        self.queries = [
            self.torch.as_tensor(np.ascontiguousarray(query, dtype=np.float32), device=self.device)
            for query in self.dataset["query_embeddings"]
        ]
        self.sync()

    def search(self, *, k: int, query_indices: tuple[int, ...] | None = None):
        queries = (
            self.queries
            if query_indices is None
            else [self.queries[index] for index in query_indices]
        )
        result = self.searcher.search(
            queries,
            top_k=int(k),
            n_full_scores=int(self.config["n_full_scores"]),
            n_ivf_probe=int(self.config["n_ivf_probe"]),
            batch_size=int(self.config["batch_size"]),
            show_progress=False,
        )
        self.sync()
        return result

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


class _PyLatePlaidAdapter:
    def __init__(self, *, index_dir: Path, device: str, config: dict[str, Any], dataset) -> None:
        try:
            import torch
            from pylate import indexes
        except ImportError as exc:  # pragma: no cover - optional benchmark dependency
            raise RuntimeError("pylate-plaid backend requires torch and pylate") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyLate PLAID CUDA benchmark requested but torch CUDA is unavailable")
        try:
            pylate_version = importlib.metadata.version("pylate")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyUnavailableError("cannot determine installed pylate version") from exc
        if _version_tuple(pylate_version) < (1, 6, 0):
            raise DependencyUnavailableError(
                f"pylate-plaid requires pylate>=1.6.0 so seed and low_memory are applied; found {pylate_version}"
            )
        self.torch = torch
        self.device = "cuda:0" if device == "cuda" else "cpu"
        self.dataset = dataset
        self.documents = [
            np.ascontiguousarray(
                dataset["doc_embeddings"][int(start) : int(end)], dtype=np.float32
            )
            for start, end in zip(dataset["doc_offsets"][:-1], dataset["doc_offsets"][1:])
        ]
        self.queries = None
        self.index = indexes.PLAID(
            index_folder=str(index_dir.parent),
            index_name=index_dir.name,
            override=True,
            use_fast=True,
            nbits=int(config["nbits"]),
            seed=int(config["seed"]),
            use_triton=config["use_triton"],
            n_ivf_probe=int(config["n_ivf_probe"]),
            n_full_scores=int(config["n_full_scores"]),
            batch_size=int(config["batch_size"]),
            show_progress=False,
            device=self.device,
            low_memory=bool(config["low_memory"]),
        )

    def build(self) -> None:
        self.index.add_documents(
            documents_ids=list(self.dataset["doc_ids"]),
            documents_embeddings=self.documents,
        )
        self.sync()

    def prepare_queries(self) -> None:
        self.queries = [
            self.torch.as_tensor(np.ascontiguousarray(query, dtype=np.float32), device=self.device)
            for query in self.dataset["query_embeddings"]
        ]
        self.sync()

    def search(self, *, k: int, query_indices: tuple[int, ...] | None = None):
        queries = (
            self.queries
            if query_indices is None
            else [self.queries[index] for index in query_indices]
        )
        result = self.index(queries, k=int(k))
        self.sync()
        return result

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


def run_benchmark(
    *,
    input_path: Path,
    output_path: Path,
    backend: str = "pylate-plaid",
    device: str = "cuda",
    k: int = 10,
    metric_ks: tuple[int, ...] = (1, 5, 10),
    repeat: int = 5,
    warmup: int = 1,
    limit_queries: int | None = None,
    query_seed: int = 20260715,
    index_dir: Path | None = None,
    keep_index: bool = False,
    overwrite_index: bool = False,
    nbits: int = 4,
    n_full_scores: int = 4096,
    n_ivf_probe: int = 4,
    seed: int = 42,
    use_triton: bool | None = False,
    batch_size: int = 1 << 18,
    low_memory: bool = False,
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    if backend not in {"fast-plaid", "pylate-plaid"}:
        raise ValueError(f"unsupported backend: {backend}")
    if repeat < 1 or warmup < 0 or k < 1:
        raise ValueError("repeat and k must be positive and warmup must be nonnegative")
    if n_full_scores < 1 or n_ivf_probe < 1 or batch_size < 1:
        raise ValueError("n_full_scores, n_ivf_probe, and batch_size must be positive")
    if limit_queries is not None and limit_queries < 0:
        raise ValueError("limit_queries must be nonnegative")
    metric_ks = tuple(sorted(set((1, int(k), *(int(value) for value in metric_ks)))))
    if any(value < 1 for value in metric_ks):
        raise ValueError("metric cutoffs must be positive")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("PLAID output must not overwrite the input embedding cache")
    if index_dir is not None:
        try:
            input_path.resolve().relative_to(index_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("PLAID input cache must not be inside the index directory")
        try:
            output_path.resolve().relative_to(index_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("PLAID output must not be inside the index directory")

    config = {
        "backend": backend,
        "device": device,
        "k": int(k),
        "metric_ks": list(metric_ks),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "limit_queries": limit_queries,
        "query_seed": int(query_seed),
        "nbits": int(nbits),
        "n_full_scores": int(n_full_scores),
        "n_ivf_probe": int(n_ivf_probe),
        "seed": int(seed),
        "use_triton": use_triton,
        "batch_size": int(batch_size),
        "low_memory": bool(low_memory),
    }
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "closest_prior_plaid",
        "status": "running",
        "current_stage": "load_dataset",
        "started_at": _utc_now(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config": config,
        "environment": _environment(backend),
        "stages": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(output_path, artifact)
    current_stage = "load_dataset"
    created_temporary_index = index_dir is None
    index_owned = created_temporary_index
    if index_dir is None:
        index_dir = Path(tempfile.mkdtemp(prefix=f"bitmax-{backend}-")) / "index"
    index_dir = Path(index_dir)
    adapter = None
    try:
        stage_start = time.perf_counter()
        dataset = _load_demo_dataset(input_path, limit_queries=None)
        selected_queries = deterministic_query_indices(
            len(dataset["query_embeddings"]), limit_queries, int(query_seed)
        )
        all_query_ids = _load_query_ids(input_path, len(dataset["query_embeddings"]))
        dataset = {
            **dataset,
            "query_embeddings": tuple(dataset["query_embeddings"][int(index)] for index in selected_queries),
            "qrels": dataset["qrels"][selected_queries],
        }
        artifact["selected_queries"] = {
            "indices": selected_queries.tolist(),
            "ids": [all_query_ids[int(index)] for index in selected_queries],
        }
        if len(set(dataset["doc_ids"])) != len(dataset["doc_ids"]):
            raise ValueError("PLAID baseline requires unique doc_ids")
        score_k = min(max(metric_ks), len(dataset["doc_ids"]))
        config["search_depth"] = int(score_k)
        config["n_full_scores"] = max(score_k, min(int(n_full_scores), len(dataset["doc_ids"])))
        artifact["dataset"] = {
            "name": dataset["name"],
            "queries": len(dataset["query_embeddings"]),
            "docs": len(dataset["doc_ids"]),
            "dim": int(dataset["doc_embeddings"].shape[1]),
            "doc_tokens": int(dataset["doc_embeddings"].shape[0]),
            "input_bytes": int(input_path.stat().st_size),
        }
        _complete_stage(artifact, current_stage, stage_start)
        _write_checkpoint(output_path, artifact)

        current_stage = "prepare_index"
        artifact["current_stage"] = current_stage
        _write_checkpoint(output_path, artifact)
        stage_start = time.perf_counter()
        if index_dir.exists():
            if not overwrite_index:
                raise FileExistsError(f"index directory already exists: {index_dir}")
            shutil.rmtree(index_dir)
            index_owned = True
        else:
            index_owned = True
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        if backend == "fast-plaid":
            # The direct FastPlaid constructor expects its index root to exist.
            index_dir.mkdir()
        factory = adapter_factory or _create_adapter
        adapter = factory(
            backend=backend,
            index_dir=index_dir,
            device=device,
            config=config,
            dataset=dataset,
        )
        artifact["index"] = {"path": str(index_dir), "retained": bool(keep_index)}
        _complete_stage(artifact, current_stage, stage_start)
        _write_checkpoint(output_path, artifact)

        current_stage = "build_index"
        artifact["current_stage"] = current_stage
        _write_checkpoint(output_path, artifact)
        stage_start = time.perf_counter()
        adapter.build()
        adapter.sync()
        build_ms = (time.perf_counter() - stage_start) * 1_000.0
        index_bytes = _directory_size(str(index_dir))
        artifact["build"] = {
            "latency_ms": float(build_ms),
            "index_bytes": int(index_bytes),
            "bytes_per_document": float(index_bytes / max(len(dataset["doc_ids"]), 1)),
        }
        _complete_stage(artifact, current_stage, stage_start, elapsed_ms=build_ms)
        _write_checkpoint(output_path, artifact)

        current_stage = "prepare_queries"
        artifact["current_stage"] = current_stage
        _write_checkpoint(output_path, artifact)
        stage_start = time.perf_counter()
        adapter.prepare_queries()
        _complete_stage(artifact, current_stage, stage_start)
        _write_checkpoint(output_path, artifact)

        current_stage = "warmup_search"
        artifact["current_stage"] = current_stage
        _write_checkpoint(output_path, artifact)
        stage_start = time.perf_counter()
        for _ in range(warmup):
            adapter.search(k=score_k, query_indices=(0,))
            adapter.search(k=score_k)
        _complete_stage(artifact, current_stage, stage_start)
        _write_checkpoint(output_path, artifact)

        current_stage = "timed_search"
        artifact["current_stage"] = current_stage
        artifact["search"] = {
            "batch1": {
                "batch_size": 1,
                "completed_queries": 0,
                "repeats_per_query": int(repeat),
                "latency_samples_ms_by_query": [],
                "latency_samples_ms": [],
            },
            "throughput": {
                "batch_queries": len(dataset["query_embeddings"]),
                "completed_repeats": 0,
                "latency_samples_ms": [],
                "result_digests": [],
            },
        }
        _write_checkpoint(output_path, artifact)

        batch1 = artifact["search"]["batch1"]
        for query_idx in range(len(dataset["query_embeddings"])):
            query_samples = []
            for _ in range(repeat):
                start = time.perf_counter()
                adapter.search(k=score_k, query_indices=(query_idx,))
                adapter.sync()
                query_samples.append(float((time.perf_counter() - start) * 1_000.0))
            batch1["latency_samples_ms_by_query"].append(query_samples)
            batch1["latency_samples_ms"].extend(query_samples)
            batch1["completed_queries"] = query_idx + 1
            if (query_idx + 1) % 16 == 0 or query_idx + 1 == len(dataset["query_embeddings"]):
                _write_checkpoint(output_path, artifact)
        batch1.update(_latency_stats(batch1["latency_samples_ms"]))

        throughput = artifact["search"]["throughput"]
        final_scores = None
        for repeat_idx in range(repeat):
            start = time.perf_counter()
            raw = adapter.search(k=score_k)
            adapter.sync()
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            scores = _scores_from_results(raw, dataset["doc_ids"], query_count=len(dataset["query_embeddings"]))
            final_scores = scores
            throughput["latency_samples_ms"].append(float(elapsed_ms))
            throughput["result_digests"].append(_score_digest(scores))
            throughput["completed_repeats"] = repeat_idx + 1
            _write_checkpoint(output_path, artifact)

        samples = throughput["latency_samples_ms"]
        stats = _latency_stats(samples)
        throughput.update(stats)
        throughput.update(
            {
                "amortized_per_query_p50_ms": float(
                    stats["latency_p50_ms"] / max(len(dataset["query_embeddings"]), 1)
                ),
                "amortized_per_query_p95_ms": float(
                    stats["latency_p95_ms"] / max(len(dataset["query_embeddings"]), 1)
                ),
                "throughput_queries_per_second_at_p50": float(
                    1_000.0 * len(dataset["query_embeddings"]) / max(stats["latency_p50_ms"], 1e-12)
                ),
                "deterministic_topk_across_repeats": len(set(throughput["result_digests"])) == 1,
            }
        )
        timed_samples = batch1["latency_samples_ms"] + samples
        _complete_stage(
            artifact,
            current_stage,
            time.perf_counter(),
            elapsed_ms=float(sum(timed_samples)),
        )

        current_stage = "quality"
        artifact["current_stage"] = current_stage
        _write_checkpoint(output_path, artifact)
        stage_start = time.perf_counter()
        assert final_scores is not None
        quality: dict[str, float] = {}
        for cutoff in metric_ks:
            effective_k = min(cutoff, len(dataset["doc_ids"]))
            metrics = _ranking_metrics(final_scores, dataset["qrels"], k=effective_k)
            quality[f"recall_at_{cutoff}"] = float(metrics["recall_at_k"])
            quality[f"mrr_at_{cutoff}"] = float(metrics["mrr_at_k"])
            quality[f"ndcg_at_{cutoff}"] = float(metrics["ndcg_at_k"])
        artifact["quality"] = quality
        artifact["per_query_quality"] = _per_query_quality(
            final_scores,
            dataset["qrels"],
            query_ids=artifact["selected_queries"]["ids"],
            metric_ks=metric_ks,
        )
        artifact["rankings"] = _rankings(
            final_scores,
            doc_ids=dataset["doc_ids"],
            query_ids=artifact["selected_queries"]["ids"],
            depth=score_k,
        )
        _complete_stage(artifact, current_stage, stage_start)
        artifact.update({"status": "ok", "current_stage": "complete", "completed_at": _utc_now()})
    except Exception as exc:
        artifact.update(
            {
                "status": "unavailable" if _is_dependency_failure(exc) else "failed",
                "current_stage": current_stage,
                "completed_at": _utc_now(),
                "failure": {
                    "stage": current_stage,
                    "classification": "dependency_unavailable" if _is_dependency_failure(exc) else "runtime_error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "cause_type": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                    "cause_message": str(exc.__cause__) if exc.__cause__ is not None else None,
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finally:
        if not keep_index and index_owned:
            cleanup_root = index_dir.parent if created_temporary_index else index_dir
            shutil.rmtree(cleanup_root, ignore_errors=True)
            artifact.setdefault("index", {})["retained"] = False
            artifact["index"]["cleaned_up"] = True
        elif not index_owned:
            artifact.setdefault("index", {})["retained"] = True
            artifact["index"]["cleaned_up"] = False
        _write_checkpoint(output_path, artifact)
    return artifact


def _create_adapter(**kwargs):
    backend = kwargs.pop("backend")
    if backend == "fast-plaid":
        return _DirectFastPlaidAdapter(**kwargs)
    if backend == "pylate-plaid":
        return _PyLatePlaidAdapter(**kwargs)
    raise ValueError(f"unsupported backend: {backend}")


def _scores_from_results(raw: Any, doc_ids: tuple[str, ...], *, query_count: int) -> np.ndarray:
    if len(raw) != query_count:
        raise ValueError(f"PLAID returned {len(raw)} result rows for {query_count} queries")
    doc_index = {str(doc_id): idx for idx, doc_id in enumerate(doc_ids)}
    scores = np.full((query_count, len(doc_ids)), -np.inf, dtype=np.float32)
    for query_idx, row in enumerate(raw):
        for item in row:
            result_id, score = _result_id_score(item)
            if isinstance(result_id, (int, np.integer)):
                doc_idx = int(result_id)
                if doc_idx < 0 or doc_idx >= len(doc_ids):
                    raise ValueError(f"PLAID returned document index outside corpus: {doc_idx}")
            else:
                key = str(result_id)
                if key not in doc_index:
                    raise ValueError(f"PLAID returned unknown document ID: {key}")
                doc_idx = doc_index[key]
            scores[query_idx, doc_idx] = float(score)
    return scores


def _result_id_score(item: Any) -> tuple[Any, float]:
    if isinstance(item, dict):
        result_id = item.get("id", item.get("doc_id", item.get("document_id")))
        if result_id is None or "score" not in item:
            raise ValueError(f"unrecognized PLAID result dictionary: {item}")
        return result_id, float(item["score"])
    if hasattr(item, "id") and hasattr(item, "score"):
        return item.id, float(item.score)
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return item[0], float(item[1])
    raise ValueError(f"unrecognized PLAID result item: {item!r}")


def _per_query_quality(
    scores: np.ndarray,
    qrels: np.ndarray,
    *,
    query_ids: list[str],
    metric_ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for query_idx, query_id in enumerate(query_ids):
        row: dict[str, Any] = {"query_id": query_id}
        for cutoff in metric_ks:
            effective_k = min(cutoff, scores.shape[1])
            metrics = _ranking_metrics(
                scores[query_idx : query_idx + 1],
                qrels[query_idx : query_idx + 1],
                k=effective_k,
            )
            row[f"recall_at_{cutoff}"] = float(metrics["recall_at_k"])
            row[f"mrr_at_{cutoff}"] = float(metrics["mrr_at_k"])
            row[f"ndcg_at_{cutoff}"] = float(metrics["ndcg_at_k"])
        rows.append(row)
    return rows


def _rankings(
    scores: np.ndarray,
    *,
    doc_ids: tuple[str, ...],
    query_ids: list[str],
    depth: int,
) -> list[dict[str, Any]]:
    tie_ids = np.arange(len(doc_ids), dtype=np.int64)
    output = []
    for query_id, row in zip(query_ids, scores):
        order = np.lexsort((tie_ids, -row))[:depth]
        hits = [
            {"doc_id": doc_ids[int(doc_idx)], "score": float(row[int(doc_idx)]), "rank": rank}
            for rank, doc_idx in enumerate(order, start=1)
            if np.isfinite(row[int(doc_idx)])
        ]
        output.append({"query_id": query_id, "hits": hits})
    return output


def _complete_stage(
    artifact: dict[str, Any], stage: str, started: float, *, elapsed_ms: float | None = None
) -> None:
    artifact["stages"].append(
        {
            "stage": stage,
            "status": "ok",
            "completed_at": _utc_now(),
            "elapsed_ms": float(elapsed_ms if elapsed_ms is not None else (time.perf_counter() - started) * 1_000.0),
        }
    )


def _score_digest(scores: np.ndarray) -> str:
    top = np.argsort(-scores, axis=1, kind="stable")[:, : min(100, scores.shape[1])]
    return hashlib.sha256(np.ascontiguousarray(top, dtype=np.int64).tobytes()).hexdigest()


def _is_dependency_failure(exc: Exception) -> bool:
    chain: BaseException | None = exc
    while chain is not None:
        if isinstance(chain, (ImportError, ModuleNotFoundError, DependencyUnavailableError)):
            return True
        chain = chain.__cause__
    message = str(exc).lower()
    return "requires torch" in message or "requires torch and" in message


def _environment(backend: str) -> dict[str, Any]:
    package = "fast-plaid" if backend == "fast-plaid" else "pylate"
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend_package": package,
        "backend_version": version,
        "backend_python_requires": ">=3.10",
    }


def _load_query_ids(path: Path, query_count: int) -> tuple[str, ...]:
    with np.load(path, allow_pickle=False) as data:
        if "query_ids" not in data.files:
            return tuple(f"query-{idx}" for idx in range(query_count))
        values = tuple(str(value) for value in np.asarray(data["query_ids"]))
    if len(values) != query_count:
        raise ValueError(f"query_ids must contain {query_count} values")
    return values


def _write_checkpoint(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for raw in value.split(".")[:3]:
        digits = ""
        for character in raw:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


def _parse_metric_ks(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("metric cutoffs must be positive comma-separated integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("fast-plaid", "pylate-plaid"), default="pylate-plaid")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric-ks", type=_parse_metric_ks, default=(1, 5, 10))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--query-seed", type=int, default=20260715)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--keep-index", action="store_true")
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--n-full-scores", type=int, default=4096)
    parser.add_argument("--n-ivf-probe", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-triton", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=1 << 18)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--allow-unavailable", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_benchmark(
        input_path=args.input,
        output_path=args.output,
        backend=args.backend,
        device=args.device,
        k=args.k,
        metric_ks=args.metric_ks,
        repeat=args.repeat,
        warmup=args.warmup,
        limit_queries=args.limit_queries,
        query_seed=args.query_seed,
        index_dir=args.index_dir,
        keep_index=args.keep_index,
        overwrite_index=args.overwrite_index,
        nbits=args.nbits,
        n_full_scores=args.n_full_scores,
        n_ivf_probe=args.n_ivf_probe,
        seed=args.seed,
        use_triton=args.use_triton,
        batch_size=args.batch_size,
        low_memory=args.low_memory,
    )
    print(args.output)
    print(f"status={artifact['status']} stage={artifact['current_stage']}")
    if artifact["status"] == "ok" or (args.allow_unavailable and artifact["status"] == "unavailable"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
