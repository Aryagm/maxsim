from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def build_vidore_embeddings(
    *,
    output_path: Path,
    dataset_name: str = "vidore/docvqa_test_subsampled",
    config: str = "default",
    split: str = "test",
    limit: int = 16,
    model_name: str = "vidore/colqwen2-v1.0-hf",
    batch_size: int = 1,
    streaming: bool = False,
) -> Path:
    rows = _load_vidore_rows(dataset_name=dataset_name, config=config, split=split, limit=limit, streaming=streaming)
    doc_keys, doc_row_indices, _query_doc_indices, qrels = _deduplicate_doc_rows(rows)
    model, processor = _load_retrieval_model(model_name)

    doc_images = [_as_rgb_image(rows[row_idx]["image"]) for row_idx in doc_row_indices]
    queries = [str(row["query"]) for row in rows]
    doc_embeddings = _encode_batches(model, processor, doc_images, batch_size=batch_size, modality="image")
    query_embeddings = _encode_batches(model, processor, queries, batch_size=batch_size, modality="text")
    flat_docs, doc_offsets = _flatten_ragged(doc_embeddings)
    flat_queries, query_offsets = _flatten_ragged(query_embeddings)

    if flat_docs.shape[1] % 8 != 0:
        raise ValueError(f"model embedding dim must be divisible by 8, got {flat_docs.shape[1]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        doc_embeddings=flat_docs,
        doc_offsets=doc_offsets,
        query_embeddings=flat_queries,
        query_offsets=query_offsets,
        qrels=qrels,
        doc_ids=np.asarray(doc_keys),
        query_ids=np.asarray([str(row.get("questionId", f"query-{idx}")) for idx, row in enumerate(rows)]),
        dataset_name=np.asarray(f"{dataset_name}:{split}:{limit}"),
        source_dataset=np.asarray(dataset_name),
        source_config=np.asarray(config),
        source_split=np.asarray(split),
        model_name=np.asarray(model_name),
    )
    return output_path


def _load_vidore_rows(*, dataset_name: str, config: str, split: str, limit: int, streaming: bool = False) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("build_vidore_embeddings requires the 'datasets' package") from exc

    if limit < 1:
        raise ValueError("limit must be >= 1")
    if streaming:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        rows = []
        for row in dataset:
            rows.append(dict(row))
            if len(rows) >= int(limit):
                break
        if len(rows) < int(limit):
            raise ValueError(f"streaming dataset only yielded {len(rows)} rows before limit {limit}")
        return rows
    dataset = load_dataset(dataset_name, config, split=split)
    count = min(int(limit), len(dataset))
    return [dict(dataset[idx]) for idx in range(count)]


def _load_retrieval_model(model_name: str):
    try:
        import torch
        _ignore_missing_torchvision_nms_fake_registration(torch)
        import transformers
        from transformers.utils.import_utils import is_flash_attn_2_available
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("build_vidore_embeddings requires torch and transformers with ColVision retrieval models") from exc

    model_key = model_name.lower()
    if "colqwen" in model_key:
        model_cls = transformers.ColQwen2ForRetrieval
        processor_cls = transformers.ColQwen2Processor
        kwargs = {"attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa"}
    elif "colpali" in model_key:
        model_cls = transformers.ColPaliForRetrieval
        processor_cls = transformers.ColPaliProcessor
        kwargs = {}
    else:
        raise ValueError("model_name must reference a ColQwen2 or ColPali retrieval checkpoint")

    model = model_cls.from_pretrained(model_name, device_map="auto", **kwargs).eval()
    processor = processor_cls.from_pretrained(model_name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, processor


def _ignore_missing_torchvision_nms_fake_registration(torch_module) -> None:
    register_fake = getattr(getattr(torch_module, "library", None), "register_fake", None)
    if register_fake is None or getattr(register_fake, "_bitmax_safe_register", False):
        return

    def safe_register_fake(op_name, *args, **kwargs):
        decorator = register_fake(op_name, *args, **kwargs)

        def wrapped(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if op_name == "torchvision::nms" and "does not exist" in str(exc):
                    return fn
                raise

        return wrapped

    safe_register_fake._bitmax_safe_register = True
    torch_module.library.register_fake = safe_register_fake


def _encode_batches(model, processor, values: list[Any], *, batch_size: int, modality: str) -> list[np.ndarray]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("encoding requires torch") from exc

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    encoded: list[np.ndarray] = []
    for batch in _chunks(values, batch_size):
        if modality == "image":
            inputs = processor(images=batch, return_tensors="pt").to(model.device)
        elif modality == "text":
            inputs = processor(text=batch, return_tensors="pt").to(model.device)
        else:
            raise ValueError("modality must be 'image' or 'text'")
        with torch.inference_mode():
            embeddings = model(**inputs).embeddings.detach().cpu().to(torch.float32).numpy()
        encoded.extend(np.ascontiguousarray(embedding, dtype=np.float32) for embedding in embeddings)
    return encoded


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _as_rgb_image(image):
    convert = getattr(image, "convert", None)
    if convert is None:
        raise ValueError("dataset image rows must contain PIL images")
    return image.convert("RGB")


def _deduplicate_doc_rows(rows: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], np.ndarray]:
    doc_to_index: dict[str, int] = {}
    doc_keys: list[str] = []
    doc_row_indices: list[int] = []
    query_doc_indices: list[int] = []
    for row_idx, row in enumerate(rows):
        key = _doc_key(row)
        if key not in doc_to_index:
            doc_to_index[key] = len(doc_keys)
            doc_keys.append(key)
            doc_row_indices.append(row_idx)
        query_doc_indices.append(doc_to_index[key])

    qrels = np.zeros((len(rows), len(doc_keys)), dtype=np.float32)
    for query_idx, doc_idx in enumerate(query_doc_indices):
        qrels[query_idx, doc_idx] = 1.0
    return tuple(doc_keys), tuple(doc_row_indices), tuple(query_doc_indices), qrels


def _doc_key(row: dict[str, Any]) -> str:
    doc_id = row.get("docId", "")
    filename = row.get("image_filename", "")
    page = row.get("page", "")
    return f"{doc_id}:{filename}:{page}"


def _flatten_ragged(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not arrays:
        raise ValueError("at least one embedding array is required")
    dim = int(arrays[0].shape[1])
    offsets = [0]
    contiguous_arrays = []
    for array in arrays:
        if array.ndim != 2 or int(array.shape[1]) != dim:
            raise ValueError("all embedding arrays must have shape [tokens, dim]")
        contiguous_arrays.append(np.ascontiguousarray(array, dtype=np.float32))
        offsets.append(offsets[-1] + int(array.shape[0]))
    return np.concatenate(contiguous_arrays, axis=0), np.asarray(offsets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ViDoRe/ColVision embedding .npz for maxsim retrieval benchmarks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--model", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--streaming", action="store_true", help="Stream source rows and stop after --limit instead of downloading the full split first.")
    args = parser.parse_args()

    path = build_vidore_embeddings(
        output_path=args.output,
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.limit,
        model_name=args.model,
        batch_size=args.batch_size,
        streaming=args.streaming,
    )
    print(path)


if __name__ == "__main__":
    main()
