from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maxsim._api import PackedDocs
from maxsim.experimental import DimCentroidCalibration

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PackedBundle:
    packed: PackedDocs
    centroid_calibration: DimCentroidCalibration | None = None
    metadata: dict[str, Any] | None = None


def save_packed(
    path: str | Path,
    packed: PackedDocs,
    *,
    calibration: DimCentroidCalibration | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if packed.device != "cpu":
        raise ValueError("save_packed requires CPU PackedDocs; call it before moving docs to CUDA")
    if not isinstance(packed.data, np.ndarray):
        raise ValueError("save_packed requires CPU PackedDocs backed by a NumPy array")
    if calibration is not None and calibration.dim != packed.dim:
        raise ValueError("calibration dim must match packed dim")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scale_kind, scale_values = _encode_scale(packed.scale)
    token_scale_kind, token_scale_values = _encode_scale(packed.token_scale)
    arrays: dict[str, object] = {
        "schema_version": np.array(_SCHEMA_VERSION, dtype=np.int64),
        "format": np.array("maxsim_packed_docs"),
        "data": np.ascontiguousarray(packed.data, dtype=np.uint8),
        "doc_offsets": np.ascontiguousarray(packed.doc_offsets, dtype=np.int64),
        "dim": np.array(packed.dim, dtype=np.int64),
        "num_docs": np.array(packed.num_docs, dtype=np.int64),
        "scale_kind": np.array(scale_kind),
        "scale_values": scale_values,
        "token_scale_kind": np.array(token_scale_kind),
        "token_scale_values": token_scale_values,
        "metadata_json": np.array(json.dumps({} if metadata is None else metadata, sort_keys=True)),
        "has_centroid_calibration": np.array(calibration is not None, dtype=np.bool_),
    }
    if calibration is not None:
        arrays.update(
            {
                "centroid_thresholds": np.ascontiguousarray(calibration.thresholds, dtype=np.float32),
                "centroid_negative": np.ascontiguousarray(calibration.negative_centroids, dtype=np.float32),
                "centroid_positive": np.ascontiguousarray(calibration.positive_centroids, dtype=np.float32),
            }
        )
    np.savez_compressed(output, **arrays)


def load_packed(path: str | Path) -> PackedBundle:
    with np.load(Path(path), allow_pickle=False) as data:
        schema_version = int(np.asarray(data["schema_version"]).item())
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported packed docs schema_version={schema_version}")
        file_format = str(np.asarray(data["format"]).item())
        if file_format != "maxsim_packed_docs":
            raise ValueError(f"unsupported packed docs format: {file_format}")

        token_scale = None
        if "token_scale_kind" in data:
            token_scale = _decode_scale(str(np.asarray(data["token_scale_kind"]).item()), np.asarray(data["token_scale_values"]))
        packed = PackedDocs(
            data=np.ascontiguousarray(data["data"], dtype=np.uint8),
            doc_offsets=np.ascontiguousarray(data["doc_offsets"], dtype=np.int64),
            dim=int(np.asarray(data["dim"]).item()),
            num_docs=int(np.asarray(data["num_docs"]).item()),
            scale=_decode_scale(str(np.asarray(data["scale_kind"]).item()), np.asarray(data["scale_values"])),
            device="cpu",
            token_scale=token_scale,
        )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        calibration = None
        if bool(np.asarray(data["has_centroid_calibration"]).item()):
            calibration = DimCentroidCalibration(
                thresholds=np.ascontiguousarray(data["centroid_thresholds"], dtype=np.float32),
                negative_centroids=np.ascontiguousarray(data["centroid_negative"], dtype=np.float32),
                positive_centroids=np.ascontiguousarray(data["centroid_positive"], dtype=np.float32),
            )
    return PackedBundle(packed=packed, centroid_calibration=calibration, metadata=metadata)


def _encode_scale(scale) -> tuple[str, np.ndarray]:
    if scale is None:
        return "none", np.empty((0,), dtype=np.float32)
    if isinstance(scale, np.ndarray):
        return "vector", np.ascontiguousarray(scale, dtype=np.float32)
    return "scalar", np.array([float(scale)], dtype=np.float32)


def _decode_scale(kind: str, values: np.ndarray):
    if kind == "none":
        return None
    if kind == "scalar":
        if values.shape != (1,):
            raise ValueError("scalar scale payload must contain one value")
        return float(values[0])
    if kind == "vector":
        return np.ascontiguousarray(values, dtype=np.float32)
    raise ValueError(f"unknown scale kind: {kind}")
