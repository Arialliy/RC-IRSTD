from __future__ import annotations

"""Fold-specific, label-free source references for permutation-invariant distances."""

from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.utils.io import list_npz


def build_source_reference(
    score_directories: list[str | Path],
    output_path: str | Path,
    *,
    context_size: int = 32,
    stride: int | None = None,
    feature_config: WindowFeatureConfig | None = None,
) -> Path:
    if not score_directories:
        raise ValueError("At least one source score directory is required")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    step = context_size if stride is None else int(stride)
    if step <= 0:
        raise ValueError("stride must be positive")
    config = feature_config or WindowFeatureConfig()
    extractor = WindowFeatureExtractor(config)
    centres: list[np.ndarray] = []
    domains: list[str] = []
    all_windows: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None

    for directory in score_directories:
        paths = list_npz(directory)
        records = [load_score_record(path, load_mask=False) for path in paths]
        records = sorted(records, key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
        if len(records) < context_size:
            raise ValueError(f"{directory} has fewer than {context_size} records")
        windows: list[np.ndarray] = []
        for start in range(0, len(records) - context_size + 1, step):
            features, names = extractor.extract(records[start : start + context_size])
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("Feature schema changed across source domains")
            windows.append(features.astype(np.float32))
            all_windows.append(features.astype(np.float32))
        if not windows:
            raise ValueError(f"No source windows produced for {directory}")
        centres.append(np.mean(np.stack(windows), axis=0, dtype=np.float64).astype(np.float32))
        domains.append(Path(directory).resolve().name)

    assert feature_names is not None
    stacked = np.stack(all_windows).astype(np.float32)
    scale = np.maximum(stacked.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        centres=np.stack(centres).astype(np.float32),
        scale=scale,
        domains=np.asarray(domains, dtype=np.str_),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        context_size=np.asarray(context_size, dtype=np.int64),
        artifact_type=np.asarray("label_free_source_feature_reference_v1"),
    )
    return output


def source_distances_from_reference(
    features: np.ndarray,
    reference_path: str | Path,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    with np.load(reference_path, allow_pickle=False) as payload:
        centres = np.asarray(payload["centres"], dtype=np.float32)
        scale = np.maximum(np.asarray(payload["scale"], dtype=np.float32), 1e-6)
        names = tuple(np.asarray(payload["feature_names"]).astype(str).tolist())
    if names != feature_names:
        raise ValueError("Source-reference feature schema differs from deployment features")
    values = np.asarray(features, dtype=np.float32).reshape(1, -1)
    return np.sqrt(np.mean(((values - centres) / scale[None, :]) ** 2, axis=1)).astype(
        np.float32
    )
