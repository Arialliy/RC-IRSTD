"""Extended, metadata-preserving dataset utilities for risk evaluation.

Dataset classes are loaded lazily so split/path audits remain usable before
the optional training stack (PyTorch/torchvision) is installed.
"""

from .split_utils import (
    read_split_entries,
    resolve_image_and_mask,
    resolve_sample_file,
    resolve_split_file,
)

__all__ = [
    "IRSTDEvalDataset",
    "SampleMeta",
    "SpatialTransform",
    "build_spatial_transform",
    "read_split_entries",
    "resolve_image_and_mask",
    "resolve_sample_file",
    "resolve_split_file",
    "restore_tensor_to_original",
    "sample_meta_from_batch",
]


def __getattr__(name: str):
    if name == "IRSTDEvalDataset":
        from .eval_dataset import IRSTDEvalDataset

        return IRSTDEvalDataset
    if name in {
        "SampleMeta",
        "SpatialTransform",
        "build_spatial_transform",
        "restore_tensor_to_original",
        "sample_meta_from_batch",
    }:
        from . import dataset_meta

        return getattr(dataset_meta, name)
    raise AttributeError(name)
