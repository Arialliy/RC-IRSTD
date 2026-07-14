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
from .dataset_identity import (
    build_dataset_record,
    dataset_identity,
    score_manifest_content_sha256,
    sha256_file,
)
from .label_manifest_artifacts import (
    label_manifest_content_sha256,
    load_label_mask,
    verify_label_attachment,
)

__all__ = [
    "IRSTDEvalDataset",
    "IRSTDInferenceDataset",
    "ImageSampleMeta",
    "SampleMeta",
    "SpatialTransform",
    "build_spatial_transform",
    "build_dataset_record",
    "dataset_identity",
    "read_split_entries",
    "resolve_image_and_mask",
    "resolve_sample_file",
    "resolve_split_file",
    "restore_tensor_to_original",
    "image_meta_from_batch",
    "label_manifest_content_sha256",
    "load_label_mask",
    "sample_meta_from_batch",
    "score_manifest_content_sha256",
    "sha256_file",
    "verify_label_attachment",
]


def __getattr__(name: str):
    if name == "IRSTDEvalDataset":
        from .eval_dataset import IRSTDEvalDataset

        return IRSTDEvalDataset
    if name == "IRSTDInferenceDataset":
        from .inference_dataset import IRSTDInferenceDataset

        return IRSTDInferenceDataset
    if name in {
        "ImageSampleMeta",
        "SampleMeta",
        "SpatialTransform",
        "build_spatial_transform",
        "restore_tensor_to_original",
        "image_meta_from_batch",
        "sample_meta_from_batch",
    }:
        from . import dataset_meta

        return getattr(dataset_meta, name)
    raise AttributeError(name)
