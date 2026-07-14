"""Export native-resolution sigmoid score maps from an MSHNet checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_ext.dataset_meta import (
    image_meta_from_batch,
    restore_tensor_to_original,
    safe_output_stem,
)
from data_ext.dataset_identity import (
    DATASET_RECORD_SCHEMA_VERSION,
    ORDERED_SAMPLE_IDS_ALGORITHM,
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    SPLIT_IMAGE_ARTIFACT_ALGORITHM,
    build_dataset_record,
    score_manifest_content_sha256,
    sha256_file,
    validate_dataset_record,
)
from data_ext.development_split_contract import (
    DETECTOR_DIAGNOSTIC_ROLE,
    DEVELOPMENT_SPLIT_CONTRACT_SCHEMA_VERSION,
    serialise_development_partition_contract,
    verify_detector_diagnostic_partition,
)
from data_ext.inference_dataset import IRSTDInferenceDataset
from data_ext.split_utils import (
    read_split_entries,
    resolve_split_file,
    sample_id_from_entry,
)
from model.MSHNet import MSHNet


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCORE_MANIFEST_SCHEMA_VERSION = 3
SPLIT_CONTRACT_SCHEMA_VERSION = 1
OFFICIAL_SPLIT_ROLES = ("official_train", "official_test")
SPLIT_ROLES = (*OFFICIAL_SPLIT_ROLES, DETECTOR_DIAGNOSTIC_ROLE)
RAW_LOGIT_ARTIFACT_SCHEMA_VERSION = 1
RAW_LOGIT_CONTENT_ALGORITHM = (
    "sha256-length-prefixed-image-raw-logit-dtype-shape-v1"
)
RAW_LOGIT_DTYPE = "float64"
RAW_LOGIT_SPACE = "native_original_hw_spatially_aligned_restored_model_logit"
RAW_LOGIT_SCORE_RELATION = (
    "score = restore(float64_sigmoid(model_grid_logit)); raw_logit = "
    "restore(float64(model_grid_logit)); sigmoid(raw_logit) is "
    "diagnostic_only_not_pointwise_equal"
)


def select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device != "auto":
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalise_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Accept repository weights, checkpoints and DataParallel prefixes."""

    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "net" in checkpoint:
            state_dict = checkpoint["net"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    if not state_dict:
        raise ValueError("Checkpoint state_dict is empty")
    normalised = {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in normalised.values()):
        raise TypeError("Checkpoint mapping contains non-tensor model parameters")
    return normalised


def load_model(
    weight_path: str | Path,
    device: torch.device,
    *,
    checkpoint: object | None = None,
) -> torch.nn.Module:
    model = MSHNet(3)
    if checkpoint is None:
        checkpoint = torch.load(
            Path(weight_path).expanduser().resolve(),
            map_location=device,
        )
    model.load_state_dict(normalise_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model


def checkpoint_provenance(checkpoint: object) -> dict[str, object]:
    """Extract detector-fold metadata without trusting a CLI label.

    Historical paper weights contain tensors only.  They remain usable for
    diagnostic reproduction but are explicitly marked unverified; RC protocol
    artifacts must use a training checkpoint carrying source-domain metadata.
    """

    if not isinstance(checkpoint, Mapping):
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "checkpoint_is_not_a_metadata_mapping",
        }
    raw_sources = checkpoint.get(
        "detector_source_domains",
        checkpoint.get("source_names"),
    )
    if raw_sources is None:
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "missing_detector_source_domains",
        }
    if isinstance(raw_sources, (str, bytes)):
        sources = [str(raw_sources)]
    else:
        sources = [str(value) for value in raw_sources]
    if not sources or any(not value for value in sources) or len(set(sources)) != len(sources):
        raise ValueError("checkpoint detector source domains are empty or duplicated")
    raw_held_out = checkpoint.get("held_out_domains", [])
    if not isinstance(raw_held_out, (list, tuple)):
        raise TypeError("checkpoint held_out_domains must be an ordered list")
    held_out = [str(value) for value in raw_held_out]
    if len(set(held_out)) != len(held_out):
        raise ValueError("checkpoint held_out_domains contain duplicates")
    overlap = sorted(set(sources) & set(held_out))
    if overlap:
        raise ValueError(
            "checkpoint source/held-out domain metadata overlap: " + ", ".join(overlap)
        )
    outer_target = checkpoint.get("outer_target")
    if outer_target is not None and str(outer_target) in sources:
        raise ValueError("checkpoint outer_target occurs in detector source domains")
    common: dict[str, object] = {
        "detector_source_domains": sources,
        "held_out_domains": held_out,
        "outer_fold_id": checkpoint.get("outer_fold_id"),
        "outer_target": outer_target,
        "checkpoint_selection": checkpoint.get("checkpoint_selection"),
        "protocol_scope": checkpoint.get("protocol_scope"),
        "training_seed": checkpoint.get("seed"),
        "run_config_sha256": checkpoint.get("run_config_sha256"),
    }
    if common["run_config_sha256"] is not None:
        digest = str(common["run_config_sha256"]).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("checkpoint run_config_sha256 is not a valid SHA-256")
        common["run_config_sha256"] = digest
    raw_records = checkpoint.get("detector_source_records")
    if raw_records is None:
        # Keep historical metadata visible for diagnostics, but old
        # source-name-only checkpoints are not eligible for the main protocol.
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "missing_detector_source_records",
            **common,
        }
    if not isinstance(raw_records, (list, tuple)):
        raise TypeError("checkpoint detector_source_records must be an ordered list")
    if len(raw_records) != len(sources):
        raise ValueError(
            "checkpoint detector_source_records must align one-to-one with "
            "detector_source_domains"
        )
    # Records from before the current raster-filtered content schema lack its
    # complete identity contract. They may be exported diagnostically but can
    # never establish main-protocol target exclusion.
    if any(
        not isinstance(record, Mapping)
        or int(record.get("record_schema_version", -1))
        != DATASET_RECORD_SCHEMA_VERSION
        for record in raw_records
    ):
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": (
                "detector_source_records_precede_supported_raster_schema_v3"
            ),
            **common,
        }
    records = [
        validate_dataset_record(
            record,
            require_source_name=True,
            require_training_artifact=True,
        )
        for record in raw_records
    ]
    record_names = [str(record["source_name"]) for record in records]
    if record_names != sources:
        raise ValueError(
            "checkpoint detector source record order/names do not match "
            "detector_source_domains"
        )
    identities = [str(record["dataset_identity_sha256"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "checkpoint detector source records contain duplicate dataset identities"
        )
    dataset_sizes = checkpoint.get("dataset_sizes")
    if dataset_sizes is not None:
        if not isinstance(dataset_sizes, Mapping):
            raise TypeError("checkpoint dataset_sizes must be a mapping")
        for record in records:
            name = str(record["source_name"])
            if name in dataset_sizes and int(dataset_sizes[name]) != int(
                record["num_samples"]
            ):
                raise ValueError(
                    f"checkpoint dataset size disagrees with source record {name!r}"
                )
    if common["checkpoint_selection"] != "fixed_last_no_test_or_target_validation":
        raise ValueError(
            "checkpoint_verified requires checkpoint_selection="
            "'fixed_last_no_test_or_target_validation'"
        )
    if common["protocol_scope"] not in {
        "multi_source_protocol_candidate",
        "single_source_inner_smoke_not_main_result",
    }:
        raise ValueError(
            "checkpoint_verified requires a known detector protocol_scope"
        )
    expected_scope = (
        "single_source_inner_smoke_not_main_result"
        if len(sources) == 1
        else "multi_source_protocol_candidate"
    )
    if common["protocol_scope"] != expected_scope:
        raise ValueError(
            "checkpoint detector source count is inconsistent with protocol_scope"
        )
    outer_fold_id = common["outer_fold_id"]
    outer_target = common["outer_target"]
    if not isinstance(outer_fold_id, str) or not outer_fold_id.strip():
        raise ValueError("checkpoint_verified requires a non-empty outer_fold_id")
    if not isinstance(outer_target, str) or not outer_target.strip():
        raise ValueError("checkpoint_verified requires a non-empty outer_target")
    if outer_target not in held_out:
        raise ValueError(
            "checkpoint_verified requires outer_target in held_out_domains"
        )
    return {
        "provenance_level": "checkpoint_verified",
        **common,
        "detector_source_records": records,
    }


def extract_logits(model_output: object) -> torch.Tensor:
    """Extract final logits from the current ``(aux, logits)`` model output."""

    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        candidate = model_output[-1]
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError("Model output does not contain a logits tensor")


def high_precision_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """Compute probabilities in float64 to preserve extreme-tail ordering.

    Detector logits are usually float32.  Casting after sigmoid cannot recover
    distinct large logits that already saturated to the same float32 value,
    which is material at per-million-pixel false-alarm budgets.
    """

    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point torch.Tensor")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits contain NaN or infinity")
    return torch.sigmoid(logits.to(dtype=torch.float64))


def raw_logit_manifest_content_sha256(
    items: Sequence[Mapping[str, object]],
) -> str:
    """Bind every optional raw-logit artifact to its ordered sample ID."""

    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("raw-logit manifest items must be a non-empty ordered list")
    digest = hashlib.sha256()
    _update_hash_frame(digest, RAW_LOGIT_CONTENT_ALGORITHM)
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"raw-logit manifest item {index} must be a mapping")
        image_id = str(item.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(f"raw-logit manifest item {index} has an empty image_id")
        if image_id in seen:
            raise ValueError(f"duplicate raw-logit image_id: {image_id!r}")
        seen.add(image_id)
        file_value = str(item.get("raw_logit_file", "")).strip()
        if not file_value or Path(file_value).expanduser().is_absolute():
            raise ValueError(
                f"items[{index}].raw_logit_file must be a relative path"
            )
        raw_sha = str(item.get("raw_logit_file_sha256", "")).strip().lower()
        if len(raw_sha) != 64 or any(
            character not in "0123456789abcdef" for character in raw_sha
        ):
            raise ValueError(
                f"items[{index}].raw_logit_file_sha256 is not a SHA-256"
            )
        dtype = str(item.get("raw_logit_dtype", "")).strip()
        if dtype != RAW_LOGIT_DTYPE:
            raise ValueError(
                f"items[{index}].raw_logit_dtype must equal {RAW_LOGIT_DTYPE!r}"
            )
        shape = item.get("raw_logit_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"items[{index}].raw_logit_shape must be [height, width]")
        dimensions: list[int] = []
        for raw_value in shape:
            if isinstance(raw_value, (bool, np.bool_)):
                raise TypeError("raw-logit shape values must be positive integers")
            value = int(raw_value)
            if float(raw_value) != float(value) or value <= 0:
                raise ValueError("raw-logit shape values must be positive integers")
            dimensions.append(value)
        for value in (
            image_id,
            file_value,
            raw_sha,
            dtype,
            str(dimensions[0]),
            str(dimensions[1]),
        ):
            _update_hash_frame(digest, value)
    return digest.hexdigest()


def _update_hash_frame(digest: "hashlib._Hash", value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _stage1_raw_logit_provenance(
    checkpoint: object,
    detector_provenance: Mapping[str, object],
    target_dataset_record: Mapping[str, object],
    split_contract: Mapping[str, object],
    *,
    dataset_name: str,
) -> dict[str, object]:
    """Fail closed unless raw logits belong to a verified development split."""

    if split_contract.get("role") != DETECTOR_DIAGNOSTIC_ROLE:
        raise ValueError(
            "raw-logit export is development-only and requires "
            "split_role='detector_diagnostic'"
        )
    if detector_provenance.get("provenance_level") != "checkpoint_verified":
        raise ValueError(
            "raw-logit export requires checkpoint_verified detector provenance"
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("raw-logit export requires a metadata checkpoint mapping")
    risk_contract = checkpoint.get("risk_objective_contract")
    if not isinstance(risk_contract, Mapping):
        raise ValueError(
            "raw-logit export requires checkpoint risk_objective_contract"
        )
    stage1_variant = str(risk_contract.get("stage1_variant", "")).strip()
    if stage1_variant not in {"D0", "D3"}:
        raise ValueError("raw-logit export accepts only Stage-1 variants D0 or D3")
    segmentation_loss = checkpoint.get("segmentation_loss_implementation")
    if not isinstance(segmentation_loss, Mapping):
        raise ValueError(
            "raw-logit export requires checkpoint segmentation_loss_implementation"
        )

    sources = [
        str(value) for value in detector_provenance.get("detector_source_domains", [])
    ]
    source_records = detector_provenance.get("detector_source_records")
    if not sources or not isinstance(source_records, list):
        raise ValueError("raw-logit export requires verified detector source records")

    partition_audit: dict[str, object] | None = None
    if dataset_name in sources:
        matching_records = [
            record
            for record in source_records
            if isinstance(record, Mapping)
            and str(record.get("source_name")) == dataset_name
        ]
        if len(matching_records) != 1:
            raise ValueError(
                "raw-logit same-domain diagnostic requires exactly one matching "
                "detector source record"
            )
        source_record = matching_records[0]
        if (
            source_record.get("dataset_identity_sha256")
            != target_dataset_record.get("dataset_identity_sha256")
        ):
            raise ValueError(
                "raw-logit same-domain diagnostic dataset identity mismatch"
            )
        training_items = source_record.get("training_artifact_items")
        diagnostic_items = target_dataset_record.get("split_image_artifact_items")
        if not isinstance(training_items, list) or not isinstance(
            diagnostic_items, list
        ):
            raise ValueError(
                "raw-logit partition audit requires training and diagnostic items"
            )
        training_ids = {str(item["sample_id"]) for item in training_items}
        diagnostic_ids = {str(item["sample_id"]) for item in diagnostic_items}
        training_hashes = {str(item["image_sha256"]) for item in training_items}
        diagnostic_hashes = {str(item["image_sha256"]) for item in diagnostic_items}
        id_overlap = sorted(training_ids & diagnostic_ids)
        image_overlap = sorted(training_hashes & diagnostic_hashes)
        if id_overlap or image_overlap:
            raise ValueError(
                "raw-logit detector-fit/diagnostic partition overlap: "
                f"IDs={len(id_overlap)}, image hashes={len(image_overlap)}"
            )
        proof_mode = "verified_same_dataset_detector_fit_disjointness"
        partition_audit = {
            "source_name": dataset_name,
            "detector_fit_count": len(training_items),
            "detector_diagnostic_count": len(diagnostic_items),
            "sample_id_overlap_count": 0,
            "image_content_overlap_count": 0,
            "disjointness_verified": True,
        }
    else:
        if detector_provenance.get("target_exclusion_verified") is not True:
            raise ValueError(
                "raw-logit held-out-domain diagnostic requires verified target exclusion"
            )
        proof_mode = "verified_held_out_domain_exclusion"

    return {
        "schema_version": RAW_LOGIT_ARTIFACT_SCHEMA_VERSION,
        "status": "verified",
        "eligible": True,
        "proof_mode": proof_mode,
        "split_role": DETECTOR_DIAGNOSTIC_ROLE,
        "development_only": True,
        "official_test_scores_consumed": False,
        "checkpoint_provenance_level": "checkpoint_verified",
        "stage1_variant": stage1_variant,
        "training_seed": detector_provenance.get("training_seed"),
        "risk_objective_contract": dict(risk_contract),
        "segmentation_loss_implementation": dict(segmentation_loss),
        "same_dataset_partition_audit": partition_audit,
    }


def build_official_split_contract(
    dataset: IRSTDInferenceDataset,
    *,
    split_role: str,
    manifest_root: str | Path,
    derived_split_manifest: str | Path | None = None,
    derived_split_manifest_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Freeze and audit an official or official-train-derived score split.

    ``split_role`` is deliberately independent of the historical ``split``
    argument.  A caller exporting training scores must opt in with
    ``official_train``; leaving the exporter defaults unchanged always creates
    an ``official_test`` artifact that a calibration consumer must reject.

    The returned dataset record is the record for the selected role.  Both
    official records are built from the concrete split files and raw image
    bytes, then reduced to the fields needed to replay the disjointness audit
    without embedding labels. ``detector_diagnostic`` is accepted only when
    ``derived_split_manifest`` names the frozen v2 manifest and
    ``dataset.split_file`` is the exact path declared there. Matching bytes
    copied to another path are insufficient.
    """

    role = str(split_role).strip()
    if role not in SPLIT_ROLES:
        raise ValueError(
            "split_role must be one of " + ", ".join(SPLIT_ROLES)
        )
    if role == DETECTOR_DIAGNOSTIC_ROLE and (
        derived_split_manifest is None or derived_split_manifest_sha256 is None
    ):
        raise ValueError(
            "split_role='detector_diagnostic' requires derived_split_manifest "
            "and derived_split_manifest_sha256"
        )
    if role in OFFICIAL_SPLIT_ROLES and (
        derived_split_manifest is not None
        or derived_split_manifest_sha256 is not None
    ):
        raise ValueError(
            "derived_split_manifest is valid only for "
            "split_role='detector_diagnostic'"
        )

    split_records: dict[str, dict[str, object]] = {}
    split_paths: dict[str, Path] = {}
    for official_role, split_name in (
        ("official_train", "train"),
        ("official_test", "test"),
    ):
        split_path = resolve_split_file(dataset.root, split_name).resolve()
        sample_ids = [
            sample_id_from_entry(entry) for entry in read_split_entries(split_path)
        ]
        split_paths[official_role] = split_path
        split_records[official_role] = validate_dataset_record(
            build_dataset_record(dataset.root, split_path, sample_ids),
            require_source_name=False,
            require_training_artifact=False,
        )

    train_record = split_records["official_train"]
    test_record = split_records["official_test"]
    if train_record["dataset_identity_sha256"] != test_record["dataset_identity_sha256"]:
        raise RuntimeError(
            "dataset image content changed while official train/test records "
            "were being frozen"
        )
    for official_role, split_path in split_paths.items():
        record = split_records[official_role]
        if sha256_file(split_path) != record["split_sha256"]:
            raise RuntimeError(
                f"official split changed while protocol contract was built: {split_path}"
            )

    selected_path = Path(dataset.split_file).expanduser().resolve()
    development_partition = None
    if role in OFFICIAL_SPLIT_ROLES:
        expected_selected_path = split_paths[role]
        if selected_path != expected_selected_path:
            raise ValueError(
                f"split_role={role!r} requires the concrete official split file "
                f"{expected_selected_path}, but the selected split resolves to "
                f"{selected_path}. Calibration exports must explicitly use "
                "split='train' together with split_role='official_train'."
            )
        selected_record = split_records[role]
    else:
        development_partition = verify_detector_diagnostic_partition(
            derived_split_manifest,
            dataset_name=dataset.dataset_name,
            dataset_root=dataset.root,
            selected_split_file=selected_path,
            official_train_split=split_paths["official_train"],
            official_test_split=split_paths["official_test"],
            expected_manifest_sha256=derived_split_manifest_sha256,
        )
        diagnostic_ids = list(
            development_partition.partitions[
                DETECTOR_DIAGNOSTIC_ROLE
            ].sample_ids
        )
        selected_record = validate_dataset_record(
            build_dataset_record(dataset.root, selected_path, diagnostic_ids),
            require_source_name=False,
            require_training_artifact=False,
        )
    selected_ids = [str(sample[0]) for sample in dataset.samples]
    selected_record_ids = [
        str(item["sample_id"])
        for item in selected_record["split_image_artifact_items"]
    ]
    if selected_ids != selected_record_ids:
        raise RuntimeError(
            "inference dataset order differs from the selected official split record"
        )

    train_items = list(train_record["split_image_artifact_items"])
    test_items = list(test_record["split_image_artifact_items"])
    train_ids = {str(item["sample_id"]) for item in train_items}
    test_ids = {str(item["sample_id"]) for item in test_items}
    train_image_hashes = {str(item["image_sha256"]) for item in train_items}
    test_image_hashes = {str(item["image_sha256"]) for item in test_items}
    id_overlap = sorted(train_ids & test_ids)
    image_overlap = sorted(train_image_hashes & test_image_hashes)
    if id_overlap or image_overlap:
        details = []
        if id_overlap:
            details.append(f"sample-ID overlap={len(id_overlap)}")
        if image_overlap:
            details.append(f"raw-image-content overlap={len(image_overlap)}")
        raise ValueError(
            "official train/test splits are not disjoint ("
            + ", ".join(details)
            + "); score export is ineligible for calibration or final evaluation"
        )

    contract: dict[str, object] = {
        "schema_version": (
            SPLIT_CONTRACT_SCHEMA_VERSION
            if role in OFFICIAL_SPLIT_ROLES
            else DEVELOPMENT_SPLIT_CONTRACT_SCHEMA_VERSION
        ),
        "role": role,
        "selected_split_file": _portable_path(selected_path, manifest_root),
        "selected_split_sha256": selected_record["split_sha256"],
        "selected_num_images": selected_record["num_samples"],
        "selected_ids_sha256": selected_record["ordered_sample_ids_sha256"],
        "ordered_sample_ids_algorithm": ORDERED_SAMPLE_IDS_ALGORITHM,
        "split_image_artifact_algorithm": SPLIT_IMAGE_ARTIFACT_ALGORITHM,
        "train_test_id_overlap_count": len(id_overlap),
        "train_test_id_overlap_ids": id_overlap,
        "train_test_image_content_overlap_count": len(image_overlap),
        "train_test_image_content_overlap_sha256_leaves": image_overlap,
        "disjointness_verified": True,
    }
    for official_role, record in split_records.items():
        contract.update(
            {
                f"{official_role}_split_file": _portable_path(
                    split_paths[official_role], manifest_root
                ),
                f"{official_role}_split_sha256": record["split_sha256"],
                f"{official_role}_num_images": record["num_samples"],
                f"{official_role}_ids_sha256": record[
                    "ordered_sample_ids_sha256"
                ],
                f"{official_role}_split_image_artifact_sha256": record[
                    "split_image_artifact_sha256"
                ],
                f"{official_role}_split_image_artifact_items": record[
                    "split_image_artifact_items"
                ],
            }
        )
    if development_partition is not None:
        contract.update(
            serialise_development_partition_contract(
                development_partition,
                path_anchor=manifest_root,
            )
        )
    return contract, selected_record


def export_score_maps(
    *,
    dataset_dir: str | Path,
    weight_path: str | Path,
    output_dir: str | Path,
    base_size: int | Sequence[int] = 256,
    resize_mode: str = "resize",
    split: str = "test",
    split_file: str | Path | None = None,
    split_role: str = "official_test",
    derived_split_manifest: str | Path | None = None,
    derived_split_manifest_sha256: str | None = None,
    source_dataset: str | None = None,
    device: str = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    save_raw_logits: bool = False,
) -> dict[str, object]:
    """Run inference and write one native-resolution ``.npz`` per image.

    ``save_raw_logits`` is an opt-in, development-only diagnostic.  In that
    mode logits are independently restored to the same native spatial frame
    as the score. The score itself always retains the historical
    sigmoid-then-restore path; enabling diagnostics therefore cannot change a
    gate metric. Because interpolation and sigmoid do not commute,
    ``sigmoid(raw_logit)`` is diagnostic-only and is not claimed to reproduce
    the saved score pointwise.
    """

    input_hw = (
        (int(base_size), int(base_size))
        if isinstance(base_size, (int, np.integer))
        else tuple(int(value) for value in base_size)
    )
    if len(input_hw) != 2 or any(value <= 0 or value % 16 != 0 for value in input_hw):
        raise ValueError(
            "MSHNet input height and width must be positive multiples of 16"
        )
    if save_raw_logits and split_role != DETECTOR_DIAGNOSTIC_ROLE:
        raise ValueError(
            "save_raw_logits is development-only and requires "
            "split_role='detector_diagnostic'"
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    incomplete_marker = output_root / ".export_incomplete"
    if incomplete_marker.exists() and not overwrite:
        raise RuntimeError(
            f"A previous score export did not finish under {output_root}; "
            "rerun with --overwrite after checking the directory."
        )
    existing = list(output_root.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains {len(existing)} score maps: "
            f"{output_root}. Pass overwrite=True to replace matching files."
        )
    raw_logit_root = output_root / "raw_logits"
    existing_raw_logits = (
        list(raw_logit_root.glob("*.npy")) if raw_logit_root.is_dir() else []
    )
    if save_raw_logits and existing_raw_logits and not overwrite:
        raise FileExistsError(
            f"Output directory already contains {len(existing_raw_logits)} raw "
            f"logit maps: {raw_logit_root}. Pass overwrite=True to replace "
            "matching files."
        )
    incomplete_marker.write_text(
        "Score-map export is in progress. Do not sweep this directory.\n",
        encoding="utf-8",
    )

    dataset = IRSTDInferenceDataset(
        dataset_dir,
        base_size=base_size,
        resize_mode=resize_mode,
        split=split,
        split_file=split_file,
    )
    split_contract, target_dataset_record = build_official_split_contract(
        dataset,
        split_role=split_role,
        manifest_root=output_root,
        derived_split_manifest=derived_split_manifest,
        derived_split_manifest_sha256=derived_split_manifest_sha256,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu" and torch.cuda.is_available()),
    )
    selected_device = select_device(device)
    weight = Path(weight_path).expanduser().resolve()
    checkpoint = torch.load(weight, map_location=selected_device)
    detector_provenance = checkpoint_provenance(checkpoint)
    detector_provenance["target_exclusion_verified"] = False
    if detector_provenance["provenance_level"] == "checkpoint_verified":
        source_domains = set(detector_provenance["detector_source_domains"])
        held_out_domains = set(detector_provenance["held_out_domains"])
        logical_exclusion_verified = (
            dataset.dataset_name in held_out_domains
            and dataset.dataset_name not in source_domains
        )
        target_leaves = set(target_dataset_record["image_content_sha256_leaves"])
        source_union: set[str] = set()
        per_source_collisions: list[dict[str, object]] = []
        for source_record in detector_provenance["detector_source_records"]:
            source_leaves = set(source_record["image_content_sha256_leaves"])
            source_union.update(source_leaves)
            collisions = sorted(target_leaves & source_leaves)
            per_source_collisions.append(
                {
                    "source_name": source_record["source_name"],
                    "source_leaf_count": len(source_leaves),
                    "collision_count": len(collisions),
                    "collision_sha256_leaves": collisions,
                }
            )
        all_collisions = sorted(target_leaves & source_union)
        identity_exclusion_verified = not all_collisions
        detector_provenance["target_identity_collision_audit"] = {
            "comparison": "raw_image_file_sha256_leaf_intersection",
            "target_leaf_count": len(target_leaves),
            "detector_source_union_leaf_count": len(source_union),
            "collision_count": len(all_collisions),
            "collision_sha256_leaves": all_collisions,
            "per_source": per_source_collisions,
        }
        detector_provenance["logical_target_exclusion_verified"] = (
            logical_exclusion_verified
        )
        detector_provenance["target_identity_exclusion_verified"] = (
            identity_exclusion_verified
        )
        detector_provenance["target_exclusion_verified"] = (
            logical_exclusion_verified and identity_exclusion_verified
        )
    raw_logit_provenance = None
    if save_raw_logits:
        raw_logit_provenance = _stage1_raw_logit_provenance(
            checkpoint,
            detector_provenance,
            target_dataset_record,
            split_contract,
            dataset_name=dataset.dataset_name,
        )
        raw_logit_root.mkdir(parents=True, exist_ok=True)
    model = load_model(weight_path, selected_device, checkpoint=checkpoint)

    items: list[dict[str, object]] = []
    used_output_names: set[str] = set()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Exporting native score maps"):
            if "mask" in batch:
                raise RuntimeError("label-free inference batch unexpectedly contains a mask")
            metadata = image_meta_from_batch(batch["meta"], 0)
            image = batch["image"].to(selected_device, non_blocking=True)
            logits = extract_logits(model(image, True))
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
                raise ValueError(f"Expected logits [1, 1, H, W], got {tuple(logits.shape)}")
            probability = high_precision_sigmoid(logits[0, 0])
            probability = restore_tensor_to_original(
                probability,
                metadata.transform,
                mode="bilinear",
            )
            raw_logit_array = None
            if save_raw_logits:
                raw_logit = restore_tensor_to_original(
                    logits[0, 0].to(dtype=torch.float64),
                    metadata.transform,
                    mode="bilinear",
                )
                if not bool(torch.isfinite(raw_logit).all().item()):
                    raise ValueError(
                        f"Restored logits contain NaN/Inf for {metadata.image_id!r}"
                    )
                raw_logit_array = raw_logit.detach().cpu().numpy().astype(
                    np.float64,
                    copy=False,
                )
            probability_array = (
                probability.clamp_(0.0, 1.0).detach().cpu().numpy().astype(np.float64)
            )
            if probability_array.shape != metadata.transform.original_hw:
                raise RuntimeError(
                    f"Restored score/image mismatch for {metadata.image_id!r}: "
                    f"{probability_array.shape} vs {metadata.transform.original_hw}"
                )
            if raw_logit_array is not None and (
                raw_logit_array.shape != metadata.transform.original_hw
            ):
                raise RuntimeError(
                    f"Restored raw-logit/image mismatch for {metadata.image_id!r}: "
                    f"{raw_logit_array.shape} vs {metadata.transform.original_hw}"
                )

            output_name = f"{safe_output_stem(metadata.image_id)}.npz"
            if output_name in used_output_names:
                raise RuntimeError(
                    f"Two sample IDs map to the same output filename: {output_name}"
                )
            used_output_names.add(output_name)
            output_path = output_root / output_name
            _write_npz_atomic(
                output_path,
                prob=probability_array,
                image_id=np.asarray(metadata.image_id),
                dataset_name=np.asarray(metadata.dataset_name),
                original_hw=np.asarray(metadata.transform.original_hw, dtype=np.int32),
                input_hw=np.asarray(metadata.transform.input_hw, dtype=np.int32),
                resized_hw=np.asarray(metadata.transform.resized_hw, dtype=np.int32),
                padding_ltrb=np.asarray(
                    metadata.transform.padding_ltrb,
                    dtype=np.int32,
                ),
                resize_mode=np.asarray(metadata.transform.mode),
            )
            score_file_sha256 = sha256_file(output_path)
            gray_file_sha256 = sha256_file(metadata.image_path)
            item: dict[str, object] = {
                "image_id": metadata.image_id,
                "file": output_name,
                "score_file_sha256": score_file_sha256,
                "image_path": _portable_path(metadata.image_path, output_root),
                "gray_file_sha256": gray_file_sha256,
                "original_hw": list(metadata.transform.original_hw),
            }
            if raw_logit_array is not None:
                raw_logit_name = f"{safe_output_stem(metadata.image_id)}.npy"
                raw_logit_path = raw_logit_root / raw_logit_name
                _write_npy_atomic(raw_logit_path, raw_logit_array)
                item.update(
                    {
                        "raw_logit_file": (
                            Path("raw_logits") / raw_logit_name
                        ).as_posix(),
                        "raw_logit_file_sha256": sha256_file(raw_logit_path),
                        "raw_logit_dtype": str(raw_logit_array.dtype),
                        "raw_logit_shape": list(raw_logit_array.shape),
                    }
                )
            items.append(item)

    verified_split_role = str(split_contract["role"])
    manifest: dict[str, object] = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "label_free_score_export",
        "source_dataset": source_dataset,
        "source_dataset_assertion": source_dataset,
        "source_dataset_assertion_authority": "informational_only",
        "target_dataset": dataset.dataset_name,
        "target_dataset_record": target_dataset_record,
        "split_contract": split_contract,
        "partition_scope": split_contract.get(
            "partition_scope",
            (
                "official_final_evaluation"
                if verified_split_role == "official_test"
                else "official_train_development_or_calibration"
            ),
        ),
        "official_test_artifact": verified_split_role == "official_test",
        "final_evaluation_eligible": verified_split_role == "official_test",
        "development_only": verified_split_role != "official_test",
        # A score export alone never establishes a paper claim.  In
        # particular, source_dataset is only an informational assertion and
        # cannot upgrade a development partition into a final artifact.
        "claim_bearing_final_evaluation": False,
        "path_anchor": "manifest_directory",
        "dataset_dir": _portable_path(dataset.root, output_root),
        "split_file": _portable_path(dataset.split_file, output_root),
        "weight_path": _portable_path(weight, output_root),
        "weight_sha256": sha256_file(weight),
        "detector_provenance": detector_provenance,
        "input_hw": list(dataset.input_hw),
        "resize_mode": resize_mode,
        "restored_to_original_hw": True,
        "score_type": "sigmoid_probability",
        "score_dtype": "float64",
        "sigmoid_compute_dtype": "float64",
        "extreme_tail_precision_verified": True,
        "extreme_tail_precision_contract": (
            "cast logits to float64 before sigmoid; restore probability in float64"
        ),
        "threshold_semantics": "prediction = probability > threshold",
        "labels_embedded": False,
        "label_contract": "external_label_attachment_manifest_required_offline",
        "num_images": len(items),
        "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": score_manifest_content_sha256(items),
        "items": items,
    }
    if save_raw_logits:
        if raw_logit_provenance is None:
            raise RuntimeError("raw-logit provenance was not constructed")
        manifest.update(
            {
                "raw_logits_exported": True,
                "raw_logit_artifact_schema_version": (
                    RAW_LOGIT_ARTIFACT_SCHEMA_VERSION
                ),
                "raw_logit_space": RAW_LOGIT_SPACE,
                "raw_logit_dtype": RAW_LOGIT_DTYPE,
                "raw_logit_score_relation": RAW_LOGIT_SCORE_RELATION,
                "raw_logit_content_sha256_algorithm": (
                    RAW_LOGIT_CONTENT_ALGORITHM
                ),
                "raw_logit_content_sha256": (
                    raw_logit_manifest_content_sha256(items)
                ),
                "raw_logit_provenance": raw_logit_provenance,
            }
        )
    # Duplicate the protocol-critical fields at the top level so every
    # downstream stage can hard-validate them without interpreting a free-form
    # nested provenance payload.  The full payload remains for audit detail.
    for field in (
        "provenance_level",
        "detector_source_domains",
        "held_out_domains",
        "outer_fold_id",
        "outer_target",
        "checkpoint_selection",
        "protocol_scope",
        "run_config_sha256",
        "detector_source_records",
        "logical_target_exclusion_verified",
        "target_identity_exclusion_verified",
        "target_identity_collision_audit",
        "target_exclusion_verified",
    ):
        if field in detector_provenance:
            manifest[field] = detector_provenance[field]
    _write_json_atomic(output_root / "manifest.json", manifest)
    incomplete_marker.unlink()
    return manifest


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _portable_path(path: str | Path, base: str | Path | None = None) -> str:
    """Render paths relative to the repository for portable manifests."""

    resolved = Path(path).expanduser().resolve()
    anchor = Path(base or REPOSITORY_ROOT).expanduser().resolve()
    return Path(os.path.relpath(resolved, start=anchor)).as_posix()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional non-square input size; overrides --base-size",
    )
    parser.add_argument("--resize-mode", choices=("resize", "letterbox"), default="resize")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-file")
    parser.add_argument(
        "--split-role",
        choices=SPLIT_ROLES,
        default="official_test",
        help=(
            "Protocol role for the selected split. Calibration requires the "
            "explicit pair --split train --split-role official_train; the safe "
            "default is final-evaluation-only official_test. The development "
            "role detector_diagnostic additionally requires --split-file and "
            "--derived-split-manifest."
        ),
    )
    parser.add_argument(
        "--derived-split-manifest",
        help=(
            "Frozen v2 official-train-derived split manifest; required only "
            "for --split-role detector_diagnostic."
        ),
    )
    parser.add_argument(
        "--derived-split-manifest-sha256",
        help=(
            "Pre-frozen SHA-256 of --derived-split-manifest; required for "
            "--split-role detector_diagnostic."
        ),
    )
    parser.add_argument("--source-dataset")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-raw-logits",
        action="store_true",
        help=(
            "Development-only: save native-resolution float64 raw logits and "
            "bind them into the manifest. Requires --split-role "
            "detector_diagnostic and verified Stage-1 D0/D3 provenance."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base_size: int | Sequence[int] = args.input_size or args.base_size
    manifest = export_score_maps(
        dataset_dir=args.dataset_dir,
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        base_size=base_size,
        resize_mode=args.resize_mode,
        split=args.split,
        split_file=args.split_file,
        split_role=args.split_role,
        derived_split_manifest=args.derived_split_manifest,
        derived_split_manifest_sha256=args.derived_split_manifest_sha256,
        source_dataset=args.source_dataset,
        device=args.device,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        save_raw_logits=args.save_raw_logits,
    )
    print(
        f"Exported {manifest['num_images']} native-resolution score maps to "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
