from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from data_ext.dataset_identity import (
    DATASET_IDENTITY_ALGORITHM,
    DATASET_IDENTITY_EXTENSIONS,
    DATASET_RECORD_SCHEMA_VERSION,
    IMAGE_CONTENT_LEAF_ALGORITHM,
    IMAGE_CONTENT_LEAF_SET_ALGORITHM,
    TRAINING_ARTIFACT_ALGORITHM,
    SPLIT_IMAGE_ARTIFACT_ALGORITHM,
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    build_dataset_record,
    dataset_identity,
    ordered_sample_ids_sha256,
    score_manifest_content_sha256,
    sha256_file,
    validate_dataset_record,
)
from data_ext.label_manifest_artifacts import (
    LABEL_MANIFEST_ARTIFACT_TYPE,
    LABEL_MANIFEST_CONTENT_ALGORITHM,
    LABEL_MANIFEST_SCHEMA_VERSION,
    label_manifest_content_sha256,
)
from data_ext.mask_alignment import align_mask_to_image
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from data_ext.split_utils import resolve_image_and_mask, resolve_split_file
from evaluation.budget_metrics import compute_budget_metrics
from evaluation.component_matching import match_components
from evaluation.evaluate_adapter_output import (
    evaluate_adapter_output,
    main as evaluate_adapter_main,
    summarise_adapter_evaluations,
)
from evaluation.operating_point import select_operating_point
from evaluation.threshold_sweep import (
    CURVE_SCHEMA_VERSION,
    ScoreMapRecord,
    THRESHOLD_GRID_VERSION,
    discover_score_maps,
    main as threshold_sweep_main,
    normalise_thresholds,
    read_curve_csv,
    sweep_thresholds,
    threshold_grid_metadata,
)

try:
    import torch
    from torch.utils.data import DataLoader

    from data_ext.dataset_meta import (
        build_spatial_transform,
        restore_tensor_to_original,
        sample_meta_from_batch,
    )
    from data_ext.eval_dataset import IRSTDEvalDataset
    from evaluation.export_score_maps import (
        checkpoint_provenance,
        export_score_maps,
        high_precision_sigmoid,
    )

    HAS_TORCH_STACK = True
except ModuleNotFoundError:
    HAS_TORCH_STACK = False


requires_torch = pytest.mark.skipif(
    not HAS_TORCH_STACK,
    reason="PyTorch/torchvision evaluation stack is not installed",
)


@requires_torch
def test_float64_sigmoid_preserves_extreme_logit_order() -> None:
    logits = torch.tensor([18.0, 19.0], dtype=torch.float32)
    saturated = torch.sigmoid(logits)
    precise = high_precision_sigmoid(logits)
    assert saturated[0] == saturated[1]
    assert precise.dtype == torch.float64
    assert precise[0] < precise[1] < 1.0


def _make_nuaa_style_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[1:3, 2:6] = 200
    mask = np.zeros((4, 8), dtype=np.uint8)
    mask[2, 4] = 255
    Image.fromarray(image).save(root / "images" / "Misc_1.png")
    Image.fromarray(mask).save(root / "masks" / "Misc_1_pixels0.png")
    (root / "img_idx" / "test_NUAA-SIRST.txt").write_text(
        "Misc_1.png\n",
        encoding="utf-8",
    )
    return root


def _add_disjoint_official_train_split(root: Path) -> None:
    with Image.open(root / "images" / "Misc_1.png") as image_file:
        train_image = np.asarray(image_file.convert("RGB")).copy()
    train_image[0, 0] = (17, 31, 47)
    Image.fromarray(train_image).save(root / "images" / "Misc_train.png")
    with Image.open(root / "masks" / "Misc_1_pixels0.png") as mask_file:
        mask = np.asarray(mask_file).copy()
    Image.fromarray(mask).save(root / "masks" / "Misc_train_pixels0.png")
    (root / "img_idx" / "train_NUAA-SIRST.txt").write_text(
        "Misc_train.png\n",
        encoding="utf-8",
    )


def _make_frozen_detector_diagnostic_partition(
    repository_root: Path,
) -> tuple[Path, Path, Path]:
    """Create a small v2 partition with a non-guessed manifest subdirectory."""

    dataset = repository_root / "datasets" / "NUAA-SIRST"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    (dataset / "img_idx").mkdir()
    image_ids = ("fit_a", "diagnostic_b", "quarantine_c", "test_d")
    for index, image_id in enumerate(image_ids, start=1):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[0, index] = (index * 31, index * 17, index * 13)
        mask = np.zeros((4, 8), dtype=np.uint8)
        mask[2, index] = 255
        Image.fromarray(image).save(dataset / "images" / f"{image_id}.png")
        Image.fromarray(mask).save(dataset / "masks" / f"{image_id}.png")

    train = dataset / "img_idx" / "train_NUAA-SIRST.txt"
    test = dataset / "img_idx" / "test_NUAA-SIRST.txt"
    train.write_text(
        "fit_a\ndiagnostic_b\nquarantine_c\n", encoding="utf-8"
    )
    test.write_text("test_d\n", encoding="utf-8")

    split_root = repository_root / "splits" / "aaai27_v2"
    # Intentionally not the dataset slug: the exporter must use the manifest
    # mapping and must not guess a directory from dataset_name.
    role_root = split_root / "manifest-selected-directory"
    role_root.mkdir(parents=True)
    role_payloads = {
        "effective_development_train.txt": "fit_a\ndiagnostic_b\n",
        "detector_fit.txt": "fit_a\n",
        "detector_diagnostic.txt": "diagnostic_b\n",
        "quarantined_official_train_ids.txt": "quarantine_c\n",
    }
    for name, content in role_payloads.items():
        (role_root / name).write_text(content, encoding="utf-8")

    manifest_path = split_root / "manifest.json"
    relative_role_root = role_root.relative_to(split_root)
    manifest = {
        "schema_version": "rc-irstd.aaai27-official-train-splits.v2",
        "artifact_type": "official_train_derived_role_splits",
        "role_contract": {
            "official_test_emitted": False,
            "official_test_labels_read_for_quarantine": False,
            "outer_target_official_train_used_for_detector_fit": False,
            "outer_target_detector_diagnostic_used_for_development_evaluation": True,
            "outer_target_diagnostic_selects_checkpoint": False,
            "detector_checkpoint_selection": "fixed_last",
            "detector_diagnostic_used_for_checkpoint_selection": False,
        },
        "datasets": [
            {
                "dataset_name": "NUAA-SIRST",
                "dataset_root": "datasets/NUAA-SIRST",
                "official_train_split": (
                    "datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
                ),
                "official_train_split_sha256": sha256_file(train),
                "official_train_count": 3,
                "official_test_split": (
                    "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
                ),
                "official_test_split_sha256": sha256_file(test),
                "official_test_count": 1,
                "official_train_test_id_overlap_count": 0,
                "detector": {
                    "fit_file": (
                        relative_role_root / "detector_fit.txt"
                    ).as_posix(),
                    "fit_sha256": sha256_file(role_root / "detector_fit.txt"),
                    "fit_count": 1,
                    "diagnostic_file": (
                        relative_role_root / "detector_diagnostic.txt"
                    ).as_posix(),
                    "diagnostic_sha256": sha256_file(
                        role_root / "detector_diagnostic.txt"
                    ),
                    "diagnostic_count": 1,
                },
                "development_quarantine": {
                    "effective_development_train_file": (
                        relative_role_root / "effective_development_train.txt"
                    ).as_posix(),
                    "effective_development_train_sha256": sha256_file(
                        role_root / "effective_development_train.txt"
                    ),
                    "effective_development_train_count": 2,
                    "quarantined_file": (
                        relative_role_root
                        / "quarantined_official_train_ids.txt"
                    ).as_posix(),
                    "quarantined_sha256": sha256_file(
                        role_root / "quarantined_official_train_ids.txt"
                    ),
                    "quarantined_count": 1,
                    "partition_of_official_train": True,
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, manifest_path, role_root / "detector_diagnostic.txt"


def _fake_source_record(source_name: str, identity_hex: str) -> dict[str, object]:
    leaf = identity_hex * 64
    mask_sha = "d" * 64

    def framed_digest(algorithm: str, values: list[str]) -> str:
        digest = hashlib.sha256()
        for value in [algorithm, *values]:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    return {
        "record_schema_version": DATASET_RECORD_SCHEMA_VERSION,
        "source_name": source_name,
        "dataset_identity_algorithm": DATASET_IDENTITY_ALGORITHM,
        "dataset_identity_sha256": identity_hex * 64,
        "dataset_num_files": 1,
        "dataset_num_bytes": 10,
        "dataset_identity_folders": ["images"],
        "dataset_identity_extensions": list(DATASET_IDENTITY_EXTENSIONS),
        "image_content_leaf_algorithm": IMAGE_CONTENT_LEAF_ALGORITHM,
        "image_content_sha256_leaves": [leaf],
        "image_content_leaf_set_algorithm": IMAGE_CONTENT_LEAF_SET_ALGORITHM,
        "image_content_leaf_set_sha256": framed_digest(
            IMAGE_CONTENT_LEAF_SET_ALGORITHM, [leaf]
        ),
        "split_sha256": "b" * 64,
        "ordered_sample_ids_algorithm": "sha256-length-prefixed-sample-ids-v1",
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256(["sample"]),
        "num_samples": 1,
        "split_image_artifact_algorithm": SPLIT_IMAGE_ARTIFACT_ALGORITHM,
        "split_image_artifact_sha256": framed_digest(
            SPLIT_IMAGE_ARTIFACT_ALGORITHM, ["sample", leaf]
        ),
        "split_image_artifact_items": [
            {"sample_id": "sample", "image_sha256": leaf}
        ],
        "training_artifact_algorithm": TRAINING_ARTIFACT_ALGORITHM,
        "training_artifact_sha256": framed_digest(
            TRAINING_ARTIFACT_ALGORITHM, ["sample", leaf, mask_sha]
        ),
        "training_artifact_num_samples": 1,
        "training_artifact_items": [
            {
                "sample_id": "sample",
                "image_sha256": leaf,
                "mask_sha256": mask_sha,
            }
        ],
    }


def test_split_and_nuaa_mask_resolution(tmp_path: Path) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    split = resolve_split_file(root, "test")
    assert split.name == "test_NUAA-SIRST.txt"
    image_path, mask_path = resolve_image_and_mask(root, "Misc_1.png")
    assert image_path.name == "Misc_1.png"
    assert mask_path.name == "Misc_1_pixels0.png"
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        assert image.size == mask.size == (8, 4)


def test_ambiguous_split_requires_explicit_path(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "img_idx").mkdir(parents=True)
    (root / "img_idx" / "test_a.txt").write_text("a\n", encoding="utf-8")
    (root / "img_idx" / "test_b.txt").write_text("b\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Multiple split files"):
        resolve_split_file(root, "test")


def test_dataset_identity_survives_directory_copy_and_detects_content_change(
    tmp_path: Path,
) -> None:
    original = _make_nuaa_style_dataset(tmp_path / "logical-A")
    (original / "images" / ".DS_Store").write_bytes(b"finder metadata")
    (original / "images" / "notes.json").write_text("{}\n", encoding="utf-8")
    copied = tmp_path / "renamed-copy"
    shutil.copytree(original, copied)

    original_identity = dataset_identity(original)
    copied_identity = dataset_identity(copied)
    assert (
        original_identity["dataset_identity_sha256"]
        == copied_identity["dataset_identity_sha256"]
    )
    assert original_identity["dataset_num_files"] == 1
    assert original_identity["dataset_identity_algorithm"] == (
        DATASET_IDENTITY_ALGORITHM
    )
    assert original_identity["dataset_identity_extensions"] == list(
        DATASET_IDENTITY_EXTENSIONS
    )

    (copied / "images" / ".DS_Store").write_bytes(b"changed finder metadata")
    (copied / "images" / "notes.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    assert (
        dataset_identity(copied)["dataset_identity_sha256"]
        == original_identity["dataset_identity_sha256"]
    )

    changed_mask = np.zeros((4, 8), dtype=np.uint8)
    changed_mask[0, 0] = 255
    Image.fromarray(changed_mask).save(copied / "masks" / "Misc_1_pixels0.png")
    assert (
        dataset_identity(copied)["dataset_identity_sha256"]
        == original_identity["dataset_identity_sha256"]
    )

    extra_raster = copied / "images" / "extra.PNG"
    extra_raster.write_bytes((copied / "images" / "Misc_1.png").read_bytes())
    assert dataset_identity(copied)["dataset_num_files"] == 2
    assert (
        dataset_identity(copied)["dataset_identity_sha256"]
        != original_identity["dataset_identity_sha256"]
    )
    extra_raster.unlink()

    changed = np.zeros((4, 8, 3), dtype=np.uint8)
    changed[0, 0] = 255
    Image.fromarray(changed).save(copied / "images" / "Misc_1.png")
    assert (
        dataset_identity(copied)["dataset_identity_sha256"]
        != original_identity["dataset_identity_sha256"]
    )


def test_dataset_record_binds_split_bytes_ordered_ids_and_count(tmp_path: Path) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "dataset")
    split = resolve_split_file(root, "test")
    record = build_dataset_record(
        root,
        split,
        ["Misc_1"],
        source_name="SOURCE",
        training_artifacts=[
            (
                root / "images" / "Misc_1.png",
                root / "masks" / "Misc_1_pixels0.png",
            )
        ],
    )
    assert record["source_name"] == "SOURCE"
    assert record["split_sha256"] == sha256_file(split)
    assert record["num_samples"] == 1
    with pytest.raises(ValueError, match="duplicates"):
        build_dataset_record(root, split, ["Misc_1", "Misc_1"])
    with pytest.raises(ValueError, match="selected split"):
        build_dataset_record(root, split, ["different-id"])


@requires_torch
def test_training_source_records_are_ordered_and_reject_dataset_aliases(
    tmp_path: Path,
) -> None:
    from scripts.train_multisource_tail import build_detector_source_records
    from utils.data import IRSTD_Dataset

    source_a = _make_nuaa_style_dataset(tmp_path / "physical-A")
    source_b = _make_nuaa_style_dataset(tmp_path / "physical-B")
    changed = np.zeros((4, 8, 3), dtype=np.uint8)
    changed[0, 0] = 255
    Image.fromarray(changed).save(source_b / "images" / "Misc_1.png")
    for root in (source_a, source_b):
        (root / "trainval.txt").write_text("Misc_1.png\n", encoding="utf-8")

    def training_dataset(root: Path) -> IRSTD_Dataset:
        return IRSTD_Dataset(
            SimpleNamespace(
                dataset_dir=str(root),
                split_file=None,
                base_size=16,
                crop_size=16,
            ),
            mode="train",
        )

    dataset_a = training_dataset(source_a)
    dataset_b = training_dataset(source_b)
    records = build_detector_source_records(
        ["A", "B"],
        {"A": dataset_a, "B": dataset_b},
    )
    assert [record["source_name"] for record in records] == ["A", "B"]
    assert [record["num_samples"] for record in records] == [1, 1]
    assert (
        records[0]["dataset_identity_sha256"]
        != records[1]["dataset_identity_sha256"]
    )
    with pytest.raises(ValueError, match="duplicate dataset content"):
        build_detector_source_records(
            ["A", "ALIAS"],
            {"A": dataset_a, "ALIAS": training_dataset(source_a)},
        )


@requires_torch
def test_formal_trainer_rejects_source_test_contamination(tmp_path: Path) -> None:
    from scripts.train_multisource_tail import audited_source_train_split

    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    contaminated = root / "trainval.txt"
    contaminated.write_text("Misc_1.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps the official test split"):
        audited_source_train_split(root, contaminated)


def test_score_manifest_content_hash_is_ordered_and_fail_closed() -> None:
    items = [
        {
            "image_id": "one",
            "score_file_sha256": "a" * 64,
            "gray_file_sha256": "b" * 64,
        },
        {
            "image_id": "two",
            "score_file_sha256": "c" * 64,
            "gray_file_sha256": "d" * 64,
        },
    ]
    digest = score_manifest_content_sha256(items)
    assert digest != score_manifest_content_sha256(list(reversed(items)))
    with pytest.raises(ValueError, match="duplicate"):
        score_manifest_content_sha256([items[0], dict(items[0])])


def test_source_record_training_images_must_belong_to_content_leaves() -> None:
    record = _fake_source_record("A", "1")
    alien = "e" * 64
    record["training_artifact_items"][0]["image_sha256"] = alien
    digest = hashlib.sha256()
    for value in [TRAINING_ARTIFACT_ALGORITHM, "sample", alien, "d" * 64]:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    record["training_artifact_sha256"] = digest.hexdigest()
    with pytest.raises(ValueError, match="leaf multiset"):
        validate_dataset_record(
            record,
            require_source_name=True,
            require_training_artifact=True,
        )


@pytest.mark.parametrize("resize_mode", ["resize", "letterbox"])
@requires_torch
def test_eval_dataset_default_collate_is_stable(
    tmp_path: Path,
    resize_mode: str,
) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    dataset = IRSTDEvalDataset(root, base_size=8, resize_mode=resize_mode)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    assert tuple(batch["image"].shape) == (1, 3, 8, 8)
    assert tuple(batch["mask"].shape) == (1, 1, 8, 8)
    meta = sample_meta_from_batch(batch["meta"], 0)
    assert meta.image_id == "Misc_1"
    assert meta.dataset_name == "NUAA-SIRST"
    assert meta.transform.original_hw == (4, 8)
    assert meta.transform.input_hw == (8, 8)
    assert tuple(dataset.load_original_mask(0).shape) == (1, 4, 8)
    if resize_mode == "letterbox":
        assert meta.transform.resized_hw == (4, 8)
        assert meta.transform.padding_ltrb == (0, 2, 0, 2)


@requires_torch
def test_nuaa_mismatched_mask_is_aligned_to_image_canvas(tmp_path: Path) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    oversized_mask = np.zeros((8, 16), dtype=np.uint8)
    oversized_mask[4, 8] = 255
    Image.fromarray(oversized_mask).save(root / "masks" / "Misc_1_pixels0.png")
    dataset = IRSTDEvalDataset(root, base_size=8, resize_mode="letterbox")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    meta = sample_meta_from_batch(batch["meta"], 0)
    assert meta.transform.original_hw == (4, 8)
    assert meta.mask_original_hw == (8, 16)
    assert tuple(dataset.load_original_mask(0).shape) == (1, 4, 8)


def test_misc_111_alignment_exactly_matches_basicirstd_nearest_rule() -> None:
    """Lock the published NUAA Misc_111 geometry to BasicIRSTD semantics."""

    image = Image.new("RGB", (325, 220))
    mask_array = np.zeros((400, 592), dtype=np.uint8)
    mask_array[73:89, 301:319] = 255
    mask_array[250, 120] = 255
    mask = Image.fromarray(mask_array)
    resampling = getattr(Image, "Resampling", Image).NEAREST

    expected = mask.resize(image.size, resampling)
    aligned = align_mask_to_image(mask, image, "Misc_111")

    assert aligned.size == image.size
    assert np.array_equal(np.asarray(aligned), np.asarray(expected))


def test_mask_alignment_rejects_true_aspect_ratio_mismatch() -> None:
    image = Image.new("RGB", (8, 4))
    mask = Image.new("L", (8, 8))
    with pytest.raises(ValueError, match="aspect-ratio mismatch"):
        align_mask_to_image(mask, image, "wrong_pair")


@requires_torch
def test_checkpoint_provenance_is_derived_from_checkpoint_metadata() -> None:
    provenance = checkpoint_provenance(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["A", "B"],
            "detector_source_records": [
                _fake_source_record("A", "1"),
                _fake_source_record("B", "2"),
            ],
            "held_out_domains": ["C"],
            "outer_fold_id": "outer-c",
            "outer_target": "C",
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
            "protocol_scope": "multi_source_protocol_candidate",
            "seed": 7,
        }
    )
    assert provenance["provenance_level"] == "checkpoint_verified"
    assert provenance["detector_source_domains"] == ["A", "B"]
    assert [
        record["source_name"] for record in provenance["detector_source_records"]
    ] == ["A", "B"]
    assert provenance["outer_target"] == "C"
    assert provenance["training_seed"] == 7


@requires_torch
def test_checkpoint_without_dataset_records_is_legacy_but_diagnostic() -> None:
    provenance = checkpoint_provenance(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["A"],
            "held_out_domains": ["B"],
        }
    )
    assert provenance["provenance_level"] == "legacy_unverified"
    assert provenance["legacy_reason"] == "missing_detector_source_records"
    assert provenance["detector_source_domains"] == ["A"]


@requires_torch
def test_checkpoint_with_pre_raster_filter_records_is_legacy() -> None:
    old_record = _fake_source_record("A", "1")
    old_record["record_schema_version"] = 2
    old_record.pop("dataset_identity_extensions")
    provenance = checkpoint_provenance(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["A"],
            "detector_source_records": [old_record],
            "held_out_domains": ["B"],
        }
    )
    assert provenance["provenance_level"] == "legacy_unverified"
    assert provenance["legacy_reason"] == (
        "detector_source_records_precede_supported_raster_schema_v3"
    )


@requires_torch
def test_checkpoint_verified_requires_outer_fold_identity() -> None:
    checkpoint = {
        "state_dict": {"unused": torch.zeros(1)},
        "detector_source_domains": ["A", "B"],
        "detector_source_records": [
            _fake_source_record("A", "1"),
            _fake_source_record("B", "2"),
        ],
        "held_out_domains": ["C"],
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "protocol_scope": "multi_source_protocol_candidate",
    }
    with pytest.raises(ValueError, match="outer_fold_id"):
        checkpoint_provenance(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_selection", "best_on_target", "checkpoint_selection"),
        ("protocol_scope", "untracked_scope", "protocol_scope"),
        (
            "protocol_scope",
            "multi_source_protocol_candidate",
            "source count is inconsistent",
        ),
    ],
)
@requires_torch
def test_checkpoint_verified_rejects_unknown_selection_or_scope(
    field: str,
    value: str,
    message: str,
) -> None:
    checkpoint = {
        "state_dict": {"unused": torch.zeros(1)},
        "detector_source_domains": ["A"],
        "detector_source_records": [_fake_source_record("A", "1")],
        "held_out_domains": ["B"],
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "protocol_scope": "single_source_inner_smoke_not_main_result",
    }
    checkpoint[field] = value
    with pytest.raises(ValueError, match=message):
        checkpoint_provenance(checkpoint)


@requires_torch
def test_export_manifest_binds_dataset_and_score_gray_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    exporter_module = importlib.import_module("evaluation.export_score_maps")
    target = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    source = _make_nuaa_style_dataset(tmp_path / "SOURCE-data")
    _add_disjoint_official_train_split(target)
    _add_disjoint_official_train_split(source)
    source_image = np.zeros((4, 8, 3), dtype=np.uint8)
    source_image[0, 0] = 255
    Image.fromarray(source_image).save(source / "images" / "Misc_1.png")
    source_train_image = np.zeros((4, 8, 3), dtype=np.uint8)
    source_train_image[0, 1] = 251
    Image.fromarray(source_train_image).save(source / "images" / "Misc_train.png")
    source_record = build_dataset_record(
        source,
        resolve_split_file(source, "test"),
        ["Misc_1"],
        source_name="SOURCE",
        training_artifacts=[
            (
                source / "images" / "Misc_1.png",
                source / "masks" / "Misc_1_pixels0.png",
            )
        ],
    )
    checkpoint = tmp_path / "detector.pt"
    torch.save(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["SOURCE"],
            "detector_source_records": [source_record],
            "held_out_domains": ["NUAA-SIRST"],
            "outer_fold_id": "outer-target",
            "outer_target": "NUAA-SIRST",
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
            "protocol_scope": "single_source_inner_smoke_not_main_result",
        },
        checkpoint,
    )

    class ZeroDetector(torch.nn.Module):
        def forward(self, image: torch.Tensor, warm_flag: bool) -> torch.Tensor:
            del warm_flag
            return torch.zeros(
                (image.shape[0], 1, image.shape[2], image.shape[3]),
                dtype=image.dtype,
                device=image.device,
            )

    monkeypatch.setattr(
        exporter_module,
        "load_model",
        lambda *args, **kwargs: ZeroDetector(),
    )
    output = tmp_path / "scores"
    manifest = export_score_maps(
        dataset_dir=target,
        weight_path=checkpoint,
        output_dir=output,
        base_size=16,
        device="cpu",
    )

    assert manifest["target_exclusion_verified"] is True
    assert manifest["target_identity_exclusion_verified"] is True
    assert manifest["target_dataset_record"]["num_samples"] == 1
    assert manifest["content_sha256_algorithm"] == SCORE_MANIFEST_CONTENT_ALGORITHM
    assert manifest["content_sha256"] == score_manifest_content_sha256(
        manifest["items"]
    )
    item = manifest["items"][0]
    assert item["score_file_sha256"] == sha256_file(output / item["file"])
    original_image = (output / item["image_path"]).resolve()
    assert original_image == (target / "images" / "Misc_1.png").resolve()
    assert item["gray_file_sha256"] == sha256_file(original_image)
    with np.load(output / item["file"], allow_pickle=False) as score_payload:
        assert "prob" in score_payload
        assert score_payload["prob"].dtype == np.float64
        assert "mask" not in score_payload
    assert manifest["score_dtype"] == "float64"
    assert manifest["sigmoid_compute_dtype"] == "float64"
    assert manifest["extreme_tail_precision_verified"] is True
    assert manifest["labels_embedded"] is False
    assert manifest["schema_version"] == 3
    split_contract = manifest["split_contract"]
    assert split_contract["role"] == "official_test"
    assert split_contract["selected_num_images"] == 1
    assert split_contract["official_train_num_images"] == 1
    assert split_contract["official_test_num_images"] == 1
    assert split_contract["train_test_id_overlap_count"] == 0
    assert split_contract["train_test_image_content_overlap_count"] == 0
    assert split_contract["disjointness_verified"] is True
    verified_test = verify_score_manifest_artifacts(
        output / "manifest.json",
        required_split_role="official_test",
    )
    assert verified_test.split_role == "official_test"
    assert verified_test.legacy_final_evaluation_only is False
    with pytest.raises(ValueError, match="cannot satisfy required role"):
        verify_score_manifest_artifacts(
            output / "manifest.json",
            required_split_role="official_train",
        )

    # Training score export requires an explicit role as well as split name.
    with pytest.raises(ValueError, match="split_role='official_train'"):
        export_score_maps(
            dataset_dir=target,
            weight_path=checkpoint,
            output_dir=tmp_path / "implicit-train-scores",
            base_size=16,
            split="train",
            device="cpu",
        )
    train_output = tmp_path / "train-scores"
    train_manifest = export_score_maps(
        dataset_dir=target,
        weight_path=checkpoint,
        output_dir=train_output,
        base_size=16,
        split="train",
        split_role="official_train",
        device="cpu",
    )
    assert train_manifest["split_contract"]["role"] == "official_train"
    assert train_manifest["items"][0]["image_id"] == "Misc_train"
    verified_train = verify_score_manifest_artifacts(
        train_output / "manifest.json",
        required_split_role="official_train",
    )
    assert verified_train.split_role == "official_train"

    # Role-less v2 remains readable only without a claim-bearing role gate.
    legacy_path = output / "legacy-manifest.json"
    legacy_payload = dict(manifest)
    legacy_payload["schema_version"] = 2
    legacy_payload.pop("split_contract")
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy = verify_score_manifest_artifacts(legacy_path)
    assert legacy.split_role is None
    assert legacy.legacy_final_evaluation_only is True
    with pytest.raises(ValueError, match="final-test-evaluation-only"):
        verify_score_manifest_artifacts(
            legacy_path,
            required_split_role="official_train",
        )

    tampered_path = output / "tampered-split-manifest.json"
    tampered_payload = json.loads(json.dumps(manifest))
    tampered_payload["split_contract"]["train_test_id_overlap_count"] = 1
    tampered_path.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sample-ID overlap audit mismatch"):
        verify_score_manifest_artifacts(
            tampered_path,
            required_split_role="official_test",
        )

    from data_ext.label_manifest_artifacts import (
        load_label_mask,
        verify_label_attachment,
    )
    from evaluation.export_label_maps import export_label_maps

    label_output = tmp_path / "labels"
    label_manifest = export_label_maps(
        dataset_dir=target,
        score_manifest=output / "manifest.json",
        output_dir=label_output,
    )
    attachment = verify_label_attachment(
        output / "manifest.json",
        label_output / "label-manifest.json",
    )
    assert label_manifest["score_manifest_sha256"] == sha256_file(
        output / "manifest.json"
    )
    assert len(attachment.selected_items) == 1
    assert load_label_mask(attachment.selected_items[0]).shape == (4, 8)
    with pytest.raises(ValueError, match="disjoint directory trees"):
        export_label_maps(
            dataset_dir=target,
            score_manifest=output / "manifest.json",
            output_dir=output / "labels",
        )
    wrong_root = tmp_path / "wrong-copy" / "NUAA-SIRST"
    shutil.copytree(target, wrong_root)
    wrong_image = np.zeros((4, 8, 3), dtype=np.uint8)
    wrong_image[0, 0] = 123
    Image.fromarray(wrong_image).save(wrong_root / "images" / "Misc_1.png")
    with pytest.raises(ValueError, match="wrong data root"):
        export_label_maps(
            dataset_dir=wrong_root,
            score_manifest=output / "manifest.json",
            output_dir=tmp_path / "wrong-labels",
        )
    curve_path = tmp_path / "attached-curve.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(output),
            "--label-manifest",
            str(label_output / "label-manifest.json"),
            "--output",
            str(curve_path),
        ]
    ) == 0
    curve_manifest = json.loads(
        curve_path.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert curve_manifest["evaluation_scope"] == (
        "score_bound_label_attachment_verified"
    )
    assert curve_manifest["score_split_role"] == "official_test"
    assert curve_manifest["final_evaluation_eligible"] is True
    assert curve_manifest["claim_bearing_final_evaluation"] is False
    assert curve_manifest["label_manifest_sha256"] == sha256_file(
        label_output / "label-manifest.json"
    )
    label_path = attachment.selected_items[0].label_path
    np.savez_compressed(
        label_path,
        mask=np.zeros((4, 8), dtype=np.uint8),
        image_id=np.asarray("Misc_1"),
        original_hw=np.asarray((4, 8), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="label-file SHA-256 mismatch"):
        verify_label_attachment(
            output / "manifest.json",
            label_output / "label-manifest.json",
        )

    # A renamed/logically held-out copy of a detector source is still caught
    # by the content identity and cannot claim target exclusion.
    target_record_as_source = build_dataset_record(
        target,
        resolve_split_file(target, "test"),
        ["Misc_1"],
        source_name="SOURCE",
        training_artifacts=[
            (
                target / "images" / "Misc_1.png",
                target / "masks" / "Misc_1_pixels0.png",
            )
        ],
    )
    collision_checkpoint = tmp_path / "collision.pt"
    collision_payload = torch.load(checkpoint, map_location="cpu")
    collision_payload["detector_source_records"] = [target_record_as_source]
    torch.save(collision_payload, collision_checkpoint)
    # Inference remains operational after the complete mask directory is
    # removed: score export must neither resolve nor open labels.
    shutil.rmtree(target / "masks")
    collision_manifest = export_score_maps(
        dataset_dir=target,
        weight_path=collision_checkpoint,
        output_dir=tmp_path / "collision-scores",
        base_size=16,
        device="cpu",
    )
    assert collision_manifest["logical_target_exclusion_verified"] is True
    assert collision_manifest["target_identity_exclusion_verified"] is False
    assert collision_manifest["target_exclusion_verified"] is False
    assert collision_manifest["target_identity_collision_audit"]["collision_count"] == 2

    # The verifier replays raw bytes for both official splits, including the
    # unselected train split of this official-test export.
    changed_train = np.zeros((4, 8, 3), dtype=np.uint8)
    changed_train[0, 2] = 199
    Image.fromarray(changed_train).save(target / "images" / "Misc_train.png")
    with pytest.raises(ValueError, match="official_train image hashes"):
        verify_score_manifest_artifacts(
            output / "manifest.json",
            required_split_role="official_test",
        )


@requires_torch
def test_detector_diagnostic_export_label_and_sweep_remain_development_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from data_ext.label_manifest_artifacts import verify_label_attachment
    from evaluation.export_label_maps import export_label_maps

    exporter_module = importlib.import_module("evaluation.export_score_maps")
    dataset, split_manifest, diagnostic_split = (
        _make_frozen_detector_diagnostic_partition(tmp_path / "repository")
    )
    split_manifest_sha = sha256_file(split_manifest)
    checkpoint = tmp_path / "detector.pt"
    torch.save({"state_dict": {"unused": torch.zeros(1)}}, checkpoint)
    # Score export and diagnostic label attachment must never need masks from
    # detector-fit, quarantine, or official-test roles.
    for image_id in ("fit_a", "quarantine_c", "test_d"):
        (dataset / "masks" / f"{image_id}.png").unlink()

    class ZeroDetector(torch.nn.Module):
        def forward(self, image: torch.Tensor, warm_flag: bool) -> torch.Tensor:
            del warm_flag
            return torch.zeros(
                (image.shape[0], 1, image.shape[2], image.shape[3]),
                dtype=image.dtype,
                device=image.device,
            )

    monkeypatch.setattr(
        exporter_module,
        "load_model",
        lambda *args, **kwargs: ZeroDetector(),
    )
    score_root = tmp_path / "diagnostic-scores"
    # The deliberately misleading free-form assertion must not affect the
    # verified partition role or final-evaluation eligibility.
    manifest = export_score_maps(
        dataset_dir=dataset,
        weight_path=checkpoint,
        output_dir=score_root,
        base_size=16,
        split_file=diagnostic_split,
        split_role="detector_diagnostic",
        derived_split_manifest=split_manifest,
        derived_split_manifest_sha256=split_manifest_sha,
        source_dataset="official_test",
        device="cpu",
    )
    assert [item["image_id"] for item in manifest["items"]] == [
        "diagnostic_b"
    ]
    assert manifest["items"][0]["original_hw"] == [4, 8]
    assert manifest["split_contract"]["schema_version"] == 2
    assert manifest["split_contract"]["role"] == "detector_diagnostic"
    assert manifest["partition_scope"] == (
        "official_train_derived_development_diagnostic"
    )
    assert manifest["official_test_artifact"] is False
    assert manifest["final_evaluation_eligible"] is False
    assert manifest["development_only"] is True
    assert manifest["claim_bearing_final_evaluation"] is False
    verified = verify_score_manifest_artifacts(
        score_root / "manifest.json",
        required_split_role="detector_diagnostic",
    )
    assert verified.split_role == "detector_diagnostic"
    with pytest.raises(ValueError, match="cannot satisfy required role"):
        verify_score_manifest_artifacts(
            score_root / "manifest.json",
            required_split_role="official_test",
        )
    with pytest.raises(ValueError, match="cannot satisfy required role"):
        verify_score_manifest_artifacts(
            score_root / "manifest.json",
            required_split_role="official_train",
        )
    tampered_score_payload = json.loads(
        (score_root / "manifest.json").read_text(encoding="utf-8")
    )
    tampered_score_payload["official_test_artifact"] = True
    tampered_score = score_root / "tampered-role-manifest.json"
    tampered_score.write_text(json.dumps(tampered_score_payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="official_test_artifact must be exactly False"
    ):
        verify_score_manifest_artifacts(
            tampered_score,
            required_split_role="detector_diagnostic",
        )

    label_root = tmp_path / "diagnostic-labels"
    label_manifest = export_label_maps(
        dataset_dir=dataset,
        score_manifest=score_root / "manifest.json",
        output_dir=label_root,
    )
    assert label_manifest["score_split_role"] == "detector_diagnostic"
    assert label_manifest["official_test_artifact"] is False
    assert label_manifest["claim_bearing_final_evaluation"] is False
    attachment = verify_label_attachment(
        score_root / "manifest.json", label_root / "label-manifest.json"
    )
    assert attachment.selected_items[0].original_hw == (4, 8)
    tampered_label_payload = json.loads(
        (label_root / "label-manifest.json").read_text(encoding="utf-8")
    )
    tampered_label_payload["claim_bearing_final_evaluation"] = True
    tampered_label = label_root / "tampered-label-manifest.json"
    tampered_label.write_text(json.dumps(tampered_label_payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="claim_bearing_final_evaluation must be exactly False"
    ):
        verify_label_attachment(score_root / "manifest.json", tampered_label)

    curve = tmp_path / "diagnostic-curve.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(score_root),
            "--label-manifest",
            str(label_root / "label-manifest.json"),
            "--output",
            str(curve),
        ]
    ) == 0
    curve_manifest = json.loads(
        curve.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert curve_manifest["score_split_role"] == "detector_diagnostic"
    assert curve_manifest["official_test_artifact"] is False
    assert curve_manifest["final_evaluation_eligible"] is False
    assert curve_manifest["development_only"] is True
    assert curve_manifest["claim_bearing_final_evaluation"] is False
    assert curve_manifest["oracle_only"] is True
    assert curve_manifest["deployable"] is False


@requires_torch
def test_detector_diagnostic_contract_rejects_unfrozen_or_copied_split(
    tmp_path: Path,
) -> None:
    from data_ext.inference_dataset import IRSTDInferenceDataset
    from evaluation.export_score_maps import build_official_split_contract

    dataset, split_manifest, diagnostic_split = (
        _make_frozen_detector_diagnostic_partition(tmp_path / "repository")
    )
    copied_split = tmp_path / "copied-diagnostic.txt"
    shutil.copyfile(diagnostic_split, copied_split)
    inference = IRSTDInferenceDataset(
        dataset,
        base_size=16,
        split_file=copied_split,
    )
    with pytest.raises(ValueError, match="requires the exact diagnostic path"):
        build_official_split_contract(
            inference,
            split_role="detector_diagnostic",
            manifest_root=tmp_path / "scores-copy",
            derived_split_manifest=split_manifest,
            derived_split_manifest_sha256=sha256_file(split_manifest),
        )

    exact_inference = IRSTDInferenceDataset(
        dataset,
        base_size=16,
        split_file=diagnostic_split,
    )
    with pytest.raises(ValueError, match="requires derived_split_manifest"):
        build_official_split_contract(
            exact_inference,
            split_role="detector_diagnostic",
            manifest_root=tmp_path / "scores-no-manifest",
        )
    with pytest.raises(ValueError, match="frozen value"):
        build_official_split_contract(
            exact_inference,
            split_role="detector_diagnostic",
            manifest_root=tmp_path / "scores-wrong-sha",
            derived_split_manifest=split_manifest,
            derived_split_manifest_sha256="0" * 64,
        )


@requires_torch
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("outer_target_official_train_used_for_detector_fit", True),
        (
            "outer_target_detector_diagnostic_used_for_development_evaluation",
            False,
        ),
        ("outer_target_diagnostic_selects_checkpoint", True),
        ("outer_target_official_train_used", False),
        ("outer_target_official_train_allowed_in_same_outer_fold", False),
    ),
)
def test_detector_diagnostic_contract_rejects_outer_target_role_drift(
    tmp_path: Path,
    field: str,
    invalid_value: bool,
) -> None:
    from data_ext.inference_dataset import IRSTDInferenceDataset
    from evaluation.export_score_maps import build_official_split_contract

    dataset, split_manifest, diagnostic_split = (
        _make_frozen_detector_diagnostic_partition(tmp_path / "repository")
    )
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    payload["role_contract"][field] = invalid_value
    split_manifest.write_text(json.dumps(payload), encoding="utf-8")
    inference = IRSTDInferenceDataset(
        dataset,
        base_size=16,
        split_file=diagnostic_split,
    )
    with pytest.raises(ValueError, match=field):
        build_official_split_contract(
            inference,
            split_role="detector_diagnostic",
            manifest_root=tmp_path / "scores-role-drift",
            derived_split_manifest=split_manifest,
            derived_split_manifest_sha256=sha256_file(split_manifest),
        )


@requires_torch
@pytest.mark.parametrize(
    "contaminating_id", ("fit_a", "quarantine_c", "test_d")
)
def test_detector_diagnostic_contract_rejects_partition_contamination(
    tmp_path: Path,
    contaminating_id: str,
) -> None:
    from data_ext.inference_dataset import IRSTDInferenceDataset
    from evaluation.export_score_maps import build_official_split_contract

    dataset, split_manifest, diagnostic_split = (
        _make_frozen_detector_diagnostic_partition(tmp_path / "repository")
    )
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    diagnostic_split.write_text(f"{contaminating_id}\n", encoding="utf-8")
    payload["datasets"][0]["detector"]["diagnostic_sha256"] = sha256_file(
        diagnostic_split
    )
    split_manifest.write_text(json.dumps(payload), encoding="utf-8")
    inference = IRSTDInferenceDataset(
        dataset,
        base_size=16,
        split_file=diagnostic_split,
    )
    with pytest.raises(ValueError, match="exact disjoint partition"):
        build_official_split_contract(
            inference,
            split_role="detector_diagnostic",
            manifest_root=tmp_path / "scores-contaminated",
            derived_split_manifest=split_manifest,
            derived_split_manifest_sha256=sha256_file(split_manifest),
        )


@requires_torch
@pytest.mark.parametrize("overlap_kind", ("sample_id", "image_content"))
def test_exporter_rejects_official_train_test_contamination(
    tmp_path: Path,
    overlap_kind: str,
) -> None:
    target = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    _add_disjoint_official_train_split(target)
    if overlap_kind == "sample_id":
        (target / "img_idx" / "train_NUAA-SIRST.txt").write_text(
            "Misc_1.png\n",
            encoding="utf-8",
        )
        expected = "sample-ID overlap=1"
    else:
        shutil.copyfile(
            target / "images" / "Misc_1.png",
            target / "images" / "Misc_train.png",
        )
        expected = "raw-image-content overlap=1"
    with pytest.raises(ValueError, match=expected):
        export_score_maps(
            dataset_dir=target,
            weight_path=tmp_path / "unused.pt",
            output_dir=tmp_path / f"scores-{overlap_kind}",
            base_size=16,
            device="cpu",
        )


@requires_torch
def test_exporter_rejects_spatial_size_not_divisible_by_sixteen(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        export_score_maps(
            dataset_dir=tmp_path / "unused-dataset",
            weight_path=tmp_path / "unused.pt",
            output_dir=tmp_path / "unused-output",
            base_size=(32, 30),
            device="cpu",
        )


@requires_torch
def test_letterbox_prediction_is_cropped_and_restored() -> None:
    transform = build_spatial_transform((2, 4), (4, 4), "letterbox")
    input_map = torch.full((4, 4), 9.0)
    input_map[1:3, :] = 1.0
    restored = restore_tensor_to_original(input_map, transform, mode="bilinear")
    assert tuple(restored.shape) == (2, 4)
    assert torch.allclose(restored, torch.ones_like(restored))


def test_overlap_matching_is_8_connected_and_one_to_one() -> None:
    target = np.zeros((5, 6), dtype=np.uint8)
    target[1, 1] = 1
    target[1, 4] = 1
    prediction = np.zeros_like(target)
    prediction[1, 1:5] = 1  # One component overlaps two separate GT objects.

    result = match_components(prediction, target, rule="overlap")
    assert result.num_gt == 2
    assert result.num_pred_components == 1
    assert result.num_tp_objects == 1
    assert result.num_fp_components == 0
    assert result.num_fp_pixels == 2

    diagonal = np.zeros((3, 3), dtype=np.uint8)
    diagonal[0, 0] = 1
    diagonal[1, 1] = 1
    connected = match_components(diagonal, diagonal, rule="overlap")
    assert connected.num_gt == 1
    assert connected.num_pred_components == 1


def test_centroid_matching_is_one_to_one() -> None:
    target = np.zeros((7, 7), dtype=np.uint8)
    target[1, 1] = 1
    target[5, 5] = 1
    prediction = np.zeros_like(target)
    prediction[1, 2] = 1
    prediction[5, 4] = 1
    result = match_components(
        prediction,
        target,
        rule="centroid",
        centroid_distance=2.0,
    )
    assert result.num_tp_objects == 2
    assert result.num_fp_components == 0
    assert result.num_fp_pixels == 2


def test_component_false_alarms_can_increase_when_threshold_tightens() -> None:
    scores = np.asarray(
        [[0.0, 0.0, 0.0], [0.9, 0.2, 0.9], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    target = np.zeros_like(scores, dtype=np.uint8)
    loose = match_components(scores > 0.1, target, rule="overlap")
    strict = match_components(scores > 0.5, target, rule="overlap")

    assert loose.num_fp_pixels == 3
    assert strict.num_fp_pixels == 2
    assert loose.num_fp_components == 1
    assert strict.num_fp_components == 2


def test_threshold_sweep_has_explicit_endpoints_and_monotone_pixel_fa() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32),
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        image_id="toy",
    )
    thresholds = normalise_thresholds([0.5])
    assert thresholds.tolist() == [0.0, 0.5, 1.0]
    rows = sweep_thresholds([record], [0.5])
    assert [row["threshold"] for row in rows] == [0.0, 0.5, 1.0]
    assert [row["fp_pixels"] for row in rows] == [3, 1, 0]
    assert all(
        float(rows[index]["fa_pixel"]) >= float(rows[index + 1]["fa_pixel"])
        for index in range(len(rows) - 1)
    )


def test_threshold_grid_contract_is_versioned() -> None:
    metadata = threshold_grid_metadata()
    assert metadata["schema_version"] == CURVE_SCHEMA_VERSION
    assert metadata["threshold_grid_version"] == THRESHOLD_GRID_VERSION
    assert metadata["thresholds"][0] == 0.0
    assert metadata["thresholds"][-1] == 1.0


def test_adaptive_events_resolve_high_tail_budget_counterexample() -> None:
    record = ScoreMapRecord(
        probability=np.asarray(
            [[0.999995, 0.999999], [0.0, 0.0]],
            dtype=np.float32,
        ),
        mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        image_id="high-tail",
    )
    rows, audit = sweep_thresholds(
        [record],
        threshold_mode="adaptive",
        event_threshold_cap=None,
        return_metadata=True,
    )
    selected = select_operating_point(rows, pixel_budget=0.1)
    assert selected is not None
    # Strict probability > threshold excludes the equal-score FP while the
    # larger TP remains, an operating point absent from the fixed grid.
    assert selected["threshold"] == pytest.approx(0.999995, abs=1e-7)
    assert selected["pd"] == 1.0
    assert selected["fp_pixels"] == 0
    assert rows[0]["threshold"] == 0.0
    assert rows[-1]["threshold"] == 1.0
    assert audit["threshold_mode"] == "adaptive"
    assert audit["event_candidate_count"] == 2
    assert audit["event_threshold_count"] == 2
    assert audit["event_coverage_fraction_lower_bound"] == 1.0
    assert audit["global_exact"] is False


def test_capped_exact_plan_never_claims_global_exact() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.2, 0.3], [0.4, 0.4]], dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
        image_id="capped",
    )
    rows, audit = sweep_thresholds(
        [record],
        thresholds=[0.5],
        threshold_mode="exact",
        event_threshold_cap=1,
        return_metadata=True,
    )
    assert [row["threshold"] for row in rows] == pytest.approx([0.0, 0.4, 0.5, 1.0])
    assert audit["threshold_mode"] == "exact_capped"
    assert audit["event_thresholds_capped"] is True
    assert audit["event_candidate_count"] == 3
    assert audit["event_threshold_count"] == 1
    assert audit["event_coverage_fraction_lower_bound"] == pytest.approx(1 / 3)
    assert audit["event_coverage_score_lower_bound"] == pytest.approx(0.4)
    assert audit["global_exact"] is False


def test_uncapped_exact_plan_is_globally_exact() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.2, 0.3], [0.4, 0.4]], dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
    )
    _, audit = sweep_thresholds(
        [record],
        thresholds=[0.5],
        threshold_mode="exact",
        event_threshold_cap=None,
        return_metadata=True,
    )
    assert audit["threshold_mode"] == "exact"
    assert audit["event_threshold_count"] == 3
    assert audit["event_coverage_fraction_lower_bound"] == 1.0
    assert audit["global_exact"] is True


def test_cli_defaults_to_query_adaptive_events_and_audits_manifest(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    probability = np.asarray(
        [[0.999995, 0.999999], [0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
    np.savez_compressed(
        score_dir / "toy.npz",
        prob=probability,
        mask=mask,
        image_id=np.asarray("toy"),
    )
    curve_path = tmp_path / "curve.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(score_dir),
            "--output",
            str(curve_path),
            "--event-threshold-cap",
            "0",
            "--diagnostic-allow-embedded-mask",
        ]
    ) == 0

    rows = read_curve_csv(curve_path)
    selected = select_operating_point(rows, pixel_budget=0.1)
    assert selected is not None
    assert selected["threshold"] == pytest.approx(0.999995, abs=1e-7)
    assert selected["pd"] == 1.0

    manifest = json.loads(
        curve_path.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["threshold_mode_requested"] == "adaptive"
    assert manifest["threshold_mode"] == "adaptive"
    assert manifest["event_candidate_count"] == 2
    assert manifest["event_threshold_count"] == 2
    assert manifest["event_coverage_score_lower_bound"] == pytest.approx(0.999995)
    assert manifest["event_coverage_fraction_lower_bound"] == 1.0
    assert manifest["global_exact"] is False
    assert manifest["matching_rule"] == "overlap"
    assert manifest["centroid_distance"] == 3.0
    assert manifest["evaluation_scope"] == "legacy_combined_npz_diagnostic"


def test_manifest_score_discovery_rejects_tamper_and_relocated_substitution(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "scores"
    image_dir = score_dir / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8), mode="L").save(image_path)
    score_path = score_dir / "sample.npz"
    np.savez_compressed(
        score_path,
        prob=np.full((4, 5), 0.5, dtype=np.float32),
        mask=np.zeros((4, 5), dtype=np.uint8),
        image_id=np.asarray("sample"),
        dataset_name=np.asarray("TARGET"),
        original_hw=np.asarray((4, 5), dtype=np.int32),
    )
    items = [
        {
            "image_id": "sample",
            "file": "sample.npz",
            "score_file_sha256": sha256_file(score_path),
            "image_path": "images/sample.png",
            "gray_file_sha256": sha256_file(image_path),
            "original_hw": [4, 5],
        }
    ]
    manifest = {
        "path_anchor": "manifest_directory",
        "target_dataset": "TARGET",
        "score_type": "sigmoid_probability",
        "restored_to_original_hw": True,
        "threshold_semantics": "prediction = probability > threshold",
        "num_images": 1,
        "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": score_manifest_content_sha256(items),
        "items": items,
    }
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert discover_score_maps(
        score_dir,
        ["sample"],
        allow_legacy_combined_diagnostic=True,
    ) == [score_path.resolve()]

    # Copying a manifest into another directory cannot silently rebind its
    # relative paths to different bytes.
    copied_dir = tmp_path / "copied-scores"
    copied_image_dir = copied_dir / "images"
    copied_image_dir.mkdir(parents=True)
    shutil.copy2(manifest_path, copied_dir / "manifest.json")
    shutil.copy2(score_path, copied_dir / score_path.name)
    Image.fromarray(np.ones((4, 5), dtype=np.uint8), mode="L").save(
        copied_image_dir / image_path.name
    )
    with pytest.raises(ValueError, match="original-image SHA-256 mismatch"):
        discover_score_maps(
            copied_dir,
            ["sample"],
            allow_legacy_combined_diagnostic=True,
        )

    np.savez_compressed(
        score_path,
        prob=np.full((4, 5), 0.75, dtype=np.float32),
        mask=np.zeros((4, 5), dtype=np.uint8),
        image_id=np.asarray("sample"),
        dataset_name=np.asarray("TARGET"),
        original_hw=np.asarray((4, 5), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="score-file SHA-256 mismatch"):
        discover_score_maps(
            score_dir,
            ["sample"],
            allow_legacy_combined_diagnostic=True,
        )


def test_threshold_sweep_labels_each_gt_only_once(monkeypatch) -> None:
    import evaluation.component_matching as component_matching

    calls = 0
    original_label = component_matching.label

    def counting_label(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_label(*args, **kwargs)

    monkeypatch.setattr(component_matching, "label", counting_label)
    record = ScoreMapRecord(
        probability=np.asarray([[0.9, 0.2], [0.1, 0.0]], dtype=np.float32),
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
    )
    rows = sweep_thresholds([record], thresholds=[0.5])
    # One GT label pass plus one prediction label pass for each of 3 points.
    assert len(rows) == 3
    assert calls == 1 + len(rows)


def test_dual_budget_operating_point_requires_both_constraints() -> None:
    rows = [
        {"threshold": 0.2, "pd": 0.9, "fa_pixel": 0.02, "fa_component_mp": 0.5},
        {"threshold": 0.5, "pd": 0.8, "fa_pixel": 0.005, "fa_component_mp": 2.0},
        {"threshold": 0.7, "pd": 0.7, "fa_pixel": 0.001, "fa_component_mp": 0.5},
    ]
    selected = select_operating_point(
        rows,
        pixel_budget=0.01,
        component_budget=1.0,
    )
    assert selected is not None
    assert selected["threshold"] == 0.7


def test_rejection_aware_dual_budget_metrics() -> None:
    metrics = compute_budget_metrics(
        [
            {"fa_pixel": 0.005, "fa_component_mp": 0.5},
            {"fa_pixel": 0.02, "fa_component_mp": 0.5},
            {"fa_pixel": 100.0, "fa_component_mp": 100.0, "rejected": True},
        ],
        pixel_budget=0.01,
        component_budget=1.0,
    )
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["bsr"] == pytest.approx(0.5)
    assert metrics["unconditional_bsr"] == pytest.approx(1 / 3)
    assert metrics["excess"] == pytest.approx(0.5)
    assert metrics["component_bsr"] == pytest.approx(1.0)


def _write_adapter_replay_fixture(
    tmp_path: Path,
    *,
    reject_bias: float = -2.0,
) -> tuple[Path, dict[str, object], Path, Path]:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    dataset_root = tmp_path / "TARGET"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "img_idx").mkdir()
    context_gray = dataset_root / "images" / "context.png"
    query_gray = dataset_root / "images" / "query.png"
    train_gray = dataset_root / "images" / "train.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(context_gray)
    Image.fromarray(
        np.asarray([[255, 128], [32, 64]], dtype=np.uint8), mode="L"
    ).save(query_gray)
    Image.fromarray(
        np.asarray([[3, 17], [101, 203]], dtype=np.uint8), mode="L"
    ).save(train_gray)
    (dataset_root / "img_idx" / "train_TARGET.txt").write_text(
        "train.png\n",
        encoding="utf-8",
    )
    (dataset_root / "img_idx" / "test_TARGET.txt").write_text(
        "context.png\nquery.png\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        score_dir / "context.npz",
        prob=np.zeros((2, 2), dtype=np.float32),
        image_id=np.asarray("context"),
        dataset_name=np.asarray("TARGET"),
        original_hw=np.asarray([2, 2], dtype=np.int32),
    )
    np.savez_compressed(
        score_dir / "query.npz",
        prob=np.asarray([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32),
        image_id=np.asarray("query"),
        dataset_name=np.asarray("TARGET"),
        original_hw=np.asarray([2, 2], dtype=np.int32),
    )
    detector_sha = "a" * 64
    manifest_items = [
        {
            "image_id": "context",
            "file": "context.npz",
            "score_file_sha256": sha256_file(score_dir / "context.npz"),
            "image_path": "../TARGET/images/context.png",
            "gray_file_sha256": sha256_file(context_gray),
            "original_hw": [2, 2],
        },
        {
            "image_id": "query",
            "file": "query.npz",
            "score_file_sha256": sha256_file(score_dir / "query.npz"),
            "image_path": "../TARGET/images/query.png",
            "gray_file_sha256": sha256_file(query_gray),
            "original_hw": [2, 2],
        },
    ]
    from data_ext.inference_dataset import IRSTDInferenceDataset
    from evaluation.export_score_maps import build_official_split_contract

    inference_dataset = IRSTDInferenceDataset(
        dataset_root,
        base_size=16,
        split="test",
    )
    split_contract, target_dataset_record = build_official_split_contract(
        inference_dataset,
        split_role="official_test",
        manifest_root=score_dir,
    )
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "artifact_type": "label_free_score_export",
                "path_anchor": "manifest_directory",
                "target_dataset": "TARGET",
                "target_dataset_record": target_dataset_record,
                "split_contract": split_contract,
                "dataset_dir": "../TARGET",
                "split_file": split_contract["selected_split_file"],
                "weight_sha256": detector_sha,
                "outer_fold_id": "fold-target",
                "outer_target": "TARGET",
                "detector_source_domains": ["SOURCE-A", "SOURCE-B"],
                "held_out_domains": ["TARGET"],
                "protocol_scope": "multi_source_protocol_candidate",
                "target_exclusion_verified": True,
                "score_type": "sigmoid_probability",
                "restored_to_original_hw": True,
                "threshold_semantics": "prediction = probability > threshold",
                "labels_embedded": False,
                "label_contract": "external_label_attachment_manifest_required_offline",
                "num_images": 2,
                "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
                "content_sha256": score_manifest_content_sha256(manifest_items),
                "items": manifest_items,
            }
        ),
        encoding="utf-8",
    )
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    label_masks = {
        "context": np.zeros((2, 2), dtype=np.uint8),
        "query": np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
    }
    label_items: list[dict[str, object]] = []
    for score_item in manifest_items:
        image_id = str(score_item["image_id"])
        file_name = f"{image_id}.label.npz"
        label_path = label_dir / file_name
        np.savez_compressed(
            label_path,
            mask=label_masks[image_id],
            image_id=np.asarray(image_id),
            original_hw=np.asarray([2, 2], dtype=np.int32),
        )
        label_items.append(
            {
                "image_id": image_id,
                "file": file_name,
                "label_file_sha256": sha256_file(label_path),
                "source_image_file_sha256": score_item["gray_file_sha256"],
                "original_hw": [2, 2],
            }
        )
    label_manifest_path = label_dir / "label-manifest.json"
    label_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": LABEL_MANIFEST_SCHEMA_VERSION,
                "artifact_type": LABEL_MANIFEST_ARTIFACT_TYPE,
                "path_anchor": "manifest_directory",
                "score_manifest_file": "../scores/manifest.json",
                "score_manifest_sha256": sha256_file(manifest_path),
                "score_manifest_content_sha256": score_manifest_content_sha256(
                    manifest_items
                ),
                "target_dataset": "TARGET",
                "labels_embedded_in_scores": False,
                "num_images": len(label_items),
                "content_sha256_algorithm": LABEL_MANIFEST_CONTENT_ALGORITHM,
                "content_sha256": label_manifest_content_sha256(label_items),
                "items": label_items,
            }
        ),
        encoding="utf-8",
    )
    import torch

    from model.threshold_calibrator import ThresholdCalibrator
    from rc.domain_statistics import BASE_FEATURE_DIM, FEATURE_NAMES
    from rc.meta_dataset import FeatureStandardizer
    from rc.online_adapter import main as online_adapter_main
    from rc.schema import (
        DeploymentProtocolContract,
        SourceContract,
        SourceReference,
        StatisticsConfig,
    )

    config = StatisticsConfig(peak_kernel_size=3, peak_min_score=0.05)
    contract = SourceContract(
        detector_checkpoint_sha=detector_sha,
        detector_source_domains=("SOURCE-A", "SOURCE-B"),
        outer_fold_id="fold-target",
        outer_target="TARGET",
        held_out_domains=("TARGET",),
        protocol_scope="multi_source_protocol_candidate",
    )
    reference = SourceReference(
        domains=contract.detector_source_domains,
        sha256="c" * 64,
        centers=tuple(
            tuple(0.0 for _ in range(BASE_FEATURE_DIM))
            for _ in contract.detector_source_domains
        ),
        scale=tuple(1.0 for _ in range(BASE_FEATURE_DIM)),
        contract=contract,
    )
    input_names = FEATURE_NAMES + (
        "budget_log10_pixel",
        "budget_log10_component",
        "budget_active_pixel",
        "budget_active_component",
    )
    standardizer = FeatureStandardizer(
        input_names,
        np.zeros(len(input_names), dtype=np.float64),
        np.ones(len(input_names), dtype=np.float64),
    )
    model = ThresholdCalibrator(len(input_names), hidden_dim=8, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.reject_head.bias.fill_(reject_bias)
    deployment_protocol_contract = DeploymentProtocolContract(
        context_size=1,
        query_size=1,
        reject_cutoff=0.5,
    )
    checkpoint_path = tmp_path / "calibrator.pt"
    torch.save(
        {
            "format_version": "rc-irstd.calibrator.v3",
            "episode_collection_provenance": {
                "mode": "combined",
                "combined": {"file": "episodes.jsonl", "sha256": "e" * 64},
            },
            "episode_collection_sha256": "e" * 64,
            "training_config": {
                "epochs": 1,
                "batch_size": 1,
                "lr": 0.001,
                "weight_decay": 0.0,
                "hidden_dim": 8,
                "dropout": 0.0,
                "under_weight": 4.0,
                "reject_weight": 1.0,
                "threshold_on_reject": False,
                "threshold_transform": "identity",
                "reject_probability": 0.5,
                "num_workers": 0,
                "seed": 1,
                "device_requested": "cpu",
                "device_resolved": "cpu",
            },
            "model_state_dict": model.state_dict(),
            "input_dim": len(input_names),
            "hidden_dim": 8,
            "dropout": 0.0,
            "standardizer": standardizer.to_dict(),
            "statistics_feature_names": list(FEATURE_NAMES),
            "input_feature_names": list(input_names),
            "statistics_config": config.to_dict(),
            "p_min": 0.5,
            "outer_fold_id": "fold-target",
            "outer_target": "TARGET",
            "calibration_pseudo_targets": ["SOURCE-A", "SOURCE-B"],
            "deployment_detector_source_domains": ["SOURCE-A", "SOURCE-B"],
            "deployment_detector_checkpoint_sha": detector_sha,
            "deployment_held_out_domains": ["TARGET"],
            "deployment_protocol_scope": "multi_source_protocol_candidate",
            "deployment_source_reference": reference.to_dict(),
            "deployment_protocol_contract": deployment_protocol_contract.to_dict(),
            "reject_probability": 0.5,
        },
        checkpoint_path,
    )
    adapter_path = tmp_path / "adapter.json"
    assert (
        online_adapter_main(
            [
                "--manifest",
                str(manifest_path),
                "--calibrator-checkpoint",
                str(checkpoint_path),
                "--target-domain",
                "TARGET",
                "--context-size",
                "1",
                "--query-size",
                "1",
                "--pixel-budget",
                "0.30",
                "--device",
                "cpu",
                "--output",
                str(adapter_path),
            ]
        )
        == 0
    )
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    return manifest_path, adapter, checkpoint_path, label_manifest_path


def test_adapter_replay_verifies_binding_and_reports_raw_counts(tmp_path: Path) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    result = evaluate_adapter_output(
        adapter,
        manifest_path,
        calibrator_checkpoint=checkpoint_path,
        label_manifest=label_manifest_path,
    )
    assert result["rejected"] is False
    assert result["query_image_ids"] == ["query"]
    assert result["pd"] == pytest.approx(1.0)
    assert result["fa_pixel"] == pytest.approx(0.25)
    assert result["fa_component_mp"] == pytest.approx(0.0)
    assert result["tp_objects"] == result["gt_objects"] == 1
    assert result["pred_components"] == 1
    assert result["fp_components"] == 0
    assert result["fp_pixels"] == 1
    assert result["total_pixels"] == 4
    assert result["num_images"] == 1


@pytest.mark.parametrize(
    ("field", "replacement", "error_type", "message"),
    [
        ("score_manifest_sha256", "0" * 64, ValueError, "SHA-256 mismatch"),
        (
            "calibrator_checkpoint_sha256",
            "0" * 64,
            ValueError,
            "Calibrator checkpoint SHA-256 mismatch",
        ),
        (
            "score_manifest_target_dataset",
            "OTHER",
            ValueError,
            "Target-domain binding mismatch",
        ),
        ("query_image_ids", ["not-query"], KeyError, "absent from manifest"),
        ("threshold", 0.7, ValueError, "deterministic calibrator replay"),
    ],
)
def test_adapter_replay_rejects_binding_mismatches(
    tmp_path: Path,
    field: str,
    replacement: object,
    error_type: type[Exception],
    message: str,
) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    adapter[field] = replacement
    with pytest.raises(error_type, match=message):
        evaluate_adapter_output(
            adapter,
            manifest_path,
            calibrator_checkpoint=checkpoint_path,
            label_manifest=label_manifest_path,
        )


def test_rejected_adapter_has_no_fabricated_metrics_and_summary_is_coverage_aware(
    tmp_path: Path,
) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    covered = evaluate_adapter_output(
        adapter,
        manifest_path,
        calibrator_checkpoint=checkpoint_path,
        label_manifest=label_manifest_path,
    )
    rejected_root = tmp_path / "genuine-rejection"
    rejected_root.mkdir()
    rejected_manifest, rejected_adapter, rejected_checkpoint, _ = (
        _write_adapter_replay_fixture(rejected_root, reject_bias=2.0)
    )
    rejected = evaluate_adapter_output(
        rejected_adapter,
        rejected_manifest,
        calibrator_checkpoint=rejected_checkpoint,
        # The rejected branch must not resolve or open any label artifact.
        label_manifest=tmp_path / "intentionally-missing-label-manifest.json",
    )
    assert rejected["rejected"] is True
    assert all(field not in rejected for field in ("pd", "fa_pixel", "tp_objects"))

    summary = summarise_adapter_evaluations([covered, rejected])
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["bsr"] == pytest.approx(1.0)
    assert summary["excess"] == pytest.approx(0.0)
    assert summary["covered_pd"] == pytest.approx(1.0)
    assert summary["covered_tp_objects"] == 1
    assert summary["covered_gt_objects"] == 1


def test_adapter_replay_rejects_final_target_cutoff_tampering(tmp_path: Path) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    adapter["reject_cutoff"] = 0.0
    adapter["reject"] = True
    with pytest.raises(ValueError, match="deterministic calibrator replay"):
        evaluate_adapter_output(
            adapter,
            manifest_path,
            calibrator_checkpoint=checkpoint_path,
            label_manifest=label_manifest_path,
        )


def test_adapter_evaluation_rejects_matching_contract_override(tmp_path: Path) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    with pytest.raises(ValueError, match="requested matching_rule differs"):
        evaluate_adapter_output(
            adapter,
            manifest_path,
            calibrator_checkpoint=checkpoint_path,
            label_manifest=label_manifest_path,
            matching_rule="centroid",
        )


def test_adapter_replay_rejects_coordinated_matching_contract_tampering(
    tmp_path: Path,
) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    tampered = json.loads(json.dumps(adapter))
    tampered["evaluation_contract"]["matching_rule"] = "centroid"
    tampered["decision_contract"]["evaluation_matching"][
        "matching_rule"
    ] = "centroid"
    tampered["deployment_protocol_contract"]["evaluation_matching"][
        "matching_rule"
    ] = "centroid"
    with pytest.raises(ValueError, match="deterministic calibrator replay"):
        evaluate_adapter_output(
            tampered,
            manifest_path,
            calibrator_checkpoint=checkpoint_path,
            label_manifest=label_manifest_path,
        )


def test_adapter_evaluation_cli_rejects_matching_contract_override(
    tmp_path: Path,
) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    with pytest.raises(ValueError, match="requested matching_rule differs"):
        evaluate_adapter_main(
            [
                "--adapter-output",
                str(adapter_path),
                "--score-manifest",
                str(manifest_path),
                "--calibrator-checkpoint",
                str(checkpoint_path),
                "--label-manifest",
                str(label_manifest_path),
                "--matching-rule",
                "centroid",
            ]
        )


def test_adapter_replay_cli_writes_json(tmp_path: Path) -> None:
    manifest_path, adapter, checkpoint_path, label_manifest_path = (
        _write_adapter_replay_fixture(tmp_path)
    )
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    output_path = tmp_path / "evaluation.json"
    assert (
        evaluate_adapter_main(
            [
                "--adapter-output",
                str(adapter_path),
                "--score-manifest",
                str(manifest_path),
                "--calibrator-checkpoint",
                str(checkpoint_path),
                "--label-manifest",
                str(label_manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pd"] == pytest.approx(1.0)
    assert payload["score_manifest_sha256"] == adapter["score_manifest_sha256"]
