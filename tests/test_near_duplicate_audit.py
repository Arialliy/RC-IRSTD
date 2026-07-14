from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_near_duplicates import (
    ImageRecord,
    audit_records,
    build_report,
    confirmation_signature,
    phash64,
)
from scripts.resolve_near_duplicate_quarantine import build_quarantine


def _record(
    name: str,
    array: np.ndarray,
    *,
    dataset: str,
    role: str,
) -> ImageRecord:
    values = np.asarray(array, dtype=np.float64)
    raw = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8).tobytes()
    return ImageRecord(
        dataset_name=dataset,
        split_role=role,
        image_id=name,
        path=Path(name),
        image_sha256=hashlib.sha256(raw).hexdigest(),
        phash=phash64(values),
        confirmation=confirmation_signature(values),
    )


def test_phash_and_confirmation_find_brightness_variant_order_independently() -> None:
    y, x = np.mgrid[0:32, 0:32]
    base = (np.sin(x / 5.0) + np.cos(y / 7.0) + 2.0) / 4.0
    brightness_variant = np.clip(base * 0.8 + 0.1, 0.0, 1.0)
    unrelated = np.random.default_rng(9).random((32, 32))
    records = [
        _record("a", base, dataset="A", role="official_train"),
        _record("b", brightness_variant, dataset="B", role="official_test"),
        _record("c", unrelated, dataset="C", role="official_train"),
    ]
    first = audit_records(records, phash_hamming_max=4, confirmation_cosine_min=0.995)
    second = audit_records(
        list(reversed(records)),
        phash_hamming_max=4,
        confirmation_cosine_min=0.995,
    )
    assert first["near_duplicate_contract_passed"] is False
    assert first["confirmed_near_duplicate_pair_count"] == 1
    assert first["confirmed_near_duplicate_pairs"][0]["left"]["image_id"] == "a"
    assert first["confirmed_near_duplicate_pairs"][0]["right"]["image_id"] == "b"
    assert len(first["confirmed_near_duplicate_pairs"][0]["candidate_id"]) == 64
    assert second["confirmed_near_duplicate_pair_count"] == 1
    assert first["image_index_sha256"] == second["image_index_sha256"]


def test_build_report_reads_images_without_requiring_masks(tmp_path: Path) -> None:
    dataset = tmp_path / "domain-a"
    (dataset / "images").mkdir(parents=True)
    (dataset / "img_idx").mkdir()
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    Image.fromarray(image).save(dataset / "images" / "train.png")
    Image.fromarray(image).save(dataset / "images" / "test.png")
    (dataset / "img_idx" / "train_domain-a.txt").write_text(
        "train\n", encoding="utf-8"
    )
    (dataset / "img_idx" / "test_domain-a.txt").write_text(
        "test\n", encoding="utf-8"
    )

    report = build_report([("domain-a", dataset)])
    assert report["labels_scores_checkpoints_or_metrics_read"] is False
    assert report["confirmed_near_duplicate_pair_count"] == 1
    assert report["confirmed_near_duplicate_pairs"][0]["relation"] == "cross_role"


def test_non_official_role_requires_explicit_split_binding(tmp_path: Path) -> None:
    dataset = tmp_path / "domain-a"
    dataset.mkdir()
    with np.testing.assert_raises_regex(ValueError, "explicit --split-file"):
        build_report(
            [("domain-a", dataset)],
            split_roles=("development_train",),
        )


def test_duplicate_dataset_names_are_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "domain-a"
    dataset.mkdir()
    with np.testing.assert_raises_regex(ValueError, "dataset names must be unique"):
        build_report(
            [("domain-a", dataset), ("domain-a", dataset)],
            split_roles=(),
        )


def test_quarantine_records_every_candidate_and_unique_train_endpoint(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.json"
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"preview")
    pair = {
        "left": {
            "dataset_name": "A",
            "split_role": "official_test",
            "image_id": "test-1",
            "image_sha256": "1" * 64,
        },
        "right": {
            "dataset_name": "A",
            "split_role": "official_train",
            "image_id": "train-1",
            "image_sha256": "2" * 64,
        },
        "candidate_id": "3" * 64,
        "phash_hamming_distance": 2,
        "confirmation_cosine": 0.999,
    }
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "rc-irstd.near-duplicate-audit.v1",
                "status": "review_required",
                "image_only": True,
                "labels_scores_checkpoints_or_metrics_read": False,
                "confirmed_near_duplicate_pair_count": 1,
                "confirmed_near_duplicate_pairs": [pair],
            }
        ),
        encoding="utf-8",
    )
    payload = build_quarantine(
        audit_path,
        preview_path,
        repository_root=tmp_path,
    )
    assert payload["total_excluded_official_train_ids"] == 1
    assert payload["candidate_decisions"][0]["candidate_id"] == "3" * 64
    assert payload["datasets"][0]["excluded_official_train_ids"] == ["train-1"]
