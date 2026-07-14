"""Freeze conservative development quarantines from a reviewed image audit.

The input audit is image-only.  This resolver never reads masks, labels,
scores, checkpoints, or metrics.  Every confirmed official-train/official-test
pair is recorded individually and its official-train member is excluded from
all derived development roles.  Raw data and official split files are never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rc-irstd.aaai27-near-duplicate-quarantine.v1"
AUDIT_SCHEMA_VERSION = "rc-irstd.near-duplicate-audit.v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"evidence path must be inside repository: {path}") from error


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def build_quarantine(
    audit_path: Path,
    preview_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    audit = _load_json(audit_path)
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported near-duplicate audit schema")
    if audit.get("status") != "review_required":
        raise ValueError("resolver requires a review_required source audit")
    if audit.get("image_only") is not True or audit.get(
        "labels_scores_checkpoints_or_metrics_read"
    ) is not False:
        raise ValueError("source audit is not an image-only integrity audit")
    pairs = audit.get("confirmed_near_duplicate_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("source audit contains no confirmed candidate pairs")
    if len(pairs) != int(audit.get("confirmed_near_duplicate_pair_count", -1)):
        raise ValueError("source audit pair count is inconsistent")

    decisions: list[dict[str, Any]] = []
    excluded: dict[str, set[str]] = {}
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, Mapping):
            raise TypeError(f"candidate pair {index} must be an object")
        left = raw_pair.get("left")
        right = raw_pair.get("right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError(f"candidate pair {index} endpoints must be objects")
        if left.get("dataset_name") != right.get("dataset_name"):
            raise ValueError(
                "cross-dataset confirmed candidates require a separate domain-"
                "independence decision; blanket quarantine is forbidden"
            )
        endpoints = {str(left.get("split_role")): left, str(right.get("split_role")): right}
        if set(endpoints) != {"official_train", "official_test"}:
            raise ValueError(
                f"candidate pair {index} is not official_train/official_test"
            )
        train = endpoints["official_train"]
        test = endpoints["official_test"]
        dataset_name = str(train["dataset_name"])
        train_id = str(train["image_id"])
        excluded.setdefault(dataset_name, set()).add(train_id)
        decisions.append(
            {
                "candidate_index": index,
                "candidate_id": str(raw_pair["candidate_id"]),
                "dataset_name": dataset_name,
                "official_train_image_id": train_id,
                "official_train_image_sha256": str(train["image_sha256"]),
                "official_test_image_id": str(test["image_id"]),
                "official_test_image_sha256": str(test["image_sha256"]),
                "phash_hamming_distance": int(
                    raw_pair["phash_hamming_distance"]
                ),
                "confirmation_cosine": float(raw_pair["confirmation_cosine"]),
                "final_decision": "same_scene_related",
                "action": (
                    "exclude_official_train_member_from_all_development_roles"
                ),
            }
        )

    datasets = [
        {
            "dataset_name": dataset_name,
            "excluded_official_train_ids": sorted(image_ids),
            "excluded_count": len(image_ids),
        }
        for dataset_name, image_ids in sorted(excluded.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "aaai27_conservative_development_quarantine",
        "status": "resolved_by_development_quarantine",
        "source_audit": {
            "path": _portable(audit_path, repository_root),
            "sha256": sha256_file(audit_path),
            "confirmed_pair_count": len(decisions),
        },
        "visual_review": {
            "status": "complete",
            "review_type": "full_contact_sheet_pairwise_visual_inspection",
            "reviewer_identity": "Codex-assisted repository audit",
            "human_signoff_claimed": False,
            "preview_path": _portable(preview_path, repository_root),
            "preview_sha256": sha256_file(preview_path),
        },
        "decision_policy": {
            "candidate_classification": "same_scene_related",
            "candidate_action": (
                "exclude every implicated official-train ID from every derived "
                "development role"
            ),
            "official_test_labels_read": False,
            "raw_data_modified": False,
            "official_split_files_modified": False,
            "conservative_without_candidate_deletion": True,
        },
        "candidate_decisions": decisions,
        "datasets": datasets,
        "total_excluded_official_train_ids": sum(
            len(values) for values in excluded.values()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve()
    audit = Path(args.audit).expanduser()
    preview = Path(args.preview).expanduser()
    output = Path(args.output).expanduser()
    if not audit.is_absolute():
        audit = root / audit
    if not preview.is_absolute():
        preview = root / preview
    if not output.is_absolute():
        output = root / output
    for path in (audit, preview):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quarantine: {output}")
    payload = build_quarantine(audit, preview, repository_root=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
