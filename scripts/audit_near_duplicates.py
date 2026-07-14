"""Image-only perceptual near-duplicate audit for frozen dataset splits.

This audit never opens masks, labels, score maps, checkpoints, or metrics.  It
uses a preregistered pHash candidate stage followed by a fixed normalized-pixel
confirmation stage.  Confirmed pairs are review candidates, never silently
deleted or reassigned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from PIL import Image
from scipy.fft import dctn

from data_ext.split_utils import (
    read_split_entries,
    resolve_sample_file,
    resolve_split_file,
    sample_id_from_entry,
)


SCHEMA_VERSION = "rc-irstd.near-duplicate-audit.v1"
PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
CONFIRMATION_SIZE = 64


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resampling_lanczos() -> int:
    return getattr(Image, "Resampling", Image).LANCZOS


def grayscale_array(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize(
            (size, size),
            resample=_resampling_lanczos(),
        )
        return np.asarray(grayscale, dtype=np.float64) / 255.0


def phash64(array: np.ndarray) -> int:
    values = np.asarray(array, dtype=np.float64)
    if values.shape != (PHASH_SIZE, PHASH_SIZE):
        raise ValueError(f"pHash input must be {PHASH_SIZE}x{PHASH_SIZE}")
    values = values - float(values.mean())
    standard_deviation = float(values.std())
    if standard_deviation > 0.0:
        values = values / standard_deviation
    coefficients = dctn(values, type=2, norm="ortho")
    low = coefficients[
        :PHASH_LOW_FREQUENCY_SIZE,
        :PHASH_LOW_FREQUENCY_SIZE,
    ].reshape(-1)
    # Analytically zero DCT terms may differ by machine epsilon after a global
    # brightness/contrast transform.  Canonicalising them prevents spurious
    # hash flips without weakening any material low-frequency coefficient.
    tolerance = max(float(np.max(np.abs(low))) * 1e-12, 1e-15)
    low[np.abs(low) <= tolerance] = 0.0
    median = float(np.median(low[1:]))
    bits = low > median
    bits[0] = False
    result = 0
    for index, enabled in enumerate(bits.tolist()):
        if enabled:
            result |= 1 << index
    return result


def confirmation_signature(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    mean = float(values.mean())
    centered = values - mean
    norm = float(np.linalg.norm(centered))
    if norm > 0.0:
        centered = centered / norm
    return centered, mean, norm


def confirmation_cosine(
    left: tuple[np.ndarray, float, float],
    right: tuple[np.ndarray, float, float],
) -> float:
    left_values, left_mean, left_norm = left
    right_values, right_mean, right_norm = right
    if left_norm == 0.0 or right_norm == 0.0:
        if left_norm == 0.0 and right_norm == 0.0:
            return 1.0 if abs(left_mean - right_mean) <= (1.0 / 255.0) else 0.0
        return 0.0
    return float(np.clip(np.dot(left_values, right_values), -1.0, 1.0))


@dataclass(frozen=True)
class ImageRecord:
    dataset_name: str
    split_role: str
    image_id: str
    path: Path
    image_sha256: str
    phash: int
    confirmation: tuple[np.ndarray, float, float]

    def index_payload(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_role": self.split_role,
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "phash64_hex": f"{self.phash:016x}",
        }

    def pair_payload(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_role": self.split_role,
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
        }


def load_records(
    dataset_specs: Sequence[tuple[str, Path]],
    *,
    split_roles: Sequence[str],
    split_files: Mapping[tuple[str, str], Path] | None = None,
    repository_root: Path | None = None,
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    split_files = split_files or {}
    repository_root = (repository_root or Path.cwd()).resolve()
    dataset_names = [name for name, _ in dataset_specs]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError(f"dataset names must be unique: {dataset_names}")
    if len(split_roles) != len(set(split_roles)):
        raise ValueError(f"split roles must be unique: {list(split_roles)}")
    expected_bindings = {
        (dataset_name, role)
        for dataset_name in dataset_names
        for role in split_roles
        if role not in {"train", "test"}
    }
    provided_bindings = set(split_files)
    missing_bindings = expected_bindings.difference(provided_bindings)
    if missing_bindings:
        raise ValueError(
            "non-official split roles require explicit --split-file bindings: "
            f"{sorted(missing_bindings)}"
        )
    allowed_bindings = {
        (dataset_name, role)
        for dataset_name in dataset_names
        for role in split_roles
    }
    unused_bindings = provided_bindings.difference(allowed_bindings)
    if unused_bindings:
        raise ValueError(
            f"unused or unknown split-file bindings: {sorted(unused_bindings)}"
        )
    records: list[ImageRecord] = []
    inputs: list[dict[str, Any]] = []
    for dataset_name, dataset_root in sorted(dataset_specs):
        for role in split_roles:
            split_path = resolve_split_file(
                dataset_root,
                split=role,
                split_file=split_files.get((dataset_name, role)),
            )
            entries = read_split_entries(split_path)
            role_label = (
                f"official_{role}" if role in {"train", "test"} else role
            )
            try:
                portable_split = split_path.relative_to(repository_root).as_posix()
            except ValueError:
                portable_split = str(split_path)
            inputs.append(
                {
                    "dataset_name": dataset_name,
                    "split_role": role_label,
                    "split_file": portable_split,
                    "split_sha256": sha256_file(split_path),
                    "num_images": len(entries),
                }
            )
            for entry in entries:
                image_path = resolve_sample_file(
                    dataset_root,
                    "images",
                    entry,
                    kind="image",
                )
                phash_input = grayscale_array(image_path, PHASH_SIZE)
                confirmation_input = grayscale_array(image_path, CONFIRMATION_SIZE)
                records.append(
                    ImageRecord(
                        dataset_name=dataset_name,
                        split_role=role_label,
                        image_id=sample_id_from_entry(entry),
                        path=image_path,
                        image_sha256=sha256_file(image_path),
                        phash=phash64(phash_input),
                        confirmation=confirmation_signature(confirmation_input),
                    )
                )
    records.sort(key=lambda item: (item.dataset_name, item.split_role, item.image_id))
    return records, inputs


def _relation(left: ImageRecord, right: ImageRecord) -> str:
    cross_domain = left.dataset_name != right.dataset_name
    cross_role = left.split_role != right.split_role
    if cross_domain and cross_role:
        return "cross_domain_and_role"
    if cross_domain:
        return "cross_domain"
    return "cross_role"


def audit_records(
    records: Sequence[ImageRecord],
    *,
    phash_hamming_max: int,
    confirmation_cosine_min: float,
) -> dict[str, Any]:
    if phash_hamming_max < 0 or phash_hamming_max > 64:
        raise ValueError("phash_hamming_max must lie in [0, 64]")
    if not np.isfinite(confirmation_cosine_min) or not -1.0 <= confirmation_cosine_min <= 1.0:
        raise ValueError("confirmation_cosine_min must lie in [-1, 1]")
    ordered_records = sorted(
        records,
        key=lambda item: (item.dataset_name, item.split_role, item.image_id),
    )
    candidate_count = 0
    confirmed: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered_records):
        for right in ordered_records[left_index + 1 :]:
            if (
                left.dataset_name == right.dataset_name
                and left.split_role == right.split_role
            ):
                continue
            distance = int((left.phash ^ right.phash).bit_count())
            if distance > phash_hamming_max:
                continue
            candidate_count += 1
            cosine = confirmation_cosine(left.confirmation, right.confirmation)
            if cosine < confirmation_cosine_min:
                continue
            confirmed.append(
                {
                    "left": left.pair_payload(),
                    "right": right.pair_payload(),
                    "relation": _relation(left, right),
                    "phash_hamming_distance": distance,
                    "confirmation_cosine": cosine,
                    "exact_image_bytes": left.image_sha256 == right.image_sha256,
                    "candidate_id": hashlib.sha256(
                        json.dumps(
                            {
                                "left": left.pair_payload(),
                                "right": right.pair_payload(),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "resolution": "unresolved_requires_provenance_review_or_quarantine",
                }
            )

    index_lines = [
        json.dumps(item.index_payload(), sort_keys=True, separators=(",", ":"))
        for item in ordered_records
    ]
    index_sha = hashlib.sha256(("\n".join(index_lines) + "\n").encode("utf-8")).hexdigest()
    passed = not confirmed
    return {
        "artifact_type": "rc_irstd_image_only_near_duplicate_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "review_required",
        "near_duplicate_contract_passed": passed,
        "image_only": True,
        "labels_scores_checkpoints_or_metrics_read": False,
        "num_indexed_images": len(ordered_records),
        "image_index_sha256": index_sha,
        "phash_candidate_pair_count": candidate_count,
        "confirmed_near_duplicate_pair_count": len(confirmed),
        "confirmed_near_duplicate_pairs": confirmed,
    }


def build_report(
    dataset_specs: Sequence[tuple[str, Path]],
    *,
    split_roles: Sequence[str] = ("train", "test"),
    split_files: Mapping[tuple[str, str], Path] | None = None,
    repository_root: Path | None = None,
    phash_hamming_max: int = 4,
    confirmation_cosine_min: float = 0.995,
) -> dict[str, Any]:
    records, inputs = load_records(
        dataset_specs,
        split_roles=split_roles,
        split_files=split_files,
        repository_root=repository_root,
    )
    report = audit_records(
        records,
        phash_hamming_max=phash_hamming_max,
        confirmation_cosine_min=confirmation_cosine_min,
    )
    report.update(
        {
            "algorithm": {
                "candidate": (
                    "grayscale_mean_std_normalized_phash64_"
                    "DCT_32x32_low8x8_excluding_DC"
                ),
                "resize": "PIL_Lanczos",
                "phash_hamming_distance_max": phash_hamming_max,
                "confirmation": "64x64_grayscale_mean_centered_cosine",
                "confirmation_cosine_min": confirmation_cosine_min,
                "pillow_version": Image.__version__,
                "scipy_version": scipy.__version__,
            },
            "inputs": inputs,
        }
    )
    return report


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid dataset specification: {value}")
    return name.strip(), path


def parse_split_file(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "--split-file must use DATASET:ROLE=PATH"
        )
    binding, raw_path = value.split("=", 1)
    dataset_name, role = binding.split(":", 1)
    path = Path(raw_path).expanduser().resolve()
    if not dataset_name.strip() or not role.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid split-file binding: {value}")
    return dataset_name.strip(), role.strip(), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument(
        "--split-role",
        action="append",
        choices=("train", "test", "development_train"),
    )
    parser.add_argument(
        "--split-file",
        action="append",
        type=parse_split_file,
        help="Override one split with DATASET:ROLE=PATH.",
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--phash-hamming-max", type=int, default=4)
    parser.add_argument("--confirmation-cosine-min", type=float, default=0.995)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    split_files: dict[tuple[str, str], Path] = {}
    for dataset_name, role, path in args.split_file or []:
        key = (dataset_name, role)
        if key in split_files:
            raise ValueError(f"duplicate split-file override: {key}")
        split_files[key] = path
    report = build_report(
        args.dataset,
        split_roles=args.split_role or ("train", "test"),
        split_files=split_files,
        repository_root=Path(args.repository_root).expanduser().resolve(),
        phash_hamming_max=args.phash_hamming_max,
        confirmation_cosine_min=args.confirmation_cosine_min,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite near-duplicate audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_pass and report["near_duplicate_contract_passed"] is not True:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
