"""Exact grouped-query replay for no-Reject calibrator model selection.

This module is used only on pseudo-target validation episodes.  It reloads
the hash-bound native-resolution score maps and their independent labels,
applies the repository-wide strict ``probability > threshold`` rule, and uses
the same deterministic 8-connected one-to-one matching as final evaluation.
No sampled tail approximation is used for checkpoint selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.dataset_identity import sha256_file
from data_ext.label_manifest_artifacts import (
    load_label_mask,
    verify_label_attachment,
)
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from rc.meta_dataset import PixelRiskEpisodeGroup
from rc.schema import OFFICIAL_TRAIN_SPLIT_ROLE

from .component_matching import (
    PreparedTarget,
    aggregate_match_results,
    match_components,
    prepare_target,
)


@dataclass(frozen=True)
class ExactHardReplaySummary:
    """Validation metrics with the pre-registered lexicographic rank."""

    budget_satisfaction_rate: float
    log_excess: float
    mean_pd: float
    worst_domain_pd: float
    num_group_budget_pairs: int
    pixel_risk: np.ndarray
    pd: np.ndarray
    satisfied: np.ndarray
    pseudo_targets: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[float, float, float]:
        return (
            float(self.budget_satisfaction_rate),
            -float(self.log_excess),
            float(self.mean_pd),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("pixel_risk", "pd", "satisfied"):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["pseudo_targets"] = list(self.pseudo_targets)
        payload["rank_key"] = list(self.rank_key)
        payload["checkpoint_selection_order"] = ["BSR", "LogExcess", "Pd"]
        payload["threshold_semantics"] = "prediction = probability > threshold"
        payload["matching"] = "8_connected_one_to_one"
        return payload


@dataclass(frozen=True)
class _ReplayImage:
    probability: np.ndarray
    target: PreparedTarget


def _resolve_file(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"episode metadata requires {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


class ExactGroupedPixelRiskReplay:
    """Cache verified pseudo-target query arrays and replay threshold curves."""

    def __init__(
        self,
        groups: Sequence[PixelRiskEpisodeGroup],
        *,
        artifact_root: str | Path,
        matching_rule: str = "overlap",
        centroid_distance: float = 3.0,
    ) -> None:
        if not groups:
            raise ValueError("hard replay requires at least one episode group")
        if matching_rule not in {"overlap", "centroid"}:
            raise ValueError("matching_rule must be 'overlap' or 'centroid'")
        if not math.isfinite(float(centroid_distance)) or centroid_distance <= 0.0:
            raise ValueError("centroid_distance must be finite and positive")
        root = Path(artifact_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"artifact_root is not a directory: {root}")
        self.groups = tuple(groups)
        self.matching_rule = matching_rule
        self.centroid_distance = float(centroid_distance)
        self._images = tuple(self._load_group(group, root) for group in self.groups)

    def _load_group(
        self,
        group: PixelRiskEpisodeGroup,
        root: Path,
    ) -> tuple[_ReplayImage, ...]:
        episode = group.representative
        curve_manifest_path = _resolve_file(
            root,
            episode.metadata.get("curve_manifest_file"),
            name="curve_manifest_file",
        )
        if sha256_file(curve_manifest_path) != (
            episode.provenance.curve_manifest_sha256
        ):
            raise ValueError(
                "hard replay curve manifest differs from episode provenance"
            )
        payload = json.loads(curve_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("curve manifest must be a JSON object")
        if payload.get("matching_rule") != self.matching_rule or not math.isclose(
            float(payload.get("centroid_distance", float("nan"))),
            self.centroid_distance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "query curve matching contract differs from exact hard replay"
            )
        score_manifest_path = _resolve_file(
            curve_manifest_path.parent,
            payload.get("score_manifest_file"),
            name="curve score_manifest_file",
        )
        label_manifest_path = _resolve_file(
            curve_manifest_path.parent,
            payload.get("label_manifest_file"),
            name="curve label_manifest_file",
        )
        verified_scores = verify_score_manifest_artifacts(
            score_manifest_path,
            image_ids=episode.query_image_ids,
            require_mask=False,
            require_native_contract=True,
            required_split_role=OFFICIAL_TRAIN_SPLIT_ROLE,
        )
        provenance = episode.provenance
        if verified_scores.manifest_sha256 != provenance.query_score_manifest_sha256:
            raise ValueError("hard replay score manifest differs from episode provenance")
        attachment = verify_label_attachment(
            score_manifest_path,
            label_manifest_path,
            image_ids=episode.query_image_ids,
        )
        if attachment.score_manifest.manifest_sha256 != (
            provenance.query_score_manifest_sha256
        ):
            raise ValueError("hard replay label attachment rebound the score manifest")
        if attachment.manifest_sha256 != provenance.label_manifest_sha256:
            raise ValueError("hard replay label manifest differs from provenance")
        if attachment.content_sha256 != provenance.label_manifest_content_sha256:
            raise ValueError("hard replay label content differs from provenance")
        score_items = attachment.score_manifest.selected_items
        label_items = attachment.selected_items
        expected_ids = episode.query_image_ids
        observed_ids = tuple(item.image_id for item in score_items)
        if observed_ids != expected_ids or tuple(
            item.image_id for item in label_items
        ) != expected_ids:
            raise ValueError("hard replay query order differs from the episode")

        images: list[_ReplayImage] = []
        for score_item, label_item in zip(score_items, label_items):
            with np.load(score_item.score_path, allow_pickle=False) as score_payload:
                probability = np.asarray(score_payload["prob"], dtype=np.float64)
            target = load_label_mask(label_item)
            if probability.shape != target.shape:
                raise ValueError(
                    "hard replay score/label shape mismatch for "
                    f"{score_item.image_id!r}: {probability.shape} != {target.shape}"
                )
            if not np.isfinite(probability).all() or np.any(
                (probability < 0.0) | (probability > 1.0)
            ):
                raise ValueError("hard replay score map is not a finite probability")
            images.append(
                _ReplayImage(
                    probability=np.ascontiguousarray(probability),
                    target=prepare_target(target),
                )
            )
        return tuple(images)

    def evaluate(
        self,
        thresholds: np.ndarray,
        *,
        pixel_budget_grid: Sequence[float],
        epsilon: float = 1e-12,
    ) -> ExactHardReplaySummary:
        eta = np.asarray(thresholds, dtype=np.float64)
        budgets = np.asarray(pixel_budget_grid, dtype=np.float64).reshape(-1)
        expected_shape = (len(self.groups), budgets.size)
        if eta.shape != expected_shape:
            raise ValueError(f"thresholds must have shape {expected_shape}")
        if budgets.size < 2 or not np.all(np.isfinite(budgets)) or np.any(
            budgets <= 0.0
        ):
            raise ValueError("pixel_budget_grid must contain finite positive values")
        if not np.all(budgets[:-1] > budgets[1:]):
            raise ValueError("pixel_budget_grid must be strictly descending")
        if not np.isfinite(eta).all() or np.any((eta < 0.0) | (eta > 1.0)):
            raise ValueError("thresholds must be finite probabilities in [0, 1]")
        if np.any(np.diff(eta, axis=1) <= 0.0):
            raise ValueError("predicted thresholds must tighten strictly across budgets")
        if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

        pixel_risk = np.zeros(expected_shape, dtype=np.float64)
        pd = np.zeros(expected_shape, dtype=np.float64)
        for group_index, images in enumerate(self._images):
            for budget_index, threshold in enumerate(eta[group_index]):
                matches = [
                    match_components(
                        image.probability > float(threshold),
                        image.target,
                        rule=self.matching_rule,
                        centroid_distance=self.centroid_distance,
                    )
                    for image in images
                ]
                metrics = aggregate_match_results(matches)
                pixel_risk[group_index, budget_index] = float(metrics["fa_pixel"])
                pd[group_index, budget_index] = float(metrics["pd"])

        budget_matrix = np.broadcast_to(budgets[None, :], expected_shape)
        satisfied = pixel_risk <= budget_matrix
        log_excess = np.maximum(
            np.log10((pixel_risk + float(epsilon)) / (budget_matrix + float(epsilon))),
            0.0,
        )
        targets = tuple(group.pseudo_target for group in self.groups)
        domain_pd = [
            float(pd[np.asarray(targets) == target].mean())
            for target in sorted(set(targets))
        ]
        return ExactHardReplaySummary(
            budget_satisfaction_rate=float(satisfied.mean()),
            log_excess=float(log_excess.mean()),
            mean_pd=float(pd.mean()),
            worst_domain_pd=float(min(domain_pd)),
            num_group_budget_pairs=int(pixel_risk.size),
            pixel_risk=pixel_risk,
            pd=pd,
            satisfied=satisfied,
            pseudo_targets=targets,
        )


__all__ = ["ExactGroupedPixelRiskReplay", "ExactHardReplaySummary"]
