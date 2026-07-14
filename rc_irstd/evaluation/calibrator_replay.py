from __future__ import annotations

"""Hard-threshold exact replay for calibrator validation and final reporting."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from rc_irstd.data.score_records import ScoreRecord, load_score_record
from rc_irstd.episodes.meta_dataset import MetaEpisodeArrays, probability_to_logit


@dataclass(frozen=True)
class HardReplaySummary:
    budget_satisfaction_rate: float
    log_excess: float
    mean_pd: float
    worst_domain_pd: float
    num_episode_budget_pairs: int
    pixel_risk: np.ndarray
    pd: np.ndarray
    satisfied: np.ndarray
    domains: np.ndarray

    @property
    def rank_key(self) -> tuple[float, float, float]:
        """Pre-registered checkpoint order: BSR -> LogExcess -> Pd."""
        return (
            float(self.budget_satisfaction_rate),
            -float(self.log_excess),
            float(self.mean_pd),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("pixel_risk", "pd", "satisfied", "domains"):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["rank_key"] = list(self.rank_key)
        return payload


def _parse_paths(score_root: str, paths_json: str) -> list[Path]:
    values = json.loads(str(paths_json))
    if not isinstance(values, list) or not values:
        raise ValueError("query_paths must encode a non-empty JSON list")
    root = Path(score_root)
    return [(root / str(value)).resolve() for value in values]


def _episode_counts(records: list[ScoreRecord], eta: float) -> tuple[float, float]:
    false_pixels = 0
    total_pixels = 0
    detected_objects = 0
    total_objects = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for record in records:
        if record.mask is None:
            raise ValueError("Hard replay requires query masks")
        logits = probability_to_logit(record.probability)
        mask = np.asarray(record.mask).squeeze() > 0
        prediction = logits >= float(eta)
        false_pixels += int(np.logical_and(prediction, ~mask).sum())
        total_pixels += int(mask.size)
        labels, count = ndimage.label(mask, structure=structure)
        total_objects += int(count)
        for component_id in range(1, count + 1):
            if np.any(prediction[labels == component_id]):
                detected_objects += 1
    pixel_risk = float(false_pixels / max(total_pixels, 1))
    pd = float(detected_objects / max(total_objects, 1)) if total_objects > 0 else 0.0
    return pixel_risk, pd


class HardReplayEvaluator:
    """Cache query score records and replay arbitrary predicted logit thresholds."""

    def __init__(self, arrays: MetaEpisodeArrays) -> None:
        if arrays.score_roots is None:
            raise ValueError("Claim-bearing hard replay requires score_roots provenance")
        self.arrays = arrays
        self._cache: dict[int, list[ScoreRecord]] = {}

    def records(self, episode_index: int) -> list[ScoreRecord]:
        if episode_index not in self._cache:
            root = str(self.arrays.score_roots[episode_index])
            paths = _parse_paths(root, str(self.arrays.query_paths[episode_index]))
            self._cache[episode_index] = [
                load_score_record(path, require_mask=True, load_mask=True) for path in paths
            ]
        return self._cache[episode_index]

    def evaluate(
        self,
        threshold_logit: np.ndarray,
        budgets: np.ndarray | None = None,
        *,
        epsilon: float = 1e-12,
    ) -> HardReplaySummary:
        eta = np.asarray(threshold_logit, dtype=np.float64)
        budget_grid = (
            np.asarray(self.arrays.budgets, dtype=np.float64)
            if budgets is None
            else np.asarray(budgets, dtype=np.float64)
        )
        if eta.shape != (len(self.arrays.features), len(budget_grid)):
            raise ValueError(
                f"threshold_logit must have shape {(len(self.arrays.features), len(budget_grid))}"
            )
        pixel = np.zeros_like(eta, dtype=np.float64)
        pd = np.zeros_like(eta, dtype=np.float64)
        for episode_index in range(len(eta)):
            records = self.records(episode_index)
            for budget_index in range(len(budget_grid)):
                pixel[episode_index, budget_index], pd[episode_index, budget_index] = (
                    _episode_counts(records, float(eta[episode_index, budget_index]))
                )
        budget_matrix = np.broadcast_to(budget_grid[None, :], pixel.shape)
        satisfied = pixel <= budget_matrix
        log_excess = np.maximum(
            np.log10((pixel + epsilon) / (budget_matrix + epsilon)), 0.0
        )
        domain_means: list[float] = []
        for domain in np.unique(self.arrays.domains):
            domain_means.append(float(pd[self.arrays.domains == domain].mean()))
        return HardReplaySummary(
            budget_satisfaction_rate=float(satisfied.mean()),
            log_excess=float(log_excess.mean()),
            mean_pd=float(pd.mean()),
            worst_domain_pd=float(min(domain_means) if domain_means else 0.0),
            num_episode_budget_pairs=int(pixel.size),
            pixel_risk=pixel.astype(np.float32),
            pd=pd.astype(np.float32),
            satisfied=satisfied,
            domains=np.asarray(self.arrays.domains).astype(str),
        )
