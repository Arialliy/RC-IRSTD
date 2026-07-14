"""Canonical, JSON-serialisable schema for RC-IRSTD meta episodes.

The schema deliberately separates the unlabeled context window from the
labeled query window used to create an oracle training target.  Context and
query image identifiers are validated as disjoint so target labels can never
leak into the statistics consumed by the calibrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rc-irstd.meta-episode.v3"
BUDGET_NAMES = ("pixel", "component")
VALID_THRESHOLD_TRANSFORMS = ("identity", "logit", "tail")
PROVENANCE_STATUSES = ("verified", "asserted_unverified")
MULTI_SOURCE_PROTOCOL_SCOPE = "multi_source_protocol_candidate"
SINGLE_SOURCE_SMOKE_SCOPE = "single_source_inner_smoke_not_main_result"
DETECTOR_PROTOCOL_SCOPES = (
    MULTI_SOURCE_PROTOCOL_SCOPE,
    SINGLE_SOURCE_SMOKE_SCOPE,
)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_sha256(value: str, name: str) -> str:
    value = str(value)
    if value != value.lower():
        raise ValueError(f"{name} must use lowercase hexadecimal")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_protocol_scope_cardinality(
    detector_source_domains: Sequence[str],
    protocol_scope: str | None,
    *,
    name: str,
    allow_none: bool,
) -> None:
    """Keep detector source count and the declared protocol scope inseparable."""

    if protocol_scope is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be present")
    if protocol_scope not in DETECTOR_PROTOCOL_SCOPES:
        raise ValueError(
            f"{name} must be one of {DETECTOR_PROTOCOL_SCOPES}, got {protocol_scope!r}"
        )
    source_count = len(tuple(detector_source_domains))
    if protocol_scope == MULTI_SOURCE_PROTOCOL_SCOPE and source_count < 2:
        raise ValueError(f"{name}=multi-source requires at least two detector sources")
    if protocol_scope == SINGLE_SOURCE_SMOKE_SCOPE and source_count != 1:
        raise ValueError(f"{name}=single-source smoke requires exactly one detector source")


@dataclass(frozen=True)
class StatisticsConfig:
    """Collection-wide contract for every statistic entering a calibrator."""

    peak_kernel_size: int
    peak_min_score: float
    probability_histogram_bins: int = 32
    peak_histogram_bins: int = 32
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)
    plateau_mode: str = "kernel_local_row_major_rank_nms"
    plateau_atol: float = 0.0
    grayscale_normalization: str = "dtype_or_robust_0_1"
    algorithm_version: str = "rc-domain-statistics-v2"

    def __post_init__(self) -> None:
        if self.peak_kernel_size <= 0 or self.peak_kernel_size % 2 == 0:
            raise ValueError("peak_kernel_size must be a positive odd integer")
        if not 0.0 <= self.peak_min_score <= 1.0:
            raise ValueError("peak_min_score must lie in [0, 1]")
        if self.probability_histogram_bins != 32 or self.peak_histogram_bins != 32:
            raise ValueError("the v2 feature schema fixes both histogram counts at 32")
        if self.quantiles != (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999):
            raise ValueError("the v2 feature schema fixes the seven probability quantiles")
        if self.plateau_mode != "kernel_local_row_major_rank_nms":
            raise ValueError("statistics plateau_mode must match detector local-peak loss")
        if self.plateau_atol < 0.0:
            raise ValueError("plateau_atol must be non-negative")
        if self.grayscale_normalization != "dtype_or_robust_0_1":
            raise ValueError("unsupported grayscale_normalization")
        if self.algorithm_version != "rc-domain-statistics-v2":
            raise ValueError("unsupported statistics algorithm_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_kernel_size": self.peak_kernel_size,
            "peak_min_score": self.peak_min_score,
            "probability_histogram_bins": self.probability_histogram_bins,
            "peak_histogram_bins": self.peak_histogram_bins,
            "quantiles": list(self.quantiles),
            "plateau_mode": self.plateau_mode,
            "plateau_atol": self.plateau_atol,
            "grayscale_normalization": self.grayscale_normalization,
            "algorithm_version": self.algorithm_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StatisticsConfig":
        return cls(
            peak_kernel_size=int(payload["peak_kernel_size"]),
            peak_min_score=float(payload["peak_min_score"]),
            probability_histogram_bins=int(payload.get("probability_histogram_bins", 32)),
            peak_histogram_bins=int(payload.get("peak_histogram_bins", 32)),
            quantiles=tuple(float(value) for value in payload.get("quantiles", cls.__dataclass_fields__["quantiles"].default)),
            plateau_mode=str(
                payload.get("plateau_mode", "kernel_local_row_major_rank_nms")
            ),
            plateau_atol=float(payload.get("plateau_atol", 0.0)),
            grayscale_normalization=str(
                payload.get("grayscale_normalization", "dtype_or_robust_0_1")
            ),
            algorithm_version=str(payload.get("algorithm_version", "rc-domain-statistics-v2")),
        )


@dataclass(frozen=True)
class SourceContract:
    """Detector/fold provenance embedded inside a source-reference NPZ."""

    detector_checkpoint_sha: str
    detector_source_domains: tuple[str, ...]
    outer_fold_id: str | None
    outer_target: str | None
    held_out_domains: tuple[str, ...]
    protocol_scope: str | None

    def __post_init__(self) -> None:
        _validate_sha256(
            self.detector_checkpoint_sha, "source_contract.detector_checkpoint_sha"
        )
        if not self.detector_source_domains or len(set(self.detector_source_domains)) != len(
            self.detector_source_domains
        ):
            raise ValueError(
                "source_contract.detector_source_domains must be non-empty and unique"
            )
        if any(not value or value != value.strip() for value in self.detector_source_domains):
            raise ValueError("source contract detector source domains must be non-empty")
        if len(set(self.held_out_domains)) != len(self.held_out_domains):
            raise ValueError("source_contract.held_out_domains must be unique")
        if any(not value or value != value.strip() for value in self.held_out_domains):
            raise ValueError("source contract held-out domains must be non-empty")
        overlap = set(self.detector_source_domains).intersection(self.held_out_domains)
        if overlap:
            raise ValueError(
                "source contract detector-source/held-out domains overlap: "
                f"{sorted(overlap)}"
            )
        outer_fields = (self.outer_fold_id, self.outer_target)
        if (outer_fields[0] is None) != (outer_fields[1] is None):
            raise ValueError(
                "source contract outer_fold_id and outer_target must both be set or null"
            )
        for name, value in (
            ("outer_fold_id", self.outer_fold_id),
            ("outer_target", self.outer_target),
            ("protocol_scope", self.protocol_scope),
        ):
            if value is not None and (not value or value != value.strip()):
                raise ValueError(f"source_contract.{name} must be non-empty when set")
        if self.outer_target is not None and self.outer_target not in self.held_out_domains:
            raise ValueError("source contract outer_target must occur in held_out_domains")
        _validate_protocol_scope_cardinality(
            self.detector_source_domains,
            self.protocol_scope,
            name="source_contract.protocol_scope",
            allow_none=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_checkpoint_sha": self.detector_checkpoint_sha,
            "detector_source_domains": list(self.detector_source_domains),
            "outer_fold_id": self.outer_fold_id,
            "outer_target": self.outer_target,
            "held_out_domains": list(self.held_out_domains),
            "protocol_scope": self.protocol_scope,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceContract":
        required = {
            "detector_checkpoint_sha",
            "detector_source_domains",
            "outer_fold_id",
            "outer_target",
            "held_out_domains",
            "protocol_scope",
        }
        missing = required.difference(payload)
        if missing:
            raise KeyError(f"source contract is missing: {sorted(missing)}")
        source_domains = payload["detector_source_domains"]
        held_out_domains = payload["held_out_domains"]
        for name, value in (
            ("detector_source_domains", source_domains),
            ("held_out_domains", held_out_domains),
        ):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"source_contract.{name} must be a sequence of strings")
        return cls(
            detector_checkpoint_sha=str(payload["detector_checkpoint_sha"]),
            detector_source_domains=tuple(str(value) for value in source_domains),
            outer_fold_id=(
                None if payload["outer_fold_id"] is None else str(payload["outer_fold_id"])
            ),
            outer_target=(
                None if payload["outer_target"] is None else str(payload["outer_target"])
            ),
            held_out_domains=tuple(str(value) for value in held_out_domains),
            protocol_scope=(
                None if payload["protocol_scope"] is None else str(payload["protocol_scope"])
            ),
        )


@dataclass(frozen=True)
class SourceReference:
    """Auditable source-domain centers used by one episode or deployment fold."""

    domains: tuple[str, ...]
    sha256: str
    centers: tuple[tuple[float, ...], ...]
    scale: tuple[float, ...]
    contract: SourceContract

    def __post_init__(self) -> None:
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("source reference domains must be non-empty and unique")
        if self.domains != self.contract.detector_source_domains:
            raise ValueError(
                "source reference domains must exactly match source contract "
                "detector_source_domains in checkpoint order"
            )
        _validate_sha256(self.sha256, "source_reference.sha256")
        if len(self.centers) != len(self.domains):
            raise ValueError("source reference needs one center per domain")
        if not self.centers or not self.scale:
            raise ValueError("source reference centers and scale must be non-empty")
        width = len(self.scale)
        if any(len(row) != width for row in self.centers):
            raise ValueError("source reference centers must match scale width")
        values = [value for row in self.centers for value in row] + list(self.scale)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("source reference centers/scale must be finite")
        if any(float(value) <= 0.0 for value in self.scale):
            raise ValueError("source reference scale must be strictly positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": list(self.domains),
            "sha256": self.sha256,
            "centers": [list(row) for row in self.centers],
            "scale": list(self.scale),
            "contract": self.contract.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceReference":
        return cls(
            domains=tuple(str(value) for value in payload["domains"]),
            sha256=str(payload["sha256"]),
            centers=tuple(
                tuple(float(value) for value in row) for row in payload["centers"]
            ),
            scale=tuple(float(value) for value in payload["scale"]),
            contract=SourceContract.from_dict(payload["contract"]),
        )


@dataclass(frozen=True)
class FoldContract:
    outer_fold_id: str
    outer_target: str
    detector_source_domains: tuple[str, ...]
    detector_checkpoint_sha: str
    held_out_domains: tuple[str, ...]
    protocol_scope: str

    def __post_init__(self) -> None:
        if not self.outer_fold_id or not self.outer_target:
            raise ValueError("outer_fold_id and outer_target must be non-empty")
        if not self.detector_source_domains or len(set(self.detector_source_domains)) != len(
            self.detector_source_domains
        ):
            raise ValueError("detector_source_domains must be non-empty and unique")
        if self.outer_target in self.detector_source_domains:
            raise ValueError("outer_target must not be a detector source domain")
        if not self.held_out_domains or len(set(self.held_out_domains)) != len(
            self.held_out_domains
        ):
            raise ValueError("held_out_domains must be non-empty and unique")
        if self.outer_target not in self.held_out_domains:
            raise ValueError("outer_target must occur in held_out_domains")
        overlap = set(self.detector_source_domains).intersection(self.held_out_domains)
        if overlap:
            raise ValueError(
                "detector source domains must be disjoint from held_out_domains: "
                f"{sorted(overlap)}"
            )
        if not self.protocol_scope or self.protocol_scope != self.protocol_scope.strip():
            raise ValueError("protocol_scope must be non-empty")
        _validate_protocol_scope_cardinality(
            self.detector_source_domains,
            self.protocol_scope,
            name="protocol_scope",
            allow_none=False,
        )
        _validate_sha256(self.detector_checkpoint_sha, "detector_checkpoint_sha")

    def assert_matches_source_reference(self, reference: SourceReference) -> None:
        contract = reference.contract
        expected = (
            self.detector_checkpoint_sha,
            self.detector_source_domains,
            self.outer_fold_id,
            self.outer_target,
            self.held_out_domains,
            self.protocol_scope,
        )
        actual = (
            contract.detector_checkpoint_sha,
            contract.detector_source_domains,
            contract.outer_fold_id,
            contract.outer_target,
            contract.held_out_domains,
            contract.protocol_scope,
        )
        if actual != expected:
            raise ValueError(
                "source reference detector/fold contract does not exactly match FoldContract"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_fold_id": self.outer_fold_id,
            "outer_target": self.outer_target,
            "detector_source_domains": list(self.detector_source_domains),
            "detector_checkpoint_sha": self.detector_checkpoint_sha,
            "held_out_domains": list(self.held_out_domains),
            "protocol_scope": self.protocol_scope,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoldContract":
        return cls(
            outer_fold_id=str(payload["outer_fold_id"]),
            outer_target=str(payload["outer_target"]),
            detector_source_domains=tuple(
                str(value) for value in payload["detector_source_domains"]
            ),
            detector_checkpoint_sha=str(payload["detector_checkpoint_sha"]),
            held_out_domains=tuple(str(value) for value in payload["held_out_domains"]),
            protocol_scope=str(payload["protocol_scope"]),
        )


@dataclass(frozen=True)
class EpisodeProvenance:
    status: str
    curve_file_sha256: str
    curve_manifest_sha256: str
    context_score_manifest_sha256: str
    query_score_manifest_sha256: str
    query_score_target_dataset: str
    label_manifest_sha256: str = ""
    label_manifest_content_sha256: str = ""

    def __post_init__(self) -> None:
        if self.status not in PROVENANCE_STATUSES:
            raise ValueError(f"provenance status must be one of {PROVENANCE_STATUSES}")
        for name in (
            "curve_file_sha256",
            "context_score_manifest_sha256",
            "query_score_manifest_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.status == "verified":
            _validate_sha256(self.curve_manifest_sha256, "curve_manifest_sha256")
            _validate_sha256(self.label_manifest_sha256, "label_manifest_sha256")
            _validate_sha256(
                self.label_manifest_content_sha256,
                "label_manifest_content_sha256",
            )
            if self.context_score_manifest_sha256 != self.query_score_manifest_sha256:
                raise ValueError(
                    "verified provenance requires one shared context/query score manifest"
                )
        elif self.curve_manifest_sha256:
            _validate_sha256(self.curve_manifest_sha256, "curve_manifest_sha256")
        for name in ("label_manifest_sha256", "label_manifest_content_sha256"):
            if getattr(self, name):
                _validate_sha256(getattr(self, name), name)
        if not self.query_score_target_dataset:
            raise ValueError("query_score_target_dataset must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "curve_file_sha256": self.curve_file_sha256,
            "curve_manifest_sha256": self.curve_manifest_sha256,
            "context_score_manifest_sha256": self.context_score_manifest_sha256,
            "query_score_manifest_sha256": self.query_score_manifest_sha256,
            "query_score_target_dataset": self.query_score_target_dataset,
            "label_manifest_sha256": self.label_manifest_sha256,
            "label_manifest_content_sha256": self.label_manifest_content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeProvenance":
        return cls(
            status=str(payload["status"]),
            curve_file_sha256=str(payload["curve_file_sha256"]),
            curve_manifest_sha256=str(payload["curve_manifest_sha256"]),
            context_score_manifest_sha256=str(
                payload["context_score_manifest_sha256"]
            ),
            query_score_manifest_sha256=str(payload["query_score_manifest_sha256"]),
            query_score_target_dataset=str(payload["query_score_target_dataset"]),
            label_manifest_sha256=str(payload.get("label_manifest_sha256", "")),
            label_manifest_content_sha256=str(
                payload.get("label_manifest_content_sha256", "")
            ),
        )


@dataclass(frozen=True)
class BudgetSpec:
    """Two risk budgets accompanied by an explicit activity mask.

    Inactive values remain present to keep every model input fixed-width, but
    they are ignored by feasibility checks and encoded as zero before their
    activity bits are appended.
    """

    values: tuple[float, float]
    active: tuple[bool, bool]

    def __post_init__(self) -> None:
        if len(self.values) != len(BUDGET_NAMES):
            raise ValueError(f"budget values must have length {len(BUDGET_NAMES)}")
        if len(self.active) != len(BUDGET_NAMES):
            raise ValueError(f"budget active mask must have length {len(BUDGET_NAMES)}")
        for name, value, active in zip(BUDGET_NAMES, self.values, self.active):
            value = _finite_float(value, f"{name}_budget")
            if active and value <= 0.0:
                raise ValueError(f"active {name} budget must be positive")

    @classmethod
    def from_optional(
        cls,
        pixel_budget: float | None = None,
        component_budget: float | None = None,
    ) -> "BudgetSpec":
        active = (pixel_budget is not None, component_budget is not None)
        if not any(active):
            raise ValueError("at least one risk budget must be active")
        values = (
            0.0 if pixel_budget is None else float(pixel_budget),
            0.0 if component_budget is None else float(component_budget),
        )
        return cls(values=values, active=active)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BudgetSpec":
        names = tuple(payload.get("names", BUDGET_NAMES))
        if names != BUDGET_NAMES:
            raise ValueError(f"budget names must be {BUDGET_NAMES}, got {names}")
        values = tuple(float(value) for value in payload["values"])
        active = tuple(bool(value) for value in payload["active"])
        return cls(values=values, active=active)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(BUDGET_NAMES),
            "values": list(self.values),
            "active": list(self.active),
        }

    def encoded(self, eps: float = 1e-12) -> tuple[float, float, float, float]:
        """Return log budgets followed by their two activity bits."""

        logs = tuple(
            math.log10(max(value, eps)) if active else 0.0
            for value, active in zip(self.values, self.active)
        )
        return logs + tuple(float(value) for value in self.active)  # type: ignore[return-value]

    def is_satisfied(self, pixel_risk: float, component_risk: float) -> bool:
        risks = (float(pixel_risk), float(component_risk))
        return all(
            (not active) or (math.isfinite(risk) and risk <= budget)
            for risk, budget, active in zip(risks, self.values, self.active)
        )


@dataclass(frozen=True)
class RCEpisode:
    """One causal context/query meta-learning episode."""

    episode_id: str
    pseudo_target: str
    context_image_ids: tuple[str, ...]
    query_image_ids: tuple[str, ...]
    statistics: tuple[float, ...]
    feature_names: tuple[str, ...]
    statistics_config: StatisticsConfig
    source_reference: SourceReference
    fold: FoldContract
    provenance: EpisodeProvenance
    budgets: BudgetSpec
    oracle_threshold: float
    oracle_pd: float
    oracle_pixel_risk: float
    oracle_component_risk: float
    p_min: float
    reject: bool
    threshold_transform: str = "identity"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported episode schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.pseudo_target:
            raise ValueError("pseudo_target must be non-empty")
        if self.pseudo_target == self.fold.outer_target:
            raise ValueError("pseudo_target must differ from the held-out outer_target")
        if self.pseudo_target in self.fold.detector_source_domains:
            raise ValueError("pseudo_target must not be a detector source domain")
        if self.pseudo_target not in self.fold.held_out_domains:
            raise ValueError("pseudo_target must occur in fold held_out_domains")
        if (
            self.provenance.status == "verified"
            and self.fold.protocol_scope != "multi_source_protocol_candidate"
        ):
            raise ValueError(
                "verified episodes require protocol_scope=multi_source_protocol_candidate"
            )
        if self.provenance.query_score_target_dataset != self.pseudo_target:
            raise ValueError("query score target_dataset must equal pseudo_target")
        self.fold.assert_matches_source_reference(self.source_reference)
        reference_domains = set(self.source_reference.domains)
        if self.pseudo_target in reference_domains or self.fold.outer_target in reference_domains:
            raise ValueError("pseudo_target/outer_target must not occur in source reference domains")
        if not self.context_image_ids:
            raise ValueError("context_image_ids must be non-empty")
        if not self.query_image_ids:
            raise ValueError("query_image_ids must be non-empty")
        if len(set(self.context_image_ids)) != len(self.context_image_ids):
            raise ValueError("context_image_ids contain duplicates")
        if len(set(self.query_image_ids)) != len(self.query_image_ids):
            raise ValueError("query_image_ids contain duplicates")
        overlap = set(self.context_image_ids).intersection(self.query_image_ids)
        if overlap:
            raise ValueError(f"context/query image IDs must be disjoint; overlap={sorted(overlap)}")
        if len(self.statistics) != len(self.feature_names):
            raise ValueError("statistics and feature_names must have identical lengths")
        if not self.statistics:
            raise ValueError("statistics must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if not all(math.isfinite(float(value)) for value in self.statistics):
            raise ValueError("statistics must contain only finite values")
        threshold = _finite_float(self.oracle_threshold, "oracle_threshold")
        pd = _finite_float(self.oracle_pd, "oracle_pd")
        p_min = _finite_float(self.p_min, "p_min")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("oracle_threshold must lie in [0, 1]")
        if not 0.0 <= pd <= 1.0:
            raise ValueError("oracle_pd must lie in [0, 1]")
        if not 0.0 <= p_min <= 1.0:
            raise ValueError("p_min must lie in [0, 1]")
        for name, risk in (
            ("oracle_pixel_risk", self.oracle_pixel_risk),
            ("oracle_component_risk", self.oracle_component_risk),
        ):
            if _finite_float(risk, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        expected_reject = pd < p_min
        if bool(self.reject) != expected_reject:
            raise ValueError(
                "reject must be defined by oracle_pd < p_min; "
                f"got reject={self.reject}, oracle_pd={pd}, p_min={p_min}"
            )
        if self.threshold_transform not in VALID_THRESHOLD_TRANSFORMS:
            raise ValueError(
                f"threshold_transform must be one of {VALID_THRESHOLD_TRANSFORMS}"
            )

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        pseudo_target: str,
        context_image_ids: Sequence[str],
        query_image_ids: Sequence[str],
        statistics: Sequence[float],
        feature_names: Sequence[str],
        statistics_config: StatisticsConfig,
        source_reference: SourceReference,
        fold: FoldContract,
        provenance: EpisodeProvenance,
        budgets: BudgetSpec,
        oracle_threshold: float,
        oracle_pd: float,
        oracle_pixel_risk: float,
        oracle_component_risk: float,
        p_min: float,
        threshold_transform: str = "identity",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RCEpisode":
        return cls(
            episode_id=str(episode_id),
            pseudo_target=str(pseudo_target),
            context_image_ids=tuple(str(value) for value in context_image_ids),
            query_image_ids=tuple(str(value) for value in query_image_ids),
            statistics=tuple(float(value) for value in statistics),
            feature_names=tuple(str(value) for value in feature_names),
            statistics_config=statistics_config,
            source_reference=source_reference,
            fold=fold,
            provenance=provenance,
            budgets=budgets,
            oracle_threshold=float(oracle_threshold),
            oracle_pd=float(oracle_pd),
            oracle_pixel_risk=float(oracle_pixel_risk),
            oracle_component_risk=float(oracle_component_risk),
            p_min=float(p_min),
            reject=float(oracle_pd) < float(p_min),
            threshold_transform=threshold_transform,
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def input_feature_names(self) -> tuple[str, ...]:
        return self.feature_names + (
            "budget_log10_pixel",
            "budget_log10_component",
            "budget_active_pixel",
            "budget_active_component",
        )

    def encoded_features(self) -> tuple[float, ...]:
        return self.statistics + self.budgets.encoded()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "pseudo_target": self.pseudo_target,
            "context_image_ids": list(self.context_image_ids),
            "query_image_ids": list(self.query_image_ids),
            "statistics": list(self.statistics),
            "feature_names": list(self.feature_names),
            "statistics_config": self.statistics_config.to_dict(),
            "source_reference": self.source_reference.to_dict(),
            "fold": self.fold.to_dict(),
            "provenance": self.provenance.to_dict(),
            "budgets": self.budgets.to_dict(),
            "oracle": {
                "threshold": self.oracle_threshold,
                "pd": self.oracle_pd,
                "pixel_risk": self.oracle_pixel_risk,
                "component_risk": self.oracle_component_risk,
                "p_min": self.p_min,
                "reject": self.reject,
            },
            "threshold_transform": self.threshold_transform,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RCEpisode":
        oracle = payload["oracle"]
        episode = cls(
            schema_version=str(payload.get("schema_version", "")),
            episode_id=str(payload["episode_id"]),
            pseudo_target=str(payload["pseudo_target"]),
            context_image_ids=tuple(str(value) for value in payload["context_image_ids"]),
            query_image_ids=tuple(str(value) for value in payload["query_image_ids"]),
            statistics=tuple(float(value) for value in payload["statistics"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            statistics_config=StatisticsConfig.from_dict(payload["statistics_config"]),
            source_reference=SourceReference.from_dict(payload["source_reference"]),
            fold=FoldContract.from_dict(payload["fold"]),
            provenance=EpisodeProvenance.from_dict(payload["provenance"]),
            budgets=BudgetSpec.from_dict(payload["budgets"]),
            oracle_threshold=float(oracle["threshold"]),
            oracle_pd=float(oracle["pd"]),
            oracle_pixel_risk=float(oracle["pixel_risk"]),
            oracle_component_risk=float(oracle["component_risk"]),
            p_min=float(oracle["p_min"]),
            reject=bool(oracle["reject"]),
            threshold_transform=str(payload.get("threshold_transform", "identity")),
            metadata=dict(payload.get("metadata", {})),
        )
        return episode

    @property
    def outer_fold_id(self) -> str:
        return self.fold.outer_fold_id

    @property
    def outer_target(self) -> str:
        return self.fold.outer_target

    @property
    def detector_source_domains(self) -> tuple[str, ...]:
        return self.fold.detector_source_domains

    @property
    def detector_checkpoint_sha(self) -> str:
        return self.fold.detector_checkpoint_sha
