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


LEGACY_EPISODE_SCHEMA_VERSION = "rc-irstd.meta-episode.v3"
SCHEMA_VERSION = "rc-irstd.meta-episode.v4"
SUPPORTED_EPISODE_SCHEMA_VERSIONS = (
    LEGACY_EPISODE_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
BUDGET_NAMES = ("pixel", "component")
VALID_THRESHOLD_TRANSFORMS = ("identity", "logit", "tail")
PROVENANCE_STATUSES = ("verified", "asserted_unverified")
MULTI_SOURCE_PROTOCOL_SCOPE = "multi_source_protocol_candidate"
SINGLE_SOURCE_SMOKE_SCOPE = "single_source_inner_smoke_not_main_result"
DETECTOR_PROTOCOL_SCOPES = (
    MULTI_SOURCE_PROTOCOL_SCOPE,
    SINGLE_SOURCE_SMOKE_SCOPE,
)
DEPLOYMENT_PROTOCOL_CONTRACT_VERSION = "rc-irstd.deployment-protocol.v1"
ONLINE_DECISION_CONTRACT_VERSION = "rc-irstd.online-decision.v1"
CAUSAL_PARTITION_RULE = "ordered_manifest_prefix_context_then_contiguous_query"
REJECT_SCORE_RULE = "sigmoid_reject_logit"
REJECT_COMPARISON_RULE = "greater_than_or_equal"
EVALUATION_MATCHING_CONTRACT_VERSION = "rc-irstd.evaluation-matching.v1"
DEFAULT_MATCHING_RULE = "overlap"
DEFAULT_CENTROID_DISTANCE = 3.0
STATISTICS_ALGORITHM_VERSION = "rc-domain-statistics-v3-bounded-quantiles"
STATISTICS_QUANTILE_ESTIMATOR = "splitmix64_priority_bottom_k_stream_index_v1"
STATISTICS_QUANTILE_SAMPLE_LIMIT = 262_144
SCORE_SPLIT_CONTRACT_VERSION = 1
OFFICIAL_TRAIN_SPLIT_ROLE = "official_train"
OFFICIAL_TEST_SPLIT_ROLE = "official_test"
SCORE_SPLIT_ROLES = (
    OFFICIAL_TRAIN_SPLIT_ROLE,
    OFFICIAL_TEST_SPLIT_ROLE,
)


_EPISODE_SCORE_SPLIT_FIELDS = (
    "schema_version",
    "role",
    "selected_split_file",
    "selected_split_sha256",
    "selected_num_images",
    "selected_ids_sha256",
    "official_train_split_file",
    "official_train_split_sha256",
    "official_train_num_images",
    "official_train_ids_sha256",
    "official_train_split_image_artifact_sha256",
    "official_test_split_file",
    "official_test_split_sha256",
    "official_test_num_images",
    "official_test_ids_sha256",
    "official_test_split_image_artifact_sha256",
    "ordered_sample_ids_algorithm",
    "split_image_artifact_algorithm",
    "train_test_id_overlap_count",
    "train_test_id_overlap_ids",
    "train_test_image_content_overlap_count",
    "train_test_image_content_overlap_sha256_leaves",
    "disjointness_verified",
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


def _strict_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def canonicalize_episode_score_split_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and compact the score manifest's official-split proof.

    The native score manifest carries ordered image-artifact leaves for both
    official splits.  Repeating those potentially large arrays in every meta
    episode would make JSONL collections quadratic in the number of windows,
    so an episode stores the immutable split/list/artifact digests plus the
    disjointness result.  The score-manifest SHA in :class:`EpisodeProvenance`
    remains the binding to the complete leaf lists.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("episode split_contract must be a mapping")
    missing = set(_EPISODE_SCORE_SPLIT_FIELDS).difference(payload)
    if missing:
        raise KeyError(
            "episode split_contract is missing required fields: "
            f"{sorted(missing)}"
        )

    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("episode split_contract schema_version must be an integer")
    if version != SCORE_SPLIT_CONTRACT_VERSION:
        raise ValueError(
            "unsupported episode split_contract schema_version: "
            f"{version!r}"
        )
    role = str(payload["role"])
    if role not in SCORE_SPLIT_ROLES:
        raise ValueError(
            f"episode split_contract role must be one of {SCORE_SPLIT_ROLES}"
        )

    result: dict[str, Any] = {
        "schema_version": version,
        "role": role,
    }
    for name in (
        "selected_split_file",
        "official_train_split_file",
        "official_test_split_file",
        "ordered_sample_ids_algorithm",
        "split_image_artifact_algorithm",
    ):
        value = str(payload[name])
        if not value or value != value.strip():
            raise ValueError(f"episode split_contract {name} must be non-empty")
        result[name] = value

    for name in (
        "selected_split_sha256",
        "selected_ids_sha256",
        "official_train_split_sha256",
        "official_train_ids_sha256",
        "official_train_split_image_artifact_sha256",
        "official_test_split_sha256",
        "official_test_ids_sha256",
        "official_test_split_image_artifact_sha256",
    ):
        result[name] = _validate_sha256(
            str(payload[name]), f"episode split_contract {name}"
        )

    for name in (
        "selected_num_images",
        "official_train_num_images",
        "official_test_num_images",
    ):
        value = _strict_nonnegative_integer(
            payload[name], f"episode split_contract {name}"
        )
        if value <= 0:
            raise ValueError(f"episode split_contract {name} must be positive")
        result[name] = value

    selected_prefix = role.removeprefix("official_")
    role_prefix = f"official_{selected_prefix}"
    expected_selected = {
        "selected_split_file": result[f"{role_prefix}_split_file"],
        "selected_split_sha256": result[f"{role_prefix}_split_sha256"],
        "selected_num_images": result[f"{role_prefix}_num_images"],
        "selected_ids_sha256": result[f"{role_prefix}_ids_sha256"],
    }
    for name, expected in expected_selected.items():
        if result[name] != expected:
            raise ValueError(
                f"episode split_contract {name} disagrees with role={role!r}"
            )

    for count_name, leaves_name in (
        ("train_test_id_overlap_count", "train_test_id_overlap_ids"),
        (
            "train_test_image_content_overlap_count",
            "train_test_image_content_overlap_sha256_leaves",
        ),
    ):
        count = _strict_nonnegative_integer(
            payload[count_name], f"episode split_contract {count_name}"
        )
        leaves = payload[leaves_name]
        if isinstance(leaves, (str, bytes)) or not isinstance(leaves, Sequence):
            raise TypeError(f"episode split_contract {leaves_name} must be a sequence")
        compact_leaves = [str(value) for value in leaves]
        if len(compact_leaves) != count:
            raise ValueError(
                f"episode split_contract {count_name} disagrees with {leaves_name}"
            )
        if count != 0:
            raise ValueError(
                "episode split_contract requires zero official train/test overlap; "
                f"{count_name}={count}"
            )
        result[count_name] = count
        result[leaves_name] = compact_leaves

    if payload["disjointness_verified"] is not True:
        raise ValueError(
            "episode split_contract disjointness_verified must be exactly true"
        )
    result["disjointness_verified"] = True
    return {name: result[name] for name in _EPISODE_SCORE_SPLIT_FIELDS}


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
    quantile_sample_limit: int = STATISTICS_QUANTILE_SAMPLE_LIMIT
    quantile_estimator: str = STATISTICS_QUANTILE_ESTIMATOR
    algorithm_version: str = STATISTICS_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.peak_kernel_size <= 0 or self.peak_kernel_size % 2 == 0:
            raise ValueError("peak_kernel_size must be a positive odd integer")
        if not 0.0 <= self.peak_min_score <= 1.0:
            raise ValueError("peak_min_score must lie in [0, 1]")
        if self.probability_histogram_bins != 32 or self.peak_histogram_bins != 32:
            raise ValueError("the v3 feature schema fixes both histogram counts at 32")
        if self.quantiles != (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999):
            raise ValueError("the v3 feature schema fixes the seven probability quantiles")
        if self.plateau_mode != "kernel_local_row_major_rank_nms":
            raise ValueError("statistics plateau_mode must match detector local-peak loss")
        if self.plateau_atol < 0.0:
            raise ValueError("plateau_atol must be non-negative")
        if self.grayscale_normalization != "dtype_or_robust_0_1":
            raise ValueError("unsupported grayscale_normalization")
        if isinstance(self.quantile_sample_limit, bool) or not isinstance(
            self.quantile_sample_limit, int
        ):
            raise TypeError("quantile_sample_limit must be an integer")
        if self.quantile_sample_limit <= 0:
            raise ValueError("quantile_sample_limit must be positive")
        if self.quantile_estimator != STATISTICS_QUANTILE_ESTIMATOR:
            raise ValueError("unsupported statistics quantile_estimator")
        if self.algorithm_version != STATISTICS_ALGORITHM_VERSION:
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
            "quantile_sample_limit": self.quantile_sample_limit,
            "quantile_estimator": self.quantile_estimator,
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
            quantile_sample_limit=payload.get(
                "quantile_sample_limit", STATISTICS_QUANTILE_SAMPLE_LIMIT
            ),
            quantile_estimator=str(
                payload.get("quantile_estimator", STATISTICS_QUANTILE_ESTIMATOR)
            ),
            algorithm_version=str(
                payload.get("algorithm_version", STATISTICS_ALGORITHM_VERSION)
            ),
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
class DeploymentProtocolContract:
    """Target-label-free parameters frozen before final-target adaptation.

    The context/query geometry is learned and validated on pseudo-target
    episodes.  The rejection cutoff is selected during calibrator training.
    Neither may be changed after the final target score manifest is opened in
    a claim-bearing run.
    """

    context_size: int
    query_size: int
    reject_cutoff: float
    partition_rule: str = CAUSAL_PARTITION_RULE
    reject_score: str = REJECT_SCORE_RULE
    reject_comparison: str = REJECT_COMPARISON_RULE
    matching_rule: str = DEFAULT_MATCHING_RULE
    centroid_distance: float = DEFAULT_CENTROID_DISTANCE
    target_reject_cutoff_override_allowed: bool = False
    schema_version: str = DEPLOYMENT_PROTOCOL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOYMENT_PROTOCOL_CONTRACT_VERSION:
            raise ValueError(
                "unsupported deployment protocol contract schema_version: "
                f"{self.schema_version!r}"
            )
        if isinstance(self.context_size, bool) or not isinstance(
            self.context_size, int
        ):
            raise TypeError("deployment context_size must be an integer")
        if isinstance(self.query_size, bool) or not isinstance(self.query_size, int):
            raise TypeError("deployment query_size must be an integer")
        if self.context_size <= 0 or self.query_size <= 0:
            raise ValueError("deployment context_size and query_size must be positive")
        cutoff = _finite_float(self.reject_cutoff, "deployment reject_cutoff")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("deployment reject_cutoff must lie in [0, 1]")
        if self.partition_rule != CAUSAL_PARTITION_RULE:
            raise ValueError("unsupported deployment causal partition rule")
        if self.reject_score != REJECT_SCORE_RULE:
            raise ValueError("unsupported deployment reject score rule")
        if self.reject_comparison != REJECT_COMPARISON_RULE:
            raise ValueError("unsupported deployment reject comparison rule")
        if self.matching_rule not in {"overlap", "centroid"}:
            raise ValueError("deployment matching_rule must be 'overlap' or 'centroid'")
        centroid_distance = _finite_float(
            self.centroid_distance, "deployment centroid_distance"
        )
        if centroid_distance <= 0.0:
            raise ValueError("deployment centroid_distance must be positive")
        if not isinstance(self.target_reject_cutoff_override_allowed, bool):
            raise TypeError(
                "target_reject_cutoff_override_allowed must be boolean"
            )
        if self.target_reject_cutoff_override_allowed:
            raise ValueError(
                "claim-bearing deployment must forbid final-target reject-cutoff overrides"
            )

    def assert_runtime_sizes(self, *, context_size: int, query_size: int) -> None:
        observed = (int(context_size), int(query_size))
        expected = (self.context_size, self.query_size)
        if observed != expected:
            raise ValueError(
                "online context/query sizes differ from the frozen deployment "
                f"contract: observed={observed}, expected={expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_size": self.context_size,
            "query_size": self.query_size,
            "partition_rule": self.partition_rule,
            "reject_score": self.reject_score,
            "reject_comparison": self.reject_comparison,
            "reject_cutoff": self.reject_cutoff,
            "evaluation_matching": {
                "schema_version": EVALUATION_MATCHING_CONTRACT_VERSION,
                "matching_rule": self.matching_rule,
                "centroid_distance": self.centroid_distance,
            },
            "target_reject_cutoff_override_allowed": (
                self.target_reject_cutoff_override_allowed
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentProtocolContract":
        required = {
            "schema_version",
            "context_size",
            "query_size",
            "partition_rule",
            "reject_score",
            "reject_comparison",
            "reject_cutoff",
            "target_reject_cutoff_override_allowed",
        }
        missing = required.difference(payload)
        if missing:
            raise KeyError(
                f"deployment protocol contract is missing: {sorted(missing)}"
            )
        for name in ("context_size", "query_size"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"deployment protocol {name} must be an integer")
        override_allowed = payload["target_reject_cutoff_override_allowed"]
        if not isinstance(override_allowed, bool):
            raise TypeError(
                "deployment protocol target_reject_cutoff_override_allowed must be boolean"
            )
        evaluation_matching = payload.get(
            "evaluation_matching",
            {
                "schema_version": EVALUATION_MATCHING_CONTRACT_VERSION,
                "matching_rule": DEFAULT_MATCHING_RULE,
                "centroid_distance": DEFAULT_CENTROID_DISTANCE,
            },
        )
        if not isinstance(evaluation_matching, Mapping):
            raise TypeError("deployment protocol evaluation_matching must be a mapping")
        matching_required = {
            "schema_version",
            "matching_rule",
            "centroid_distance",
        }
        matching_missing = matching_required.difference(evaluation_matching)
        if matching_missing:
            raise KeyError(
                "deployment evaluation matching contract is missing: "
                f"{sorted(matching_missing)}"
            )
        if (
            evaluation_matching["schema_version"]
            != EVALUATION_MATCHING_CONTRACT_VERSION
        ):
            raise ValueError("unsupported evaluation matching contract schema_version")
        return cls(
            schema_version=str(payload["schema_version"]),
            context_size=payload["context_size"],
            query_size=payload["query_size"],
            partition_rule=str(payload["partition_rule"]),
            reject_score=str(payload["reject_score"]),
            reject_comparison=str(payload["reject_comparison"]),
            reject_cutoff=float(payload["reject_cutoff"]),
            matching_rule=str(evaluation_matching["matching_rule"]),
            centroid_distance=float(evaluation_matching["centroid_distance"]),
            target_reject_cutoff_override_allowed=override_allowed,
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
    split_contract: Mapping[str, Any] | None = None

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
        if self.split_contract is not None:
            object.__setattr__(
                self,
                "split_contract",
                canonicalize_episode_score_split_contract(self.split_contract),
            )

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
            "split_contract": (
                None if self.split_contract is None else dict(self.split_contract)
            ),
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
            split_contract=(
                None
                if payload.get("split_contract") is None
                else payload["split_contract"]
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
        if self.schema_version not in SUPPORTED_EPISODE_SCHEMA_VERSIONS:
            raise ValueError(
                "unsupported episode schema "
                f"{self.schema_version!r}; expected one of "
                f"{SUPPORTED_EPISODE_SCHEMA_VERSIONS!r}"
            )
        if (
            self.schema_version == SCHEMA_VERSION
            and self.provenance.split_contract is None
        ):
            raise ValueError(
                f"episode schema {SCHEMA_VERSION!r} requires provenance.split_contract"
            )
        if (
            self.schema_version == LEGACY_EPISODE_SCHEMA_VERSION
            and self.provenance.split_contract is not None
        ):
            raise ValueError(
                "legacy meta-episode v3 must not claim the v4 official-split proof"
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
