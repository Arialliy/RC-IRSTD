"""Risk-calibration components for RC-IRSTD."""

from .domain_statistics import (
    BASE_FEATURE_DIM,
    FEATURE_DIM,
    FEATURE_NAMES,
    DomainStatistics,
    extract_domain_statistics,
    extract_unlabeled_statistics,
    load_source_reference,
)
from .oracle_threshold import OracleResult, oracle_safe_threshold, select_oracle_operating_point
from .schema import (
    BudgetSpec,
    EpisodeProvenance,
    FoldContract,
    RCEpisode,
    SCHEMA_VERSION,
    SourceContract,
    SourceReference,
    StatisticsConfig,
)

__all__ = [
    "BASE_FEATURE_DIM",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "BudgetSpec",
    "DomainStatistics",
    "EpisodeProvenance",
    "FoldContract",
    "OracleResult",
    "RCEpisode",
    "SCHEMA_VERSION",
    "SourceContract",
    "SourceReference",
    "StatisticsConfig",
    "extract_domain_statistics",
    "extract_unlabeled_statistics",
    "load_source_reference",
    "oracle_safe_threshold",
    "select_oracle_operating_point",
]


def build_source_reference(*args, **kwargs):
    """Lazily expose the builder without pre-importing its ``python -m`` module."""

    from .build_source_reference import build_source_reference as implementation

    return implementation(*args, **kwargs)


__all__.append("build_source_reference")
