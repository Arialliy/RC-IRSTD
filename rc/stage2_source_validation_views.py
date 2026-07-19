"""Non-interchangeable RC5 source-validation views.

Epoch selection uses an exhaustive C14/Q28 cyclic view over every start in
each of the two source-validation domains. BSR and LogExcess are averaged over
starts; Pd pools TP/GT within a domain; domains then receive weight 1/2.
Mandatory variable-Q/all-once validation is evaluated separately as an
extrapolation sanity record and is explicitly excluded from epoch ranking.
No cyclic start is claimed independent and no cyclic-start CI is produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model.endpoint_aware_threshold import EndpointAwareThresholdError, decode_coordinate_numpy
from rc.stage2_calibrator_generation_v2 import build_selection_record
from rc.stage2_compositional_curve_provider import (
    PerImageExactEventCurve,
    PerImageExactEventCurveBank,
    assert_per_image_exact_event_curve,
    build_compositional_exact_curve_provider,
    build_per_image_exact_event_curve_bank,
)
from rc.stage2_cyclic_training_geometry import build_stage2_cyclic_training_geometry
from rc.stage2_domain_balanced_cyclic_sampler import DOMAIN_ORDER, OUTER_TARGETS
from rc.stage2_variable_query_geometry import SCHEMA_VERSION as VARIABLE_QUERY_SCHEMA


CYCLIC_SELECTION_SCHEMA = "rc-irstd.stage2-source-validation-cyclic-selection-view.v1"
CYCLIC_SELECTION_GEOMETRY = "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
VARIABLE_QUERY_SANITY_SCHEMA = "rc-irstd.stage2-source-variable-query-sanity-view.v1"
_CYCLIC_CAPABILITY = object()
_SANITY_CAPABILITY = object()


class Stage2SourceValidationViewError(ValueError):
    """A source validation geometry or ranking boundary failed closed."""


def _sources(outer_fold_id: str) -> tuple[str, str]:
    if outer_fold_id not in OUTER_TARGETS:
        raise Stage2SourceValidationViewError("unsupported outer fold")
    target = OUTER_TARGETS[outer_fold_id]
    return tuple(domain for domain in DOMAIN_ORDER if domain != target)  # type: ignore[return-value]


def _features(value: Any, name: str, *, minimum: int = 1) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or \
            value.ndim != 2 or value.shape[1] != 93 or value.shape[0] < minimum:
        raise TypeError(f"{name} must be explicit float32[N,93]")
    if not np.isfinite(value).all():
        raise Stage2SourceValidationViewError(f"{name} contains NaN/Inf")
    owned = np.array(value, dtype=np.float32, order="C", copy=True); owned.setflags(write=False)
    return owned


def _anchors(value: Any, count: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or \
            value.shape != (count, 3) or not np.isfinite(value).all():
        raise TypeError(f"{name} must be finite float64[N,3]")
    try: decode_coordinate_numpy(value)
    except EndpointAwareThresholdError as error:
        raise Stage2SourceValidationViewError(f"{name} is not canonical EATC-v2") from error
    if np.any(value[:, 1:] < value[:, :-1]):
        raise Stage2SourceValidationViewError(f"{name} decreased")
    owned = np.array(value, dtype=np.float64, order="C", copy=True); owned.setflags(write=False)
    return owned


@dataclass(frozen=True, init=False)
class VerifiedSourceValidationCyclicSelectionView:
    outer_fold_id: str
    outer_target: str
    source_domains: tuple[str, str]
    domain_materials: Mapping[str, Mapping[str, Any]]
    artifact_scope: str
    identity_sha256: str
    boundary_values: Mapping[str, frozenset[str]]
    upstream_bindings: Mapping[str, str]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("cyclic selection view is verifier-issued only")

    def provider_for_start(self, domain: str, start: int):
        view = assert_verified_source_validation_cyclic_selection_view(self)
        if domain not in view.domain_materials:
            raise KeyError("unknown source-validation domain")
        material = view.domain_materials[domain]
        geometry = material["geometry"]
        if type(start) is not int or not 0 <= start < len(geometry["episodes"]):
            raise IndexError("cyclic validation start is out of range")
        query = geometry["episodes"][start]["query_indices"]
        identities = material["image_identities"]
        return build_compositional_exact_curve_provider(
            curve_bank=material["curve_bank"],
            ordered_image_identities=[identities[index] for index in query],
        )


def build_synthetic_source_validation_cyclic_selection_view(
    *, outer_fold_id: str, domain_materials: Mapping[str, Mapping[str, Any]]
) -> VerifiedSourceValidationCyclicSelectionView:
    """Build a test-only full-fit source selection view; target is forbidden."""
    sources = _sources(outer_fold_id)
    if not isinstance(domain_materials, Mapping) or set(domain_materials) != set(sources):
        raise Stage2SourceValidationViewError("selection view needs exactly two sources")
    checked: dict[str, Mapping[str, Any]] = {}
    globally_seen: set[str] = set()
    for domain in sources:
        row = domain_materials[domain]
        if not isinstance(row, Mapping) or set(row) != {
            "image_identities", "context_features", "anchor_coordinates", "per_image_curves"
        }:
            raise Stage2SourceValidationViewError("cyclic domain material closure mismatch")
        identities = tuple(str(item) for item in row["image_identities"])
        if len(identities) < 42 or len(set(identities)) != len(identities) or \
                globally_seen.intersection(identities):
            raise Stage2SourceValidationViewError("validation identities invalid/overlap")
        globally_seen.update(identities)
        features = _features(row["context_features"], f"{domain}.features", minimum=42)
        anchors = _anchors(row["anchor_coordinates"], len(identities), f"{domain}.anchors")
        curves = tuple(assert_per_image_exact_event_curve(item)
                       for item in row["per_image_curves"])
        if features.shape[0] != len(identities) or len(curves) != len(identities) or \
                tuple(item.image_identity_sha256 for item in curves) != identities:
            raise Stage2SourceValidationViewError("validation material cardinality/order mismatch")
        checked[domain] = MappingProxyType({
            "image_identities": identities, "context_features": features,
            "anchor_coordinates": anchors,
            "curve_bank": build_per_image_exact_event_curve_bank(curves),
            "geometry": build_stage2_cyclic_training_geometry(len(identities)),
            "role": "source_diagnostic_validation_detector_full_fit_only",
        })
    projection = {
        "schema_version": CYCLIC_SELECTION_SCHEMA,
        "outer_fold_id": outer_fold_id,
        "source_domains": list(sources),
        "artifact_scope": "synthetic_cpu_contract_test",
        "domains": [{
            "source_domain": domain,
            "record_count": len(checked[domain]["image_identities"]),
            "ordered_image_identities": list(checked[domain]["image_identities"]),
            "context_features_sha256": hashlib.sha256(
                checked[domain]["context_features"].tobytes(order="C")).hexdigest(),
            "anchor_coordinates_sha256": hashlib.sha256(
                checked[domain]["anchor_coordinates"].tobytes(order="C")).hexdigest(),
            "curve_bank_id": checked[domain]["curve_bank"].bank_id,
        } for domain in sources],
        "outer_target_records_present": False,
    }
    identity = hashlib.sha256(json.dumps(
        projection, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    synthetic_boundaries = frozenset(globally_seen)
    value = object.__new__(VerifiedSourceValidationCyclicSelectionView)
    fields = {"outer_fold_id": outer_fold_id, "outer_target": OUTER_TARGETS[outer_fold_id],
              "source_domains": sources, "domain_materials": MappingProxyType(checked),
              "artifact_scope": "synthetic_cpu_contract_test", "identity_sha256": identity,
              "boundary_values": MappingProxyType({
                  "canonical_id": synthetic_boundaries,
                  "original_image_sha256": synthetic_boundaries,
                  "near_duplicate_cluster_id_or_unique_sentinel": synthetic_boundaries,
                  "exclusion_group_id": synthetic_boundaries,
              }), "upstream_bindings": MappingProxyType({
                  "statistics_config_sha256": hashlib.sha256(
                      b"synthetic-statistics-config").hexdigest(),
                  "source_reference_set_sha256": hashlib.sha256(
                      b"synthetic-source-reference-set").hexdigest(),
                  "detector_run_complete_set_sha256": hashlib.sha256(
                      b"synthetic-validation-run-complete-set").hexdigest(),
              }), "_capability": _CYCLIC_CAPABILITY}
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


def _replay_production_context_bundle(context_bundle: Any, score_bundle: Any):
    from rc.build_stage2_rc5_context import (
        assert_verified_stage2_rc5_context_bundle,
        verify_stage2_rc5_context_bundle,
    )
    context = assert_verified_stage2_rc5_context_bundle(context_bundle)
    metadata = score_bundle.score_manifest_metadata
    if (
        context.score_bundle.attestation_sha256
        != score_bundle.attestation_sha256
        or context.score_bundle.run_complete.sha256
        != score_bundle.run_complete.sha256
        or context.score_manifest_metadata.manifest_sha256
        != metadata.manifest_sha256
    ):
        raise Stage2SourceValidationViewError("context/score attestation mismatch")
    binding = context.producer_manifest["inputs"]["statistics_config"]
    replayed = verify_stage2_rc5_context_bundle(
        context.commit_path, context.commit_sha256,
        variable_query_window=context.variable_query_window,
        score_bundle=score_bundle,
        source_reference=context.source_reference,
        statistics_config=context.statistics_config,
        statistics_config_path=metadata.repository_root / str(binding["path"]),
        statistics_config_sha256=str(binding["sha256"]),
        repository_root=metadata.repository_root,
    )
    return (
        replayed,
        str(binding["sha256"]),
        replayed.source_reference.attestation_sha256,
    )


def build_source_validation_cyclic_selection_view_from_verified_bundles(
    *, outer_fold_id: str, domain_bundles: Mapping[str, Mapping[str, Any]],
) -> VerifiedSourceValidationCyclicSelectionView:
    """Production full-fit selection from non-interchangeable cyclic authority."""
    from rc.stage2_rc5_cyclic_context import (
        assert_verified_stage2_rc5_cyclic_context_collection,
        replay_verified_stage2_rc5_cyclic_context_collection,
    )

    sources = _sources(outer_fold_id)
    if not isinstance(domain_bundles, Mapping) or set(domain_bundles) != set(sources):
        raise Stage2SourceValidationViewError("production selection needs two sources")
    checked: dict[str, Mapping[str, Any]] = {}
    boundary_fields = ("canonical_id", "original_image_sha256",
                       "near_duplicate_cluster_id_or_unique_sentinel",
                       "exclusion_group_id")
    boundaries = {field: set() for field in boundary_fields}
    statistics_values: set[str] = set()
    source_references: set[str] = set()
    run_bindings = []
    for domain in sources:
        supplied = domain_bundles[domain]
        if not isinstance(supplied, Mapping) or set(supplied) != {
            "cyclic_context_collection", "per_image_curves"
        }:
            raise Stage2SourceValidationViewError("production domain closure mismatch")
        cyclic = replay_verified_stage2_rc5_cyclic_context_collection(
            assert_verified_stage2_rc5_cyclic_context_collection(
                supplied["cyclic_context_collection"]
            )
        )
        score = cyclic.score_bundle
        metadata = score.score_manifest_metadata
        if metadata.role != "source_diagnostic_validation" or \
                metadata.payload["outer_fold_id"] != outer_fold_id or \
                metadata.payload["source_domain"] != domain:
            raise Stage2SourceValidationViewError("source validation score role mismatch")
        identities = tuple(str(record["original_image_sha256"])
                           for record in metadata.records)
        geometry = build_stage2_cyclic_training_geometry(len(identities))
        if cyclic.manifest["cyclic_geometry_sha256"] != hashlib.sha256(
            json.dumps(
                geometry,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest():
            raise Stage2SourceValidationViewError(
                "cyclic validation geometry identity mismatch"
            )
        for start, episode in enumerate(cyclic.episodes):
            expected = geometry["episodes"][start]
            if tuple(episode["context_indices"]) != tuple(expected["context_indices"]) or \
                    tuple(episode["query_indices"]) != tuple(expected["query_indices"]):
                raise Stage2SourceValidationViewError(
                    "cyclic validation episode differs from geometry replay"
                )
        inputs = cyclic.manifest["input_bindings"]
        statistics_values.add(str(inputs["statistics_config"]["sha256"]))
        source_references.add(str(inputs["source_reference_v3"]["sha256"]))
        curves = tuple(assert_per_image_exact_event_curve(item)
                       for item in supplied["per_image_curves"])
        if len(curves) != len(identities) or tuple(
            item.image_identity_sha256 for item in curves
        ) != identities:
            raise Stage2SourceValidationViewError("validation curve/image order mismatch")
        feature_array = _features(
            np.asarray(cyclic.context_features, dtype=np.float32),
            f"{domain}.features", minimum=42,
        )
        anchor_array = _anchors(
            np.asarray(cyclic.anchor_coordinates, dtype=np.float64),
            len(identities), f"{domain}.anchors",
        )
        bank = build_per_image_exact_event_curve_bank(curves)
        checked[domain] = MappingProxyType({
            "image_identities": identities, "context_features": feature_array,
            "anchor_coordinates": anchor_array, "curve_bank": bank,
            "geometry": geometry,
            "cyclic_context_collection_sha256": cyclic.commit_sha256,
            "role": "source_diagnostic_validation_detector_full_fit_only",
        })
        for field in boundary_fields:
            boundaries[field].update(str(record[field]) for record in metadata.records)
        run_identity = score.attestation["run_complete"]["identity"]
        run_bindings.append({
            "score_attestation_sha256": score.attestation_sha256,
            "run_complete_artifact_sha256": score.run_complete.sha256,
            "run_complete_identity_sha256": str(run_identity["identity_sha256"]),
        })
    if len(statistics_values) != 1:
        raise Stage2SourceValidationViewError("validation contexts use different statistics configs")
    upstream = {
        "statistics_config_sha256": next(iter(statistics_values)),
        "source_reference_set_sha256": hashlib.sha256(json.dumps(
            sorted(source_references), separators=(",", ":")
        ).encode()).hexdigest(),
        "detector_run_complete_set_sha256": hashlib.sha256(json.dumps(
            sorted(run_bindings, key=lambda row: row["run_complete_identity_sha256"]),
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }
    projection = {
        "schema_version": CYCLIC_SELECTION_SCHEMA, "outer_fold_id": outer_fold_id,
        "source_domains": list(sources), "artifact_scope": "production",
        "domains": [{"source_domain": domain,
                     "record_count": len(checked[domain]["image_identities"]),
                     "ordered_image_identities": list(checked[domain]["image_identities"]),
                     "context_features_sha256": hashlib.sha256(
                         checked[domain]["context_features"].tobytes()).hexdigest(),
                     "anchor_coordinates_sha256": hashlib.sha256(
                         checked[domain]["anchor_coordinates"].tobytes()).hexdigest(),
                     "curve_bank_id": checked[domain]["curve_bank"].bank_id,
                     "cyclic_context_collection_sha256": checked[domain][
                         "cyclic_context_collection_sha256"
                     ]}
                    for domain in sources],
        "upstream_bindings": upstream, "outer_target_records_present": False,
    }
    identity = hashlib.sha256(json.dumps(
        projection, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    value = object.__new__(VerifiedSourceValidationCyclicSelectionView)
    fields = {"outer_fold_id": outer_fold_id, "outer_target": OUTER_TARGETS[outer_fold_id],
              "source_domains": sources, "domain_materials": MappingProxyType(checked),
              "artifact_scope": "production", "identity_sha256": identity,
              "boundary_values": MappingProxyType(
                  {field: frozenset(values) for field, values in boundaries.items()}),
              "upstream_bindings": MappingProxyType(upstream),
              "_capability": _CYCLIC_CAPABILITY}
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


def assert_verified_source_validation_cyclic_selection_view(
    value: object,
) -> VerifiedSourceValidationCyclicSelectionView:
    if type(value) is not VerifiedSourceValidationCyclicSelectionView or getattr(
        value, "_capability", None
    ) is not _CYCLIC_CAPABILITY:
        raise TypeError("a verified source cyclic selection view is required")
    return value


@dataclass(frozen=True, init=False)
class VerifiedSourceVariableQuerySanityView:
    outer_fold_id: str
    outer_target: str
    source_domains: tuple[str, str]
    rows: tuple[Mapping[str, Any], ...]
    artifact_scope: str
    identity_sha256: str
    boundary_values: Mapping[str, frozenset[str]]
    upstream_bindings: Mapping[str, str]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("variable-Q sanity view is verifier-issued only")


def build_synthetic_source_variable_query_sanity_view(
    *, outer_fold_id: str, rows: Sequence[Mapping[str, Any]]
) -> VerifiedSourceVariableQuerySanityView:
    """Build mandatory variable-Q/all-once sanity data, never ranking data."""
    sources = _sources(outer_fold_id)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TypeError("rows must be a nonempty sequence")
    checked = []
    domains_seen: set[str] = set()
    for index, row in enumerate(rows):
        required = {"source_domain", "query_size", "context_features",
                    "anchor_coordinates", "aggregate_curve"}
        if not isinstance(row, Mapping) or set(row) != required:
            raise Stage2SourceValidationViewError("sanity row closure mismatch")
        domain = row["source_domain"]
        if domain not in sources:
            raise Stage2SourceValidationViewError("sanity row is not source-only")
        query_size = row["query_size"]
        if type(query_size) is not int or query_size < 28:
            raise Stage2SourceValidationViewError("variable-Q query_size must be >=28")
        feature = _features(np.asarray(row["context_features"])[None, :]
                            if np.asarray(row["context_features"]).ndim == 1
                            else row["context_features"], f"sanity[{index}].feature")
        if feature.shape[0] != 1:
            raise Stage2SourceValidationViewError("one feature per sanity window required")
        anchor_raw = np.asarray(row["anchor_coordinates"])
        anchor = _anchors(anchor_raw[None, :] if anchor_raw.ndim == 1 else anchor_raw,
                          1, f"sanity[{index}].anchor")
        curve = assert_per_image_exact_event_curve(row["aggregate_curve"])
        checked.append(MappingProxyType({
            "source_domain": domain, "query_size": query_size,
            "context_features": feature[0], "anchor_coordinates": anchor[0],
            "aggregate_curve": curve,
        }))
        domains_seen.add(domain)
    if domains_seen != set(sources):
        raise Stage2SourceValidationViewError("sanity view misses a source domain")
    projection = {
        "schema_version": VARIABLE_QUERY_SANITY_SCHEMA,
        "outer_fold_id": outer_fold_id,
        "artifact_scope": "synthetic_cpu_contract_test",
        "rows": [{"source_domain": row["source_domain"],
                  "query_size": row["query_size"],
                  "feature_sha256": hashlib.sha256(
                      row["context_features"].tobytes()).hexdigest(),
                  "anchor_sha256": hashlib.sha256(
                      row["anchor_coordinates"].tobytes()).hexdigest(),
                  "curve_content_sha256": row["aggregate_curve"].content_sha256}
                 for row in checked],
        "excluded_from_epoch_ranking": True,
    }
    identity = hashlib.sha256(json.dumps(
        projection, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    sanity_boundaries = frozenset(
        row["aggregate_curve"].image_identity_sha256 for row in checked)
    value = object.__new__(VerifiedSourceVariableQuerySanityView)
    fields = {"outer_fold_id": outer_fold_id, "outer_target": OUTER_TARGETS[outer_fold_id],
              "source_domains": sources, "rows": tuple(checked),
              "artifact_scope": "synthetic_cpu_contract_test", "identity_sha256": identity,
              "boundary_values": MappingProxyType({
                  "canonical_id": sanity_boundaries,
                  "original_image_sha256": sanity_boundaries,
                  "near_duplicate_cluster_id_or_unique_sentinel": sanity_boundaries,
                  "exclusion_group_id": sanity_boundaries,
              }), "upstream_bindings": MappingProxyType({
                  "statistics_config_sha256": hashlib.sha256(
                      b"synthetic-statistics-config").hexdigest(),
                  "source_reference_set_sha256": hashlib.sha256(
                      b"synthetic-source-reference-set").hexdigest(),
                  "detector_run_complete_set_sha256": hashlib.sha256(
                      b"synthetic-validation-run-complete-set").hexdigest(),
              }), "_capability": _SANITY_CAPABILITY}
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


def build_source_variable_query_sanity_view_from_verified_bundles(
    *, outer_fold_id: str, domain_bundles: Mapping[str, Mapping[str, Any]],
) -> VerifiedSourceVariableQuerySanityView:
    """Production all-once sanity view from full-fit score/context wrappers."""
    from data_ext.stage2_rc5_score_bundle_v2 import (
        assert_verified_stage2_rc5_score_bundle_v2,
        replay_verified_stage2_rc5_score_bundle_v2,
    )
    from rc.stage2_crossfit_schema_v6 import context_inference_material_v2

    sources = _sources(outer_fold_id)
    if not isinstance(domain_bundles, Mapping) or set(domain_bundles) != set(sources):
        raise Stage2SourceValidationViewError("production sanity needs two sources")
    checked_rows: list[Mapping[str, Any]] = []
    boundary_fields = ("canonical_id", "original_image_sha256",
                       "near_duplicate_cluster_id_or_unique_sentinel",
                       "exclusion_group_id")
    boundaries = {field: set() for field in boundary_fields}
    statistics_values: set[str] = set()
    source_references: set[str] = set()
    run_bindings = []
    for domain in sources:
        supplied = domain_bundles[domain]
        if not isinstance(supplied, Mapping) or set(supplied) != {
            "score_bundle", "context_bundles", "aggregate_curves"
        }:
            raise Stage2SourceValidationViewError("production sanity closure mismatch")
        score = replay_verified_stage2_rc5_score_bundle_v2(
            assert_verified_stage2_rc5_score_bundle_v2(supplied["score_bundle"]))
        metadata = score.score_manifest_metadata
        if metadata.role != "source_diagnostic_validation" or \
                metadata.payload["outer_fold_id"] != outer_fold_id or \
                metadata.payload["source_domain"] != domain:
            raise Stage2SourceValidationViewError("sanity score role mismatch")
        contexts = supplied["context_bundles"]
        curves = supplied["aggregate_curves"]
        if isinstance(contexts, (str, bytes)) or not isinstance(contexts, Sequence) or \
                isinstance(curves, (str, bytes)) or not isinstance(curves, Sequence) or \
                not contexts or len(contexts) != len(curves):
            raise Stage2SourceValidationViewError("sanity context/curve windows mismatch")
        consumed_canonical: list[str] = []
        consumed_original: list[str] = []
        for context, raw_curve in zip(contexts, curves, strict=True):
            replayed, statistics_sha, reference_sha = _replay_production_context_bundle(
                context, score)
            payload = replayed.context.payload
            if payload["expected_role"] != "source_diagnostic_validation" or \
                    payload["outer_fold_id"] != outer_fold_id or \
                    payload["source_domain"] != domain:
                raise Stage2SourceValidationViewError("sanity context identity drifted")
            query_size = len(payload["query_identity_records"])
            if query_size < 28:
                raise Stage2SourceValidationViewError("sanity variable-Q is below Qmin28")
            all_records = [*payload["context_records"], *payload["query_identity_records"]]
            consumed_canonical.extend(str(row["canonical_id"]) for row in all_records)
            consumed_original.extend(str(row["original_image_sha256"]) for row in all_records)
            material = context_inference_material_v2(replayed.context)
            checked_rows.append(MappingProxyType({
                "source_domain": domain, "query_size": query_size,
                "context_features": np.asarray(material.feature_values, dtype=np.float32),
                "anchor_coordinates": np.asarray(
                    replayed.anchor.coordinates, dtype=np.float64),
                "aggregate_curve": assert_per_image_exact_event_curve(raw_curve),
            }))
            statistics_values.add(statistics_sha); source_references.add(reference_sha)
        expected_canonical = [str(row["canonical_id"]) for row in metadata.records]
        expected_original = [str(row["original_image_sha256"]) for row in metadata.records]
        if len(consumed_canonical) != len(set(consumed_canonical)) or \
                set(consumed_canonical) != set(expected_canonical) or \
                len(consumed_original) != len(set(consumed_original)) or \
                set(consumed_original) != set(expected_original):
            raise Stage2SourceValidationViewError(
                "variable-Q sanity did not consume every validation record exactly once"
            )
        for field in boundary_fields:
            boundaries[field].update(str(record[field]) for record in metadata.records)
        run_identity = score.attestation["run_complete"]["identity"]
        run_bindings.append({
            "score_attestation_sha256": score.attestation_sha256,
            "run_complete_artifact_sha256": score.run_complete.sha256,
            "run_complete_identity_sha256": str(run_identity["identity_sha256"]),
        })
    if len(statistics_values) != 1:
        raise Stage2SourceValidationViewError("sanity contexts use different statistics configs")
    upstream = {
        "statistics_config_sha256": next(iter(statistics_values)),
        "source_reference_set_sha256": hashlib.sha256(json.dumps(
            sorted(source_references), separators=(",", ":")
        ).encode()).hexdigest(),
        "detector_run_complete_set_sha256": hashlib.sha256(json.dumps(
            sorted(run_bindings, key=lambda row: row["run_complete_identity_sha256"]),
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }
    projection = {
        "schema_version": VARIABLE_QUERY_SANITY_SCHEMA,
        "outer_fold_id": outer_fold_id, "artifact_scope": "production",
        "rows": [{"source_domain": row["source_domain"],
                  "query_size": row["query_size"],
                  "feature_sha256": hashlib.sha256(
                      row["context_features"].tobytes()).hexdigest(),
                  "anchor_sha256": hashlib.sha256(
                      row["anchor_coordinates"].tobytes()).hexdigest(),
                  "curve_content_sha256": row["aggregate_curve"].content_sha256}
                 for row in checked_rows],
        "upstream_bindings": upstream,
        "all_records_consumed_once": True,
        "excluded_from_epoch_ranking": True,
    }
    identity = hashlib.sha256(json.dumps(
        projection, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    value = object.__new__(VerifiedSourceVariableQuerySanityView)
    fields = {"outer_fold_id": outer_fold_id, "outer_target": OUTER_TARGETS[outer_fold_id],
              "source_domains": sources, "rows": tuple(checked_rows),
              "artifact_scope": "production", "identity_sha256": identity,
              "boundary_values": MappingProxyType(
                  {field: frozenset(values) for field, values in boundaries.items()}),
              "upstream_bindings": MappingProxyType(upstream),
              "_capability": _SANITY_CAPABILITY}
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


def assert_verified_source_variable_query_sanity_view(
    value: object,
) -> VerifiedSourceVariableQuerySanityView:
    if type(value) is not VerifiedSourceVariableQuerySanityView or getattr(
        value, "_capability", None
    ) is not _SANITY_CAPABILITY:
        raise TypeError("a verified source variable-Q sanity view is required")
    return value


def source_validation_collection_identity_sha256(
    selection_view: VerifiedSourceValidationCyclicSelectionView,
    sanity_view: VerifiedSourceVariableQuerySanityView,
) -> str:
    selection = assert_verified_source_validation_cyclic_selection_view(selection_view)
    sanity = assert_verified_source_variable_query_sanity_view(sanity_view)
    if selection.outer_fold_id != sanity.outer_fold_id or \
            selection.artifact_scope != sanity.artifact_scope:
        raise Stage2SourceValidationViewError("validation view identities cannot combine")
    payload = {
        "schema_version": "rc-irstd.stage2-source-validation-combined-identity.v1",
        "outer_fold_id": selection.outer_fold_id,
        "cyclic_selection_view_identity_sha256": selection.identity_sha256,
        "variable_query_sanity_view_identity_sha256": sanity.identity_sha256,
        "geometries_interchangeable": False,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _standardizer(mean: Any, scale: Any) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean)
    scale_array = np.asarray(scale)
    if mean_array.dtype != np.float64 or mean_array.shape != (93,) or \
            scale_array.dtype != np.float64 or scale_array.shape != (93,) or \
            not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all() or \
            np.any(scale_array < 1e-8):
        raise Stage2SourceValidationViewError("standardizer must be finite float64[93]")
    return mean_array, scale_array


def _predict(model: nn.Module, features: np.ndarray, anchors: np.ndarray,
             mean: np.ndarray, scale: np.ndarray, device: torch.device,
             batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive exact int")
    coordinates = []
    thresholds = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                stop = min(start + batch_size, features.shape[0])
                standardized = ((features[start:stop].astype(np.float64) - mean) / scale).astype(
                    np.float32)
                output = model(
                    torch.from_numpy(standardized).to(device=device),
                    anchor_coordinates=torch.from_numpy(
                        np.array(anchors[start:stop], dtype=np.float64, copy=True)
                    ).to(device=device),
                )
                coordinates.append(output.grid_coordinates.detach().cpu().numpy())
                thresholds.append(output.grid_thresholds.detach().cpu().numpy())
    finally:
        if was_training: model.train()
    coordinate = np.concatenate(coordinates).astype(np.float64, copy=False)
    threshold = np.concatenate(thresholds).astype(np.float64, copy=False)
    if coordinate.shape != (features.shape[0], 3) or threshold.shape != coordinate.shape or \
            not np.isfinite(coordinate).all() or not np.isfinite(threshold).all():
        raise Stage2SourceValidationViewError("model validation output is invalid")
    return coordinate, threshold


def _primary_from_provider(provider: Any, coordinate: float) -> tuple[float, int, int]:
    compact = provider.compact_brackets(np.asarray([coordinate], dtype=np.float64))
    index = int(np.searchsorted(compact.coordinates, coordinate, side="right") - 1)
    index = max(0, min(index, compact.coordinates.size - 1))
    return (float(compact.pixel_false_alarm_rate[index]),
            int(compact.matched_objects[index]), int(compact.ground_truth_objects))


def evaluate_source_validation_cyclic_selection_view(
    *, model: nn.Module, view: VerifiedSourceValidationCyclicSelectionView,
    standardizer_mean: np.ndarray, standardizer_scale: np.ndarray,
    device: str | torch.device = "cpu", batch_size: int = 16,
) -> dict[str, Any]:
    """Return the only metrics authorized for epoch ranking."""
    verified = assert_verified_source_validation_cyclic_selection_view(view)
    mean, scale = _standardizer(standardizer_mean, standardizer_scale)
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu" and verified.artifact_scope == "synthetic_cpu_contract_test":
        raise Stage2SourceValidationViewError("synthetic validation is CPU-only")
    domain_metrics: dict[str, dict[str, Any]] = {}
    for domain in verified.source_domains:
        material = verified.domain_materials[domain]
        features = material["context_features"]
        anchors = material["anchor_coordinates"]
        coordinates, _ = _predict(model, features, anchors, mean, scale,
                                  resolved_device, batch_size)
        satisfied: list[float] = []
        excess: list[float] = []
        pooled_tp = 0
        pooled_gt = 0
        for start in range(features.shape[0]):
            provider = verified.provider_for_start(domain, start)
            risk, tp, gt = _primary_from_provider(provider, float(coordinates[start, 1]))
            satisfied.append(float(risk <= 1e-5))
            excess.append(math.log(max(risk / 1e-5, 1.0)))
            pooled_tp += tp; pooled_gt += gt
        if pooled_gt <= 0:
            raise Stage2SourceValidationViewError("cyclic source domain has zero pooled GT")
        domain_metrics[domain] = {
            "BSR": float(np.mean(satisfied, dtype=np.float64)),
            "LogExcess": float(np.mean(excess, dtype=np.float64)),
            "Pd": float(pooled_tp / pooled_gt),
            "pooled_tp_objects": pooled_tp, "pooled_gt_objects": pooled_gt,
            "exhaustive_cyclic_start_count": len(satisfied),
        }
    macro_bsr = sum(row["BSR"] for row in domain_metrics.values()) / 2.0
    macro_excess = sum(row["LogExcess"] for row in domain_metrics.values()) / 2.0
    macro_pd = sum(row["Pd"] for row in domain_metrics.values()) / 2.0
    selection = build_selection_record(
        macro_source_bsr=macro_bsr,
        macro_source_log_excess=macro_excess,
        macro_source_pd=macro_pd,
    )
    return {
        "schema_version": CYCLIC_SELECTION_SCHEMA,
        "selection_geometry": CYCLIC_SELECTION_GEOMETRY,
        "selection_pixel_budget": 1e-5, "selection_budget_index": 1,
        "source_domain_weighting": "equal_one_half",
        "within_domain_BSR_LogExcess": "equal_exhaustive_cyclic_start_mean",
        "within_domain_Pd": "pooled_tp_divided_by_pooled_gt_across_all_cyclic_starts",
        "domain_metrics": domain_metrics,
        "macro_source_BSR": macro_bsr,
        "macro_source_LogExcess": macro_excess,
        "macro_source_Pd": macro_pd,
        "selection_record": selection,
        "cyclic_starts_claimed_independent": False,
        "cyclic_start_confidence_interval_reported": False,
        "source_variable_query_sanity_excluded_from_epoch_ranking": True,
        "outer_target_accessed": False, "official_test_accessed": False,
    }


def evaluate_source_variable_query_sanity_view(
    *, model: nn.Module, view: VerifiedSourceVariableQuerySanityView,
    standardizer_mean: np.ndarray, standardizer_scale: np.ndarray,
    device: str | torch.device = "cpu", batch_size: int = 16,
) -> dict[str, Any]:
    """Execute mandatory all-once sanity; output cannot be ranked."""
    verified = assert_verified_source_variable_query_sanity_view(view)
    mean, scale = _standardizer(standardizer_mean, standardizer_scale)
    features = np.stack([row["context_features"] for row in verified.rows]).astype(np.float32)
    anchors = np.stack([row["anchor_coordinates"] for row in verified.rows]).astype(np.float64)
    _, thresholds = _predict(model, features, anchors, mean, scale,
                             torch.device(device), batch_size)
    per_domain: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(verified.rows):
        curve = row["aggregate_curve"]
        curve_index = curve.resolve_threshold(float(thresholds[index, 1]))
        risk = float(curve.false_positive_pixels[curve_index] / curve.total_native_pixels)
        tp = int(curve.matched_objects[curve_index]); gt = int(curve.ground_truth_objects)
        aggregate = per_domain.setdefault(row["source_domain"],
            {"satisfied": [], "log_excess": [], "tp": 0, "gt": 0, "windows": 0})
        aggregate["satisfied"].append(float(risk <= 1e-5))
        aggregate["log_excess"].append(math.log(max(risk / 1e-5, 1.0)))
        aggregate["tp"] += tp; aggregate["gt"] += gt; aggregate["windows"] += 1
    metrics = {}
    for domain in verified.source_domains:
        row = per_domain[domain]
        metrics[domain] = {
            "BSR": float(np.mean(row["satisfied"], dtype=np.float64)),
            "LogExcess": float(np.mean(row["log_excess"], dtype=np.float64)),
            "Pd": float(row["tp"] / row["gt"]) if row["gt"] else 0.0,
            "window_count": row["windows"],
        }
    return {
        "schema_version": VARIABLE_QUERY_SANITY_SCHEMA,
        "geometry_schema_version": VARIABLE_QUERY_SCHEMA,
        "geometry": "mandatory_variable_query_all_records_consumed_once",
        "excluded_from_epoch_ranking": True,
        "domain_metrics": metrics,
        "window_count": len(verified.rows),
        "outer_target_accessed": False, "official_test_accessed": False,
    }


__all__ = [
    "CYCLIC_SELECTION_GEOMETRY", "CYCLIC_SELECTION_SCHEMA",
    "VARIABLE_QUERY_SANITY_SCHEMA", "Stage2SourceValidationViewError",
    "VerifiedSourceValidationCyclicSelectionView", "VerifiedSourceVariableQuerySanityView",
    "assert_verified_source_validation_cyclic_selection_view",
    "assert_verified_source_variable_query_sanity_view",
    "build_synthetic_source_validation_cyclic_selection_view",
    "build_synthetic_source_variable_query_sanity_view",
    "build_source_validation_cyclic_selection_view_from_verified_bundles",
    "build_source_variable_query_sanity_view_from_verified_bundles",
    "evaluate_source_validation_cyclic_selection_view",
    "evaluate_source_variable_query_sanity_view",
    "source_validation_collection_identity_sha256",
]
