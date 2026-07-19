"""Memory-bounded exact-curve composition for cyclic C14/Q28 training.

Each query image owns one immutable ascending exact-event curve.  A cyclic
episode references exactly 28 such curves; it never materializes their
potentially million-row aggregate threshold union.

For RC5 one to three, or RC5+ one to nine, live EATC-v2 predictions, the
provider binary-searches every image curve in coordinate space.  The maximum
local predecessor and minimum local successor are the adjacent rows in the
global event union.  Their union has at most twice the prediction count.
Counts at those thresholds are then resolved
independently on every image curve and summed as Python integers before Pd or
pixel risk is derived.

The RC5 three-row or RC5+ nine-row exact-rational oracle is selected by a deterministic
descending k-way tail merge over per-image events.  The merge retains only
the current aggregate count state and three best rows, and stops as soon as
the loosest budget first becomes infeasible.  It does not retain an aggregate
curve.
Artifact serialization and external file verification are intentionally out
of scope for this pure in-memory provider.  The causal collection layer must
therefore supply a verified source-only curve bank; this module never opens
scores, images, labels, manifests, or target-domain artifacts itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import heapq
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_numpy,
    encode_probability_numpy,
)
from rc.stage2_context_tail_anchor import BUDGET_RATIONALS


PER_IMAGE_CURVE_SCHEMA = (
    "rc-irstd.stage2-per-image-exact-event-curve.v1"
)
PER_IMAGE_CURVE_BANK_SCHEMA = (
    "rc-irstd.stage2-per-image-exact-event-curve-bank.v1"
)
COMPOSITIONAL_PROVIDER_SCHEMA = (
    "rc-irstd.stage2-compositional-exact-curve-provider.v1"
)
PER_IMAGE_CURVE_CONTENT_ALGORITHM = (
    "sha256-u64be-framed-metadata-f64le-i64le-per-image-curve-v1"
)
PROVIDER_ID_ALGORITHM = (
    "sha256-canonical-json-ordered-image-curve-bindings-v1"
)
CURVE_BANK_ID_ALGORITHM = (
    "sha256-canonical-json-identity-sorted-image-curve-bindings-v1"
)
COMPOSITIONAL_ORACLE_SELECTION_RULE = (
    "descending-k-way-tail-merge-exact-rational-"
    "max-tp-min-fp-max-threshold-stop-after-loose-infeasible-v1"
)
BUDGET_COUNT_RULE = "(numerator * total_native_pixels) // denominator"
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"
CYCLIC_QUERY_SIZE = 28
MAX_LIVE_PREDICTIONS = 3
RC5PLUS_MAX_LIVE_PREDICTIONS = len(BUDGET_KNOT_RATIONALS)

_SHA256_HEX = frozenset("0123456789abcdef")
_PER_IMAGE_CAPABILITY = object()
_BANK_CAPABILITY = object()
_PROVIDER_CAPABILITY = object()
_INT64_MAX = int(np.iinfo(np.int64).max)


class Stage2CompositionalCurveError(ValueError):
    """A per-image curve or its exact composition failed closed."""


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2CompositionalCurveError(
            f"{name} must be lowercase SHA-256"
        )
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _INT64_MAX:
        raise Stage2CompositionalCurveError(
            f"{name} must be an exact int in [{minimum}, {_INT64_MAX}]"
        )
    return value


def _float64_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.float64 or value.ndim != 1 or value.size < 2:
        raise Stage2CompositionalCurveError(
            f"{name} must be one float64 vector with at least two rows"
        )
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    result = np.frombuffer(owned.tobytes(order="C"), dtype=np.float64)
    if not np.isfinite(result).all():
        raise Stage2CompositionalCurveError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _int64_vector(value: Any, name: str, *, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.int64 or value.ndim != 1 or value.size != size:
        raise Stage2CompositionalCurveError(
            f"{name} must be one int64 vector aligned with thresholds"
        )
    owned = np.array(value, dtype=np.int64, order="C", copy=True)
    result = np.frombuffer(owned.tobytes(order="C"), dtype=np.int64)
    if np.any(result < 0):
        raise Stage2CompositionalCurveError(
            f"{name} must contain non-negative exact counts"
        )
    result.setflags(write=False)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest_frame(digest: Any, value: bytes | memoryview) -> None:
    view = memoryview(value).cast("B")
    digest.update(int(view.nbytes).to_bytes(8, "big"))
    digest.update(view)


def _curve_content_sha256(
    *,
    image_identity_sha256: str,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    matched_objects: np.ndarray,
    total_native_pixels: int,
    ground_truth_objects: int,
) -> str:
    header = _canonical_json_bytes(
        {
            "schema_version": PER_IMAGE_CURVE_SCHEMA,
            "content_algorithm": PER_IMAGE_CURVE_CONTENT_ALGORITHM,
            "image_identity_sha256": image_identity_sha256,
            "row_count": int(thresholds.size),
            "total_native_pixels": total_native_pixels,
            "ground_truth_objects": ground_truth_objects,
            "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        }
    )
    digest = hashlib.sha256()
    _digest_frame(digest, header)
    for name, values, dtype in (
        (b"thresholds", thresholds, "<f8"),
        (b"false_positive_pixels", false_positive_pixels, "<i8"),
        (b"matched_objects", matched_objects, "<i8"),
    ):
        canonical = np.ascontiguousarray(values, dtype=dtype)
        _digest_frame(digest, name)
        _digest_frame(digest, memoryview(canonical))
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class PerImageExactEventCurve:
    """Verifier-built immutable exact-event curve for one source image."""

    image_identity_sha256: str
    thresholds: np.ndarray
    false_positive_pixels: np.ndarray
    matched_objects: np.ndarray
    total_native_pixels: int
    ground_truth_objects: int
    content_sha256: str
    _capability: object

    def __init__(
        self,
        *,
        image_identity_sha256: str,
        thresholds: np.ndarray,
        false_positive_pixels: np.ndarray,
        matched_objects: np.ndarray,
        total_native_pixels: int,
        ground_truth_objects: int,
        content_sha256: str,
        _capability: object,
    ) -> None:
        if _capability is not _PER_IMAGE_CAPABILITY:
            raise TypeError("PerImageExactEventCurve is verifier-built only")
        object.__setattr__(
            self, "image_identity_sha256", image_identity_sha256
        )
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(
            self, "false_positive_pixels", false_positive_pixels
        )
        object.__setattr__(self, "matched_objects", matched_objects)
        object.__setattr__(
            self, "total_native_pixels", total_native_pixels
        )
        object.__setattr__(
            self, "ground_truth_objects", ground_truth_objects
        )
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(self, "_capability", _capability)

    def coordinate_at(self, index: int) -> float:
        return float(
            encode_probability_numpy(self.thresholds[int(index)])
        )

    def search_coordinate_right(self, coordinate: float) -> int:
        left = 0
        right = int(self.thresholds.size)
        while left < right:
            middle = (left + right) // 2
            if self.coordinate_at(middle) <= coordinate:
                left = middle + 1
            else:
                right = middle
        return left

    def resolve_threshold(self, threshold: float) -> int:
        index = int(
            np.searchsorted(
                self.thresholds,
                np.float64(threshold),
                side="right",
            )
            - 1
        )
        return max(0, min(index, int(self.thresholds.size) - 1))


def build_per_image_exact_event_curve(
    *,
    image_identity_sha256: str,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    matched_objects: np.ndarray,
    total_native_pixels: int,
    ground_truth_objects: int,
) -> PerImageExactEventCurve:
    """Validate, own, and SHA-bind one image's exact event curve."""

    identity = _sha256(
        image_identity_sha256, "image_identity_sha256"
    )
    threshold = _float64_vector(thresholds, "thresholds")
    fp = _int64_vector(
        false_positive_pixels,
        "false_positive_pixels",
        size=threshold.size,
    )
    tp = _int64_vector(
        matched_objects,
        "matched_objects",
        size=threshold.size,
    )
    total = _strict_int(
        total_native_pixels, "total_native_pixels", minimum=1
    )
    objects = _strict_int(
        ground_truth_objects, "ground_truth_objects"
    )
    if (
        threshold[0] != 0.0
        or np.signbit(threshold[0])
        or threshold[-1] != 1.0
        or np.any((threshold < 0.0) | (threshold > 1.0))
        or np.any(np.diff(threshold) <= 0.0)
    ):
        raise Stage2CompositionalCurveError(
            "thresholds must be strictly ascending with exact +0/1 endpoints"
        )
    coordinates = encode_probability_numpy(threshold)
    if np.any(np.diff(coordinates) <= 0.0):
        raise Stage2CompositionalCurveError(
            "thresholds must remain strict in EATC-v2 coordinate space"
        )
    if np.any(fp > total):
        raise Stage2CompositionalCurveError(
            "false-positive pixels exceed total native pixels"
        )
    if np.any(tp > objects):
        raise Stage2CompositionalCurveError(
            "matched objects exceed ground-truth objects"
        )
    if np.any(fp[1:] > fp[:-1]):
        raise Stage2CompositionalCurveError(
            "false-positive pixels must be nonincreasing"
        )
    if fp[-1] != 0 or tp[-1] != 0:
        raise Stage2CompositionalCurveError(
            "strict threshold=1 endpoint must predict nothing"
        )
    content = _curve_content_sha256(
        image_identity_sha256=identity,
        thresholds=threshold,
        false_positive_pixels=fp,
        matched_objects=tp,
        total_native_pixels=total,
        ground_truth_objects=objects,
    )
    return PerImageExactEventCurve(
        image_identity_sha256=identity,
        thresholds=threshold,
        false_positive_pixels=fp,
        matched_objects=tp,
        total_native_pixels=total,
        ground_truth_objects=objects,
        content_sha256=content,
        _capability=_PER_IMAGE_CAPABILITY,
    )


def assert_per_image_exact_event_curve(
    value: object,
) -> PerImageExactEventCurve:
    if (
        type(value) is not PerImageExactEventCurve
        or getattr(value, "_capability", None) is not _PER_IMAGE_CAPABILITY
    ):
        raise TypeError("a verifier-built per-image curve is required")
    try:
        replayed = build_per_image_exact_event_curve(
            image_identity_sha256=value.image_identity_sha256,
            thresholds=value.thresholds,
            false_positive_pixels=value.false_positive_pixels,
            matched_objects=value.matched_objects,
            total_native_pixels=value.total_native_pixels,
            ground_truth_objects=value.ground_truth_objects,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "per-image exact-event curve retained-token state is invalid"
        ) from error
    if (
        value.content_sha256 != replayed.content_sha256
        or not np.array_equal(value.thresholds, replayed.thresholds)
        or not np.array_equal(
            value.false_positive_pixels,
            replayed.false_positive_pixels,
        )
        or not np.array_equal(value.matched_objects, replayed.matched_objects)
    ):
        raise TypeError(
            "per-image exact-event curve differs from content replay"
        )
    return value


@dataclass(frozen=True, init=False)
class PerImageExactEventCurveBank:
    """Immutable identity-indexed bank; aggregate episode curves are absent."""

    curves: tuple[PerImageExactEventCurve, ...]
    bank_id: str
    image_count: int
    total_curve_rows: int
    _curves_by_identity: MappingProxyType
    _capability: object

    def __init__(
        self,
        *,
        curves: tuple[PerImageExactEventCurve, ...],
        bank_id: str,
        image_count: int,
        total_curve_rows: int,
        curves_by_identity: MappingProxyType,
        _capability: object,
    ) -> None:
        if _capability is not _BANK_CAPABILITY:
            raise TypeError(
                "PerImageExactEventCurveBank is verifier-built only"
            )
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "bank_id", bank_id)
        object.__setattr__(self, "image_count", image_count)
        object.__setattr__(self, "total_curve_rows", total_curve_rows)
        object.__setattr__(
            self, "_curves_by_identity", curves_by_identity
        )
        object.__setattr__(self, "_capability", _capability)

    def curve_for_identity(
        self, image_identity_sha256: str
    ) -> PerImageExactEventCurve:
        identity = _sha256(
            image_identity_sha256, "image_identity_sha256"
        )
        try:
            return self._curves_by_identity[identity]
        except KeyError as error:
            raise Stage2CompositionalCurveError(
                "query image identity is absent from the exact-curve bank"
            ) from error


def build_per_image_exact_event_curve_bank(
    curves: Sequence[PerImageExactEventCurve],
) -> PerImageExactEventCurveBank:
    """Bind one reusable curve per image, sorted independently of input order."""

    if isinstance(curves, (str, bytes)) or not isinstance(curves, Sequence):
        raise TypeError("curves must be a nonempty ordered sequence")
    if len(curves) == 0:
        raise Stage2CompositionalCurveError(
            "per-image exact-event curve bank must be nonempty"
        )
    verified = tuple(
        assert_per_image_exact_event_curve(curve) for curve in curves
    )
    identities = [curve.image_identity_sha256 for curve in verified]
    if len(set(identities)) != len(identities):
        raise Stage2CompositionalCurveError(
            "per-image exact-event curve bank contains duplicate identity"
        )
    ordered = tuple(
        sorted(verified, key=lambda curve: curve.image_identity_sha256)
    )
    total_rows = sum(int(curve.thresholds.size) for curve in ordered)
    _strict_int(total_rows, "total_curve_rows", minimum=2)
    bindings = [
        {
            "image_identity_sha256": curve.image_identity_sha256,
            "curve_content_sha256": curve.content_sha256,
            "curve_row_count": int(curve.thresholds.size),
            "total_native_pixels": curve.total_native_pixels,
            "ground_truth_objects": curve.ground_truth_objects,
        }
        for curve in ordered
    ]
    bank_id = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": PER_IMAGE_CURVE_BANK_SCHEMA,
                "bank_id_algorithm": CURVE_BANK_ID_ALGORITHM,
                "image_count": len(ordered),
                "total_curve_rows": total_rows,
                "identity_sorted_curve_bindings": bindings,
            }
        )
    ).hexdigest()
    return PerImageExactEventCurveBank(
        curves=ordered,
        bank_id=bank_id,
        image_count=len(ordered),
        total_curve_rows=total_rows,
        curves_by_identity=MappingProxyType(
            {curve.image_identity_sha256: curve for curve in ordered}
        ),
        _capability=_BANK_CAPABILITY,
    )


def assert_per_image_exact_event_curve_bank(
    value: object,
) -> PerImageExactEventCurveBank:
    if (
        type(value) is not PerImageExactEventCurveBank
        or getattr(value, "_capability", None) is not _BANK_CAPABILITY
    ):
        raise TypeError("a verifier-built per-image curve bank is required")
    if value.image_count != len(value.curves):
        raise TypeError("per-image curve bank cardinality was corrupted")
    for curve in value.curves:
        assert_per_image_exact_event_curve(curve)
        if value._curves_by_identity.get(curve.image_identity_sha256) is not curve:
            raise TypeError("per-image curve bank index was corrupted")
    try:
        replayed = build_per_image_exact_event_curve_bank(value.curves)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "per-image curve bank retained-token state is invalid"
        ) from error
    if (
        value.bank_id != replayed.bank_id
        or value.image_count != replayed.image_count
        or value.total_curve_rows != replayed.total_curve_rows
        or tuple(
            curve.image_identity_sha256 for curve in value.curves
        )
        != tuple(
            curve.image_identity_sha256 for curve in replayed.curves
        )
    ):
        raise TypeError("per-image curve bank differs from identity replay")
    return value


@dataclass(frozen=True)
class CompositionalCurveBrackets:
    """At-most-six exact aggregate rows around one to three predictions."""

    thresholds: np.ndarray
    coordinates: np.ndarray
    false_positive_pixels: np.ndarray
    matched_objects: np.ndarray
    pixel_false_alarm_rate: np.ndarray
    detection_probability: np.ndarray
    total_native_pixels: int
    ground_truth_objects: int


@dataclass(frozen=True)
class CompositionalOracleRows:
    """Only the requested rows selected by exact-rational k-way merge."""

    budget_rationals: tuple[tuple[int, int], ...]
    allowed_false_positive_counts: tuple[int, ...]
    thresholds: np.ndarray
    coordinates: np.ndarray
    false_positive_pixels: np.ndarray
    matched_objects: np.ndarray
    pixel_false_alarm_rate: np.ndarray
    detection_probability: np.ndarray
    total_native_pixels: int
    ground_truth_objects: int
    global_rows_examined: int
    stopped_at_first_loose_infeasible: bool


def _immutable_array(value: Any, dtype: Any) -> np.ndarray:
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype)


@dataclass(frozen=True, init=False)
class CompositionalExactCurveProvider:
    """Verifier-built ordered Q28 reference to per-image curve capabilities."""

    curves: tuple[PerImageExactEventCurve, ...]
    curve_bank_id: str
    provider_id: str
    total_native_pixels: int
    ground_truth_objects: int
    _capability: object

    def __init__(
        self,
        *,
        curves: tuple[PerImageExactEventCurve, ...],
        curve_bank_id: str,
        provider_id: str,
        total_native_pixels: int,
        ground_truth_objects: int,
        _capability: object,
    ) -> None:
        if _capability is not _PROVIDER_CAPABILITY:
            raise TypeError(
                "CompositionalExactCurveProvider is verifier-built only"
            )
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "curve_bank_id", curve_bank_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(
            self, "total_native_pixels", total_native_pixels
        )
        object.__setattr__(
            self, "ground_truth_objects", ground_truth_objects
        )
        object.__setattr__(self, "_capability", _capability)

    def _aggregate_rows(
        self, thresholds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        fp_rows: list[int] = []
        tp_rows: list[int] = []
        for threshold in thresholds:
            fp = 0
            tp = 0
            value = float(threshold)
            for curve in self.curves:
                index = curve.resolve_threshold(value)
                fp += int(curve.false_positive_pixels[index])
                tp += int(curve.matched_objects[index])
            fp_rows.append(fp)
            tp_rows.append(tp)
        return (
            _immutable_array(fp_rows, np.int64),
            _immutable_array(tp_rows, np.int64),
        )

    def _compact_brackets(
        self,
        predicted_coordinates: np.ndarray,
        *,
        maximum_predictions: int,
    ) -> CompositionalCurveBrackets:
        """Compose global adjacent rows without materializing the union curve.

        Let ``U`` be the sorted union of all 28 per-image threshold sets.  For
        one query coordinate, the predecessor in ``U`` is the maximum of the
        28 local predecessors and the successor is the minimum of the 28
        local successors.  Resolving every image at those global thresholds
        and summing integer counts is exactly the corresponding conceptual
        materialized aggregate row.
        """

        if not isinstance(predicted_coordinates, np.ndarray):
            raise TypeError(
                "predicted_coordinates must be an explicit numpy array"
            )
        query = predicted_coordinates
        if (
            query.dtype != np.float64
            or query.ndim != 1
            or not 1 <= query.size <= maximum_predictions
        ):
            raise Stage2CompositionalCurveError(
                "predicted_coordinates must be one float64 vector of length "
                f"1..{maximum_predictions}"
            )
        if not np.isfinite(query).all():
            raise Stage2CompositionalCurveError(
                "predicted_coordinates must be finite"
            )
        try:
            decode_coordinate_numpy(query)
        except EndpointAwareThresholdError as error:
            raise Stage2CompositionalCurveError(
                "predicted_coordinates must be canonical EATC-v2"
            ) from error

        candidates: set[float] = set()
        for coordinate in query:
            lower: tuple[float, float] | None = None
            upper: tuple[float, float] | None = None
            for curve in self.curves:
                right = curve.search_coordinate_right(float(coordinate))
                right = max(1, min(right, int(curve.thresholds.size) - 1))
                left = right - 1
                local_lower = (
                    curve.coordinate_at(left),
                    float(curve.thresholds[left]),
                )
                local_upper = (
                    curve.coordinate_at(right),
                    float(curve.thresholds[right]),
                )
                if lower is None or local_lower[0] > lower[0]:
                    lower = local_lower
                if upper is None or local_upper[0] < upper[0]:
                    upper = local_upper
            if lower is None or upper is None:
                raise RuntimeError("compositional provider has no image curves")
            if not lower[0] < upper[0]:
                raise RuntimeError(
                    "global exact-event bracket is not strictly ordered"
                )
            candidates.add(lower[1])
            candidates.add(upper[1])

        threshold = _immutable_array(sorted(candidates), np.float64)
        maximum_rows = 2 * maximum_predictions
        if not 2 <= threshold.size <= maximum_rows:
            raise RuntimeError(
                "compositional bracket union must contain between two and "
                f"{maximum_rows} rows"
            )
        coordinate = _immutable_array(
            encode_probability_numpy(threshold), np.float64
        )
        if np.any(np.diff(coordinate) <= 0.0):
            raise RuntimeError(
                "compositional bracket coordinates are not strictly ordered"
            )
        fp, tp = self._aggregate_rows(threshold)
        risk = _immutable_array(
            fp.astype(np.float64) / float(self.total_native_pixels),
            np.float64,
        )
        detection = _immutable_array(
            (
                tp.astype(np.float64) / float(self.ground_truth_objects)
                if self.ground_truth_objects > 0
                else np.zeros(tp.size, dtype=np.float64)
            ),
            np.float64,
        )
        return CompositionalCurveBrackets(
            thresholds=threshold,
            coordinates=coordinate,
            false_positive_pixels=fp,
            matched_objects=tp,
            pixel_false_alarm_rate=risk,
            detection_probability=detection,
            total_native_pixels=self.total_native_pixels,
            ground_truth_objects=self.ground_truth_objects,
        )

    def compact_brackets(
        self, predicted_coordinates: np.ndarray
    ) -> CompositionalCurveBrackets:
        """Frozen RC5 path: adjacent union around one to three predictions."""

        return self._compact_brackets(
            predicted_coordinates,
            maximum_predictions=MAX_LIVE_PREDICTIONS,
        )

    def compact_brackets_v2(
        self, predicted_coordinates: np.ndarray
    ) -> CompositionalCurveBrackets:
        """RC5+ path: adjacent union around one to nine predictions."""

        return self._compact_brackets(
            predicted_coordinates,
            maximum_predictions=RC5PLUS_MAX_LIVE_PREDICTIONS,
        )

    def _select_exact_oracle_rows(
        self,
        budgets: tuple[tuple[int, int], ...],
    ) -> CompositionalOracleRows:
        """Select exact rows with a descending storage-O(Q) tail merge.

        The merge starts at the exact ``threshold=1`` no-prediction row and
        enumerates global union events in strictly descending threshold
        order.  At every union event it resolves all retained per-image
        curves at that exact global threshold.  This is necessary because a
        curve can move to a lower local row at a threshold contributed by a
        different image.  Aggregate FP is therefore nondecreasing.  Once it
        first exceeds the loosest exact-rational allowance, every unvisited
        lower threshold is provably infeasible for every requested budget and
        traversal stops.  The provider remains storage-O(Q): no aggregate
        curve is materialized or retained.
        """

        allowed = tuple(
            (numerator * self.total_native_pixels) // denominator
            for numerator, denominator in budgets
        )
        if any(right > left for left, right in zip(allowed, allowed[1:])):
            raise RuntimeError("frozen exact budget counts are not nested")
        current_fp = sum(
            int(curve.false_positive_pixels[-1]) for curve in self.curves
        )
        current_tp = sum(
            int(curve.matched_objects[-1]) for curve in self.curves
        )
        best: list[tuple[tuple[int, int, float], float, int, int] | None] = [
            None for _ in budgets
        ]

        def consider(threshold: float, fp: int, tp: int) -> None:
            key = (tp, -fp, threshold)
            for index, maximum in enumerate(allowed):
                if fp <= maximum and (
                    best[index] is None or key > best[index][0]
                ):
                    best[index] = (key, threshold, fp, tp)

        # The endpoint is a real decision row, not an infinity sentinel.
        consider(1.0, current_fp, current_tp)
        global_rows_examined = 1
        stopped = False
        heap: list[tuple[float, int, int]] = []
        for curve_index, curve in enumerate(self.curves):
            if curve.thresholds.size > 1:
                row_index = int(curve.thresholds.size) - 2
                heapq.heappush(
                    heap,
                    (
                        -float(curve.thresholds[row_index]),
                        curve_index,
                        row_index,
                    ),
                )
        while heap:
            threshold = -heap[0][0]
            events: list[tuple[int, int]] = []
            while heap and -heap[0][0] == threshold:
                _, curve_index, row_index = heapq.heappop(heap)
                events.append((curve_index, row_index))
            for curve_index, row_index in events:
                curve = self.curves[curve_index]
                next_index = row_index - 1
                if next_index >= 0:
                    heapq.heappush(
                        heap,
                        (
                            -float(curve.thresholds[next_index]),
                            curve_index,
                            next_index,
                        ),
                    )
            fp_row, tp_row = self._aggregate_rows(
                np.asarray([threshold], dtype=np.float64)
            )
            current_fp = int(fp_row[0])
            current_tp = int(tp_row[0])
            global_rows_examined += 1
            if current_fp > allowed[0]:
                stopped = True
                break
            consider(threshold, current_fp, current_tp)

        if any(row is None for row in best):
            raise RuntimeError("threshold=1 failed exact oracle feasibility")
        selected = [row for row in best if row is not None]
        thresholds = _immutable_array(
            [row[1] for row in selected], np.float64
        )
        if np.any(np.diff(thresholds) < 0.0):
            raise Stage2CompositionalCurveError(
                "tighter-budget exact oracle thresholds decreased"
            )
        fp = _immutable_array([row[2] for row in selected], np.int64)
        tp = _immutable_array([row[3] for row in selected], np.int64)
        coordinates = _immutable_array(
            encode_probability_numpy(thresholds), np.float64
        )
        risk = _immutable_array(
            fp.astype(np.float64) / float(self.total_native_pixels),
            np.float64,
        )
        detection = _immutable_array(
            (
                tp.astype(np.float64) / float(self.ground_truth_objects)
                if self.ground_truth_objects > 0
                else np.zeros(len(budgets), dtype=np.float64)
            ),
            np.float64,
        )
        return CompositionalOracleRows(
            budget_rationals=budgets,
            allowed_false_positive_counts=allowed,
            thresholds=thresholds,
            coordinates=coordinates,
            false_positive_pixels=fp,
            matched_objects=tp,
            pixel_false_alarm_rate=risk,
            detection_probability=detection,
            total_native_pixels=self.total_native_pixels,
            ground_truth_objects=self.ground_truth_objects,
            global_rows_examined=global_rows_examined,
            stopped_at_first_loose_infeasible=stopped,
        )

    def select_exact_oracle_rows(self) -> CompositionalOracleRows:
        """Frozen RC5 path: select the three original exact budget rows."""

        return self._select_exact_oracle_rows(tuple(BUDGET_RATIONALS))

    def select_exact_oracle_rows_v2(self) -> CompositionalOracleRows:
        """RC5+ path: select all nine frozen exact budget-knot rows."""

        return self._select_exact_oracle_rows(tuple(BUDGET_KNOT_RATIONALS))


def build_compositional_exact_curve_provider(
    *,
    curve_bank: PerImageExactEventCurveBank,
    ordered_image_identities: Sequence[str],
) -> CompositionalExactCurveProvider:
    """Project one cyclic Q28 episode from a reusable per-image curve bank."""

    bank = assert_per_image_exact_event_curve_bank(curve_bank)
    if (
        isinstance(ordered_image_identities, (str, bytes))
        or not isinstance(ordered_image_identities, Sequence)
    ):
        raise TypeError("ordered_image_identities must be an ordered sequence")
    if len(ordered_image_identities) != CYCLIC_QUERY_SIZE:
        raise Stage2CompositionalCurveError(
            f"cyclic provider requires exactly Q={CYCLIC_QUERY_SIZE} curves"
        )
    identities = tuple(
        _sha256(value, f"ordered_image_identities[{index}]")
        for index, value in enumerate(ordered_image_identities)
    )
    if len(set(identities)) != len(identities):
        raise Stage2CompositionalCurveError(
            "cyclic provider contains duplicate image identity"
        )
    verified = tuple(bank.curve_for_identity(identity) for identity in identities)
    return build_compositional_exact_curve_provider_from_verified_curves(
        curve_bank_id=bank.bank_id,
        curves=verified,
    )


def build_compositional_exact_curve_provider_from_verified_curves(
    *,
    curve_bank_id: str,
    curves: Sequence[PerImageExactEventCurve],
) -> CompositionalExactCurveProvider:
    """Build one Q28 provider without materializing the complete curve bank.

    Persistent training collections use this narrow constructor after their
    verifier has SHA-bound and memory-mapped the complete flattened bank.  It
    validates only the 28 image capabilities referenced by the live episode;
    the externally verified ``curve_bank_id`` still enters the provider ID.
    """

    bank_id = _sha256(curve_bank_id, "curve_bank_id")
    if isinstance(curves, (str, bytes)) or not isinstance(curves, Sequence):
        raise TypeError("curves must be an ordered sequence")
    if len(curves) != CYCLIC_QUERY_SIZE:
        raise Stage2CompositionalCurveError(
            f"cyclic provider requires exactly Q={CYCLIC_QUERY_SIZE} curves"
        )
    verified = tuple(
        assert_per_image_exact_event_curve(curve) for curve in curves
    )
    identities = tuple(curve.image_identity_sha256 for curve in verified)
    if len(set(identities)) != len(identities):
        raise Stage2CompositionalCurveError(
            "cyclic provider contains duplicate image identity"
        )
    total = sum(curve.total_native_pixels for curve in verified)
    objects = sum(curve.ground_truth_objects for curve in verified)
    _strict_int(total, "aggregate total_native_pixels", minimum=1)
    _strict_int(objects, "aggregate ground_truth_objects")
    provider_id = hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": COMPOSITIONAL_PROVIDER_SCHEMA,
                "provider_id_algorithm": PROVIDER_ID_ALGORITHM,
                "curve_bank_id": bank_id,
                "query_size": CYCLIC_QUERY_SIZE,
                "ordered_curve_bindings": [
                    {
                        "ordinal": index,
                        "image_identity_sha256": (
                            curve.image_identity_sha256
                        ),
                        "curve_content_sha256": curve.content_sha256,
                    }
                    for index, curve in enumerate(verified)
                ],
            }
        )
    ).hexdigest()
    return CompositionalExactCurveProvider(
        curves=verified,
        curve_bank_id=bank_id,
        provider_id=provider_id,
        total_native_pixels=total,
        ground_truth_objects=objects,
        _capability=_PROVIDER_CAPABILITY,
    )


def assert_compositional_exact_curve_provider(
    value: object,
) -> CompositionalExactCurveProvider:
    if (
        type(value) is not CompositionalExactCurveProvider
        or getattr(value, "_capability", None) is not _PROVIDER_CAPABILITY
    ):
        raise TypeError("a verifier-built compositional provider is required")
    _sha256(value.curve_bank_id, "curve_bank_id")
    _sha256(value.provider_id, "provider_id")
    if len(value.curves) != CYCLIC_QUERY_SIZE:
        raise TypeError("compositional provider query cardinality was corrupted")
    for curve in value.curves:
        assert_per_image_exact_event_curve(curve)
    try:
        replayed = build_compositional_exact_curve_provider_from_verified_curves(
            curve_bank_id=value.curve_bank_id,
            curves=value.curves,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "compositional provider retained-token state is invalid"
        ) from error
    if (
        value.provider_id != replayed.provider_id
        or value.total_native_pixels != replayed.total_native_pixels
        or value.ground_truth_objects != replayed.ground_truth_objects
    ):
        raise TypeError(
            "compositional provider differs from identity replay"
        )
    return value


__all__ = [
    "BUDGET_COUNT_RULE",
    "COMPOSITIONAL_ORACLE_SELECTION_RULE",
    "COMPOSITIONAL_PROVIDER_SCHEMA",
    "CURVE_BANK_ID_ALGORITHM",
    "CYCLIC_QUERY_SIZE",
    "MAX_LIVE_PREDICTIONS",
    "RC5PLUS_MAX_LIVE_PREDICTIONS",
    "PER_IMAGE_CURVE_BANK_SCHEMA",
    "PER_IMAGE_CURVE_CONTENT_ALGORITHM",
    "PER_IMAGE_CURVE_SCHEMA",
    "PROVIDER_ID_ALGORITHM",
    "CompositionalCurveBrackets",
    "CompositionalExactCurveProvider",
    "CompositionalOracleRows",
    "PerImageExactEventCurve",
    "PerImageExactEventCurveBank",
    "Stage2CompositionalCurveError",
    "assert_compositional_exact_curve_provider",
    "assert_per_image_exact_event_curve",
    "assert_per_image_exact_event_curve_bank",
    "build_compositional_exact_curve_provider",
    "build_compositional_exact_curve_provider_from_verified_curves",
    "build_per_image_exact_event_curve",
    "build_per_image_exact_event_curve_bank",
]
