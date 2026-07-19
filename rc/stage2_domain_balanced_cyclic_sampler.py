"""Deterministic equal-domain epoch sampling for cyclic RC5 source episodes.

Each outer fold has exactly two source domains.  Their complete cyclic
training collections can differ substantially in size, so concatenation would
silently let the larger source dominate the loss.  For epoch ``e`` this
sampler selects ``min(n_0, n_1)`` distinct episodes from each domain, using a
SHA-256-defined fixed permutation and a rotating cyclic slice.  It then emits
domain pairs, which keeps every even-sized minibatch exactly domain balanced.

The construction is source-only, result-free, and contains no Python
``hash()`` or caller-supplied manual seed override.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any


SCHEMA_VERSION = "rc-irstd.stage2-domain-balanced-cyclic-epoch-sampler.v1"
ALGORITHM_ID = "sha256_fixed_permutation_rotating_slice_domain_pairs_v1"
DOMAIN_ORDER = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
OUTER_TARGETS = {
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
}
MINIMUM_DOMAIN_EPISODES = 84
_VERIFIED_EPOCH_CAPABILITY = object()


class Stage2DomainBalancedSamplerError(ValueError):
    """The frozen equal-domain epoch sampling contract was violated."""


@dataclass(frozen=True, init=False)
class VerifiedDomainBalancedCyclicEpoch:
    """Verifier-issued epoch order; trainers must not consume raw mappings."""

    canonical_payload: bytes
    ordered_selection_sha256: str
    _capability: object

    @property
    def payload(self) -> dict[str, Any]:
        assert_verified_domain_balanced_cyclic_epoch(self)
        value = json.loads(self.canonical_payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("verified sampler capability was corrupted")
        return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2DomainBalancedSamplerError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _source_domains(outer_fold_id: str) -> tuple[str, str]:
    if not isinstance(outer_fold_id, str) or outer_fold_id not in OUTER_TARGETS:
        raise Stage2DomainBalancedSamplerError("unsupported outer_fold_id")
    target = OUTER_TARGETS[outer_fold_id]
    result = tuple(domain for domain in DOMAIN_ORDER if domain != target)
    if len(result) != 2:
        raise RuntimeError("outer fold did not resolve to two source domains")
    return result


def _episode_counts(
    value: Mapping[str, Any], source_domains: tuple[str, str]
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(source_domains):
        raise Stage2DomainBalancedSamplerError(
            "episode_counts must contain exactly the two fixed source domains"
        )
    return {
        domain: _strict_int(
            value[domain],
            f"episode_counts[{domain!r}]",
            minimum=MINIMUM_DOMAIN_EPISODES,
        )
        for domain in source_domains
    }


def _fixed_permutation(
    *, outer_fold_id: str, domain: str, derived_seed: int, count: int
) -> list[int]:
    tagged = [
        (
            _digest(
                {
                    "algorithm_id": ALGORITHM_ID,
                    "outer_fold_id": outer_fold_id,
                    "source_domain": domain,
                    "derived_seed": derived_seed,
                    "episode_index": index,
                }
            ),
            index,
        )
        for index in range(count)
    ]
    tagged.sort()
    return [index for _, index in tagged]


def build_domain_balanced_cyclic_epoch(
    *,
    outer_fold_id: str,
    derived_seed: int,
    epoch: int,
    episode_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one deterministic, without-replacement, equal-domain epoch."""

    domains = _source_domains(outer_fold_id)
    seed = _strict_int(derived_seed, "derived_seed", minimum=1)
    epoch_index = _strict_int(epoch, "epoch")
    counts = _episode_counts(episode_counts, domains)
    per_domain = min(counts.values())
    selected: dict[str, list[int]] = {}
    for domain in domains:
        count = counts[domain]
        permutation = _fixed_permutation(
            outer_fold_id=outer_fold_id,
            domain=domain,
            derived_seed=seed,
            count=count,
        )
        start = (epoch_index * per_domain) % count
        chosen = [
            permutation[(start + offset) % count]
            for offset in range(per_domain)
        ]
        if len(set(chosen)) != per_domain:
            raise RuntimeError("rotating domain slice repeated an episode")
        selected[domain] = chosen

    ordered: list[dict[str, Any]] = []
    for pair_index in range(per_domain):
        pair = [
            {
                "source_domain": domain,
                "domain_episode_index": selected[domain][pair_index],
            }
            for domain in domains
        ]
        swap = int(
            _digest(
                {
                    "algorithm_id": ALGORITHM_ID,
                    "outer_fold_id": outer_fold_id,
                    "derived_seed": seed,
                    "epoch": epoch_index,
                    "domain_pair_index": pair_index,
                }
            )[-1],
            16,
        ) % 2
        if swap:
            pair.reverse()
        ordered.extend(pair)

    selection_digest = _digest(ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "outer_fold_id": outer_fold_id,
        "outer_target": OUTER_TARGETS[outer_fold_id],
        "source_domain_order": list(domains),
        "derived_seed": seed,
        "epoch": epoch_index,
        "episode_counts": counts,
        "draws_per_domain": per_domain,
        "epoch_size": 2 * per_domain,
        "replacement_within_domain_epoch": False,
        "domain_pairing": "one_episode_per_source_domain_per_pair",
        "ordered_selection": ordered,
        "ordered_selection_sha256_algorithm": (
            "sha256-canonical-json-domain-balanced-epoch-selection-v1"
        ),
        "ordered_selection_sha256": selection_digest,
    }


def verify_domain_balanced_cyclic_epoch(
    payload: Mapping[str, Any],
) -> VerifiedDomainBalancedCyclicEpoch:
    """Replay the frozen sampler and grant a capability only for exact output."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    required = {
        "outer_fold_id",
        "derived_seed",
        "epoch",
        "episode_counts",
    }
    missing = required.difference(payload)
    if missing:
        raise Stage2DomainBalancedSamplerError(
            f"sampler payload is missing fields: {sorted(missing)}"
        )
    expected = build_domain_balanced_cyclic_epoch(
        outer_fold_id=payload["outer_fold_id"],
        derived_seed=payload["derived_seed"],
        epoch=payload["epoch"],
        episode_counts=payload["episode_counts"],
    )
    canonical = _canonical_json_bytes(dict(payload))
    if canonical != _canonical_json_bytes(expected):
        raise Stage2DomainBalancedSamplerError(
            "sampler payload is not the exact frozen replay"
        )
    value = object.__new__(VerifiedDomainBalancedCyclicEpoch)
    object.__setattr__(value, "canonical_payload", canonical)
    object.__setattr__(
        value,
        "ordered_selection_sha256",
        expected["ordered_selection_sha256"],
    )
    object.__setattr__(value, "_capability", _VERIFIED_EPOCH_CAPABILITY)
    return value


def assert_verified_domain_balanced_cyclic_epoch(
    value: object,
) -> VerifiedDomainBalancedCyclicEpoch:
    if (
        type(value) is not VerifiedDomainBalancedCyclicEpoch
        or getattr(value, "_capability", None) is not _VERIFIED_EPOCH_CAPABILITY
    ):
        raise TypeError("a verified domain-balanced cyclic epoch is required")
    payload = json.loads(value.canonical_payload.decode("utf-8"))
    if payload.get("ordered_selection_sha256") != value.ordered_selection_sha256:
        raise TypeError("verified sampler capability was corrupted")
    return value


__all__ = [
    "ALGORITHM_ID",
    "DOMAIN_ORDER",
    "MINIMUM_DOMAIN_EPISODES",
    "OUTER_TARGETS",
    "SCHEMA_VERSION",
    "Stage2DomainBalancedSamplerError",
    "VerifiedDomainBalancedCyclicEpoch",
    "assert_verified_domain_balanced_cyclic_epoch",
    "build_domain_balanced_cyclic_epoch",
    "verify_domain_balanced_cyclic_epoch",
]
