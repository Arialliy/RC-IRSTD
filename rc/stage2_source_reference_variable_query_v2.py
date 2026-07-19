"""Exact variable-Q consumer binding for verified Stage-2 source references.

The base source-reference bundle remains unchanged.  This additive capability
promotes it for RC5 only after every consumer-window binding has independently
passed the variable-query v2 public verifier.  Consequently a bundle mixing
fixed-Q v1 and variable-Q v2 consumers cannot acquire this capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from data_ext.stage2_variable_query_window import (
    SCHEMA_VERSION as VARIABLE_QUERY_WINDOW_SCHEMA,
    VerifiedStage2VariableQueryWindow,
    assert_verified_stage2_variable_query_window,
    verify_stage2_variable_query_window,
)
from rc.build_stage2_source_reference import VerifiedStage2SourceReference


CAPABILITY_SCHEMA = "rc-irstd.stage2-source-reference-variable-query-binding.v2"
_CAPABILITY_TOKEN = object()


class Stage2SourceReferenceVariableQueryV2Error(ValueError):
    """A source-reference/variable-Q consumer closure failed closed."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, init=False)
class VerifiedStage2SourceReferenceVariableQueryV2:
    """Verifier-issued source reference whose complete consumer set is v2."""

    source_reference_bundle: VerifiedStage2SourceReference
    consumer_windows: tuple[VerifiedStage2VariableQueryWindow, ...]
    consumer_bindings: tuple[Mapping[str, Any], ...]
    capability_schema: str
    mixed_consumer_schemas_allowed: bool
    _capability: object

    def __init__(
        self,
        *,
        source_reference_bundle: VerifiedStage2SourceReference | None = None,
        consumer_windows: tuple[VerifiedStage2VariableQueryWindow, ...] = (),
        consumer_bindings: tuple[Mapping[str, Any], ...] = (),
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedStage2SourceReferenceVariableQueryV2 is verifier-issued only"
            )
        if source_reference_bundle is None or not consumer_windows:
            raise RuntimeError(
                "source-reference variable-Q capability construction is incomplete"
            )
        for window in consumer_windows:
            assert_verified_stage2_variable_query_window(window)
        object.__setattr__(
            self, "source_reference_bundle", source_reference_bundle
        )
        object.__setattr__(self, "consumer_windows", tuple(consumer_windows))
        object.__setattr__(
            self,
            "consumer_bindings",
            tuple(_freeze(item) for item in consumer_bindings),
        )
        object.__setattr__(self, "capability_schema", CAPABILITY_SCHEMA)
        object.__setattr__(self, "mixed_consumer_schemas_allowed", False)
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)

    @property
    def path(self) -> Path:
        return self.source_reference_bundle.path

    @property
    def npz_sha256(self) -> str:
        return self.source_reference_bundle.npz_sha256

    @property
    def audit_path(self) -> Path:
        return self.source_reference_bundle.audit_path

    @property
    def audit_sha256(self) -> str:
        return self.source_reference_bundle.audit_sha256

    @property
    def statistics_config(self) -> Any:
        return self.source_reference_bundle.statistics_config

    @property
    def source_reference(self) -> Any:
        return self.source_reference_bundle.source_reference

    @property
    def stage2_contract(self) -> Mapping[str, Any]:
        return self.source_reference_bundle.stage2_contract

    @property
    def detector_identity(self) -> Mapping[str, Any]:
        return self.source_reference_bundle.detector_identity

    @property
    def checkpoint_binding(self) -> Mapping[str, Any]:
        return self.source_reference_bundle.checkpoint_binding

    @property
    def reference_role(self) -> str:
        return self.source_reference_bundle.reference_role


def assert_verified_stage2_source_reference_variable_query_v2(
    value: Any,
) -> VerifiedStage2SourceReferenceVariableQueryV2:
    if (
        type(value) is not VerifiedStage2SourceReferenceVariableQueryV2
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
        or value.capability_schema != CAPABILITY_SCHEMA
        or value.mixed_consumer_schemas_allowed is not False
    ):
        raise TypeError(
            "a verifier-issued variable-Q v2 source-reference capability is required"
        )
    for window in value.consumer_windows:
        assert_verified_stage2_variable_query_window(window)
    return value


def verify_stage2_source_reference_variable_query_v2(
    source_reference: VerifiedStage2SourceReference,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2SourceReferenceVariableQueryV2:
    """Require every bound consumer manifest to be exact variable-Q v2.

    ``source_reference`` must first have been produced by the public base
    source-reference verifier.  Each of its bound consumers is then reopened
    only as result-free metadata and independently promoted by the public
    variable-query verifier.  Unknown, v1, mixed, stale, cross-domain or
    summary-drifted consumer sets fail closed.
    """

    if type(source_reference) is not VerifiedStage2SourceReference:
        raise TypeError(
            "source_reference must be a public VerifiedStage2SourceReference"
        )
    root = Path(repository_root).expanduser() if repository_root is not None else None
    if repository_root is None:
        output = source_reference.audit.get("output")
        binding = (
            output.get("source_reference_npz")
            if isinstance(output, Mapping)
            else None
        )
        raw_relative = binding.get("path") if isinstance(binding, Mapping) else None
        if type(raw_relative) is not str:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "source-reference audit lacks its canonical NPZ output path"
            )
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise Stage2SourceReferenceVariableQueryV2Error(
                "source-reference audit NPZ output path is not canonical"
            )
        candidate = source_reference.path
        for _ in relative.parts:
            candidate = candidate.parent
        if candidate.joinpath(*relative.parts) != source_reference.path:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "source-reference audit NPZ path does not locate the capability"
            )
        root = candidate
    assert root is not None
    if root.is_symlink():
        raise Stage2SourceReferenceVariableQueryV2Error(
            "repository_root must not be a symlink"
        )
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise Stage2SourceReferenceVariableQueryV2Error(
            "repository_root is not a directory"
        )
    try:
        source_reference.path.relative_to(root)
        source_reference.audit_path.relative_to(root)
    except ValueError as error:
        raise Stage2SourceReferenceVariableQueryV2Error(
            "source-reference bundle is outside repository_root"
        ) from error

    windows: list[VerifiedStage2VariableQueryWindow] = []
    expected_bindings: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, binding in enumerate(source_reference.consumer_bindings):
        if not isinstance(binding, Mapping):
            raise Stage2SourceReferenceVariableQueryV2Error(
                f"consumer_bindings[{index}] is not a mapping"
            )
        relative = str(binding.get("path"))
        if relative in seen_paths:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "source-reference consumer paths are not unique"
            )
        seen_paths.add(relative)
        try:
            window = verify_stage2_variable_query_window(
                relative,
                str(binding.get("sha256")),
                repository_root=root,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "every source-reference consumer must be exact variable-Q v2"
            ) from error
        payload = window.payload
        if payload["schema_version"] != VARIABLE_QUERY_WINDOW_SCHEMA:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "consumer window schema drifted"
            )
        expected = {
            "path": window.path.relative_to(root).as_posix(),
            "sha256": window.manifest_sha256,
            "domain": payload["domain"],
            "episode_role": payload["episode_role"],
            "complete_window_count": payload["complete_window_count"],
            "record_count": payload["window_record_count"],
        }
        if dict(binding) != expected:
            raise Stage2SourceReferenceVariableQueryV2Error(
                "source-reference variable-Q consumer summary mismatch"
            )
        windows.append(window)
        expected_bindings.append(expected)

    if not windows:
        raise Stage2SourceReferenceVariableQueryV2Error(
            "source-reference has no consumer windows"
        )
    return VerifiedStage2SourceReferenceVariableQueryV2(
        source_reference_bundle=source_reference,
        consumer_windows=tuple(windows),
        consumer_bindings=tuple(expected_bindings),
        _capability=_CAPABILITY_TOKEN,
    )


__all__ = [
    "CAPABILITY_SCHEMA",
    "Stage2SourceReferenceVariableQueryV2Error",
    "VerifiedStage2SourceReferenceVariableQueryV2",
    "assert_verified_stage2_source_reference_variable_query_v2",
    "verify_stage2_source_reference_variable_query_v2",
]
