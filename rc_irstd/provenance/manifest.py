from __future__ import annotations

from pathlib import Path
from typing import Any

from rc_irstd.utils.io import atomic_json_dump, load_json


def run_manifest_path(expected: str | Path) -> Path:
    path = Path(expected)
    return path.with_name(path.name + ".run_manifest.json")


def load_run_manifest(expected: str | Path) -> dict[str, Any] | None:
    path = run_manifest_path(expected)
    return load_json(path) if path.is_file() else None


def write_run_manifest(
    expected: str | Path,
    fingerprint: str,
    payload: dict[str, object],
) -> Path:
    path = run_manifest_path(expected)
    atomic_json_dump(
        {"fingerprint": fingerprint, "provenance": payload},
        path,
    )
    return path
