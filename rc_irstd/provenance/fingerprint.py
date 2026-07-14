from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _path_descriptor(value: str, working_directory: Path) -> dict[str, object] | None:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = working_directory / path
    if not path.exists():
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    # Dataset directories can be huge. Hash split/manifests and directory metadata
    # rather than every image byte; checkpoint/config files are still fully hashed.
    manifest_files = []
    for pattern in ("manifest.json", "*.yaml", "*.yml", "img_idx/*.txt", "*.txt"):
        for item in sorted(resolved.glob(pattern)):
            if item.is_file():
                manifest_files.append(
                    {
                        "path": item.relative_to(resolved).as_posix(),
                        "sha256": sha256_file(item),
                    }
                )
    return {
        "path": str(resolved),
        "kind": "directory",
        "mtime_ns": resolved.stat().st_mtime_ns,
        "manifests": manifest_files,
    }


def command_fingerprint(
    command: Iterable[str],
    working_directory: str | Path,
    source_root: str | Path,
) -> tuple[str, dict[str, object]]:
    command_values = [str(value) for value in command]
    cwd = Path(working_directory).resolve()
    descriptors = []
    seen: set[str] = set()
    for value in command_values:
        descriptor = _path_descriptor(value, cwd)
        if descriptor is None:
            continue
        key = str(descriptor["path"])
        if key not in seen:
            descriptors.append(descriptor)
            seen.add(key)
    payload = {
        "command": command_values,
        "working_directory": str(cwd),
        "source_tree": source_tree_fingerprint(source_root),
        "inputs": descriptors,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload
