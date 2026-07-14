from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
    "artifacts",
    "dist",
    "repro_runs",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".pkl",
    ".npz",
    ".npy",
    ".csv",
    ".zip",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".gitignore",
}
_POSIX_HOME_PREFIX = "/" + "home" + "/"
_MACOS_USERS_PREFIX = "/" + "Users" + "/"
DEFAULT_FORBIDDEN_PATTERNS = (
    re.escape(_POSIX_HOME_PREFIX) + r"[^/\s]+/",
    re.escape(_MACOS_USERS_PREFIX) + r"[^/\s]+/",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an anonymous source-only supplement ZIP."
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="RC-IRSTD package root; defaults to the installed source tree.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--archive-root",
        default="RC_IRSTD_Anonymous",
        help="Top-level directory name inside the ZIP.",
    )
    parser.add_argument(
        "--forbid",
        action="append",
        default=None,
        help="Additional regular expression that must not appear in text files.",
    )
    return parser


def _is_excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return True
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if path.name in {"RC_IRSTD_AAAI_Implementation.zip"}:
        return True
    return False


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not _is_excluded(path, root):
            yield path


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore"


def _scan_forbidden(
    files: Iterable[Path],
    patterns: list[re.Pattern[str]],
) -> list[str]:
    violations: list[str] = []
    for path in files:
        if not _is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                violations.append(
                    f"{path}: pattern {pattern.pattern!r} matched {match.group(0)!r}"
                )
    return violations


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.source_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "rc_irstd").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like the RC-IRSTD source root"
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_root = PurePosixPath(args.archive_root)
    if archive_root.is_absolute() or ".." in archive_root.parts:
        raise ValueError("archive-root must be a safe relative directory name")

    files = list(_iter_source_files(root))
    expressions = list(DEFAULT_FORBIDDEN_PATTERNS)
    expressions.extend(args.forbid or [])
    patterns = [re.compile(expression) for expression in expressions]
    violations = _scan_forbidden(files, patterns)
    if violations:
        formatted = "\n".join(f"- {item}" for item in violations)
        raise RuntimeError(f"Anonymization scan failed:\n{formatted}")

    manifest_entries: list[dict[str, object]] = []
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w", compression=compression, compresslevel=9) as archive:
        for path in files:
            data = path.read_bytes()
            relative = PurePosixPath(path.relative_to(root).as_posix())
            archive_name = str(archive_root / relative)
            archive.writestr(archive_name, data)
            manifest_entries.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(data),
                    "sha256": _sha256(data),
                }
            )
        manifest = {
            "archive_root": archive_root.as_posix(),
            "source_root_redacted": True,
            "file_count": len(manifest_entries),
            "excluded": {
                "directories": sorted(EXCLUDED_DIRECTORY_NAMES),
                "suffixes": sorted(EXCLUDED_SUFFIXES),
            },
            "files": manifest_entries,
        }
        archive.writestr(
            str(archive_root / "ANONYMOUS_MANIFEST.json"),
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        )

    summary = {
        "output": str(output),
        "archive_root": archive_root.as_posix(),
        "source_files": len(files),
        "archive_sha256": _sha256(output.read_bytes()),
        "status": "passed",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
