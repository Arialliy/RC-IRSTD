"""Dataset split and sample-path resolution helpers.

The public IRSTD datasets use several slightly different layouts.  In
particular, split entries may include an extension or a relative path and the
NUAA-SIRST masks commonly append ``_pixels0`` to the image stem.  These helpers
keep that compatibility logic out of the evaluation dataset itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


def resolve_split_file(
    dataset_dir: str | Path,
    split: str = "test",
    split_file: str | Path | None = None,
) -> Path:
    """Resolve one split file without silently choosing an ambiguous match.

    Explicit paths take precedence.  Otherwise the conventional root-level
    file, then exact ``img_idx`` names, then a unique glob match are tried.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if split_file is not None:
        candidate = Path(split_file).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Split file does not exist: {candidate}")
        return candidate

    split = split.strip()
    if not split:
        raise ValueError("split must be a non-empty name")

    exact_candidates = (
        root / f"{split}.txt",
        root / "img_idx" / f"{split}.txt",
        root / "img_idx" / f"{split}_{root.name}.txt",
        root / "img_idx" / f"{root.name}_{split}.txt",
    )
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(
        {
            path.resolve()
            for base in (root, root / "img_idx")
            if base.is_dir()
            for path in base.glob(f"{split}*.txt")
            if path.is_file()
        },
        key=lambda path: str(path).lower(),
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple split files match '{split}' under {root}. "
            f"Pass split_file explicitly:\n  - {rendered}"
        )

    expected = ", ".join(str(path) for path in exact_candidates)
    raise FileNotFoundError(
        f"No split file for '{split}' under {root}. Tried {expected} and "
        f"the pattern {split}*.txt"
    )


def read_split_entries(split_file: str | Path) -> list[str]:
    """Read non-empty split entries, accepting comments and whitespace.

    Some split files contain a second, whitespace-separated class column.  The
    first token is the sample identifier used by the repository datasets.
    Duplicate entries are rejected because they would bias evaluation counts.
    """

    path = Path(split_file)
    entries: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        entry = line.split()[0].replace("\\", "/")
        if entry in seen:
            raise ValueError(
                f"Duplicate split entry {entry!r} at {path}:{line_number}"
            )
        seen.add(entry)
        entries.append(entry)

    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    return entries


def sample_id_from_entry(entry: str) -> str:
    """Return a stable, extension-free identifier from a split entry."""

    normalized = entry.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("Split entry must not be empty")
    path = Path(normalized)
    parts = list(path.parts)
    if parts and parts[0].lower() in {"images", "masks"}:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid split entry: {entry!r}")
    parts[-1] = Path(parts[-1]).stem
    return Path(*parts).as_posix()


def resolve_sample_file(
    dataset_dir: str | Path,
    folder: str,
    entry: str,
    *,
    kind: str,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> Path:
    """Resolve an image or mask path deterministically.

    For masks, both the image stem and the NUAA ``<stem>_pixels0`` alias are
    considered.  Recursive fallback is allowed only when it yields one unique
    file, avoiding accidental pairing of duplicate stems from subdirectories.
    """

    if kind not in {"image", "mask"}:
        raise ValueError("kind must be either 'image' or 'mask'")

    root = Path(dataset_dir).expanduser().resolve()
    folder_root = root / folder
    if not folder_root.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder_root}")

    sample_id = sample_id_from_entry(entry)
    relative = Path(sample_id)
    stem = relative.name
    stem_candidates = [stem]
    if kind == "mask":
        if stem.endswith("_pixels0"):
            stem_candidates.append(stem[: -len("_pixels0")])
        else:
            stem_candidates.append(f"{stem}_pixels0")

    extension_candidates = _normalise_extensions(extensions)
    direct_candidates: list[Path] = []
    for candidate_stem in stem_candidates:
        relative_stem = relative.with_name(candidate_stem)
        for extension in extension_candidates:
            direct_candidates.append(folder_root / relative_stem.with_suffix(extension))

    # Preserve an explicitly supplied extension as the first candidate.
    raw_path = Path(entry.replace("\\", "/"))
    if raw_path.suffix:
        raw_parts = list(raw_path.parts)
        if raw_parts and raw_parts[0].lower() in {"images", "masks"}:
            raw_parts = raw_parts[1:]
        if raw_parts:
            direct_candidates.insert(0, folder_root / Path(*raw_parts))

    for candidate in _deduplicate_paths(direct_candidates):
        if candidate.is_file():
            return candidate.resolve()

    recursive_matches: list[Path] = []
    for candidate_stem in stem_candidates:
        for extension in extension_candidates:
            recursive_matches.extend(folder_root.rglob(f"{candidate_stem}{extension}"))
            # Some filesystems/datasets use upper-case extensions.
            recursive_matches.extend(folder_root.rglob(f"{candidate_stem}{extension.upper()}"))

    matches = sorted(
        {path.resolve() for path in recursive_matches if path.is_file()},
        key=lambda path: str(path).lower(),
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Ambiguous {kind} for split entry {entry!r}:\n  - {rendered}"
        )
    raise FileNotFoundError(
        f"Cannot resolve {kind} for split entry {entry!r} under {folder_root}"
    )


def resolve_image_and_mask(
    dataset_dir: str | Path,
    entry: str,
    image_folder: str = "images",
    mask_folder: str = "masks",
) -> tuple[Path, Path]:
    """Resolve the paired image and mask for one split entry."""

    image_path = resolve_sample_file(
        dataset_dir,
        image_folder,
        entry,
        kind="image",
    )
    mask_path = resolve_sample_file(
        dataset_dir,
        mask_folder,
        entry,
        kind="mask",
    )
    return image_path, mask_path


def _normalise_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
    normalised: list[str] = []
    for extension in extensions:
        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in normalised:
            normalised.append(extension)
    return tuple(normalised)


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result
