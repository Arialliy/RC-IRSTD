"""Fixed-width statistics extracted from unlabeled target-domain context.

No function in this module accepts a mask.  Ground-truth masks belong only to
the disjoint query window used to construct oracle labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, maximum_filter

from .schema import SourceContract, SourceReference, StatisticsConfig


PROBABILITY_HISTOGRAM_BINS = 32
PEAK_HISTOGRAM_BINS = 32
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)
# Exact empirical quantiles require memory proportional to the complete domain.
# The collection-wide StatisticsConfig binds the deterministic sample limit and
# estimator so an old exact-v2 artifact cannot be silently mixed with v3.
STREAM_CHUNK_SIZE = 65_536
SOURCE_DISTANCE_NAMES = (
    "source_distance_available",
    "source_distance_min",
    "source_distance_mean",
    "source_distance_max",
    "source_distance_std",
    "source_cosine_distance_min",
)


def _histogram_names(prefix: str, bins: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_hist_{index:02d}" for index in range(bins))


BASE_FEATURE_NAMES = (
    _histogram_names("prob", PROBABILITY_HISTOGRAM_BINS)
    + tuple(f"prob_q{int(q * 1000):03d}" for q in QUANTILES)
    + _histogram_names("peak", PEAK_HISTOGRAM_BINS)
    + tuple(f"peak_q{int(q * 1000):03d}" for q in QUANTILES)
    + (
        "peaks_per_megapixel",
        "gray_available",
        "gray_mean",
        "gray_std",
        "gray_mad",
        "gradient_mean",
        "gradient_q950",
        "laplacian_mad",
        "high_frequency_energy_ratio",
    )
)
FEATURE_NAMES = BASE_FEATURE_NAMES + SOURCE_DISTANCE_NAMES
BASE_FEATURE_DIM = len(BASE_FEATURE_NAMES)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class DomainStatistics:
    vector: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: Mapping[str, Any] | None = None
    statistics_config: StatisticsConfig | None = None

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.shape != (len(self.feature_names),):
            raise ValueError(
                f"statistics shape must be {(len(self.feature_names),)}, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("statistics must be finite")
        object.__setattr__(self, "vector", vector)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _iter_images(values: Any, name: str) -> Iterator[np.ndarray]:
    """Yield validated images without materialising an input iterable."""

    if isinstance(values, np.ndarray) or hasattr(values, "detach"):
        source_array = _to_numpy(values)
        if source_array.ndim == 2:
            raw = (source_array,)
        elif source_array.ndim == 3:
            raw = (
                source_array[index] for index in range(source_array.shape[0])
            )
        elif source_array.ndim == 4 and source_array.shape[1] == 1:
            raw = (
                source_array[index, 0] for index in range(source_array.shape[0])
            )
        else:
            raise ValueError(f"{name} must be 2D, [N,H,W], or [N,1,H,W]")
    else:
        try:
            raw = iter(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be an image or iterable of images") from exc
    for index, value in enumerate(raw):
        image_array = np.squeeze(_to_numpy(value))
        if image_array.ndim != 2:
            raise ValueError(f"{name}[{index}] must be two-dimensional")
        if image_array.size == 0 or not np.isfinite(image_array).all():
            raise ValueError(f"{name}[{index}] must be non-empty and finite")
        yield image_array


def _normalise_grayscale(image: np.ndarray) -> np.ndarray:
    original_dtype = image.dtype
    result = image.astype(np.float64, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        maximum = float(np.iinfo(original_dtype).max)
        if maximum > 0:
            result = result / maximum
    elif result.min() < 0.0 or result.max() > 1.0:
        low, high = np.quantile(result, [0.001, 0.999])
        if high > low:
            result = (result - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def _normalised_histogram_counts(counts: np.ndarray) -> np.ndarray:
    total = max(int(counts.sum()), 1)
    return counts.astype(np.float64) / total


def _splitmix64_priorities(indices: np.ndarray) -> np.ndarray:
    """Return deterministic, collision-free priorities for uint64 indices."""

    values = np.asarray(indices, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class _BoundedQuantileAccumulator:
    """Deterministic priority sample with exact small-window behaviour."""

    def __init__(self, limit: int, chunk_size: int) -> None:
        if limit <= 0 or chunk_size <= 0:
            raise ValueError("quantile sample limit and chunk size must be positive")
        self.limit = int(limit)
        self.chunk_size = int(chunk_size)
        self.count = 0
        self._values = np.empty(0, dtype=np.float64)
        self._priorities = np.empty(0, dtype=np.uint64)

    @property
    def sample_size(self) -> int:
        return int(self._values.size)

    @property
    def exact(self) -> bool:
        return self.count <= self.limit

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        for start in range(0, flat.size, self.chunk_size):
            block = flat[start : start + self.chunk_size]
            stop_index = self.count + int(block.size)
            indices = np.arange(self.count, stop_index, dtype=np.uint64)
            priorities = _splitmix64_priorities(indices)
            combined_values = np.concatenate((self._values, block))
            combined_priorities = np.concatenate((self._priorities, priorities))
            if combined_values.size > self.limit:
                keep = np.argpartition(combined_priorities, self.limit - 1)[: self.limit]
                combined_values = combined_values[keep]
                combined_priorities = combined_priorities[keep]
            self._values = combined_values
            self._priorities = combined_priorities
            self.count = stop_index

    def quantiles(self, quantiles: Sequence[float]) -> np.ndarray:
        if self._values.size == 0:
            return np.zeros(len(quantiles), dtype=np.float64)
        return np.asarray(np.quantile(self._values, quantiles), dtype=np.float64)


class _StreamingMoments:
    """Numerically stable population mean/standard-deviation accumulator."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        batch_count = int(flat.size)
        batch_mean = float(flat.mean())
        centered = flat - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return float(np.sqrt(max(self.m2 / self.count, 0.0)))


def _plateau_representative_values(
    image: np.ndarray,
    candidates: np.ndarray,
    *,
    kernel_size: int,
) -> np.ndarray:
    """Mirror losses.local_peak_cvar's kernel-local row-major rank NMS."""

    rank = np.arange(image.size, dtype=np.float64).reshape(image.shape)
    ranked_candidates = np.where(candidates, rank, -1.0)
    local_rank_max = maximum_filter(
        ranked_candidates,
        size=kernel_size,
        mode="constant",
        cval=-1.0,
    )
    representatives = candidates & (ranked_candidates == local_rank_max)
    return np.asarray(image[representatives], dtype=np.float64)


def _source_center_matrix(source_centers: Any) -> np.ndarray:
    if source_centers is None:
        return np.empty((0, BASE_FEATURE_DIM), dtype=np.float64)
    if isinstance(source_centers, Mapping):
        values = list(source_centers.values())
    else:
        array = np.asarray(source_centers)
        values = [array] if array.ndim == 1 else list(array)
    rows = []
    for value in values:
        if isinstance(value, DomainStatistics):
            row = value.vector[:BASE_FEATURE_DIM]
        elif isinstance(value, Mapping) and "vector" in value:
            row = np.asarray(value["vector"])
        else:
            row = np.asarray(value)
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        if row.size == FEATURE_DIM:
            row = row[:BASE_FEATURE_DIM]
        if row.size != BASE_FEATURE_DIM:
            raise ValueError(
                f"source center dimension must be {BASE_FEATURE_DIM} or {FEATURE_DIM}, got {row.size}"
            )
        if not np.isfinite(row).all():
            raise ValueError("source centers must be finite")
        rows.append(row)
    if not rows:
        return np.empty((0, BASE_FEATURE_DIM), dtype=np.float64)
    return np.stack(rows, axis=0)


def aggregate_source_distances(
    base_features: Sequence[float] | np.ndarray,
    source_centers: Any = None,
    source_scale: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate an arbitrary number of source distances into six features."""

    feature = np.asarray(base_features, dtype=np.float64).reshape(-1)
    if feature.size != BASE_FEATURE_DIM:
        raise ValueError(f"base feature dimension must be {BASE_FEATURE_DIM}")
    centers = _source_center_matrix(source_centers)
    if centers.shape[0] == 0:
        return np.zeros(len(SOURCE_DISTANCE_NAMES), dtype=np.float32)
    if source_scale is None:
        scale = np.ones(BASE_FEATURE_DIM, dtype=np.float64)
    else:
        scale = np.asarray(source_scale, dtype=np.float64).reshape(-1)
        if scale.size == FEATURE_DIM:
            scale = scale[:BASE_FEATURE_DIM]
        if scale.size != BASE_FEATURE_DIM:
            raise ValueError(f"source_scale dimension must be {BASE_FEATURE_DIM}")
        scale = np.where(np.abs(scale) < 1e-8, 1.0, np.abs(scale))
    distances = np.linalg.norm((centers - feature[None, :]) / scale[None, :], axis=1)
    distances = distances / np.sqrt(BASE_FEATURE_DIM)
    feature_norm = max(float(np.linalg.norm(feature)), 1e-12)
    center_norms = np.maximum(np.linalg.norm(centers, axis=1), 1e-12)
    cosine_distance = 1.0 - (centers @ feature) / (center_norms * feature_norm)
    return np.asarray(
        [
            1.0,
            distances.min(),
            distances.mean(),
            distances.max(),
            distances.std(),
            cosine_distance.min(),
        ],
        dtype=np.float32,
    )


def extract_unlabeled_statistics(
    probabilities: Any,
    grayscale_images: Any | None = None,
    *,
    source_centers: Any = None,
    source_scale: Sequence[float] | np.ndarray | None = None,
    source_reference: SourceReference | None = None,
    statistics_config: StatisticsConfig | None = None,
    peak_kernel_size: int = 3,
    peak_min_score: float = 0.05,
) -> DomainStatistics:
    """Extract the canonical fixed-length context vector without label input."""

    if statistics_config is None:
        statistics_config = StatisticsConfig(
            peak_kernel_size=peak_kernel_size,
            peak_min_score=peak_min_score,
        )
    elif peak_kernel_size != 3 or peak_min_score != 0.05:
        if (
            peak_kernel_size != statistics_config.peak_kernel_size
            or peak_min_score != statistics_config.peak_min_score
        ):
            raise ValueError("explicit peak settings disagree with statistics_config")
    peak_kernel_size = statistics_config.peak_kernel_size
    peak_min_score = statistics_config.peak_min_score
    if source_reference is not None:
        if source_centers is not None or source_scale is not None:
            raise ValueError("use source_reference or raw source centers/scale, not both")
        source_centers = np.asarray(source_reference.centers, dtype=np.float64)
        source_scale = np.asarray(source_reference.scale, dtype=np.float64)
    sample_limit = int(statistics_config.quantile_sample_limit)
    chunk_size = int(STREAM_CHUNK_SIZE)
    probability_samples = _BoundedQuantileAccumulator(sample_limit, chunk_size)
    peak_samples = _BoundedQuantileAccumulator(sample_limit, chunk_size)
    gray_samples = _BoundedQuantileAccumulator(sample_limit, chunk_size)
    gradient_samples = _BoundedQuantileAccumulator(sample_limit, chunk_size)
    laplacian_samples = _BoundedQuantileAccumulator(sample_limit, chunk_size)
    probability_counts = np.zeros(PROBABILITY_HISTOGRAM_BINS, dtype=np.int64)
    peak_counts = np.zeros(PEAK_HISTOGRAM_BINS, dtype=np.int64)
    gray_moments = _StreamingMoments()
    gradient_sum = 0.0
    gradient_count = 0
    high_frequency_ratio_sum = 0.0
    total_pixels = 0
    total_peaks = 0
    num_images = 0

    probability_iterator = iter(_iter_images(probabilities, "probabilities"))
    gray_iterator = (
        iter(_iter_images(grayscale_images, "grayscale_images"))
        if grayscale_images is not None
        else None
    )
    for image in probability_iterator:
        image = np.clip(image.astype(np.float64, copy=False), 0.0, 1.0)
        probability_counts += np.histogram(
            image, bins=PROBABILITY_HISTOGRAM_BINS, range=(0.0, 1.0)
        )[0]
        probability_samples.update(image)
        pooled = maximum_filter(image, size=peak_kernel_size, mode="nearest")
        if statistics_config.plateau_atol == 0.0:
            reaches_local_max = image == pooled
        else:
            reaches_local_max = np.isclose(
                image,
                pooled,
                rtol=0.0,
                atol=statistics_config.plateau_atol,
            )
        candidates = reaches_local_max & (image >= peak_min_score)
        # Mirror the detector loss exactly: use row-major ranks only as a
        # deterministic tie-break inside the same local pooling kernel.
        peaks = _plateau_representative_values(
            image, candidates, kernel_size=statistics_config.peak_kernel_size
        )
        peak_counts += np.histogram(
            peaks, bins=PEAK_HISTOGRAM_BINS, range=(0.0, 1.0)
        )[0]
        peak_samples.update(peaks)
        total_pixels += int(image.size)
        total_peaks += int(peaks.size)
        num_images += 1

        if gray_iterator is not None:
            try:
                gray = next(gray_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "probabilities and grayscale_images must have the same length"
                ) from exc
            gray = _normalise_grayscale(gray)
            gray_moments.update(gray)
            gray_samples.update(gray)
            grad_y, grad_x = np.gradient(gray)
            gradients = np.hypot(grad_x, grad_y)
            gradient_sum += float(gradients.sum(dtype=np.float64))
            gradient_count += int(gradients.size)
            gradient_samples.update(gradients)
            laplacian = laplace(gray, mode="nearest")
            laplacian_samples.update(np.abs(laplacian - np.median(laplacian)))
            high = gray - gaussian_filter(gray, sigma=1.0, mode="nearest")
            high_frequency_ratio_sum += float(
                np.mean(np.square(high))
                / max(float(np.mean(np.square(gray))), 1e-12)
            )

    if num_images == 0:
        raise ValueError("probabilities must contain at least one image")
    if gray_iterator is not None:
        try:
            next(gray_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("probabilities and grayscale_images must have the same length")

    probability_histogram = _normalised_histogram_counts(probability_counts)
    probability_quantiles = probability_samples.quantiles(QUANTILES)
    if total_peaks:
        peak_histogram = _normalised_histogram_counts(peak_counts)
        peak_quantiles = peak_samples.quantiles(QUANTILES)
    else:
        peak_histogram = np.zeros(PEAK_HISTOGRAM_BINS, dtype=np.float64)
        peak_quantiles = np.zeros(len(QUANTILES), dtype=np.float64)
    peaks_per_megapixel = total_peaks / max(total_pixels / 1_000_000.0, 1e-12)

    gray_features = np.zeros(8, dtype=np.float64)
    gray_available = float(gray_iterator is not None)
    if gray_iterator is not None:
        gray_median = float(gray_samples.quantiles((0.5,))[0])
        # When sampling is active this is the MAD of the same deterministic
        # sample.  Below the cap it remains exactly the legacy global MAD.
        gray_mad = float(
            np.median(np.abs(gray_samples._values - gray_median))
        )
        gray_features = np.asarray(
            [
                gray_moments.mean,
                gray_moments.std,
                gray_mad,
                gradient_sum / max(gradient_count, 1),
                gradient_samples.quantiles((0.95,))[0],
                laplacian_samples.quantiles((0.5,))[0],
                high_frequency_ratio_sum / num_images,
                0.0,
            ],
            dtype=np.float64,
        )
    # The final placeholder above keeps assignment readable; availability is
    # inserted explicitly to match BASE_FEATURE_NAMES.
    base = np.concatenate(
        [
            probability_histogram,
            probability_quantiles,
            peak_histogram,
            peak_quantiles,
            np.asarray([peaks_per_megapixel, gray_available]),
            gray_features[:7],
        ]
    )
    if base.shape != (BASE_FEATURE_DIM,):
        raise RuntimeError(f"internal feature layout mismatch: {base.shape} != {(BASE_FEATURE_DIM,)}")
    distances = aggregate_source_distances(base, source_centers, source_scale)
    vector = np.concatenate([base, distances]).astype(np.float32)
    return DomainStatistics(
        vector=vector,
        metadata={
            "num_images": num_images,
            "num_pixels": int(total_pixels),
            "num_peaks": int(total_peaks),
            "has_grayscale": bool(gray_available),
            "num_source_centers": int(_source_center_matrix(source_centers).shape[0]),
            "statistics_config": statistics_config.to_dict(),
            "streaming_memory_contract": {
                "mode": "single_image_plus_bounded_deterministic_priority_samples",
                "quantile_sample_limit_per_family": sample_limit,
                "quantile_estimator": statistics_config.quantile_estimator,
                "stream_chunk_size": chunk_size,
                "auxiliary_memory_independent_of_domain_pixel_count": True,
            },
            "quantile_sample_counts": {
                "probability": probability_samples.sample_size,
                "peak": peak_samples.sample_size,
                "grayscale": gray_samples.sample_size,
                "gradient": gradient_samples.sample_size,
                "laplacian_deviation": laplacian_samples.sample_size,
            },
            "quantiles_exact": {
                "probability": probability_samples.exact,
                "peak": peak_samples.exact,
                "grayscale": gray_samples.exact,
                "gradient": gradient_samples.exact,
                "laplacian_deviation": laplacian_samples.exact,
            },
        },
        statistics_config=statistics_config,
    )


def extract_domain_statistics(
    probabilities: Any,
    grayscale_images: Any | None = None,
    *,
    source_centers: Any = None,
    source_scale: Sequence[float] | np.ndarray | None = None,
    source_reference: SourceReference | None = None,
    statistics_config: StatisticsConfig | None = None,
    peak_kernel_size: int = 3,
    peak_min_score: float = 0.05,
) -> DomainStatistics:
    """Backward-friendly alias retaining an explicit no-mask signature."""

    return extract_unlabeled_statistics(
        probabilities,
        grayscale_images,
        source_centers=source_centers,
        source_scale=source_scale,
        source_reference=source_reference,
        statistics_config=statistics_config,
        peak_kernel_size=peak_kernel_size,
        peak_min_score=peak_min_score,
    )


def load_probability_and_grayscale(
    probability_path: str | Path,
    grayscale_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an exported probability map and optional original grayscale image.

    Any ``mask`` member present in an exported NPZ is intentionally ignored.
    """

    probability_path = Path(probability_path)
    with np.load(probability_path, allow_pickle=False) as payload:
        if "prob" not in payload:
            raise KeyError(f"{probability_path} does not contain a 'prob' array")
        probability = np.asarray(payload["prob"]).squeeze()
        grayscale = None
        for key in ("gray", "grayscale", "original_gray"):
            if key in payload:
                grayscale = np.asarray(payload[key]).squeeze()
                break
    if grayscale_path is not None:
        from PIL import Image

        with Image.open(grayscale_path) as image:
            grayscale = np.asarray(image.convert("L"))
    if probability.ndim != 2:
        raise ValueError(f"probability map in {probability_path} must be 2D")
    if grayscale is not None and grayscale.ndim != 2:
        raise ValueError("grayscale image must be 2D")
    return probability, grayscale


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_reference(
    path: str | Path,
    *,
    statistics_config: StatisticsConfig,
) -> SourceReference:
    """Load and audit a fold-specific source reference artifact."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "domains",
            "centers",
            "scale",
            "statistics_config_json",
            "source_contract_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"source reference NPZ is missing: {sorted(missing)}")
        domains = tuple(str(value) for value in np.asarray(payload["domains"]).reshape(-1))
        centers = np.asarray(payload["centers"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64).reshape(-1)
        config_text = str(np.asarray(payload["statistics_config_json"]).item())
        contract_text = str(np.asarray(payload["source_contract_json"]).item())
    artifact_config = StatisticsConfig.from_dict(json.loads(config_text))
    if artifact_config != statistics_config:
        raise ValueError("source reference statistics_config does not match episode config")
    if centers.ndim != 2:
        raise ValueError("source reference centers must be a 2D array")
    if centers.shape[1] == FEATURE_DIM:
        centers = centers[:, :BASE_FEATURE_DIM]
    if scale.size == FEATURE_DIM:
        scale = scale[:BASE_FEATURE_DIM]
    if centers.shape[1] != BASE_FEATURE_DIM or scale.size != BASE_FEATURE_DIM:
        raise ValueError(
            f"source reference feature width must be {BASE_FEATURE_DIM} or {FEATURE_DIM}"
        )
    contract_payload = json.loads(contract_text)
    if not isinstance(contract_payload, Mapping):
        raise ValueError("source_contract_json must contain a JSON object")
    contract = SourceContract.from_dict(contract_payload)
    return SourceReference(
        domains=domains,
        sha256=_sha256_file(path),
        centers=tuple(tuple(float(value) for value in row) for row in centers),
        scale=tuple(float(value) for value in scale),
        contract=contract,
    )
