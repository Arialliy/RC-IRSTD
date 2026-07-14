# RC-IRSTD 完整代码清单

> 本文件将发布 ZIP 中的核心工程源码按文件完整展开，便于离线审查、复制和代码评审。
> 二进制权重、数据集、缓存、输出和自动生成的 package metadata 不包含在本文档中。

## 1. 汇总

- 源码/配置/启动/测试文件：**110**
- Python 包文件：**72**
- 测试文件：**17**
- Shell 启动器：**14**
- YAML 配置：**4**

## 2. 文件目录

```text
.gitignore
configs/lodo_example.yaml
configs/lodo_fold.example.yaml
configs/lodo_smoke.yaml
configs/lodo_temporal_example.yaml
pyproject.toml
rc_irstd/__init__.py
rc_irstd/calibration/__init__.py
rc_irstd/calibration/crc.py
rc_irstd/calibration/samples.py
rc_irstd/candidates/__init__.py
rc_irstd/candidates/peaks.py
rc_irstd/data/__init__.py
rc_irstd/data/dataset.py
rc_irstd/data/sampler.py
rc_irstd/data/score_records.py
rc_irstd/data/transforms.py
rc_irstd/data/windows.py
rc_irstd/deployment/__init__.py
rc_irstd/deployment/monitor.py
rc_irstd/deployment/session.py
rc_irstd/engine/__init__.py
rc_irstd/engine/worker_seed.py
rc_irstd/episodes/__init__.py
rc_irstd/episodes/builder.py
rc_irstd/episodes/dataset.py
rc_irstd/episodes/splits.py
rc_irstd/evaluation/__init__.py
rc_irstd/evaluation/budget.py
rc_irstd/evaluation/component_curves.py
rc_irstd/evaluation/curves.py
rc_irstd/evaluation/detector_selection.py
rc_irstd/evaluation/irstd_metrics.py
rc_irstd/evaluation/operating_point.py
rc_irstd/evaluation/risk_curve_metrics.py
rc_irstd/evaluation/segmentation.py
rc_irstd/features/__init__.py
rc_irstd/features/image_stats.py
rc_irstd/features/window_stats.py
rc_irstd/losses/__init__.py
rc_irstd/losses/cvar.py
rc_irstd/losses/quantile.py
rc_irstd/losses/risk_aware.py
rc_irstd/losses/sls.py
rc_irstd/models/__init__.py
rc_irstd/models/detector_adapter.py
rc_irstd/models/mshnet.py
rc_irstd/models/risk_curve.py
rc_irstd/models/risk_io.py
rc_irstd/models/tiny_detector.py
rc_irstd/pipelines/__init__.py
rc_irstd/pipelines/aggregate_results.py
rc_irstd/pipelines/apply_operating_point.py
rc_irstd/pipelines/build_episodes.py
rc_irstd/pipelines/build_supplement.py
rc_irstd/pipelines/calibrate_and_evaluate.py
rc_irstd/pipelines/evaluate_baselines.py
rc_irstd/pipelines/evaluate_scores.py
rc_irstd/pipelines/evaluate_zero_label.py
rc_irstd/pipelines/export_scores.py
rc_irstd/pipelines/make_synthetic_data.py
rc_irstd/pipelines/predict_unlabeled.py
rc_irstd/pipelines/run_deployment.py
rc_irstd/pipelines/run_lodo.py
rc_irstd/pipelines/smoke.py
rc_irstd/pipelines/train_curve.py
rc_irstd/pipelines/train_detector.py
rc_irstd/provenance/__init__.py
rc_irstd/provenance/fingerprint.py
rc_irstd/provenance/manifest.py
rc_irstd/utils/__init__.py
rc_irstd/utils/arguments.py
rc_irstd/utils/checkpoint.py
rc_irstd/utils/config.py
rc_irstd/utils/device.py
rc_irstd/utils/io.py
rc_irstd/utils/logging.py
rc_irstd/utils/seed.py
requirements.txt
scripts/aggregate_paper_results.sh
scripts/build_anonymous_supplement.sh
scripts/deploy_target.sh
scripts/full_pipeline_start.sh
scripts/launch_lodo_fold.sh
scripts/mshnet_integration_test.sh
scripts/run_lodo.sh
scripts/setup.sh
scripts/smoke_pipeline.sh
scripts/smoke_test.sh
scripts/start_training.sh
scripts/train_detector.sh
scripts/train_detector_mshnet.sh
scripts/validate_release.sh
tests/conftest.py
tests/test_crc.py
tests/test_data_protocols.py
tests/test_dataset_paths.py
tests/test_deployment_and_calibration_units.py
tests/test_episode_metrics.py
tests/test_feature_config.py
tests/test_lodo_protocol.py
tests/test_mshnet_integration.py
tests/test_new_evaluation_and_provenance.py
tests/test_operating_point.py
tests/test_peaks.py
tests/test_risk_aware_loss.py
tests/test_risk_curve.py
tests/test_sampler.py
tests/test_splits.py
tests/test_windows.py
```

## 3. 完整源码

### 3.1 `.gitignore`

- SHA-256：`ef21e63ed0c7623d02f24f5d4d2f6f75565bef5cfbbd6b3e98c5cdbb7db74831`
- 行数：`16`

````gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
outputs/
artifacts/
repro_runs/
*.pth
*.pt
*.pkl
*.npz
*.npy
*.csv
.DS_Store

*.egg-info/
````

### 3.2 `configs/lodo_example.yaml`

- SHA-256：`d4ae81ab41568b32ffc52a38f96894d0d1d6b7926283935f9673db55e503e7e7`
- 行数：`131`

````yaml
# RC-IRSTD 完整 Nested LODO 配置（静态图像主协议）
# 所有相对路径以本 YAML 所在目录为基准。
python: python
working_directory: ..
output_root: ../outputs/rc_irstd_nested_lodo

datasets:
  NUAA-SIRST:
    path: /data/NUAA-SIRST
    train_split: train
    eval_split: test
    dataset_type: iid_images
  NUDT-SIRST:
    path: /data/NUDT-SIRST
    train_split: train
    eval_split: test
    dataset_type: iid_images
  IRSTD-1K:
    path: /data/IRSTD-1K
    train_split: train
    eval_split: test
    dataset_type: iid_images
  SIRST-UAVB:
    path: /data/SIRST-UAVB
    train_split: train
    eval_split: test
    dataset_type: iid_images
  RealScene-ISTD:
    path: /data/RealScene-ISTD
    train_split: train
    eval_split: test
    dataset_type: iid_images
  NUDT-SIRST-Sea:
    path: /data/NUDT-SIRST-Sea
    train_split: train
    eval_split: test
    dataset_type: iid_images

outer_targets:
  - NUAA-SIRST
  - NUDT-SIRST
  - IRSTD-1K
  - SIRST-UAVB
  - RealScene-ISTD
  - NUDT-SIRST-Sea

detector:
  name: mshnet                 # 内置完整 MSHNet；无需外部仓库
  base_loss: auto              # mshnet -> 内置稳定 SLS
  resize: [256, 256]
  normalization: imagenet      # 16 位原图也会先按位深归一化
  dataset_type: iid_images
  per_domain_batch: 2
  epochs: 400
  warm_epoch: 5
  optimizer: adagrad
  lr: 0.05
  weight_decay: 0.0
  lambda_tail: 0.10
  lambda_miss: 0.10
  auxiliary_weight: 1.0
  tail_quantile: 0.95
  miss_quantile: 0.80
  peak_kernel: 5
  exclusion_radius: 2
  worst_gamma: 10.0
  # detector checkpoint 按以下源域预算工作点选择，而非只按 IoU
  pixel_budget: 1.0e-5
  peak_budget: 5.0
  grad_clip: 5.0
  val_every: 1
  save_every: 20
  num_workers: 4
  device: cuda
  amp: true
  deterministic: true
  seed: 42
  # 分数导出/最终评测协议
  inference_mode: native_pad   # resize | native_pad | tiled
  restore_original: true
  stride_multiple: 32
  tile_size: [512, 512]
  tile_overlap: 64

episodes:
  # iid_images: 固定随机排列后的 support/query；不宣称时间因果。
  # temporal: 真实过去 context -> 未来 horizon。
  protocol: auto
  seed: 42
  context_size: 32
  horizon: 16
  train_stride: 16
  # 正式目标域评测必须 >= context_size + horizon，防止图像重叠。
  eval_stride: 48
  peak_min_distance: 2
  peak_min_score: 0.0
  peak_border: 0
  peak_tolerance: 2.0
  max_candidates: 0

curve:
  quantile: 0.90
  hidden_dim: 256
  dropout: 0.10
  lambda_peak: 1.0
  lambda_crossing: 0.25
  crossing_temperature: 0.25
  focus_base_weight: 1.0
  focus_weight: 4.0
  focus_log_scale: 1.0
  empty_action_weight: 0.10
  batch_size: 64
  epochs: 300
  lr: 0.001
  weight_decay: 0.0001
  val_fraction: 0.20
  patience: 40
  num_workers: 0
  device: cuda
  seed: 42

budgets:
  pixel: 1.0e-6
  peak_per_mp: 1.0

calibration:
  alpha: 0.10
  unit: image                  # image = 真正按标注图像数计数
  sizes: [10, 20, 50]
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  offset_step: 1
````

### 3.3 `configs/lodo_fold.example.yaml`

- SHA-256：`518ccc972323cac05d35d072a48a7a436502f0efa745f4705a655c15a0df79a0`
- 行数：`112`

````yaml
# RC-IRSTD nested leave-one-domain-out configuration.
# Paths may be absolute or relative to this YAML file.
# For MSHNet, working_directory must be the repository root containing
# model/MSHNet.py and model/loss.py.
python: python
working_directory: /absolute/path/to/MSHNet
output_root: ../outputs/rc_irstd_nested_lodo

datasets:
  NUAA-SIRST:
    path: /data/NUAA-SIRST
    train_split: train
    eval_split: test
  NUDT-SIRST:
    path: /data/NUDT-SIRST
    train_split: train
    eval_split: test
  IRSTD-1K:
    path: /data/IRSTD-1K
    train_split: train
    eval_split: test
  SIRST-UAVB:
    path: /data/SIRST-UAVB
    train_split: train
    eval_split: test
  RealScene-ISTD:
    path: /data/RealScene-ISTD
    train_split: train
    eval_split: test
  NUDT-SIRST-Sea:
    path: /data/NUDT-SIRST-Sea
    train_split: train
    eval_split: test

outer_targets:
  - NUAA-SIRST
  - NUDT-SIRST
  - IRSTD-1K
  - SIRST-UAVB
  - RealScene-ISTD
  - NUDT-SIRST-Sea

detector:
  name: mshnet
  base_loss: auto
  train_split: train
  val_split: test
  resize: [256, 256]
  per_domain_batch: 2
  epochs: 400
  warm_epoch: 5
  optimizer: adagrad
  lr: 0.05
  weight_decay: 0.0
  lambda_tail: 0.10
  lambda_miss: 0.10
  auxiliary_weight: 1.0
  tail_quantile: 0.95
  miss_quantile: 0.80
  peak_kernel: 5
  exclusion_radius: 2
  worst_gamma: 10.0
  grad_clip: 5.0
  val_every: 1
  save_every: 20
  num_workers: 4
  device: cuda
  amp: true
  deterministic: true
  seed: 42

episodes:
  # Pseudo-target training episodes may overlap; target evaluation and CRC
  # episodes must use disjoint image blocks.
  context_size: 32
  horizon: 16
  train_stride: 16
  eval_stride: 48
  peak_min_distance: 2
  peak_min_score: 0.0
  peak_border: 0
  peak_tolerance: 2.0
  # 0 means no candidate truncation. Keep this for formal risk-control runs.
  max_candidates: 0

curve:
  quantile: 0.90
  hidden_dim: 256
  dropout: 0.10
  lambda_peak: 1.0
  batch_size: 64
  epochs: 300
  lr: 0.001
  weight_decay: 0.0001
  val_fraction: 0.20
  patience: 40
  num_workers: 0
  device: cuda
  seed: 42

budgets:
  pixel: 1.0e-6
  peak_per_mp: 1.0

calibration:
  alpha: 0.10
  # m=5 is impossible under the standard correction at alpha=.10 because
  # 1/(m+1)=.1667. Treat m=10 as a severe small-sample diagnostic; use 20/50
  # for the principal formally controlled results.
  sizes: [10, 20, 50]
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  offset_step: 1
````

### 3.4 `configs/lodo_smoke.yaml`

- SHA-256：`e4f89b62c79dfa09f4411a4b06c18c2509a97f327ac5ff12780ab0c35e7f062d`
- 行数：`64`

````yaml
# Requires synthetic data created with:
# python -m rc_irstd.pipelines.make_synthetic_data \
#   --output-root ../outputs/smoke/data \
#   --domains DomainA DomainB DomainC \
#   --height 16 --width 16 --sequences 4 --frames-per-sequence 6
python: python
working_directory: ..
output_root: ../outputs/lodo_smoke

datasets:
  DomainA: {path: ../outputs/smoke/data/DomainA, train_split: train, eval_split: test}
  DomainB: {path: ../outputs/smoke/data/DomainB, train_split: train, eval_split: test}
  DomainC: {path: ../outputs/smoke/data/DomainC, train_split: train, eval_split: test}
outer_targets: [DomainC]

detector:
  name: tiny
  base_loss: bce_dice
  resize: [16, 16]
  per_domain_batch: 12
  epochs: 1
  warm_epoch: 0
  optimizer: adamw
  lr: 0.001
  lambda_tail: 0.05
  lambda_miss: 0.05
  auxiliary_weight: 1.0
  num_workers: 0
  device: cpu
  amp: false
  deterministic: true
  seed: 42

episodes:
  # Training episodes may overlap to increase pseudo-target sample count.
  # Evaluation/CRC episodes must be image-disjoint.
  context_size: 2
  horizon: 1
  train_stride: 1
  eval_stride: 3
  peak_min_distance: 2
  peak_min_score: 0.0
  max_candidates: 1024

curve:
  quantile: 0.90
  hidden_dim: 32
  dropout: 0.0
  batch_size: 4
  epochs: 3
  lr: 0.001
  patience: 3
  device: cpu
  seed: 42

budgets:
  pixel: 1.0
  peak_per_mp: 1000000000.0

calibration:
  alpha: 0.50
  sizes: [2]
  seeds: [0]
  offset_step: 1
````

### 3.5 `configs/lodo_temporal_example.yaml`

- SHA-256：`5268ecccb263e3139c17223bc779d7cc194bb525fd902d85fae5f557c3fddc6a`
- 行数：`95`

````yaml
# 视频/连续帧数据示例。每个数据集必须通过文件名或子目录提供 sequence_id。
python: python
working_directory: ..
output_root: ../outputs/rc_irstd_temporal_lodo

datasets:
  Sensor-A:
    path: /data/Sensor-A
    train_split: train
    eval_split: test
    dataset_type: temporal
  Sensor-B:
    path: /data/Sensor-B
    train_split: train
    eval_split: test
    dataset_type: temporal
  Sensor-C:
    path: /data/Sensor-C
    train_split: train
    eval_split: test
    dataset_type: temporal

outer_targets: [Sensor-A, Sensor-B, Sensor-C]

detector:
  name: mshnet
  base_loss: auto
  resize: [256, 256]
  normalization: imagenet
  dataset_type: temporal
  per_domain_batch: 2
  epochs: 400
  warm_epoch: 5
  optimizer: adagrad
  lr: 0.05
  lambda_tail: 0.10
  lambda_miss: 0.10
  tail_quantile: 0.95
  miss_quantile: 0.80
  peak_kernel: 5
  exclusion_radius: 2
  worst_gamma: 10.0
  auxiliary_weight: 1.0
  pixel_budget: 1.0e-5
  peak_budget: 5.0
  num_workers: 4
  device: cuda
  amp: true
  deterministic: true
  seed: 42
  inference_mode: native_pad
  stride_multiple: 32

episodes:
  protocol: temporal
  context_size: 32
  horizon: 16
  train_stride: 16
  eval_stride: 48
  seed: 42
  peak_min_distance: 2
  peak_tolerance: 2.0
  max_candidates: 0

curve:
  quantile: 0.90
  hidden_dim: 256
  dropout: 0.10
  lambda_peak: 1.0
  lambda_crossing: 0.25
  crossing_temperature: 0.25
  focus_base_weight: 1.0
  focus_weight: 4.0
  focus_log_scale: 1.0
  empty_action_weight: 0.10
  batch_size: 64
  epochs: 300
  lr: 0.001
  weight_decay: 0.0001
  val_fraction: 0.20
  patience: 40
  device: cuda
  seed: 42

budgets:
  pixel: 1.0e-6
  peak_per_mp: 1.0

calibration:
  # temporal image units仍按 sequence 阻断划分；也可改 episode 做 block CRC。
  alpha: 0.10
  unit: image
  sizes: [10, 20, 50]
  seeds: [0, 1, 2, 3, 4]
  offset_step: 1
````

### 3.6 `pyproject.toml`

- SHA-256：`bfe328c6af710520ae8a82ef137517975f01863e9f372b94cad9bf74a54cd4d5`
- 行数：`51`

````toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rc-irstd"
version = "0.3.0"
description = "Complete risk-aware MSHNet, monotone risk-curve adaptation and calibrated deployment for IRSTD"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "scikit-image>=0.20",
  "pandas>=2.0",
  "Pillow>=9.0",
  "PyYAML>=6.0",
  "torch>=2.0",
  "tqdm>=4.65",
  "tabulate>=0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
rc-irstd-train-detector = "rc_irstd.pipelines.train_detector:main"
rc-irstd-export-scores = "rc_irstd.pipelines.export_scores:main"
rc-irstd-evaluate-scores = "rc_irstd.pipelines.evaluate_scores:main"
rc-irstd-evaluate-baselines = "rc_irstd.pipelines.evaluate_baselines:main"
rc-irstd-build-episodes = "rc_irstd.pipelines.build_episodes:main"
rc-irstd-train-curve = "rc_irstd.pipelines.train_curve:main"
rc-irstd-eval-zero = "rc_irstd.pipelines.evaluate_zero_label:main"
rc-irstd-predict-unlabeled = "rc_irstd.pipelines.predict_unlabeled:main"
rc-irstd-calibrate = "rc_irstd.pipelines.calibrate_and_evaluate:main"
rc-irstd-run-lodo = "rc_irstd.pipelines.run_lodo:main"
rc-irstd-aggregate = "rc_irstd.pipelines.aggregate_results:main"
rc-irstd-make-synthetic = "rc_irstd.pipelines.make_synthetic_data:main"
rc-irstd-smoke = "rc_irstd.pipelines.smoke:main"
rc-irstd-build-supplement = "rc_irstd.pipelines.build_supplement:main"
rc-irstd-apply-threshold = "rc_irstd.pipelines.apply_operating_point:main"
rc-irstd-deploy = "rc_irstd.pipelines.run_deployment:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["rc_irstd*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"
````

### 3.7 `rc_irstd/__init__.py`

- SHA-256：`c11770d882e74d3be2f6a9ff64ba6911b3ca837962d56c281c6c65dbad15881d`
- 行数：`3`

````python
"""RC-IRSTD research implementation."""

__version__ = "0.3.0"
````

### 3.8 `rc_irstd/calibration/__init__.py`

- SHA-256：`bf9862ef9f65222f89b3499a36321656afc947bedbabc75d2a3de0ee692fc95f`
- 行数：`19`

````python
from rc_irstd.calibration.crc import (
    CRCResult,
    adaptive_offset_loss_matrix,
    joint_budget_violation_losses,
    minimum_calibration_size,
    raw_global_threshold_loss_matrix,
    select_crc_parameter,
    selected_indices_from_offsets,
)

__all__ = [
    "CRCResult",
    "adaptive_offset_loss_matrix",
    "joint_budget_violation_losses",
    "minimum_calibration_size",
    "raw_global_threshold_loss_matrix",
    "select_crc_parameter",
    "selected_indices_from_offsets",
]
````

### 3.9 `rc_irstd/calibration/crc.py`

- SHA-256：`ac7251d206a9a8abb9b2fdc432f5e19b1e2a9794bff351a8ebfc1dc2649ccc26`
- 行数：`199`

````python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CRCResult:
    selected_parameter: int
    selected_position: int
    empirical_risk: float
    corrected_risk: float
    alpha: float
    calibration_size: int
    feasible: bool
    minimum_possible_corrected_risk: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimum_calibration_size(alpha: float) -> int:
    """Smallest ``m`` for which ``1/(m+1) <= alpha`` can hold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return int(np.ceil(1.0 / alpha - 1.0))


def corrected_empirical_risk(empirical_risk: np.ndarray, calibration_size: int) -> np.ndarray:
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    empirical = np.asarray(empirical_risk, dtype=np.float64)
    return (
        calibration_size / (calibration_size + 1.0) * empirical
        + 1.0 / (calibration_size + 1.0)
    )


def _validate_loss_matrix(losses: np.ndarray) -> np.ndarray:
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("losses must have shape [calibration_samples, parameters]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("losses must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("losses contain NaN or infinity")
    if np.any(matrix < -1e-12) or np.any(matrix > 1.0 + 1e-12):
        raise ValueError("CRC losses must lie in [0, 1]")
    return np.clip(matrix, 0.0, 1.0)


def select_crc_parameter(
    losses: np.ndarray,
    parameters: Iterable[int],
    alpha: float,
    require_monotone: bool = True,
) -> CRCResult:
    """Select the least conservative parameter satisfying standard CRC.

    The columns of ``losses`` must follow increasing conservatism. For the
    adaptive-threshold implementation, columns correspond to non-negative
    threshold-index offsets. For the raw global baseline, columns correspond to
    ascending threshold indices.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    matrix = _validate_loss_matrix(losses)
    parameter_values = np.asarray(list(parameters), dtype=np.int64)
    if len(parameter_values) != matrix.shape[1]:
        raise ValueError("Number of parameters must equal the loss-matrix width")
    if np.any(np.diff(parameter_values) < 0):
        raise ValueError("parameters must be ascending")
    if require_monotone and np.any(np.diff(matrix, axis=1) > 1e-10):
        bad = int(np.sum(np.diff(matrix, axis=1) > 1e-10))
        raise ValueError(
            f"Loss family is not nested/monotone; found {bad} increasing entries"
        )

    calibration_size = matrix.shape[0]
    empirical = matrix.mean(axis=0)
    corrected = corrected_empirical_risk(empirical, calibration_size)
    feasible_positions = np.flatnonzero(corrected <= alpha + 1e-12)
    minimum_possible = 1.0 / (calibration_size + 1.0)

    if len(feasible_positions):
        position = int(feasible_positions[0])
        return CRCResult(
            selected_parameter=int(parameter_values[position]),
            selected_position=position,
            empirical_risk=float(empirical[position]),
            corrected_risk=float(corrected[position]),
            alpha=float(alpha),
            calibration_size=int(calibration_size),
            feasible=True,
            minimum_possible_corrected_risk=float(minimum_possible),
            message="A CRC-feasible parameter was found.",
        )

    position = len(parameter_values) - 1
    if minimum_possible > alpha + 1e-12:
        message = (
            f"No formal solution is possible with m={calibration_size} and "
            f"alpha={alpha:g}: even zero empirical loss gives "
            f"1/(m+1)={minimum_possible:.6g}."
        )
    else:
        message = (
            "No parameter in the supplied nested family satisfies the corrected "
            "risk bound; the most conservative parameter is returned as a "
            "fallback and must not be labelled certified."
        )
    return CRCResult(
        selected_parameter=int(parameter_values[position]),
        selected_position=position,
        empirical_risk=float(empirical[position]),
        corrected_risk=float(corrected[position]),
        alpha=float(alpha),
        calibration_size=int(calibration_size),
        feasible=False,
        minimum_possible_corrected_risk=float(minimum_possible),
        message=message,
    )


def selected_indices_from_offsets(
    base_indices: np.ndarray,
    offsets: Iterable[int],
    num_thresholds: int,
) -> np.ndarray:
    base = np.asarray(base_indices, dtype=np.int64)
    offset_values = np.asarray(list(offsets), dtype=np.int64)
    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    if np.any(base < 0) or np.any(base >= num_thresholds):
        raise ValueError("base_indices are outside the threshold grid")
    if np.any(offset_values < 0):
        raise ValueError("offsets must be non-negative")
    return np.minimum(base[:, None] + offset_values[None, :], num_thresholds - 1)


def joint_budget_violation_losses(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    selected_indices: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> np.ndarray:
    pixel = np.asarray(pixel_risk_curves, dtype=np.float64)
    peak = np.asarray(peak_risk_curves, dtype=np.float64)
    indices = np.asarray(selected_indices, dtype=np.int64)
    if pixel.shape != peak.shape:
        raise ValueError("Pixel and peak risk curves must have equal shapes")
    if pixel.ndim != 2:
        raise ValueError("Risk curves must have shape [samples, thresholds]")
    if indices.ndim == 1:
        indices = indices[:, None]
    if indices.shape[0] != pixel.shape[0]:
        raise ValueError("selected_indices must have one row per sample")
    rows = np.arange(pixel.shape[0])[:, None]
    selected_pixel = pixel[rows, indices]
    selected_peak = peak[rows, indices]
    return ((selected_pixel > pixel_budget) | (selected_peak > peak_budget)).astype(np.float64)


def adaptive_offset_loss_matrix(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    base_indices: np.ndarray,
    offsets: Iterable[int],
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_risk_curves)
    indices = selected_indices_from_offsets(base_indices, offsets, pixel.shape[1])
    losses = joint_budget_violation_losses(
        pixel_risk_curves,
        peak_risk_curves,
        indices,
        pixel_budget,
        peak_budget,
    )
    return losses, indices


def raw_global_threshold_loss_matrix(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> np.ndarray:
    pixel = np.asarray(pixel_risk_curves, dtype=np.float64)
    peak = np.asarray(peak_risk_curves, dtype=np.float64)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Risk curves must share shape [samples, thresholds]")
    return ((pixel > pixel_budget) | (peak > peak_budget)).astype(np.float64)
````

### 3.10 `rc_irstd/calibration/samples.py`

- SHA-256：`91ee86ff1fd54abd16807e308c2ce0d41026e1b197d660f9d32b49678bb3f0ad`
- 行数：`181`

````python
from __future__ import annotations

"""Explicit calibration units for block-level and true image-shot CRC."""

from dataclasses import dataclass
import json

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.episodes.splits import split_iid_images


@dataclass(frozen=True)
class CalibrationSamples:
    pixel_risk: np.ndarray
    peak_risk: np.ndarray
    pd: np.ndarray
    domains: np.ndarray
    sequences: np.ndarray
    sample_ids: np.ndarray
    base_indices: np.ndarray
    base_rejected: np.ndarray
    parent_episode: np.ndarray
    protocols: np.ndarray
    label_count_per_sample: np.ndarray
    unit: str

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)


def _parse_ids(values: np.ndarray) -> list[list[str]]:
    result: list[list[str]] = []
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list):
            raise ValueError("future_ids must encode JSON lists")
        result.append([str(item) for item in parsed])
    return result


def episode_calibration_samples(
    arrays: EpisodeArrays,
    base_indices: np.ndarray,
    base_rejected: np.ndarray,
) -> CalibrationSamples:
    protocols = (
        arrays.protocols
        if arrays.protocols is not None
        else np.asarray(["temporal"] * len(arrays.features))
    )
    return CalibrationSamples(
        pixel_risk=arrays.pixel_risk,
        peak_risk=arrays.peak_risk,
        pd=arrays.pd,
        domains=arrays.domains,
        sequences=arrays.sequences,
        sample_ids=np.asarray([f"episode_{index:08d}" for index in range(len(arrays.features))]),
        base_indices=np.asarray(base_indices, dtype=np.int64),
        base_rejected=np.asarray(base_rejected, dtype=bool),
        parent_episode=np.arange(len(arrays.features), dtype=np.int64),
        protocols=np.asarray(protocols).astype(str),
        label_count_per_sample=np.asarray(
            [len(ids) for ids in _parse_ids(arrays.future_ids)], dtype=np.int64
        ),
        unit="episode_block",
    )


def image_calibration_samples(
    arrays: EpisodeArrays,
    base_indices: np.ndarray,
    base_rejected: np.ndarray,
) -> CalibrationSamples:
    required = (
        arrays.future_pixel_risk,
        arrays.future_peak_risk,
        arrays.future_pd,
        arrays.future_gt_count,
    )
    if any(value is None for value in required):
        raise ValueError(
            "Episode file does not contain per-future-image curves. Rebuild it "
            "with the current rc-irstd-build-episodes command."
        )
    pixel = np.asarray(arrays.future_pixel_risk)
    peak = np.asarray(arrays.future_peak_risk)
    pd = np.asarray(arrays.future_pd).copy()
    gt_count = np.asarray(arrays.future_gt_count)
    if not (pixel.shape == peak.shape == pd.shape):
        raise ValueError("Per-image risk arrays have inconsistent shapes")
    if gt_count.shape != pixel.shape[:2]:
        raise ValueError("future_gt_count must have shape [episodes, horizon]")
    episodes, horizon, thresholds = pixel.shape
    ids_nested = _parse_ids(arrays.future_ids)
    if any(len(ids) != horizon for ids in ids_nested):
        raise ValueError("future_ids length does not match stored horizon")
    image_ids = np.asarray([item for row in ids_nested for item in row], dtype=np.str_)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(
            "Image-shot CRC requires each labelled image once. Rebuild evaluation "
            "episodes with non-overlapping windows (stride >= context+horizon)."
        )
    parent = np.repeat(np.arange(episodes), horizon)
    domains = np.repeat(arrays.domains, horizon)
    sequences = np.repeat(arrays.sequences, horizon)
    protocols_source = (
        arrays.protocols
        if arrays.protocols is not None
        else np.asarray(["temporal"] * episodes)
    )
    protocols = np.repeat(protocols_source, horizon)
    flattened_pd = pd.reshape(-1, thresholds)
    flattened_gt = gt_count.reshape(-1)
    # Empty-target images provide false-alarm evidence but no target-detection
    # denominator. Mark Pd as NaN so they do not depress target-bearing Pd.
    flattened_pd[flattened_gt == 0] = np.nan
    return CalibrationSamples(
        pixel_risk=pixel.reshape(-1, thresholds),
        peak_risk=peak.reshape(-1, thresholds),
        pd=flattened_pd,
        domains=domains,
        sequences=sequences,
        sample_ids=image_ids,
        base_indices=np.repeat(np.asarray(base_indices, dtype=np.int64), horizon),
        base_rejected=np.repeat(np.asarray(base_rejected, dtype=bool), horizon),
        parent_episode=parent,
        protocols=np.asarray(protocols).astype(str),
        label_count_per_sample=np.ones(episodes * horizon, dtype=np.int64),
        unit="image",
    )


def split_calibration_samples(
    samples: CalibrationSamples,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split exact labelled units while retaining an independent test partition."""
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    if samples.unit == "image" and np.all(samples.protocols == "iid"):
        calibration, test = split_iid_images(
            samples.sample_ids, calibration_size, seed
        )
        return calibration, test, {
            "strategy": "iid_unique_images",
            "independent_groups": samples.num_samples,
        }

    # Temporal images and block episodes are sequence-blocked. calibration_size
    # still counts the selected labelled samples, while unused samples from a
    # selected sequence are discarded rather than leaked into test.
    groups = np.asarray(
        [f"{d}::{s}" for d, s in zip(samples.domains, samples.sequences, strict=True)]
    )
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Sequence-blocked calibration requires at least two groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    selected_groups: list[str] = []
    available = 0
    for group in unique[:-1]:
        selected_groups.append(str(group))
        available += int(np.sum(groups == group))
        if available >= calibration_size:
            break
    if available < calibration_size:
        raise ValueError("Not enough samples while retaining a disjoint test group")
    pool = np.flatnonzero(np.isin(groups, selected_groups))
    rng.shuffle(pool)
    calibration = np.sort(pool[:calibration_size])
    test = np.flatnonzero(~np.isin(groups, selected_groups))
    return calibration, test, {
        "strategy": "sequence_blocked_exact_samples",
        "independent_groups": len(unique),
        "selected_calibration_groups": selected_groups,
    }
````

### 3.11 `rc_irstd/candidates/__init__.py`

- SHA-256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- 行数：`0`

````python

````

### 3.12 `rc_irstd/candidates/peaks.py`

- SHA-256：`df3fe15b33b02c5b1e9b6f2345d3985341741768a281ee2283d88696316f4059`
- 行数：`182`

````python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FixedPeakSet:
    scores: np.ndarray
    ys: np.ndarray
    xs: np.ndarray
    gt_ids: np.ndarray
    num_gt: int

    def __post_init__(self) -> None:
        lengths = {len(self.scores), len(self.ys), len(self.xs), len(self.gt_ids)}
        if len(lengths) != 1:
            raise ValueError("Peak arrays must have the same length")


def _one_point_per_plateau(mask: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask)
    ys: list[int] = []
    xs: list[int] = []
    for component_id in range(1, count + 1):
        coords = np.argwhere(labels == component_id)
        if coords.size == 0:
            continue
        values = score[coords[:, 0], coords[:, 1]]
        best_value = values.max()
        best = coords[values == best_value]
        # Lexicographic tie-breaking is deterministic across platforms.
        order = np.lexsort((best[:, 1], best[:, 0]))
        y, x = best[order[0]]
        ys.append(int(y))
        xs.append(int(x))
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def extract_fixed_peaks(
    score_map: np.ndarray,
    min_distance: int = 2,
    min_score: float = 0.0,
    border: int = 0,
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a threshold-independent set of deterministic local maxima.

    Once this set is extracted, increasing a score threshold can only remove
    candidates. Consequently, candidate count, false-candidate count and matched
    target count are monotone with respect to the threshold.
    """
    score = np.asarray(score_map, dtype=np.float32).squeeze()
    if score.ndim != 2:
        raise ValueError(f"score_map must be 2-D, got shape {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("score_map contains NaN or infinity")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")
    size = 2 * min_distance + 1
    local_max = score >= ndimage.maximum_filter(score, size=size, mode="nearest")
    candidate_mask = local_max & (score >= float(min_score))
    if border > 0:
        candidate_mask[:border, :] = False
        candidate_mask[-border:, :] = False
        candidate_mask[:, :border] = False
        candidate_mask[:, -border:] = False
    ys, xs = _one_point_per_plateau(candidate_mask, score)
    scores = score[ys, xs] if len(ys) else np.empty((0,), dtype=np.float32)
    order = np.lexsort((xs, ys, -scores))
    scores, ys, xs = scores[order], ys[order], xs[order]
    if max_candidates is not None:
        scores = scores[:max_candidates]
        ys = ys[:max_candidates]
        xs = xs[:max_candidates]
    return scores.astype(np.float32), ys, xs


def assign_peaks_to_gt(
    ys: np.ndarray,
    xs: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: float = 2.0,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(gt_mask).squeeze() > 0
    if mask.ndim != 2:
        raise ValueError("gt_mask must be 2-D")
    gt_labels, num_gt = ndimage.label(mask)
    gt_ids = np.zeros(len(ys), dtype=np.int32)
    if num_gt == 0 or len(ys) == 0:
        return gt_ids, int(num_gt)

    distance, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    height, width = mask.shape
    for index, (y, x) in enumerate(zip(ys, xs, strict=True)):
        y_i, x_i = int(y), int(x)
        if not (0 <= y_i < height and 0 <= x_i < width):
            raise ValueError("Peak coordinate is outside the mask")
        direct = int(gt_labels[y_i, x_i])
        if direct > 0:
            gt_ids[index] = direct
            continue
        if distance[y_i, x_i] <= tolerance:
            near_y = int(nearest[0, y_i, x_i])
            near_x = int(nearest[1, y_i, x_i])
            gt_ids[index] = int(gt_labels[near_y, near_x])
    return gt_ids, int(num_gt)


def keep_one_peak_per_gt(
    gt_ids: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Turn a provisional many-to-one assignment into a fixed one-to-one match.

    Several local maxima can fall inside, or within the tolerance radius of, the
    same target. Counting every such maximum as a true candidate would make the
    false-candidate metric artificially permissive. We therefore retain only the
    highest-scoring candidate for each GT component and mark all duplicate
    candidates as background. The assignment is computed once, before any score
    threshold is applied, so the resulting true/false candidate labels remain a
    nested family as the threshold increases.
    """

    assignments = np.asarray(gt_ids, dtype=np.int32).copy()
    candidate_scores = np.asarray(scores, dtype=np.float32)
    if assignments.shape != candidate_scores.shape:
        raise ValueError("gt_ids and scores must have the same shape")
    for gt_id in np.unique(assignments[assignments > 0]):
        members = np.flatnonzero(assignments == gt_id)
        if len(members) <= 1:
            continue
        # Stable first-index tie-breaking is deterministic because extracted
        # candidates are already sorted by score, then coordinates.
        best = int(members[np.argmax(candidate_scores[members])])
        assignments[members] = 0
        assignments[best] = int(gt_id)
    return assignments


def build_fixed_peak_set(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    min_distance: int = 2,
    min_score: float = 0.0,
    border: int = 0,
    tolerance: float = 2.0,
    max_candidates: int | None = None,
) -> FixedPeakSet:
    scores, ys, xs = extract_fixed_peaks(
        score_map,
        min_distance=min_distance,
        min_score=min_score,
        border=border,
        max_candidates=max_candidates,
    )
    gt_ids, num_gt = assign_peaks_to_gt(ys, xs, gt_mask, tolerance=tolerance)
    gt_ids = keep_one_peak_per_gt(gt_ids, scores)
    return FixedPeakSet(scores=scores, ys=ys, xs=xs, gt_ids=gt_ids, num_gt=num_gt)


def fixed_peak_curves(
    peak_set: FixedPeakSet,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be ascending")
    total = np.zeros(len(thresholds), dtype=np.int64)
    false = np.zeros(len(thresholds), dtype=np.int64)
    matched = np.zeros(len(thresholds), dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        active = peak_set.scores >= threshold
        active_ids = peak_set.gt_ids[active]
        total[index] = int(active.sum())
        false[index] = int((active_ids == 0).sum())
        matched[index] = len(np.unique(active_ids[active_ids > 0]))
    return total, false, matched
````

### 3.13 `rc_irstd/data/__init__.py`

- SHA-256：`65aaaa6f5d31649bc5f4f70fec31dfb3969864461b1acdb34c34cf679c11b526`
- 行数：`11`

````python
from rc_irstd.data.dataset import IRSTDDataset, SampleMeta, collate_samples
from rc_irstd.data.score_records import ScoreRecord, load_score_record, save_score_record

__all__ = [
    "IRSTDDataset",
    "SampleMeta",
    "collate_samples",
    "ScoreRecord",
    "load_score_record",
    "save_score_record",
]
````

### 3.14 `rc_irstd/data/dataset.py`

- SHA-256：`29e243bc3fdf414ec3b903d1a4d0d77d35c0736631e2c6241e39217592636c0c`
- 行数：`239`

````python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from rc_irstd.data.transforms import (
    image_array_to_tensor,
    load_image_preserve_depth,
    resize_image_array,
    target_preserving_resize_mask,
)


@dataclass(frozen=True)
class SampleMeta:
    image_id: str
    dataset_name: str
    original_hw: tuple[int, int]
    sequence_id: str
    frame_index: int
    image_path: str
    mask_path: str | None
    input_hw: tuple[int, int]
    bit_depth: int
    dataset_type: str


def _resolve_file(folder: Path, image_id: str, required: bool = True) -> Path | None:
    """Resolve split entries with or without extensions and nested directories."""
    relative = Path(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Split entry must be a safe relative path, got: {image_id}")
    direct = folder / relative
    if direct.is_file():
        return direct
    parent = folder / relative.parent
    stem = relative.stem
    matches = sorted(parent.glob(f"{stem}.*")) if parent.is_dir() else []
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"No file for '{image_id}' under {folder}")
    return None


def read_split(root: Path, split: str | Path) -> list[str]:
    split_path = Path(split)
    candidates: list[Path] = []
    if split_path.is_file():
        candidates.append(split_path)
    else:
        candidates.extend(
            [
                root / str(split),
                root / f"{split}.txt",
                root / "img_idx" / str(split),
                root / "img_idx" / f"{split}.txt",
            ]
        )
        if (root / "img_idx").is_dir():
            candidates.extend(sorted((root / "img_idx").glob(f"{split}*.txt")))
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            f"Cannot resolve split '{split}' in {root}. "
            "Pass an explicit split file when the dataset uses a custom layout."
        )
    names = [
        line.strip()
        for line in selected.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"Split file {selected} is empty")
    return names


def infer_sequence_and_index(image_id: str, fallback_index: int) -> tuple[str, int]:
    path = Path(image_id)
    stem = path.stem
    parent = path.parent.as_posix()
    parent_sequence = parent if parent not in {"", "."} else "default"
    for separator in ("_", "-"):
        parts = stem.split(separator)
        if len(parts) > 1 and parts[-1].isdigit():
            prefix = separator.join(parts[:-1])
            if parent_sequence != "default":
                prefix = f"{parent_sequence}/{prefix}" if prefix else parent_sequence
            return prefix or "default", int(parts[-1])
    if stem.isdigit():
        return parent_sequence, int(stem)
    return parent_sequence, fallback_index


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)[None])


class IRSTDDataset(Dataset[dict[str, Any]]):
    """BasicIRSTD-style dataset with bit-depth and target-preserving loading."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str | Path = "train",
        resize_hw: tuple[int, int] | None = None,
        augment: bool = False,
        domain_id: int = 0,
        require_mask: bool = True,
        sequence_parser: Callable[[str, int], tuple[str, int]] | None = None,
        normalization: str = "imagenet",
        dataset_type: str = "iid_images",
        include_component_labels: bool = True,
    ) -> None:
        self.root = Path(dataset_dir)
        self.dataset_name = self.root.name
        self.names = read_split(self.root, split)
        self.resize_hw = resize_hw
        self.augment = augment
        self.domain_id = int(domain_id)
        self.require_mask = bool(require_mask)
        self.sequence_parser = sequence_parser or infer_sequence_and_index
        self.normalization = normalization
        if dataset_type not in {"iid_images", "temporal"}:
            raise ValueError("dataset_type must be iid_images or temporal")
        self.dataset_type = dataset_type
        self.include_component_labels = bool(include_component_labels)
        if not (self.root / "images").is_dir():
            raise FileNotFoundError(f"Expected images/ under {self.root}")
        if self.require_mask and not (self.root / "masks").is_dir():
            raise FileNotFoundError(f"Expected masks/ under {self.root}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.names[index]
        image_path = _resolve_file(self.root / "images", entry, required=True)
        assert image_path is not None
        mask_path = _resolve_file(self.root / "masks", entry, required=self.require_mask)
        loaded = load_image_preserve_depth(image_path)
        image_array = loaded.array
        if image_array.ndim == 3 and image_array.shape[-1] == 4:
            image_array = image_array[..., :3]
        mask_array: np.ndarray | None = None
        if mask_path is not None:
            with Image.open(mask_path) as mask_image:
                mask_array = (np.asarray(mask_image) > 0).astype(np.uint8)
                if mask_array.ndim == 3:
                    mask_array = mask_array[..., 0]
        original_hw = tuple(int(value) for value in image_array.shape[:2])

        if self.resize_hw is not None:
            image_array = resize_image_array(image_array, self.resize_hw)
            if mask_array is not None:
                mask_array = target_preserving_resize_mask(mask_array, self.resize_hw)

        if self.augment:
            if np.random.rand() < 0.5:
                image_array = np.flip(image_array, axis=1)
                if mask_array is not None:
                    mask_array = np.flip(mask_array, axis=1)
            if np.random.rand() < 0.5:
                image_array = np.flip(image_array, axis=0)
                if mask_array is not None:
                    mask_array = np.flip(mask_array, axis=0)
            rotations = int(np.random.randint(0, 4))
            image_array = np.rot90(image_array, rotations).copy()
            if mask_array is not None:
                mask_array = np.rot90(mask_array, rotations).copy()

        sequence_id, frame_index = self.sequence_parser(entry, index)
        if self.dataset_type == "iid_images":
            # Keep path-derived grouping metadata for provenance, but callers must
            # not interpret it as temporal ordering under the iid protocol.
            frame_index = index
        component_labels = None
        if mask_array is not None and self.include_component_labels:
            component_labels = ndimage.label(mask_array > 0)[0].astype(np.int64)

        input_hw = tuple(int(value) for value in image_array.shape[:2])
        return {
            "image": image_array_to_tensor(image_array, normalization=self.normalization),
            "mask": mask_to_tensor(mask_array) if mask_array is not None else None,
            "component_labels": (
                torch.from_numpy(component_labels[None])
                if component_labels is not None
                else None
            ),
            "domain_id": self.domain_id,
            "meta": SampleMeta(
                image_id=Path(entry).with_suffix("").as_posix(),
                dataset_name=self.dataset_name,
                original_hw=original_hw,
                sequence_id=sequence_id,
                frame_index=frame_index,
                image_path=str(image_path),
                mask_path=str(mask_path) if mask_path is not None else None,
                input_hw=input_hw,
                bit_depth=loaded.bit_depth,
                dataset_type=self.dataset_type,
            ),
        }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shapes = {tuple(item["image"].shape[-2:]) for item in batch}
    if len(shapes) != 1:
        raise ValueError(
            "Batches require equal tensor shapes. Configure resize_hw, "
            "or use batch_size=1 for original-resolution export."
        )
    masks = [item["mask"] for item in batch]
    if any(mask is None for mask in masks) and not all(mask is None for mask in masks):
        raise ValueError("A batch cannot mix labelled and unlabelled samples")
    mask_batch = None if all(mask is None for mask in masks) else torch.stack(masks)  # type: ignore[arg-type]

    labels = [item.get("component_labels") for item in batch]
    if all(label is None for label in labels):
        label_batch = None
    elif any(label is None for label in labels):
        raise ValueError("A batch cannot mix component-labelled and unlabelled samples")
    else:
        label_batch = torch.stack(labels)  # type: ignore[arg-type]

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": mask_batch,
        "component_labels": label_batch,
        "domain_id": torch.tensor([item["domain_id"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }
````

### 3.15 `rc_irstd/data/sampler.py`

- SHA-256：`594d7a75c61a28fa1760c8e2a8feb78f82b3082793a155218116f2372cb92c3a`
- 行数：`76`

````python
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Create batches with the same number of examples from each domain."""

    def __init__(
        self,
        domain_ids: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        self.domains = sorted(np.unique(self.domain_ids).tolist())
        if not self.domains:
            raise ValueError("domain_ids must not be empty")
        if batch_size % len(self.domains) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by num_domains={len(self.domains)}"
            )
        self.per_domain = batch_size // len(self.domains)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.indices = {
            domain: np.flatnonzero(self.domain_ids == domain)
            for domain in self.domains
        }
        if any(len(values) == 0 for values in self.indices.values()):
            raise ValueError("Every domain must contain at least one sample")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        max_size = max(len(values) for values in self.indices.values())
        if self.drop_last:
            # Sampling is with replacement for smaller domains, so a positive
            # dataset should still yield one balanced batch even when
            # ``per_domain`` exceeds the largest raw domain size.
            return max(1, max_size // self.per_domain)
        return math.ceil(max_size / self.per_domain)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools: dict[int, np.ndarray] = {}
        target_size = len(self) * self.per_domain
        for domain, values in self.indices.items():
            selected = values.copy()
            if self.shuffle:
                rng.shuffle(selected)
            if len(selected) < target_size:
                extra = rng.choice(selected, size=target_size - len(selected), replace=True)
                selected = np.concatenate([selected, extra])
            pools[domain] = selected[:target_size]

        for batch_index in range(len(self)):
            batch: list[int] = []
            start = batch_index * self.per_domain
            stop = start + self.per_domain
            for domain in self.domains:
                batch.extend(pools[domain][start:stop].tolist())
            if self.shuffle:
                rng.shuffle(batch)
            yield batch
````

### 3.16 `rc_irstd/data/score_records.py`

- SHA-256：`94c738e5114710d1e244e6911f2735d1a51801a87e8068ea8ea8aeef1414d919`
- 行数：`84`

````python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreRecord:
    probability: np.ndarray
    mask: np.ndarray | None
    image_stats: np.ndarray
    image_stat_names: tuple[str, ...]
    image_id: str
    dataset_name: str
    sequence_id: str
    frame_index: int
    original_hw: tuple[int, int]
    source_checkpoint: str = ""
    dataset_type: str = "iid_images"
    inference_mode: str = "resize"

    @property
    def total_pixels(self) -> int:
        return int(self.probability.size)


def save_score_record(record: ScoreRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "probability": np.asarray(record.probability, dtype=np.float32),
        "image_stats": np.asarray(record.image_stats, dtype=np.float32),
        "image_stat_names": np.asarray(record.image_stat_names, dtype=np.str_),
        "image_id": np.asarray(record.image_id),
        "dataset_name": np.asarray(record.dataset_name),
        "sequence_id": np.asarray(record.sequence_id),
        "frame_index": np.asarray(record.frame_index, dtype=np.int64),
        "original_hw": np.asarray(record.original_hw, dtype=np.int32),
        "source_checkpoint": np.asarray(record.source_checkpoint),
        "dataset_type": np.asarray(record.dataset_type),
        "inference_mode": np.asarray(record.inference_mode),
        "has_mask": np.asarray(record.mask is not None),
    }
    if record.mask is not None:
        payload["mask"] = np.asarray(record.mask, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_score_record(path: str | Path, require_mask: bool = False) -> ScoreRecord:
    with np.load(path, allow_pickle=False) as payload:
        has_mask = bool(np.asarray(payload.get("has_mask", "mask" in payload)).item())
        mask = np.asarray(payload["mask"], dtype=np.uint8) if has_mask and "mask" in payload else None
        if require_mask and mask is None:
            raise ValueError(f"Score record {path} does not contain a mask")
        probability = np.asarray(payload["probability"], dtype=np.float32).squeeze()
        if probability.ndim != 2:
            raise ValueError(f"Probability in {path} must be 2-D")
        if not np.isfinite(probability).all():
            raise ValueError(f"Probability in {path} contains invalid values")
        return ScoreRecord(
            probability=probability,
            mask=mask.squeeze() if mask is not None else None,
            image_stats=np.asarray(payload["image_stats"], dtype=np.float32),
            image_stat_names=tuple(np.asarray(payload["image_stat_names"]).astype(str).tolist()),
            image_id=_scalar_string(payload["image_id"]),
            dataset_name=_scalar_string(payload["dataset_name"]),
            sequence_id=_scalar_string(payload["sequence_id"]),
            frame_index=int(np.asarray(payload["frame_index"]).item()),
            original_hw=tuple(int(x) for x in np.asarray(payload["original_hw"]).tolist()),
            source_checkpoint=_scalar_string(payload.get("source_checkpoint", np.asarray(""))),
            dataset_type=_scalar_string(
                payload.get("dataset_type", np.asarray("iid_images"))
            ),
            inference_mode=_scalar_string(
                payload.get("inference_mode", np.asarray("resize"))
            ),
        )
````

### 3.17 `rc_irstd/data/transforms.py`

- SHA-256：`a4ebbf9cf1b24f0f8fe81acea7fe744a983320300dad7dfe63c375e288a4aecd`
- 行数：`157`

````python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    array: np.ndarray
    bit_depth: int


def load_image_preserve_depth(path: str | Path) -> LoadedImage:
    """Load an image without silently reducing 16-bit infrared data to 8 bit."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype == np.uint16:
        bit_depth = 16
    elif array.dtype == np.uint8:
        bit_depth = 8
    elif np.issubdtype(array.dtype, np.integer):
        bit_depth = int(np.iinfo(array.dtype).bits)
    else:
        bit_depth = 32
    return LoadedImage(array=np.asarray(array), bit_depth=bit_depth)


def _scale_to_unit(array: np.ndarray, mode: str = "dtype") -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if mode == "dtype":
        original = np.asarray(array)
        if np.issubdtype(original.dtype, np.integer):
            maximum = float(np.iinfo(original.dtype).max)
        else:
            maximum = float(np.nanmax(values))
        return np.clip(values / max(maximum, 1.0), 0.0, 1.0)
    if mode == "minmax":
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    elif mode == "percentile":
        low, high = [float(x) for x in np.nanpercentile(values, [0.5, 99.5])]
    else:
        raise ValueError(f"Unknown intensity scaling mode: {mode}")
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def image_array_to_tensor(
    array: np.ndarray,
    normalization: str = "imagenet",
) -> torch.Tensor:
    """Convert grayscale/RGB 8- or 16-bit input to a three-channel tensor."""
    if normalization == "imagenet":
        scaled = _scale_to_unit(array, mode="dtype")
    elif normalization in {"minmax", "percentile"}:
        scaled = _scale_to_unit(array, mode=normalization)
    elif normalization == "none":
        scaled = np.asarray(array, dtype=np.float32)
    else:
        raise ValueError(
            "normalization must be one of imagenet, minmax, percentile or none"
        )

    if scaled.ndim == 2:
        scaled = np.repeat(scaled[..., None], 3, axis=2)
    elif scaled.ndim == 3 and scaled.shape[-1] == 1:
        scaled = np.repeat(scaled, 3, axis=2)
    elif scaled.ndim != 3:
        raise ValueError(f"Unsupported image shape: {scaled.shape}")
    if scaled.shape[-1] > 3:
        scaled = scaled[..., :3]
    tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))).float()
    if normalization == "imagenet":
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


def resize_image_array(
    array: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize an image while retaining its numeric dynamic range."""
    target_h, target_w = [int(value) for value in target_hw]
    source = np.asarray(array)
    tensor = torch.from_numpy(source.astype(np.float32, copy=False))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
        resized = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)[None]
        resized = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported image shape: {source.shape}")
    output = resized.cpu().numpy()
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        output = np.rint(output).clip(info.min, info.max).astype(source.dtype)
    else:
        output = output.astype(source.dtype)
    return output


def target_preserving_resize_mask(
    mask: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize a binary mask without deleting small positive targets.

    Adaptive max pooling is used when reducing resolution. It guarantees that a
    positive input pixel contributes to at least one output bin, unlike a plain
    nearest-neighbour sample that can skip a one-pixel target.
    """
    binary = (np.asarray(mask) > 0).astype(np.float32)
    source_h, source_w = binary.shape[-2:]
    target_h, target_w = [int(value) for value in target_hw]
    tensor = torch.from_numpy(binary)[None, None]
    if target_h < source_h or target_w < source_w:
        # Adaptive max pooling also supports a larger output along the other
        # axis.  Using it whenever either axis is reduced prevents a one-pixel
        # target from disappearing in mixed resize cases such as H down / W up.
        resized = F.adaptive_max_pool2d(tensor, (target_h, target_w))
    else:
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
    return (resized[0, 0].numpy() > 0.5).astype(np.uint8)


def pad_tensor_to_stride(
    tensor: torch.Tensor,
    stride: int = 32,
    value: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if tensor.ndim not in {3, 4}:
        raise ValueError("tensor must be CHW or BCHW")
    if stride <= 0:
        raise ValueError("stride must be positive")
    height, width = tensor.shape[-2:]
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value), (height, width)
````

### 3.18 `rc_irstd/data/windows.py`

- SHA-256：`9bb725b7c40968b43e7fe8fe344d411421586792d025a52dd2b38eb621ae6b9b`
- 行数：`81`

````python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CausalWindow:
    context_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    sequence_id: str
    protocol: str = "temporal"


def build_causal_windows(
    sequence_ids: Sequence[str],
    frame_indices: Sequence[int],
    context_size: int,
    horizon: int,
    stride: int = 1,
) -> list[CausalWindow]:
    """Build prefix-to-future windows within each real temporal sequence."""
    if context_size <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("context_size, horizon and stride must be positive")
    groups: dict[str, list[tuple[int, int]]] = {}
    for global_index, (sequence, frame) in enumerate(
        zip(sequence_ids, frame_indices, strict=True)
    ):
        groups.setdefault(str(sequence), []).append((int(frame), global_index))

    windows: list[CausalWindow] = []
    for sequence, pairs in sorted(groups.items()):
        ordered = [index for _, index in sorted(pairs)]
        total = context_size + horizon
        for start in range(0, len(ordered) - total + 1, stride):
            context = tuple(ordered[start : start + context_size])
            future = tuple(ordered[start + context_size : start + total])
            if set(context).intersection(future):
                raise RuntimeError("Context and future windows overlap")
            windows.append(CausalWindow(context, future, sequence, "temporal"))
    return windows


def build_iid_windows(
    num_samples: int,
    context_size: int,
    horizon: int,
    stride: int | None = None,
    seed: int = 0,
) -> list[CausalWindow]:
    """Build deterministic support/query blocks for unordered static images.

    The permutation is fixed by ``seed``.  There is no claim of temporal
    causality: each output is explicitly tagged ``protocol='iid'``.  Setting
    ``stride=context_size+horizon`` yields non-overlapping statistical blocks;
    smaller strides are permitted for meta-training and are handled by the
    overlap-aware split utilities.
    """
    if num_samples <= 0 or context_size <= 0 or horizon <= 0:
        raise ValueError("num_samples, context_size and horizon must be positive")
    total = int(context_size + horizon)
    step = total if stride is None else int(stride)
    if step <= 0:
        raise ValueError("stride must be positive")
    if num_samples < total:
        return []
    permutation = np.random.default_rng(seed).permutation(num_samples).tolist()
    windows: list[CausalWindow] = []
    for block_index, start in enumerate(range(0, num_samples - total + 1, step)):
        context = tuple(int(value) for value in permutation[start : start + context_size])
        future = tuple(
            int(value) for value in permutation[start + context_size : start + total]
        )
        if set(context).intersection(future):
            raise RuntimeError("IID context and future blocks overlap")
        windows.append(
            CausalWindow(context, future, f"iid_block_{block_index:06d}", "iid")
        )
    return windows
````

### 3.19 `rc_irstd/deployment/__init__.py`

- SHA-256：`e4fa87d0cba845489b3d3916b9473dd20edc68130d7914fad1af669684925ff5`
- 行数：`5`

````python
"""Stateful zero-label deployment utilities."""

from rc_irstd.deployment.session import DeploymentState, ThresholdUpdate

__all__ = ["DeploymentState", "ThresholdUpdate"]
````

### 3.20 `rc_irstd/deployment/monitor.py`

- SHA-256：`d0deecaeb5364ea4b55b19e3793ba2aa685585f7f4de5a0125460245fe5aa6d3`
- 行数：`25`

````python
from __future__ import annotations

import numpy as np

from rc_irstd.models.risk_curve import FeatureNormaliser


def feature_ood_score(
    feature: np.ndarray,
    normaliser: FeatureNormaliser,
    clip: float = 50.0,
) -> float:
    """RMS standardised distance from the meta-training feature centre."""
    transformed = normaliser.transform(np.asarray(feature, dtype=np.float32)[None])[0]
    transformed = np.clip(transformed, -float(clip), float(clip))
    return float(np.sqrt(np.mean(np.square(transformed, dtype=np.float64))))


def score_drift(previous: np.ndarray, current: np.ndarray) -> float:
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("Feature vectors must have equal shapes")
    denominator = max(float(np.linalg.norm(previous)), 1e-12)
    return float(np.linalg.norm(current - previous) / denominator)
````

### 3.21 `rc_irstd/deployment/session.py`

- SHA-256：`4090a163b9776d756f609c7357826b58749f5f66b61a3c131aeeecc0dea3be4b`
- 行数：`56`

````python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ThresholdUpdate:
    sequence_id: str
    update_index: int
    warmup_ids: tuple[str, ...]
    base_threshold_index: int
    offset_index: int
    final_threshold_index: int
    threshold: float
    predicted_pixel_risk: float
    predicted_peak_risk_per_mp: float
    rejected: bool
    feature_ood_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warmup_ids"] = list(self.warmup_ids)
        return payload


@dataclass
class DeploymentState:
    detector_checkpoint: str
    curve_checkpoint: str
    score_directory: str
    pixel_budget: float
    peak_budget_per_mp: float
    warmup_size: int
    offset_index: int = 0
    updates: list[ThresholdUpdate] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, update: ThresholdUpdate) -> None:
        self.updates.append(update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_checkpoint": self.detector_checkpoint,
            "curve_checkpoint": self.curve_checkpoint,
            "score_directory": self.score_directory,
            "pixel_budget": self.pixel_budget,
            "peak_budget_per_mp": self.peak_budget_per_mp,
            "warmup_size": self.warmup_size,
            "offset_index": self.offset_index,
            "created_at": self.created_at,
            "updates": [item.to_dict() for item in self.updates],
        }
````

### 3.22 `rc_irstd/engine/__init__.py`

- SHA-256：`e7b7dd02b7e4c46b453a371fc865766d7771b4b699d4dfa7176937a9b7068ccc`
- 行数：`15`

````python
"""Training-engine utilities."""

from rc_irstd.engine.worker_seed import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
)

__all__ = [
    "capture_rng_state",
    "make_generator",
    "restore_rng_state",
    "seed_worker",
]
````

### 3.23 `rc_irstd/engine/worker_seed.py`

- SHA-256：`ef73ee5f99639c3ae475cce96b1c0c8509a57f93581a37ac5dfa8b0860b7c0ce`
- 行数：`47`

````python
from __future__ import annotations

"""Deterministic DataLoader and checkpoint RNG helpers."""

import random
from typing import Any

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's per-worker seed."""
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
````

### 3.24 `rc_irstd/episodes/__init__.py`

- SHA-256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- 行数：`0`

````python

````

### 3.25 `rc_irstd/episodes/builder.py`

- SHA-256：`0eba256b610ba07f9cc0b00e91fa233780a0e6f13f1a8105871cb2aa55e9a16b`
- 行数：`245`

````python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.data.windows import build_causal_windows, build_iid_windows
from rc_irstd.evaluation.curves import (
    aggregate_curve_counts,
    compute_image_curves,
    rates_from_counts,
)
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.utils.io import list_npz


@dataclass(frozen=True)
class EpisodeBuildConfig:
    context_size: int = 32
    horizon: int = 16
    stride: int = 16
    protocol: str = "auto"  # auto | iid | temporal
    seed: int = 0
    peak_min_distance: int = 2
    peak_min_score: float = 0.0
    peak_border: int = 0
    peak_tolerance: float = 2.0
    max_candidates_per_image: int | None = None
    pixel_epsilon: float = 1e-12
    peak_epsilon: float = 1e-6

    def validate(self) -> None:
        if self.context_size <= 0 or self.horizon <= 0 or self.stride <= 0:
            raise ValueError("context_size, horizon and stride must be positive")
        if self.protocol not in {"auto", "iid", "temporal"}:
            raise ValueError("protocol must be auto, iid or temporal")


def default_threshold_grid() -> np.ndarray:
    empty_threshold = np.nextafter(np.float32(1.0), np.float32(2.0))
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.90, 64, endpoint=False),
                np.linspace(0.90, 0.99, 64, endpoint=False),
                np.linspace(0.99, 0.999, 64, endpoint=False),
                np.linspace(0.999, 1.0, 61),
                np.asarray([empty_threshold], dtype=np.float32),
            ]
        )
    ).astype(np.float32)


def _resolve_protocol(records, requested: str) -> str:
    if requested != "auto":
        return requested
    types = {getattr(record, "dataset_type", "iid_images") for record in records}
    return "temporal" if types == {"temporal"} else "iid"


def build_episode_file(
    score_directory: str | Path,
    output_path: str | Path,
    thresholds: np.ndarray | None = None,
    config: EpisodeBuildConfig | None = None,
) -> Path:
    config = config or EpisodeBuildConfig()
    config.validate()
    thresholds = (
        default_threshold_grid()
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float32)
    )
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be a strictly increasing 1-D array")

    records = [load_score_record(path, require_mask=True) for path in list_npz(score_directory)]
    ordered = sorted(records, key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    protocol = _resolve_protocol(ordered, config.protocol)
    if protocol == "temporal":
        windows = build_causal_windows(
            [item.sequence_id for item in ordered],
            [item.frame_index for item in ordered],
            context_size=config.context_size,
            horizon=config.horizon,
            stride=config.stride,
        )
    else:
        windows = build_iid_windows(
            len(ordered),
            context_size=config.context_size,
            horizon=config.horizon,
            stride=config.stride,
            seed=config.seed,
        )
    if not windows:
        raise ValueError(
            f"No {protocol} support/query windows from {len(ordered)} records with "
            f"context={config.context_size}, horizon={config.horizon}, stride={config.stride}"
        )

    feature_config = WindowFeatureConfig(
        peak_min_distance=config.peak_min_distance,
        peak_min_score=config.peak_min_score,
        peak_border=config.peak_border,
        max_candidates_per_image=config.max_candidates_per_image,
    )
    feature_extractor = WindowFeatureExtractor(feature_config)

    image_curves = []
    image_rates: list[dict[str, np.ndarray]] = []
    for record in ordered:
        assert record.mask is not None
        curve = compute_image_curves(
            record.probability,
            record.mask,
            thresholds,
            peak_min_distance=config.peak_min_distance,
            peak_min_score=config.peak_min_score,
            peak_border=config.peak_border,
            peak_tolerance=config.peak_tolerance,
            max_candidates=config.max_candidates_per_image,
        )
        image_curves.append(curve)
        image_rates.append(
            rates_from_counts(
                curve,
                pixel_epsilon=config.pixel_epsilon,
                peak_epsilon=config.peak_epsilon,
            )
        )

    feature_rows: list[np.ndarray] = []
    pixel_log_rows: list[np.ndarray] = []
    peak_log_rows: list[np.ndarray] = []
    pixel_rate_rows: list[np.ndarray] = []
    peak_rate_rows: list[np.ndarray] = []
    pd_rows: list[np.ndarray] = []
    context_pixel_upper_rows: list[np.ndarray] = []
    context_peak_upper_rows: list[np.ndarray] = []
    future_pixel_rows: list[np.ndarray] = []
    future_peak_rows: list[np.ndarray] = []
    future_pd_rows: list[np.ndarray] = []
    future_gt_rows: list[np.ndarray] = []
    domains: list[str] = []
    sequences: list[str] = []
    protocols: list[str] = []
    context_ids: list[str] = []
    future_ids: list[str] = []
    feature_names: tuple[str, ...] | None = None

    for window in windows:
        context_records = [ordered[index] for index in window.context_indices]
        future_records = [ordered[index] for index in window.future_indices]
        context_counts = aggregate_curve_counts(
            [image_curves[index] for index in window.context_indices]
        )
        future_counts = aggregate_curve_counts(
            [image_curves[index] for index in window.future_indices]
        )
        rates = rates_from_counts(
            future_counts,
            pixel_epsilon=config.pixel_epsilon,
            peak_epsilon=config.peak_epsilon,
        )
        features, names = feature_extractor.extract(context_records)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature schema changed between episodes")

        feature_rows.append(features)
        pixel_log_rows.append(rates["pixel_log_risk"].astype(np.float32))
        peak_log_rows.append(rates["peak_log_risk"].astype(np.float32))
        pixel_rate_rows.append(rates["pixel_false_rate"].astype(np.float32))
        peak_rate_rows.append(rates["peak_false_per_mp"].astype(np.float32))
        pd_rows.append(rates["pd"].astype(np.float32))
        context_pixel_upper_rows.append(
            (context_counts.predicted_pixels / max(context_counts.total_pixels, 1)).astype(np.float32)
        )
        context_peak_upper_rows.append(
            (
                context_counts.predicted_peaks
                / max(context_counts.total_pixels / 1_000_000.0, 1e-12)
            ).astype(np.float32)
        )
        future_pixel_rows.append(
            np.stack(
                [image_rates[index]["pixel_false_rate"] for index in window.future_indices]
            ).astype(np.float32)
        )
        future_peak_rows.append(
            np.stack(
                [image_rates[index]["peak_false_per_mp"] for index in window.future_indices]
            ).astype(np.float32)
        )
        future_pd_rows.append(
            np.stack([image_rates[index]["pd"] for index in window.future_indices]).astype(np.float32)
        )
        future_gt_rows.append(
            np.asarray([image_curves[index].total_gt for index in window.future_indices], dtype=np.int32)
        )
        domains.append(future_records[0].dataset_name)
        sequences.append(window.sequence_id)
        protocols.append(window.protocol)
        context_ids.append(json.dumps([item.image_id for item in context_records]))
        future_ids.append(json.dumps([item.image_id for item in future_records]))

    assert feature_names is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.stack(feature_rows).astype(np.float32),
        pixel_log_risk=np.stack(pixel_log_rows).astype(np.float32),
        peak_log_risk=np.stack(peak_log_rows).astype(np.float32),
        pixel_risk=np.stack(pixel_rate_rows).astype(np.float32),
        peak_risk=np.stack(peak_rate_rows).astype(np.float32),
        pd=np.stack(pd_rows).astype(np.float32),
        context_pixel_upper=np.stack(context_pixel_upper_rows).astype(np.float32),
        context_peak_upper=np.stack(context_peak_upper_rows).astype(np.float32),
        future_pixel_risk=np.stack(future_pixel_rows).astype(np.float32),
        future_peak_risk=np.stack(future_peak_rows).astype(np.float32),
        future_pd=np.stack(future_pd_rows).astype(np.float32),
        future_gt_count=np.stack(future_gt_rows).astype(np.int32),
        thresholds=thresholds,
        domains=np.asarray(domains, dtype=np.str_),
        sequences=np.asarray(sequences, dtype=np.str_),
        protocols=np.asarray(protocols, dtype=np.str_),
        context_ids=np.asarray(context_ids, dtype=np.str_),
        future_ids=np.asarray(future_ids, dtype=np.str_),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        feature_config_json=np.asarray(json.dumps(feature_config.to_dict(), sort_keys=True)),
        build_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        context_size=np.asarray(config.context_size, dtype=np.int64),
        horizon=np.asarray(config.horizon, dtype=np.int64),
        stride=np.asarray(config.stride, dtype=np.int64),
        protocol=np.asarray(protocol),
        risk_definition=np.asarray("pixel_false_rate_and_fixed_false_peaks_per_mp"),
        context_upper_definition=np.asarray("all_context_detections_treated_as_false_upper_bound"),
    )
    return output_path
````

### 3.26 `rc_irstd/episodes/dataset.py`

- SHA-256：`1465991503b38d7e0be6f6fa2db0eb9ee9e73a3506b7f5a66c035543f848db6b`
- 行数：`178`

````python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeArrays:
    features: np.ndarray
    pixel_log_risk: np.ndarray
    peak_log_risk: np.ndarray
    pixel_risk: np.ndarray
    peak_risk: np.ndarray
    pd: np.ndarray
    context_pixel_upper: np.ndarray
    context_peak_upper: np.ndarray
    thresholds: np.ndarray
    domains: np.ndarray
    sequences: np.ndarray
    context_ids: np.ndarray
    future_ids: np.ndarray
    feature_names: tuple[str, ...]
    feature_config: dict[str, object] = field(default_factory=dict)
    protocols: np.ndarray | None = None
    future_pixel_risk: np.ndarray | None = None
    future_peak_risk: np.ndarray | None = None
    future_pd: np.ndarray | None = None
    future_gt_count: np.ndarray | None = None

    def subset(self, indices: np.ndarray) -> "EpisodeArrays":
        indices = np.asarray(indices, dtype=np.int64)
        optional = lambda value: None if value is None else value[indices]
        return EpisodeArrays(
            features=self.features[indices],
            pixel_log_risk=self.pixel_log_risk[indices],
            peak_log_risk=self.peak_log_risk[indices],
            pixel_risk=self.pixel_risk[indices],
            peak_risk=self.peak_risk[indices],
            pd=self.pd[indices],
            context_pixel_upper=self.context_pixel_upper[indices],
            context_peak_upper=self.context_peak_upper[indices],
            thresholds=self.thresholds,
            domains=self.domains[indices],
            sequences=self.sequences[indices],
            context_ids=self.context_ids[indices],
            future_ids=self.future_ids[indices],
            feature_names=self.feature_names,
            feature_config=dict(self.feature_config),
            protocols=optional(self.protocols),
            future_pixel_risk=optional(self.future_pixel_risk),
            future_peak_risk=optional(self.future_peak_risk),
            future_pd=optional(self.future_pd),
            future_gt_count=optional(self.future_gt_count),
        )


def _optional(payload, name: str, dtype):
    return np.asarray(payload[name], dtype=dtype) if name in payload else None


def load_episode_file(path: str | Path) -> EpisodeArrays:
    with np.load(path, allow_pickle=False) as payload:
        pixel_risk = np.asarray(payload["pixel_risk"], dtype=np.float32)
        peak_risk = np.asarray(payload["peak_risk"], dtype=np.float32)
        context_pixel_upper = np.asarray(
            payload["context_pixel_upper"]
            if "context_pixel_upper" in payload
            else np.full_like(pixel_risk, np.nan),
            dtype=np.float32,
        )
        context_peak_upper = np.asarray(
            payload["context_peak_upper"]
            if "context_peak_upper" in payload
            else np.full_like(peak_risk, np.nan),
            dtype=np.float32,
        )
        return EpisodeArrays(
            features=np.asarray(payload["features"], dtype=np.float32),
            pixel_log_risk=np.asarray(payload["pixel_log_risk"], dtype=np.float32),
            peak_log_risk=np.asarray(payload["peak_log_risk"], dtype=np.float32),
            pixel_risk=pixel_risk,
            peak_risk=peak_risk,
            pd=np.asarray(payload["pd"], dtype=np.float32),
            context_pixel_upper=context_pixel_upper,
            context_peak_upper=context_peak_upper,
            thresholds=np.asarray(payload["thresholds"], dtype=np.float32),
            domains=np.asarray(payload["domains"]).astype(str),
            sequences=np.asarray(payload["sequences"]).astype(str),
            context_ids=np.asarray(payload["context_ids"]).astype(str),
            future_ids=np.asarray(payload["future_ids"]).astype(str),
            feature_names=tuple(np.asarray(payload["feature_names"]).astype(str).tolist()),
            feature_config=(
                json.loads(str(np.asarray(payload["feature_config_json"]).item()))
                if "feature_config_json" in payload
                else {}
            ),
            protocols=(
                np.asarray(payload["protocols"]).astype(str)
                if "protocols" in payload
                else None
            ),
            future_pixel_risk=_optional(payload, "future_pixel_risk", np.float32),
            future_peak_risk=_optional(payload, "future_peak_risk", np.float32),
            future_pd=_optional(payload, "future_pd", np.float32),
            future_gt_count=_optional(payload, "future_gt_count", np.int32),
        )


def _concat_optional(arrays: list[EpisodeArrays], name: str):
    values = [getattr(item, name) for item in arrays]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Episode files disagree on optional field {name}")
    return np.concatenate(values, axis=0)


def concatenate_episode_files(paths: Sequence[str | Path]) -> EpisodeArrays:
    arrays = [load_episode_file(path) for path in paths]
    if not arrays:
        raise ValueError("At least one episode file is required")
    reference = arrays[0]
    for current in arrays[1:]:
        if not np.array_equal(current.thresholds, reference.thresholds):
            raise ValueError("Episode files use different threshold grids")
        if current.feature_names != reference.feature_names:
            raise ValueError("Episode files use different feature schemas")
        if current.feature_config != reference.feature_config:
            raise ValueError("Episode files use different feature configurations")
    return EpisodeArrays(
        features=np.concatenate([item.features for item in arrays], axis=0),
        pixel_log_risk=np.concatenate([item.pixel_log_risk for item in arrays], axis=0),
        peak_log_risk=np.concatenate([item.peak_log_risk for item in arrays], axis=0),
        pixel_risk=np.concatenate([item.pixel_risk for item in arrays], axis=0),
        peak_risk=np.concatenate([item.peak_risk for item in arrays], axis=0),
        pd=np.concatenate([item.pd for item in arrays], axis=0),
        context_pixel_upper=np.concatenate([item.context_pixel_upper for item in arrays], axis=0),
        context_peak_upper=np.concatenate([item.context_peak_upper for item in arrays], axis=0),
        thresholds=reference.thresholds,
        domains=np.concatenate([item.domains for item in arrays]),
        sequences=np.concatenate([item.sequences for item in arrays]),
        context_ids=np.concatenate([item.context_ids for item in arrays]),
        future_ids=np.concatenate([item.future_ids for item in arrays]),
        feature_names=reference.feature_names,
        feature_config=dict(reference.feature_config),
        protocols=_concat_optional(arrays, "protocols"),
        future_pixel_risk=_concat_optional(arrays, "future_pixel_risk"),
        future_peak_risk=_concat_optional(arrays, "future_peak_risk"),
        future_pd=_concat_optional(arrays, "future_pd"),
        future_gt_count=_concat_optional(arrays, "future_gt_count"),
    )


class RiskCurveDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: EpisodeArrays,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.features = ((arrays.features - feature_mean) / feature_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]),
            "pixel_log_risk": torch.from_numpy(self.arrays.pixel_log_risk[index]),
            "peak_log_risk": torch.from_numpy(self.arrays.peak_log_risk[index]),
        }
````

### 3.27 `rc_irstd/episodes/splits.py`

- SHA-256：`29db696e6f06817367bfa3c8152d3c394d225e75bab95c558840dc360bed1848`
- 行数：`194`

````python
from __future__ import annotations

import json

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays


def _json_ids(value: str) -> set[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, list) else set()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _iid_overlap_groups(arrays: EpisodeArrays, indices: np.ndarray) -> dict[int, str]:
    """Group IID episodes connected by any shared context/future image."""
    union = _UnionFind(len(indices))
    owners: dict[str, int] = {}
    for local, global_index in enumerate(indices):
        ids = _json_ids(arrays.context_ids[global_index]) | _json_ids(
            arrays.future_ids[global_index]
        )
        for image_id in ids:
            if image_id in owners:
                union.union(local, owners[image_id])
            else:
                owners[image_id] = local
    return {
        int(global_index): f"iid_overlap_{union.find(local):08d}"
        for local, global_index in enumerate(indices)
    }


def _group_labels(arrays: EpisodeArrays) -> np.ndarray:
    protocols = (
        np.asarray(arrays.protocols).astype(str)
        if arrays.protocols is not None
        else np.asarray(["temporal"] * len(arrays.domains), dtype=np.str_)
    )
    labels = np.empty(len(arrays.domains), dtype=object)
    iid_indices = np.flatnonzero(protocols == "iid")
    iid_groups = _iid_overlap_groups(arrays, iid_indices) if len(iid_indices) else {}
    for index, (domain, sequence, protocol) in enumerate(
        zip(arrays.domains, arrays.sequences, protocols, strict=True)
    ):
        if protocol == "iid":
            labels[index] = f"{domain}::{iid_groups[index]}"
        else:
            labels[index] = f"{domain}::{sequence}"
    return labels.astype(np.str_)


def leave_domains_out(
    arrays: EpisodeArrays,
    held_out_domains: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    held = np.isin(arrays.domains, np.asarray(held_out_domains).astype(str))
    return np.flatnonzero(~held), np.flatnonzero(held)


def grouped_train_val_split(
    arrays: EpisodeArrays,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split independent temporal sequences or IID overlap components."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    groups = _group_labels(arrays)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            "The episode graph has fewer than two independent groups. For IID "
            "data, build non-overlapping windows with stride >= context+horizon; "
            "for temporal data, provide at least two sequences."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(len(unique) - 1, max(1, int(round(len(unique) * val_fraction))))
    val_groups = set(unique[:count].tolist())
    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    train, validation = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if not len(train) or not len(validation):
        raise RuntimeError("Grouped train/validation split produced an empty partition")
    return train, validation


def grouped_calibration_test_split(
    arrays: EpisodeArrays,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact calibration episodes and an independent-group test set.

    ``calibration_size`` counts episodes/blocks.  Use the image-calibration path
    in :mod:`rc_irstd.calibration.samples` when a number of labelled images is
    being claimed.
    """
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    groups = _group_labels(arrays)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            "Calibration/test splitting needs at least two independent groups. "
            "Use non-overlapping IID windows or multiple temporal sequences."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    group_indices: dict[str, np.ndarray] = {}
    for group in unique:
        values = np.flatnonzero(groups == group)
        rng.shuffle(values)
        group_indices[str(group)] = values

    selected_groups: list[str] = []
    available = 0
    for group in unique[:-1]:
        selected_groups.append(str(group))
        available += len(group_indices[str(group)])
        if available >= calibration_size:
            break
    if available < calibration_size:
        max_possible = sum(len(group_indices[str(group)]) for group in unique[:-1])
        raise ValueError(
            f"Requested {calibration_size} calibration episodes, but only "
            f"{max_possible} are available while retaining an independent test group"
        )

    pool = np.concatenate([group_indices[group] for group in selected_groups])
    rng.shuffle(pool)
    calibration = np.sort(pool[:calibration_size].astype(np.int64))
    selected_group_set = set(selected_groups)
    test = np.flatnonzero(
        np.asarray([str(group) not in selected_group_set for group in groups], dtype=bool)
    ).astype(np.int64)
    if len(test) == 0:
        raise RuntimeError("Independent-group split produced an empty test set")
    if set(groups[calibration]).intersection(set(groups[test])):
        raise RuntimeError("Calibration and test groups overlap")
    return calibration, test


def split_iid_images(
    image_ids: np.ndarray,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact labelled-image split for IID calibration samples."""
    image_ids = np.asarray(image_ids).astype(str)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError("Image-shot calibration requires unique image IDs")
    if calibration_size <= 0 or calibration_size >= len(image_ids):
        raise ValueError("calibration_size must be in [1, num_images-1]")
    permutation = np.random.default_rng(seed).permutation(len(image_ids))
    calibration = np.sort(permutation[:calibration_size].astype(np.int64))
    test = np.sort(permutation[calibration_size:].astype(np.int64))
    return calibration, test


def split_sequence_units(
    sequences: np.ndarray,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select complete temporal sequences; size counts independent sequences."""
    sequences = np.asarray(sequences).astype(str)
    unique = np.unique(sequences)
    if calibration_size <= 0 or calibration_size >= len(unique):
        raise ValueError("calibration_size must leave at least one test sequence")
    selected = set(np.random.default_rng(seed).permutation(unique)[:calibration_size].tolist())
    calibration = np.flatnonzero(np.asarray([item in selected for item in sequences]))
    test = np.flatnonzero(np.asarray([item not in selected for item in sequences]))
    return calibration, test
````

### 3.28 `rc_irstd/evaluation/__init__.py`

- SHA-256：`65eef48be4602ab39f2b1571a7491c31fb5319e8b1b93b26d1caae7009498a33`
- 行数：`9`

````python
from rc_irstd.evaluation.budget import BudgetSummary, summarise_selected_points
from rc_irstd.evaluation.operating_point import OperatingPoint, select_dual_budget_threshold

__all__ = [
    "BudgetSummary",
    "summarise_selected_points",
    "OperatingPoint",
    "select_dual_budget_threshold",
]
````

### 3.29 `rc_irstd/evaluation/budget.py`

- SHA-256：`752fd5f0db882ed8578ae8a278a08de43fb2d652b47a2a9640198e0369f892ee`
- 行数：`72`

````python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BudgetSummary:
    joint_bsr: float
    pixel_bsr: float
    peak_bsr: float
    pixel_excess: float
    peak_excess: float
    mean_pd_selected: float
    effective_pd_with_rejects: float
    conditional_pd_non_rejected: float
    worst_domain_pd_selected: float
    rejection_rate: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise_selected_points(
    pixel_risk: np.ndarray,
    peak_risk: np.ndarray,
    pd: np.ndarray,
    rejected: np.ndarray,
    domains: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> BudgetSummary:
    pixel_risk = np.asarray(pixel_risk, dtype=np.float64)
    peak_risk = np.asarray(peak_risk, dtype=np.float64)
    pd = np.asarray(pd, dtype=np.float64)
    rejected = np.asarray(rejected, dtype=bool)
    domains = np.asarray(domains).astype(str)
    if not (len(pixel_risk) == len(peak_risk) == len(pd) == len(rejected) == len(domains)):
        raise ValueError("All selected-point arrays must have equal length")
    if len(pd) == 0:
        raise ValueError("Cannot summarise an empty selection")
    pixel_ok = pixel_risk <= pixel_budget
    peak_ok = peak_risk <= peak_budget
    joint = pixel_ok & peak_ok
    valid_pd = np.isfinite(pd)
    domain_means: list[float] = []
    for domain in np.unique(domains):
        values = pd[(domains == domain) & valid_pd]
        if len(values):
            domain_means.append(float(values.mean()))
    non_rejected_valid = (~rejected) & valid_pd
    conditional = float(pd[non_rejected_valid].mean()) if non_rejected_valid.any() else 0.0
    effective_values = pd[valid_pd].copy()
    effective_values[rejected[valid_pd]] = 0.0
    return BudgetSummary(
        joint_bsr=float(joint.mean()),
        pixel_bsr=float(pixel_ok.mean()),
        peak_bsr=float(peak_ok.mean()),
        pixel_excess=float(np.maximum(pixel_risk - pixel_budget, 0.0).mean()),
        peak_excess=float(np.maximum(peak_risk - peak_budget, 0.0).mean()),
        mean_pd_selected=float(pd[valid_pd].mean()) if valid_pd.any() else 0.0,
        effective_pd_with_rejects=(
            float(effective_values.mean()) if len(effective_values) else 0.0
        ),
        conditional_pd_non_rejected=conditional,
        worst_domain_pd_selected=float(min(domain_means)) if domain_means else float("nan"),
        rejection_rate=float(rejected.mean()),
        count=len(pd),
    )
````

### 3.30 `rc_irstd/evaluation/component_curves.py`

- SHA-256：`45b150cdd73662ea609d7840c0e8b8108f8e636745e0ea3d2ea91d342c14b7e5`
- 行数：`55`

````python
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold


@dataclass(frozen=True)
class ComponentCurveRow:
    threshold: float
    pd: float
    false_components_per_mp: float
    false_pixel_rate: float
    iou: float
    niou: float
    hiou: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_component_curve(
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    thresholds: np.ndarray,
    object_tolerance: float = 2.0,
) -> list[ComponentCurveRow]:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be an ascending 1-D array")
    rows: list[ComponentCurveRow] = []
    for threshold in thresholds:
        metrics = evaluate_irstd_at_threshold(
            probabilities, masks, float(threshold), object_tolerance
        )
        rows.append(
            ComponentCurveRow(
                threshold=float(threshold),
                pd=metrics.pd,
                false_components_per_mp=metrics.false_components_per_mp,
                false_pixel_rate=metrics.false_pixel_rate,
                iou=metrics.iou,
                niou=metrics.niou,
                hiou=metrics.hiou,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
            )
        )
    return rows
````

### 3.31 `rc_irstd/evaluation/curves.py`

- SHA-256：`ff9e9737bd5440062ff02ccee6fd5dddddf9b8cf43cd6ababd2c9263d57a1f59`
- 行数：`111`

````python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_irstd.candidates.peaks import build_fixed_peak_set, fixed_peak_curves


@dataclass(frozen=True)
class ImageCurveCounts:
    pixel_false: np.ndarray
    peak_false: np.ndarray
    matched_gt: np.ndarray
    predicted_pixels: np.ndarray
    predicted_peaks: np.ndarray
    total_pixels: int
    total_gt: int

    @property
    def num_thresholds(self) -> int:
        return len(self.pixel_false)


def _counts_at_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float32).reshape(-1))
    positions = np.searchsorted(ordered, thresholds, side="left")
    return (len(ordered) - positions).astype(np.int64)


def compute_image_curves(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    thresholds: np.ndarray,
    peak_min_distance: int = 2,
    peak_min_score: float = 1e-6,
    peak_border: int = 0,
    peak_tolerance: float = 2.0,
    max_candidates: int | None = None,
) -> ImageCurveCounts:
    score = np.asarray(score_map, dtype=np.float32).squeeze()
    mask = np.asarray(gt_mask).squeeze() > 0
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if score.shape != mask.shape:
        raise ValueError(f"score and mask shapes differ: {score.shape} vs {mask.shape}")
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be ascending")

    background_values = score[~mask]
    all_values = score.reshape(-1)
    pixel_false = _counts_at_thresholds(background_values, thresholds)
    predicted_pixels = _counts_at_thresholds(all_values, thresholds)

    peak_set = build_fixed_peak_set(
        score,
        mask,
        min_distance=peak_min_distance,
        min_score=peak_min_score,
        border=peak_border,
        tolerance=peak_tolerance,
        max_candidates=max_candidates,
    )
    predicted_peaks, peak_false, matched_gt = fixed_peak_curves(peak_set, thresholds)
    return ImageCurveCounts(
        pixel_false=pixel_false,
        peak_false=peak_false,
        matched_gt=matched_gt,
        predicted_pixels=predicted_pixels,
        predicted_peaks=predicted_peaks,
        total_pixels=int(score.size),
        total_gt=int(peak_set.num_gt),
    )


def aggregate_curve_counts(records: list[ImageCurveCounts]) -> ImageCurveCounts:
    if not records:
        raise ValueError("Cannot aggregate an empty record list")
    num_thresholds = records[0].num_thresholds
    if any(record.num_thresholds != num_thresholds for record in records):
        raise ValueError("All records must use the same threshold grid")
    return ImageCurveCounts(
        pixel_false=np.sum([item.pixel_false for item in records], axis=0),
        peak_false=np.sum([item.peak_false for item in records], axis=0),
        matched_gt=np.sum([item.matched_gt for item in records], axis=0),
        predicted_pixels=np.sum([item.predicted_pixels for item in records], axis=0),
        predicted_peaks=np.sum([item.predicted_peaks for item in records], axis=0),
        total_pixels=sum(item.total_pixels for item in records),
        total_gt=sum(item.total_gt for item in records),
    )


def rates_from_counts(
    counts: ImageCurveCounts,
    pixel_epsilon: float = 1e-12,
    peak_epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    pixel_rate = counts.pixel_false / max(counts.total_pixels, 1)
    peak_per_mp = counts.peak_false / max(counts.total_pixels / 1_000_000.0, 1e-12)
    pd = counts.matched_gt / max(counts.total_gt, 1)
    return {
        "pixel_false_rate": pixel_rate.astype(np.float64),
        "peak_false_per_mp": peak_per_mp.astype(np.float64),
        "pd": pd.astype(np.float64),
        "pixel_log_risk": np.log10(pixel_rate + pixel_epsilon),
        "peak_log_risk": np.log10(peak_per_mp + peak_epsilon),
    }


def monotone_nonincreasing_envelope(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.maximum.accumulate(values[::-1], axis=-1)[::-1]
````

### 3.32 `rc_irstd/evaluation/detector_selection.py`

- SHA-256：`a8dacde18940d0355a9938542e312b8b18a37846bcd7335558dbce6f3eb3ba49`
- 行数：`138`

````python
from __future__ import annotations

"""Budget-aligned detector checkpoint selection.

The detector is selected from labelled *source validation* data only.  For each
source domain, the evaluator finds the earliest threshold that simultaneously
satisfies the pixel and fixed-local-peak budgets, then reports object detection
probability at that working point.  A checkpoint is preferred lexicographically
by worst-domain Pd, mean-domain Pd, and IoU.
"""

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from rc_irstd.evaluation.curves import (
    ImageCurveCounts,
    aggregate_curve_counts,
    rates_from_counts,
)


@dataclass(frozen=True)
class DomainBudgetPoint:
    domain: str
    index: int
    threshold: float
    pd: float
    pixel_risk: float
    peak_risk: float
    rejected: bool


@dataclass(frozen=True)
class DetectorBudgetSelection:
    pixel_budget: float
    peak_budget: float
    mean_domain_pd: float
    worst_domain_pd: float
    mean_threshold: float
    rejection_rate: float
    domain_points: tuple[DomainBudgetPoint, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["domain_points"] = [asdict(item) for item in self.domain_points]
        return payload

    def rank_key(self, iou: float) -> tuple[float, float, float, float]:
        # Lower rejection is a final tie-breaker. Empty-action predictions are
        # not allowed to make an otherwise weak detector appear strong.
        return (
            float(self.worst_domain_pd),
            float(self.mean_domain_pd),
            float(iou),
            -float(self.rejection_rate),
        )


def validation_threshold_grid(num_points: int = 96) -> np.ndarray:
    if num_points < 16:
        raise ValueError("num_points must be at least 16")
    empty = np.nextafter(np.float32(1.0), np.float32(2.0))
    # More support is allocated near one, where IRSTD low-FA crossings occur.
    coarse = max(8, num_points // 4)
    medium = max(8, num_points // 4)
    fine = max(8, num_points - coarse - medium)
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.90, coarse, endpoint=False),
                np.linspace(0.90, 0.99, medium, endpoint=False),
                np.linspace(0.99, 1.0, fine),
                np.asarray([empty], dtype=np.float32),
            ]
        )
    ).astype(np.float32)


def _select_first_feasible(
    thresholds: np.ndarray,
    pixel: np.ndarray,
    peak: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> int:
    feasible = np.flatnonzero((pixel <= pixel_budget) & (peak <= peak_budget))
    return int(feasible[0]) if len(feasible) else len(thresholds) - 1


def summarise_detector_budget(
    domain_curves: dict[str, Iterable[ImageCurveCounts]],
    thresholds: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> DetectorBudgetSelection:
    if pixel_budget <= 0 or peak_budget <= 0:
        raise ValueError("budgets must be positive")
    points: list[DomainBudgetPoint] = []
    thresholds = np.asarray(thresholds, dtype=np.float32)
    for domain, values in sorted(domain_curves.items()):
        records = list(values)
        if not records:
            continue
        rates = rates_from_counts(aggregate_curve_counts(records))
        index = _select_first_feasible(
            thresholds,
            rates["pixel_false_rate"],
            rates["peak_false_per_mp"],
            pixel_budget,
            peak_budget,
        )
        points.append(
            DomainBudgetPoint(
                domain=str(domain),
                index=index,
                threshold=float(thresholds[index]),
                pd=float(rates["pd"][index]),
                pixel_risk=float(rates["pixel_false_rate"][index]),
                peak_risk=float(rates["peak_false_per_mp"][index]),
                rejected=bool(thresholds[index] > 1.0),
            )
        )
    if not points:
        raise ValueError("No validation curves were supplied")
    pds = np.asarray([item.pd for item in points], dtype=np.float64)
    thresholds_selected = np.asarray([item.threshold for item in points], dtype=np.float64)
    rejected = np.asarray([item.rejected for item in points], dtype=np.float64)
    return DetectorBudgetSelection(
        pixel_budget=float(pixel_budget),
        peak_budget=float(peak_budget),
        mean_domain_pd=float(pds.mean()),
        worst_domain_pd=float(pds.min()),
        mean_threshold=float(thresholds_selected.mean()),
        rejection_rate=float(rejected.mean()),
        domain_points=tuple(points),
    )
````

### 3.33 `rc_irstd/evaluation/irstd_metrics.py`

- SHA-256：`4c27540dfd3f6c38c5ffa4b98f6b987b6b53277eb534f7cf30fe6adf6aac5735`
- 行数：`110`

````python
from __future__ import annotations

"""Literature-compatible IRSTD segmentation and object metrics."""

from dataclasses import asdict, dataclass

import numpy as np

from rc_irstd.evaluation.segmentation import evaluate_binary_segmentation


@dataclass(frozen=True)
class IRSTDMetrics:
    iou: float
    niou: float
    hiou: float
    foreground_iou: float
    background_iou: float
    precision: float
    recall: float
    f1: float
    pd: float
    false_components_per_mp: float
    false_pixel_rate: float
    gt_objects: int
    detected_objects: int
    false_components: int
    total_pixels: int
    num_images: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate_irstd_at_threshold(
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    threshold: float,
    object_tolerance: float = 2.0,
) -> IRSTDMetrics:
    if len(probabilities) != len(masks) or not probabilities:
        raise ValueError("probabilities and masks must be non-empty and equally sized")
    tp = fp = fn = tn = 0
    union = 0
    total_pixels = 0
    gt_objects = detected_objects = false_components = 0
    per_image_iou: list[float] = []

    for probability, mask in zip(probabilities, masks, strict=True):
        pred = np.asarray(probability).squeeze() >= float(threshold)
        target = np.asarray(mask).squeeze() > 0
        if pred.shape != target.shape:
            raise ValueError("Probability and mask shapes differ")
        metrics = evaluate_binary_segmentation(pred, target, object_tolerance)
        current_tp = metrics.true_positive_pixels
        current_fp = metrics.false_positive_pixels
        current_fn = metrics.false_negative_pixels
        current_tn = int(target.size - current_tp - current_fp - current_fn)
        tp += current_tp
        fp += current_fp
        fn += current_fn
        tn += current_tn
        union += metrics.union
        total_pixels += int(target.size)
        gt_objects += metrics.gt_objects
        detected_objects += metrics.detected_objects
        false_components += metrics.false_components
        # nIoU is the mean image IoU over non-empty unions. Fully empty images
        # are excluded and reported through false-alarm metrics instead.
        if metrics.union > 0:
            per_image_iou.append(metrics.intersection / metrics.union)

    foreground_iou = _safe_ratio(tp, tp + fp + fn)
    background_iou = _safe_ratio(tn, tn + fp + fn)
    hiou = (
        2.0 * foreground_iou * background_iou / (foreground_iou + background_iou)
        if foreground_iou + background_iou > 0
        else 0.0
    )
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return IRSTDMetrics(
        iou=_safe_ratio(tp, union),
        niou=float(np.mean(per_image_iou)) if per_image_iou else 0.0,
        hiou=float(hiou),
        foreground_iou=foreground_iou,
        background_iou=background_iou,
        precision=precision,
        recall=recall,
        f1=f1,
        pd=_safe_ratio(detected_objects, gt_objects),
        false_components_per_mp=_safe_ratio(
            false_components, total_pixels / 1_000_000.0
        ),
        false_pixel_rate=_safe_ratio(fp, total_pixels),
        gt_objects=gt_objects,
        detected_objects=detected_objects,
        false_components=false_components,
        total_pixels=total_pixels,
        num_images=len(probabilities),
    )
````

### 3.34 `rc_irstd/evaluation/operating_point.py`

- SHA-256：`5b3263363f38c5e65a760ed6f90a94b6f60a4aeead6ee66e4527053faa110155`
- 行数：`53`

````python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    index: int
    threshold: float
    rejected: bool
    predicted_pixel_risk: float
    predicted_peak_risk: float


def select_dual_budget_threshold(
    thresholds: np.ndarray,
    pixel_log_risk: np.ndarray,
    peak_log_risk: np.ndarray,
    pixel_budget: float,
    peak_budget_per_mp: float,
) -> OperatingPoint:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pixel = np.asarray(pixel_log_risk, dtype=np.float64)
    peak = np.asarray(peak_log_risk, dtype=np.float64)
    if not (len(thresholds) == len(pixel) == len(peak)):
        raise ValueError("Threshold and risk curves must have equal lengths")
    if pixel_budget <= 0 or peak_budget_per_mp <= 0:
        raise ValueError("Budgets must be positive")
    feasible = np.flatnonzero(
        (pixel <= np.log10(pixel_budget))
        & (peak <= np.log10(peak_budget_per_mp))
    )
    if len(feasible) == 0:
        index = len(thresholds) - 1
        return OperatingPoint(
            index=index,
            threshold=float(thresholds[index]),
            rejected=True,
            predicted_pixel_risk=float(10 ** pixel[index]),
            predicted_peak_risk=float(10 ** peak[index]),
        )
    index = int(feasible[0])
    return OperatingPoint(
        index=index,
        threshold=float(thresholds[index]),
        # A threshold above one is the explicit empty-prediction/abstention
        # action included in the formal threshold grid.
        rejected=bool(thresholds[index] > 1.0),
        predicted_pixel_risk=float(10 ** pixel[index]),
        predicted_peak_risk=float(10 ** peak[index]),
    )
````

### 3.35 `rc_irstd/evaluation/risk_curve_metrics.py`

- SHA-256：`b384b8b3f5049737b442c8fa07bc353559421324e1d8a71f5d38a2ca68c676d4`
- 行数：`115`

````python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rc_irstd.evaluation.budget import BudgetSummary, summarise_selected_points
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold


@dataclass(frozen=True)
class RiskCurveMetrics:
    pixel_log_mae: float
    peak_log_mae: float
    pixel_pointwise_coverage: float
    peak_pointwise_coverage: float
    joint_pointwise_coverage: float
    pixel_underestimation_mae: float
    peak_underestimation_mae: float
    monotonicity_violations: int
    selected: BudgetSummary

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected"] = self.selected.to_dict()
        return payload


def select_indices_from_predictions(
    thresholds: np.ndarray,
    predicted_pixel_log: np.ndarray,
    predicted_peak_log: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(predicted_pixel_log)
    peak = np.asarray(predicted_peak_log)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Predicted risk curves must share shape [samples, thresholds]")
    indices: list[int] = []
    rejected: list[bool] = []
    for pixel_curve, peak_curve in zip(pixel, peak, strict=True):
        point = select_dual_budget_threshold(
            thresholds,
            pixel_curve,
            peak_curve,
            pixel_budget,
            peak_budget,
        )
        indices.append(point.index)
        rejected.append(point.rejected)
    return np.asarray(indices, dtype=np.int64), np.asarray(rejected, dtype=bool)


def evaluate_risk_curve_predictions(
    thresholds: np.ndarray,
    predicted_pixel_log: np.ndarray,
    predicted_peak_log: np.ndarray,
    true_pixel_log: np.ndarray,
    true_peak_log: np.ndarray,
    true_pixel_risk: np.ndarray,
    true_peak_risk: np.ndarray,
    true_pd: np.ndarray,
    domains: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[RiskCurveMetrics, np.ndarray, np.ndarray]:
    predicted_pixel_log = np.asarray(predicted_pixel_log, dtype=np.float64)
    predicted_peak_log = np.asarray(predicted_peak_log, dtype=np.float64)
    true_pixel_log = np.asarray(true_pixel_log, dtype=np.float64)
    true_peak_log = np.asarray(true_peak_log, dtype=np.float64)
    if not (
        predicted_pixel_log.shape
        == predicted_peak_log.shape
        == true_pixel_log.shape
        == true_peak_log.shape
    ):
        raise ValueError("All log-risk arrays must have equal shapes")

    indices, rejected = select_indices_from_predictions(
        thresholds,
        predicted_pixel_log,
        predicted_peak_log,
        pixel_budget,
        peak_budget,
    )
    rows = np.arange(len(indices))
    selected_summary = summarise_selected_points(
        np.asarray(true_pixel_risk)[rows, indices],
        np.asarray(true_peak_risk)[rows, indices],
        np.asarray(true_pd)[rows, indices],
        rejected,
        domains,
        pixel_budget,
        peak_budget,
    )
    pixel_error = predicted_pixel_log - true_pixel_log
    peak_error = predicted_peak_log - true_peak_log
    monotonicity = int(
        np.sum(np.diff(predicted_pixel_log, axis=1) > 1e-8)
        + np.sum(np.diff(predicted_peak_log, axis=1) > 1e-8)
    )
    metrics = RiskCurveMetrics(
        pixel_log_mae=float(np.abs(pixel_error).mean()),
        peak_log_mae=float(np.abs(peak_error).mean()),
        pixel_pointwise_coverage=float((pixel_error >= 0).mean()),
        peak_pointwise_coverage=float((peak_error >= 0).mean()),
        joint_pointwise_coverage=float(((pixel_error >= 0) & (peak_error >= 0)).mean()),
        pixel_underestimation_mae=float(np.maximum(-pixel_error, 0.0).mean()),
        peak_underestimation_mae=float(np.maximum(-peak_error, 0.0).mean()),
        monotonicity_violations=monotonicity,
        selected=selected_summary,
    )
    return metrics, indices, rejected
````

### 3.36 `rc_irstd/evaluation/segmentation.py`

- SHA-256：`cff8b8406ae809c9eab77cc8e20d75bfd3f624fcd1da7dcdb66ea590d4916bf2`
- 行数：`72`

````python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SegmentationMetrics:
    intersection: int
    union: int
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    gt_objects: int
    detected_objects: int
    false_components: int


def evaluate_binary_segmentation(
    prediction: np.ndarray,
    target: np.ndarray,
    object_tolerance: float = 2.0,
) -> SegmentationMetrics:
    pred = np.asarray(prediction).squeeze() > 0
    gt = np.asarray(target).squeeze() > 0
    if pred.shape != gt.shape:
        raise ValueError("prediction and target must have equal shapes")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = int((pred | gt).sum())

    gt_labels, gt_count = ndimage.label(gt)
    pred_labels, pred_count = ndimage.label(pred)
    detected: set[int] = set()
    false_components = 0
    if gt_count > 0:
        distance, nearest = ndimage.distance_transform_edt(~gt, return_indices=True)
    else:
        distance = np.full(gt.shape, np.inf)
        nearest = np.zeros((2,) + gt.shape, dtype=np.int64)

    for component_id in range(1, pred_count + 1):
        coords = np.argwhere(pred_labels == component_id)
        overlapping = np.unique(gt_labels[pred_labels == component_id])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping):
            detected.update(int(value) for value in overlapping)
            continue
        centroid = np.rint(coords.mean(axis=0)).astype(int)
        y, x = int(centroid[0]), int(centroid[1])
        if distance[y, x] <= object_tolerance:
            near_y = int(nearest[0, y, x])
            near_x = int(nearest[1, y, x])
            gt_id = int(gt_labels[near_y, near_x])
            if gt_id > 0:
                detected.add(gt_id)
                continue
        false_components += 1

    return SegmentationMetrics(
        intersection=tp,
        union=union,
        true_positive_pixels=tp,
        false_positive_pixels=fp,
        false_negative_pixels=fn,
        gt_objects=int(gt_count),
        detected_objects=len(detected),
        false_components=false_components,
    )
````

### 3.37 `rc_irstd/features/__init__.py`

- SHA-256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- 行数：`0`

````python

````

### 3.38 `rc_irstd/features/image_stats.py`

- SHA-256：`7d6de7da10aefab53f22b76a50caadaae4eea93112a8d7fa1004fd7238b22742`
- 行数：`88`

````python
from __future__ import annotations

import numpy as np
from scipy import ndimage


IMAGE_STAT_NAMES = (
    "gray_mean",
    "gray_std",
    "gray_mad",
    "gray_q01",
    "gray_q10",
    "gray_q50",
    "gray_q90",
    "gray_q99",
    "gradient_mean",
    "gradient_std",
    "gradient_q95",
    "laplacian_std",
    "local_contrast_mean",
    "local_contrast_q95",
    "entropy_256",
    "high_frequency_energy",
)


def _to_gray(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    array = source.astype(np.float32, copy=False)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[-1] >= 3:
        gray = 0.2989 * array[..., 0] + 0.5870 * array[..., 1] + 0.1140 * array[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {array.shape}")
    if np.issubdtype(source.dtype, np.integer):
        gray = gray / max(float(np.iinfo(source.dtype).max), 1.0)
    elif gray.max(initial=0.0) > 1.5:
        gray = gray / max(float(np.nanmax(gray)), 1.0)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def compute_image_statistics(image: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    gray = _to_gray(image)
    median = float(np.median(gray))
    mad = float(np.median(np.abs(gray - median)))
    q01, q10, q50, q90, q99 = np.quantile(gray, [0.01, 0.10, 0.50, 0.90, 0.99])

    grad_y = ndimage.sobel(gray, axis=0, mode="reflect")
    grad_x = ndimage.sobel(gray, axis=1, mode="reflect")
    gradient = np.hypot(grad_x, grad_y)
    laplacian = ndimage.laplace(gray, mode="reflect")
    smooth = ndimage.gaussian_filter(gray, sigma=1.5, mode="reflect")
    local_contrast = np.abs(gray - smooth)

    histogram, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    spectrum = np.fft.rfft2(gray - gray.mean())
    power = np.abs(spectrum) ** 2
    height, width = power.shape
    y = np.fft.fftfreq(gray.shape[0])[:, None]
    x = np.fft.rfftfreq(gray.shape[1])[None, :]
    radial = np.sqrt(y * y + x * x)
    high_frequency = power[radial >= 0.25].sum() / max(power.sum(), 1e-12)

    values = np.asarray([
        gray.mean(),
        gray.std(),
        mad,
        q01,
        q10,
        q50,
        q90,
        q99,
        gradient.mean(),
        gradient.std(),
        np.quantile(gradient, 0.95),
        laplacian.std(),
        local_contrast.mean(),
        np.quantile(local_contrast, 0.95),
        entropy,
        high_frequency,
    ], dtype=np.float32)
    return values, IMAGE_STAT_NAMES
````

### 3.39 `rc_irstd/features/window_stats.py`

- SHA-256：`fef4afaf3311ddd273185b45314c2fde70afa4b2780c72a173631f5cd05b0eb1`
- 行数：`195`

````python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import ScoreRecord


DEFAULT_SURVIVAL_THRESHOLDS = np.asarray([
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.97,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
], dtype=np.float32)
DEFAULT_QUANTILES = np.asarray([
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.995,
    0.999,
    0.9995,
], dtype=np.float32)


@dataclass(frozen=True)
class WindowFeatureConfig:
    survival_thresholds: np.ndarray = field(
        default_factory=lambda: DEFAULT_SURVIVAL_THRESHOLDS.copy()
    )
    quantiles: np.ndarray = field(default_factory=lambda: DEFAULT_QUANTILES.copy())
    peak_min_distance: int = 2
    peak_min_score: float = 0.0
    peak_border: int = 0
    max_candidates_per_image: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "survival_thresholds": np.asarray(
                self.survival_thresholds, dtype=np.float32
            ).tolist(),
            "quantiles": np.asarray(self.quantiles, dtype=np.float32).tolist(),
            "peak_min_distance": int(self.peak_min_distance),
            "peak_min_score": float(self.peak_min_score),
            "peak_border": int(self.peak_border),
            "max_candidates_per_image": (
                None
                if self.max_candidates_per_image is None
                else int(self.max_candidates_per_image)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WindowFeatureConfig":
        payload = payload or {}
        return cls(
            survival_thresholds=np.asarray(
                payload.get("survival_thresholds", DEFAULT_SURVIVAL_THRESHOLDS),
                dtype=np.float32,
            ),
            quantiles=np.asarray(
                payload.get("quantiles", DEFAULT_QUANTILES), dtype=np.float32
            ),
            peak_min_distance=int(payload.get("peak_min_distance", 2)),
            peak_min_score=float(payload.get("peak_min_score", 0.0)),
            peak_border=int(payload.get("peak_border", 0)),
            max_candidates_per_image=(
                None
                if payload.get("max_candidates_per_image") is None
                else int(payload["max_candidates_per_image"])
            ),
        )


class WindowFeatureExtractor:
    """Label-free score, peak and acquisition statistics for a deployment window."""

    def __init__(self, config: WindowFeatureConfig | None = None) -> None:
        self.config = config or WindowFeatureConfig()

    @staticmethod
    def _mean_std(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(np.float32)

    def extract(self, records: Sequence[ScoreRecord]) -> tuple[np.ndarray, tuple[str, ...]]:
        if not records:
            raise ValueError("A window must contain at least one score record")
        config = self.config
        threshold_count = len(config.survival_thresholds)
        quantile_count = len(config.quantiles)

        pixel_survival: list[np.ndarray] = []
        pixel_quantiles: list[np.ndarray] = []
        peak_survival_per_mp: list[np.ndarray] = []
        peak_quantiles: list[np.ndarray] = []
        image_statistics: list[np.ndarray] = []
        total_pixels: list[float] = []
        peak_counts: list[float] = []

        expected_stat_names = records[0].image_stat_names
        for record in records:
            if record.image_stat_names != expected_stat_names:
                raise ValueError("All records in a window must use the same image statistics")
            scores = record.probability.reshape(-1)
            pixel_survival.append(
                np.asarray([(scores >= threshold).mean() for threshold in config.survival_thresholds])
            )
            pixel_quantiles.append(np.quantile(scores, config.quantiles))

            peak_scores, _, _ = extract_fixed_peaks(
                record.probability,
                min_distance=config.peak_min_distance,
                min_score=config.peak_min_score,
                border=config.peak_border,
                max_candidates=config.max_candidates_per_image,
            )
            denominator_mp = max(record.total_pixels / 1_000_000.0, 1e-12)
            peak_survival_per_mp.append(
                np.asarray([
                    (peak_scores >= threshold).sum() / denominator_mp
                    for threshold in config.survival_thresholds
                ])
            )
            if len(peak_scores):
                peak_quantiles.append(np.quantile(peak_scores, config.quantiles))
            else:
                peak_quantiles.append(np.zeros(quantile_count, dtype=np.float32))
            image_statistics.append(record.image_stats)
            total_pixels.append(float(record.total_pixels))
            peak_counts.append(float(len(peak_scores) / denominator_mp))

        pixel_survival_array = np.log10(np.asarray(pixel_survival) + 1e-12)
        pixel_quantile_array = np.asarray(pixel_quantiles)
        peak_survival_array = np.log10(np.asarray(peak_survival_per_mp) + 1e-6)
        peak_quantile_array = np.asarray(peak_quantiles)
        image_stat_array = np.asarray(image_statistics)

        features = np.concatenate([
            self._mean_std(pixel_survival_array),
            self._mean_std(pixel_quantile_array),
            self._mean_std(peak_survival_array),
            self._mean_std(peak_quantile_array),
            self._mean_std(image_stat_array),
            np.asarray([
                np.log1p(len(records)),
                np.log1p(np.sum(total_pixels)),
                np.mean(np.log1p(total_pixels)),
                np.std(np.log1p(total_pixels)),
                np.mean(np.log1p(peak_counts)),
                np.std(np.log1p(peak_counts)),
            ], dtype=np.float32),
        ]).astype(np.float32)

        names: list[str] = []
        for prefix, base_names in (
            ("pixel_survival_log10", [f"t{value:g}" for value in config.survival_thresholds]),
            ("pixel_quantile", [f"q{value:g}" for value in config.quantiles]),
            ("peak_survival_per_mp_log10", [f"t{value:g}" for value in config.survival_thresholds]),
            ("peak_quantile", [f"q{value:g}" for value in config.quantiles]),
            ("image", list(expected_stat_names)),
        ):
            names.extend([f"{prefix}_{name}_mean" for name in base_names])
            names.extend([f"{prefix}_{name}_std" for name in base_names])
        names.extend([
            "window_log1p_num_images",
            "window_log1p_total_pixels",
            "image_log1p_pixels_mean",
            "image_log1p_pixels_std",
            "peak_log1p_per_mp_mean",
            "peak_log1p_per_mp_std",
        ])
        if len(names) != len(features):
            raise RuntimeError(f"Feature name mismatch: {len(names)} != {len(features)}")
        if not np.isfinite(features).all():
            raise ValueError("Extracted features contain NaN or infinity")
        return features, tuple(names)
````

### 3.40 `rc_irstd/losses/__init__.py`

- SHA-256：`67c16361cf3c79f837707ef1ea4b6c562cc3eab9e7d3d388f808753bb432fd4f`
- 行数：`16`

````python
from rc_irstd.losses.cvar import smooth_upper_max, smooth_worst_group, upper_cvar
from rc_irstd.losses.quantile import budget_focused_weight, crossing_loss, pinball_loss
from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss
from rc_irstd.losses.sls import SLSIoULoss, location_loss

__all__ = [
    "RiskAwareDetectorLoss",
    "SLSIoULoss",
    "budget_focused_weight",
    "crossing_loss",
    "location_loss",
    "pinball_loss",
    "smooth_upper_max",
    "smooth_worst_group",
    "upper_cvar",
]
````

### 3.41 `rc_irstd/losses/cvar.py`

- SHA-256：`3f48f39ab11adba85e5bc36f867300ab7fb4e257784f78fbdaa99b59e916473b`
- 行数：`47`

````python
from __future__ import annotations

import math

import torch


def upper_cvar(values: torch.Tensor, quantile: float = 0.95) -> torch.Tensor:
    """Mean of the upper ``1-quantile`` fraction, preserving gradients."""
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must be in [0, 1)")
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    count = max(1, int(math.ceil((1.0 - quantile) * flat.numel())))
    return torch.topk(flat, k=count, largest=True, sorted=False).values.mean()


def smooth_upper_max(group_risks: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Log-sum-exp upper approximation of the maximum group risk."""
    if group_risks.numel() == 0:
        raise ValueError("group_risks must not be empty")
    if gamma <= 0:
        return group_risks.mean()
    return torch.logsumexp(gamma * group_risks, dim=0) / gamma


def normalized_log_mean_exp(
    group_risks: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Normalised log-mean-exp; useful when an upper bound is not required."""
    if group_risks.numel() == 0:
        raise ValueError("group_risks must not be empty")
    if gamma <= 0:
        return group_risks.mean()
    count = torch.as_tensor(
        float(group_risks.numel()),
        dtype=group_risks.dtype,
        device=group_risks.device,
    )
    return (torch.logsumexp(gamma * group_risks, dim=0) - torch.log(count)) / gamma


def smooth_worst_group(group_risks: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Backward-compatible name for the smooth upper maximum."""
    return smooth_upper_max(group_risks, gamma=gamma)
````

### 3.42 `rc_irstd/losses/quantile.py`

- SHA-256：`152815acbe73f965e500edf190362bd89ee116fe8a0690345e29c9ea383a9156`
- 行数：`70`

````python
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.9,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    error = target - prediction
    loss = torch.maximum(quantile * error, (quantile - 1.0) * error)
    if weight is not None:
        broadcast = torch.broadcast_to(weight, loss.shape)
        denominator = broadcast.sum().clamp_min(1e-12)
        return (loss * broadcast).sum() / denominator
    return loss.mean()


def budget_focused_weight(
    target_log_risk: torch.Tensor,
    budget: float,
    base_weight: float = 1.0,
    focus_weight: float = 4.0,
    log_scale: float = 1.0,
    empty_action_weight: float = 0.1,
) -> torch.Tensor:
    """Emphasise risk-curve points near the deployment budget crossing."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if base_weight < 0 or focus_weight < 0 or log_scale <= 0:
        raise ValueError("Invalid budget weighting parameters")
    log_budget = math.log10(float(budget))
    weight = base_weight + focus_weight * torch.exp(
        -torch.abs(target_log_risk - log_budget) / log_scale
    )
    if weight.shape[-1] > 0:
        weight = weight.clone()
        weight[..., -1] = weight[..., -1] * float(empty_action_weight)
    return weight


def crossing_loss(
    prediction_log_risk: torch.Tensor,
    target_log_risk: torch.Tensor,
    budget: float,
    temperature: float = 0.25,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary loss for predicting which thresholds satisfy a risk budget."""
    if prediction_log_risk.shape != target_log_risk.shape:
        raise ValueError("Prediction and target curves must have the same shape")
    if budget <= 0 or temperature <= 0:
        raise ValueError("budget and temperature must be positive")
    log_budget = math.log10(float(budget))
    safe_target = (target_log_risk <= log_budget).to(prediction_log_risk.dtype)
    safe_logits = (log_budget - prediction_log_risk) / temperature
    loss = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    if weight is not None:
        broadcast = torch.broadcast_to(weight, loss.shape)
        return (loss * broadcast).sum() / broadcast.sum().clamp_min(1e-12)
    return loss.mean()
````

### 3.43 `rc_irstd/losses/risk_aware.py`

- SHA-256：`fa7297770c547db03bf351c62b1d75d7583750daf8ef5e27ca8bae382ea3e202`
- 行数：`234`

````python
from __future__ import annotations

from collections.abc import Callable
import inspect

import numpy as np
import torch
from scipy import ndimage
from torch import nn
import torch.nn.functional as F

from rc_irstd.losses.cvar import smooth_worst_group, upper_cvar


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def fallback_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


def differentiable_local_peak_mask(
    probability: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer")
    pooled = F.max_pool2d(probability, kernel_size, stride=1, padding=kernel_size // 2)
    return probability >= pooled


def background_peak_cvar_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
    quantile: float = 0.95,
    kernel_size: int = 5,
    exclusion_radius: int = 2,
    gamma: float = 10.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if exclusion_radius > 0:
        kernel = 2 * exclusion_radius + 1
        excluded = F.max_pool2d(target, kernel, stride=1, padding=exclusion_radius) > 0
    else:
        excluded = target > 0
    peak_mask = differentiable_local_peak_mask(probability, kernel_size) & (~excluded)

    domain_risks: list[torch.Tensor] = []
    for domain in torch.unique(domain_ids):
        member_indices = torch.nonzero(domain_ids == domain, as_tuple=False).flatten()
        image_risks: list[torch.Tensor] = []
        for image_index in member_indices:
            values = probability[image_index][peak_mask[image_index]]
            if values.numel() == 0:
                # Fall back to valid background values for this image only. This
                # keeps candidate-rich images from dominating the entire domain.
                values = probability[image_index][~excluded[image_index]]
            if values.numel() == 0:
                image_risks.append(probability[image_index].sum() * 0.0)
            else:
                image_risks.append(upper_cvar(values, quantile))
        domain_risks.append(torch.stack(image_risks).mean())
    return smooth_worst_group(torch.stack(domain_risks), gamma=gamma)


def _component_scores(
    probability: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    component_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    if component_labels is not None:
        labels_batch = component_labels.to(device=probability.device, dtype=torch.long)
        if labels_batch.shape != target.shape:
            labels_batch = F.interpolate(
                labels_batch.to(torch.float32),
                size=target.shape[-2:],
                mode="nearest",
            ).to(torch.long)
    else:
        masks = target.detach().cpu().numpy() > 0.5
        label_arrays = [ndimage.label(masks[index, 0])[0] for index in range(target.shape[0])]
        labels_batch = torch.from_numpy(np.stack(label_arrays)[:, None]).to(
            probability.device, dtype=torch.long
        )

    for batch_index in range(target.shape[0]):
        labels = labels_batch[batch_index, 0]
        component_ids = torch.unique(labels)
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            component = labels == component_id
            values = probability[batch_index, 0][component]
            if values.numel() == 0:
                continue
            if temperature <= 0:
                scores.append(values.max())
            else:
                # Normalised LSE pooling remains within a small additive constant
                # of the maximum and sends gradients to the whole target.
                pooled = torch.logsumexp(values * temperature, dim=0) / temperature
                pooled = pooled - torch.log(
                    torch.tensor(float(values.numel()), device=values.device)
                ) / temperature
                scores.append(pooled)
    if not scores:
        return probability.new_empty((0,))
    return torch.stack(scores)


def hard_target_miss_cvar_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.8,
    temperature: float = 10.0,
    component_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    scores = _component_scores(
        probability,
        target,
        temperature,
        component_labels=component_labels,
    )
    if scores.numel() == 0:
        return probability.sum() * 0.0
    return upper_cvar(1.0 - scores, quantile)


class RiskAwareDetectorLoss(nn.Module):
    """SLS/base loss plus worst-domain false-peak and hard-target tails."""

    def __init__(
        self,
        base_loss: Callable[..., torch.Tensor] | None = None,
        lambda_tail: float = 0.1,
        lambda_miss: float = 0.1,
        tail_quantile: float = 0.95,
        miss_quantile: float = 0.8,
        peak_kernel: int = 5,
        exclusion_radius: int = 2,
        worst_gamma: float = 10.0,
        auxiliary_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss or fallback_segmentation_loss
        self.lambda_tail = lambda_tail
        self.lambda_miss = lambda_miss
        self.tail_quantile = tail_quantile
        self.miss_quantile = miss_quantile
        self.peak_kernel = peak_kernel
        self.exclusion_radius = exclusion_radius
        self.worst_gamma = worst_gamma
        self.auxiliary_weight = float(auxiliary_weight)
        try:
            signature = inspect.signature(self.base_loss.forward)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            signature = inspect.signature(self.base_loss)
        parameter_names = set(signature.parameters)
        self._base_accepts_schedule = "warm_epoch" in parameter_names or len(parameter_names) >= 4

    def _call_base(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int,
        epoch: int,
    ) -> torch.Tensor:
        if self._base_accepts_schedule:
            return self.base_loss(logits, target, warm_epoch, epoch)
        return self.base_loss(logits, target)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        domain_ids: torch.Tensor,
        auxiliary_logits: list[torch.Tensor] | None = None,
        component_labels: torch.Tensor | None = None,
        warm_epoch: int = 0,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        final_base = self._call_base(logits, target, warm_epoch, epoch)
        # The reference MSHNet path supervises the final map and every
        # multi-scale auxiliary map, then averages those SLS terms. Adaptive
        # max pooling preserves tiny positive masks when matching an auxiliary
        # resolution and is equivalent to the original repeated max-pooling
        # path for the usual integer scale factors.
        if auxiliary_logits and self.auxiliary_weight > 0.0:
            auxiliary_terms: list[torch.Tensor] = []
            for auxiliary in auxiliary_logits:
                scaled_target = F.adaptive_max_pool2d(
                    target, output_size=auxiliary.shape[-2:]
                )
                auxiliary_terms.append(
                    self._call_base(auxiliary, scaled_target, warm_epoch, epoch)
                )
            auxiliary_sum = torch.stack(auxiliary_terms).sum()
            denominator = 1.0 + self.auxiliary_weight * len(auxiliary_terms)
            base = (
                final_base + self.auxiliary_weight * auxiliary_sum
            ) / denominator
        else:
            base = final_base

        if self.lambda_tail != 0.0:
            tail = background_peak_cvar_loss(
                logits,
                target,
                domain_ids,
                quantile=self.tail_quantile,
                kernel_size=self.peak_kernel,
                exclusion_radius=self.exclusion_radius,
                gamma=self.worst_gamma,
            )
        else:
            tail = logits.sum() * 0.0
        if self.lambda_miss != 0.0:
            miss = hard_target_miss_cvar_loss(
                logits,
                target,
                quantile=self.miss_quantile,
                component_labels=component_labels,
            )
        else:
            miss = logits.sum() * 0.0
        total = base + self.lambda_tail * tail + self.lambda_miss * miss
        return {"total": total, "base": base, "tail": tail, "miss": miss}
````

### 3.44 `rc_irstd/losses/sls.py`

- SHA-256：`8c173465ff2a371635d2ec833219f73402e5340f66ab2ed80ce81a2d45b65b97`
- 行数：`90`

````python
from __future__ import annotations

"""Numerically robust internal implementation of MSHNet's SLS-IoU loss."""

import torch
from torch import nn
import torch.nn.functional as F


def location_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if probability.shape != target.shape:
        raise ValueError("probability and target must have identical shapes")
    batch, _, height, width = probability.shape
    dtype = probability.dtype
    device = probability.device
    x_index = (
        torch.arange(width, device=device, dtype=dtype)[None, None, :]
        .expand(1, height, width)
        / max(width, 1)
    )
    y_index = (
        torch.arange(height, device=device, dtype=dtype)[None, :, None]
        .expand(1, height, width)
        / max(height, 1)
    )
    smooth = torch.finfo(dtype).eps
    losses: list[torch.Tensor] = []
    for index in range(batch):
        pred_map = probability[index]
        target_map = target[index]
        pred_centerx = (x_index * pred_map).mean()
        pred_centery = (y_index * pred_map).mean()
        target_centerx = (x_index * target_map).mean()
        target_centery = (y_index * target_map).mean()
        pred_angle = torch.atan2(pred_centery, pred_centerx + smooth)
        target_angle = torch.atan2(target_centery, target_centerx + smooth)
        angle_loss = (4.0 / (torch.pi**2)) * (pred_angle - target_angle).square()
        pred_length = torch.sqrt(pred_centerx.square() + pred_centery.square() + smooth)
        target_length = torch.sqrt(target_centerx.square() + target_centery.square() + smooth)
        length_similarity = torch.minimum(pred_length, target_length) / (
            torch.maximum(pred_length, target_length) + smooth
        )
        losses.append(1.0 - length_similarity + angle_loss)
    return torch.stack(losses).mean() if losses else probability.sum() * 0.0


class SLSIoULoss(nn.Module):
    """Scale and Location Sensitive IoU loss.

    The interface matches the public MSHNet implementation. Empty-target images
    are handled without NaNs by the epsilon terms; during the warm-up phase this
    reduces to soft IoU, and afterwards the scale factor and optional location
    penalty are enabled.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_log: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int = 1,
        epoch: int = 1,
        with_shape: bool = True,
    ) -> torch.Tensor:
        target = target.to(dtype=pred_log.dtype)
        if target.shape[-2:] != pred_log.shape[-2:]:
            target = F.interpolate(target, size=pred_log.shape[-2:], mode="nearest")
        probability = torch.sigmoid(pred_log)
        intersection_sum = (probability * target).sum(dim=(1, 2, 3))
        pred_sum = probability.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        denominator = pred_sum + target_sum - intersection_sum
        iou = (intersection_sum + self.eps) / (denominator + self.eps)
        if epoch <= warm_epoch:
            return 1.0 - iou.mean()

        distance = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + distance + self.eps) / (
            torch.maximum(pred_sum, target_sum) + distance + self.eps
        )
        loss = 1.0 - (alpha * iou).mean()
        if with_shape:
            # For all-background images the location term is not meaningful.
            nonempty = target_sum > 0
            if torch.any(nonempty):
                loss = loss + location_loss(probability[nonempty], target[nonempty])
        return loss
````

### 3.45 `rc_irstd/models/__init__.py`

- SHA-256：`568e0eb088f7992900dd0f821f358267ded5c92047582a8b82282dd43f43c670`
- 行数：`15`

````python
from rc_irstd.models.detector_adapter import DetectorAdapter, DetectorOutput, build_detector
from rc_irstd.models.mshnet import MSHNet, MSHNetFeatures
from rc_irstd.models.risk_curve import FeatureNormaliser, RiskCurvePredictor
from rc_irstd.models.tiny_detector import TinyUNet

__all__ = [
    "DetectorAdapter",
    "DetectorOutput",
    "FeatureNormaliser",
    "MSHNet",
    "MSHNetFeatures",
    "RiskCurvePredictor",
    "TinyUNet",
    "build_detector",
]
````

### 3.46 `rc_irstd/models/detector_adapter.py`

- SHA-256：`bdb8b9e0e815649d53ce0c3f8573d0b502dd3426a4ef79642170f62273658a19`
- 行数：`126`

````python
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rc_irstd.models.mshnet import MSHNet
from rc_irstd.models.tiny_detector import TinyUNet
from rc_irstd.utils.io import normalise_state_dict


@dataclass
class DetectorOutput:
    logits: torch.Tensor
    auxiliary_logits: list[torch.Tensor]


class DetectorAdapter(nn.Module):
    """Normalise heterogeneous detector APIs to final and auxiliary logits.

    Forward dispatch is determined once from the signature. This avoids masking
    genuine model-internal ``TypeError`` exceptions, which the previous generic
    try/except implementation could silently reinterpret as an API mismatch.
    """

    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.model = model
        self.name = name
        parameters = inspect.signature(model.forward).parameters
        if "warm_flag" in parameters:
            self.forward_mode = "warm_flag"
        elif "training_tag" in parameters:
            self.forward_mode = "training_tag"
        else:
            self.forward_mode = "plain"

    def forward(self, x: torch.Tensor, training_tag: bool = True) -> DetectorOutput:
        if self.forward_mode == "warm_flag":
            raw = self.model(x, warm_flag=training_tag)
        elif self.forward_mode == "training_tag":
            raw = self.model(x, training_tag=training_tag)
        else:
            raw = self.model(x)

        if isinstance(raw, torch.Tensor):
            return DetectorOutput(raw, [])
        if isinstance(raw, dict):
            final = raw.get("logits", raw.get("pred", raw.get("out")))
            if final is None:
                raise KeyError("Detector dictionary must contain logits, pred or out")
            auxiliary = raw.get("auxiliary_logits", raw.get("aux", []))
            return DetectorOutput(final, list(auxiliary))
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2 and isinstance(raw[0], (tuple, list)):
                auxiliary, final = raw[0], raw[1]
                return DetectorOutput(final, list(auxiliary))
            tensors = [item for item in raw if isinstance(item, torch.Tensor)]
            if not tensors:
                raise TypeError("Detector returned a tuple without tensors")
            return DetectorOutput(tensors[-1], tensors[:-1])
        raise TypeError(f"Unsupported detector output type: {type(raw).__name__}")


def _external_mshnet(in_channels: int) -> nn.Module:
    try:
        module = importlib.import_module("model.MSHNet")
        constructor = getattr(module, "MSHNet")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "mshnet_external was requested, but model.MSHNet.MSHNet is unavailable. "
            "Use --detector mshnet for the bundled implementation or add the "
            "external MSHNet root to PYTHONPATH."
        ) from exc
    return constructor(in_channels)


def build_detector(
    name: str,
    in_channels: int = 3,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> DetectorAdapter:
    normalised = name.lower().replace("-", "_")
    if normalised in {"tiny", "tiny_unet"}:
        model: nn.Module = TinyUNet(in_channels=in_channels)
    elif normalised in {"mshnet", "mshnet_internal"}:
        model = MSHNet(input_channels=in_channels)
        normalised = "mshnet"
    elif normalised == "mshnet_external":
        model = _external_mshnet(in_channels)
    else:
        raise ValueError(f"Unknown detector '{name}'")

    adapter = DetectorAdapter(model, normalised)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = normalise_state_dict(payload)
        attempts = [
            state,
            {key.removeprefix("model."): value for key, value in state.items()},
            {key.removeprefix("module."): value for key, value in state.items()},
        ]
        last_error: RuntimeError | None = None
        for candidate in attempts:
            try:
                adapter.model.load_state_dict(candidate, strict=strict)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    return adapter.to(device)


def resize_logits(logits: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(logits.shape[-2:]) == tuple(target_hw):
        return logits
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
````

### 3.47 `rc_irstd/models/mshnet.py`

- SHA-256：`17908d8ccb691f7e8f40b9457ad88be58bcb1ed3dc41c754a296a54498d6e283`
- 行数：`196`

````python
from __future__ import annotations

"""Self-contained MSHNet implementation used by RC-IRSTD.

The module mirrors the public CVPR 2024 MSHNet parameter names so that official
and common fork checkpoints can be loaded without requiring a second repository.
RC-IRSTD wraps the network through :mod:`rc_irstd.models.detector_adapter` and
keeps the original warm-up API: before ``warm_flag`` is enabled, only the
full-resolution head is used; afterwards four decoder heads are fused.
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(int(in_planes) // int(ratio), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("SpatialAttention kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)


@dataclass(frozen=True)
class MSHNetFeatures:
    decoder_0: torch.Tensor
    decoder_1: torch.Tensor
    decoder_2: torch.Tensor
    decoder_3: torch.Tensor
    middle: torch.Tensor


class MSHNet(nn.Module):
    """Multi-Scale Head Network with official checkpoint-compatible names."""

    def __init__(
        self,
        input_channels: int = 3,
        block: type[nn.Module] = ResNet,
        channels: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256),
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(channels) != 5 or len(blocks) != 4:
            raise ValueError("channels must contain 5 entries and blocks 4 entries")
        c0, c1, c2, c3, c4 = [int(value) for value in channels]
        b0, b1, b2, b3 = [int(value) for value in blocks]
        self.input_channels = int(input_channels)
        self.channels = (c0, c1, c2, c3, c4)
        self.blocks = (b0, b1, b2, b3)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(self.input_channels, c0, 1, 1)
        self.encoder_0 = self._make_layer(c0, c0, block)
        self.encoder_1 = self._make_layer(c0, c1, block, b0)
        self.encoder_2 = self._make_layer(c1, c2, block, b1)
        self.encoder_3 = self._make_layer(c2, c3, block, b2)
        self.middle_layer = self._make_layer(c3, c4, block, b3)
        self.decoder_3 = self._make_layer(c3 + c4, c3, block, b2)
        self.decoder_2 = self._make_layer(c2 + c3, c2, block, b1)
        self.decoder_1 = self._make_layer(c1 + c2, c1, block, b0)
        self.decoder_0 = self._make_layer(c0 + c1, c0, block)
        self.output_0 = nn.Conv2d(c0, 1, 1)
        self.output_1 = nn.Conv2d(c1, 1, 1)
        self.output_2 = nn.Conv2d(c2, 1, 1)
        self.output_3 = nn.Conv2d(c3, 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        block: type[nn.Module],
        block_num: int = 1,
    ) -> nn.Sequential:
        if block_num < 1:
            raise ValueError("block_num must be positive")
        layers: list[nn.Module] = [block(in_channels, out_channels)]
        layers.extend(block(out_channels, out_channels) for _ in range(block_num - 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=True)

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_feature: bool = False,
    ):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self._resize_like(x_m, x_e3)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self._resize_like(x_d3, x_e2)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self._resize_like(x_d2, x_e1)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self._resize_like(x_d1, x_e0)], dim=1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(
                torch.cat(
                    [
                        mask0,
                        self._resize_like(mask1, mask0),
                        self._resize_like(mask2, mask0),
                        self._resize_like(mask3, mask0),
                    ],
                    dim=1,
                )
            )
            auxiliary = [mask0, mask1, mask2, mask3]
        else:
            auxiliary = []
            output = self.output_0(x_d0)

        if return_feature:
            features = MSHNetFeatures(x_d0, x_d1, x_d2, x_d3, x_m)
            return auxiliary, output, features
        return auxiliary, output

    def export_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "channels": list(self.channels),
            "blocks": list(self.blocks),
        }
````

### 3.48 `rc_irstd/models/risk_curve.py`

- SHA-256：`0cd55e44c741e983f7aea8bd17622d9f7ef74c6a1bac4462f10e100b3adbcfce`
- 行数：`94`

````python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MonotoneLogRiskHead(nn.Module):
    """Produce a non-increasing log-risk curve over ascending thresholds.

    A naive cumulative-softplus head accumulates roughly ``0.69`` at every
    threshold at initialization, which explodes when the grid contains hundreds
    of points. Here a learned positive *total drop* is distributed across the
    threshold intervals by a softmax. This preserves exact monotonicity while
    keeping the initial curve numerically well scaled.
    """

    def __init__(self, hidden_dim: int, num_thresholds: int) -> None:
        super().__init__()
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2")
        self.num_thresholds = num_thresholds
        # start value + total positive drop + interval allocation logits
        self.projection = nn.Linear(hidden_dim, num_thresholds + 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        start = raw[:, :1]
        total_drop = F.softplus(raw[:, 1:2])
        allocation = torch.softmax(raw[:, 2:], dim=1)
        decrements = total_drop * allocation
        tail = start - torch.cumsum(decrements, dim=1)
        return torch.cat([start, tail], dim=1)


class RiskCurvePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_thresholds: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_thresholds = int(num_thresholds)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pixel_head = MonotoneLogRiskHead(hidden_dim, num_thresholds)
        self.peak_head = MonotoneLogRiskHead(hidden_dim, num_thresholds)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "pixel_log_risk": self.pixel_head(hidden),
            "peak_log_risk": self.peak_head(hidden),
        }


@dataclass(frozen=True)
class FeatureNormaliser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, min_std: float = 1e-6) -> "FeatureNormaliser":
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = features.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, min_std)
        return cls(mean, std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureNormaliser":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
        )
````

### 3.49 `rc_irstd/models/risk_io.py`

- SHA-256：`46763b519cf276ae5750429a728b69a9004484061aed478279243bc148463469`
- 行数：`74`

````python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rc_irstd.models.risk_curve import FeatureNormaliser, RiskCurvePredictor


@dataclass(frozen=True)
class LoadedRiskModel:
    model: RiskCurvePredictor
    normaliser: FeatureNormaliser
    thresholds: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def load_risk_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedRiskModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "normaliser", "thresholds", "feature_names", "model_config"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Risk checkpoint is missing fields: {sorted(missing)}")
    config = payload["model_config"]
    model = RiskCurvePredictor(
        input_dim=int(config["input_dim"]),
        num_thresholds=int(config["num_thresholds"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return LoadedRiskModel(
        model=model,
        normaliser=FeatureNormaliser.from_dict(payload["normaliser"]),
        thresholds=np.asarray(payload["thresholds"], dtype=np.float32),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"model", "normaliser", "thresholds", "feature_names"}
        },
    )


def predict_risk_curves(
    loaded: LoadedRiskModel,
    features: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array[None]
    normalised = loaded.normaliser.transform(array)
    outputs: dict[str, list[np.ndarray]] = {
        "pixel_log_risk": [],
        "peak_log_risk": [],
    }
    with torch.inference_mode():
        for start in range(0, len(normalised), batch_size):
            tensor = torch.from_numpy(normalised[start : start + batch_size]).to(device)
            prediction = loaded.model(tensor)
            for key in outputs:
                outputs[key].append(prediction[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}
````

### 3.50 `rc_irstd/models/tiny_detector.py`

- SHA-256：`828736e1a247c06eda9e5f3bb50b9533a70295d8615a752dbf7314009ce2125d`
- 行数：`49`

````python
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """Small fallback model for tests and pipeline validation.

    The research experiments should use MSHNet or another established IRSTD
    backbone. TinyUNet exists so every pipeline can be smoke-tested without the
    external repository.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 2) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bridge = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        bridge = self.bridge(F.max_pool2d(e2, 2))
        d2 = F.interpolate(bridge, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
````

### 3.51 `rc_irstd/pipelines/__init__.py`

- SHA-256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- 行数：`0`

````python

````

### 3.52 `rc_irstd/pipelines/aggregate_results.py`

- SHA-256：`9a9476f764e5fe96f41c1694ecb486670e9a5a57b3e8e1a7acd81a1b16a3471a`
- 行数：`170`

````python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rc_irstd.utils.io import atomic_json_dump, ensure_dir


METRICS = [
    "joint_bsr",
    "pixel_bsr",
    "peak_bsr",
    "pixel_excess",
    "peak_excess",
    "mean_pd_selected",
    "effective_pd_with_rejects",
    "conditional_pd_non_rejected",
    "worst_domain_pd_selected",
    "rejection_rate",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate nested-LODO outputs into paper-ready CSV and Markdown tables."
    )
    parser.add_argument("--lodo-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _outer_name(path: Path) -> str:
    return path.name.removeprefix("outer_")


def _zero_rows(outer_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "zero_label" / "summary.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        selected = payload.get("metrics", {}).get("selected", {})
        rows.append(
            {
                "outer_target": _outer_name(outer_dir),
                "method": "risk_curve_zero_label",
                **selected,
                "pixel_log_mae": payload.get("metrics", {}).get("pixel_log_mae"),
                "peak_log_mae": payload.get("metrics", {}).get("peak_log_mae"),
                "joint_pointwise_coverage": payload.get("metrics", {}).get(
                    "joint_pointwise_coverage"
                ),
            }
        )
    return rows


def _baseline_rows(outer_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "baselines" / "summary.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        for method, metrics in payload.get("methods", {}).items():
            rows.append(
                {
                    "outer_target": _outer_name(outer_dir),
                    "method": method,
                    **metrics,
                }
            )
    return rows


def _crc_rows(outer_dirs: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "few_shot_crc" / "results.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "outer_target", _outer_name(outer_dir))
        frames.append(frame)
    return frames


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.lodo_root).expanduser().resolve()
    output = ensure_dir(
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "paper_tables"
    )
    outer_dirs = sorted(path for path in root.glob("outer_*") if path.is_dir())
    if not outer_dirs:
        raise FileNotFoundError(f"No outer_* directories found under {root}")

    manifest: dict[str, Any] = {
        "lodo_root": str(root),
        "outer_targets": [_outer_name(path) for path in outer_dirs],
        "files": {},
    }

    zero = pd.DataFrame(_zero_rows(outer_dirs))
    if not zero.empty:
        zero_path = output / "zero_label_by_domain.csv"
        zero.to_csv(zero_path, index=False)
        summary = zero.groupby("method", dropna=False)[METRICS].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = output / "zero_label_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "zero_label_summary.md")
        manifest["files"]["zero_label"] = [str(zero_path), str(summary_path)]

    baselines = pd.DataFrame(_baseline_rows(outer_dirs))
    if not baselines.empty:
        baseline_path = output / "baselines_by_domain.csv"
        baselines.to_csv(baseline_path, index=False)
        summary = baselines.groupby("method", dropna=False)[METRICS].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = output / "baselines_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "baselines_summary.md")
        manifest["files"]["baselines"] = [str(baseline_path), str(summary_path)]

    crc_frames = _crc_rows(outer_dirs)
    if crc_frames:
        crc = pd.concat(crc_frames, ignore_index=True)
        crc_path = output / "few_shot_crc_all_runs.csv"
        crc.to_csv(crc_path, index=False)
        group_keys = ["method", "calibration_size"]
        available_metrics = [metric for metric in METRICS if metric in crc.columns]
        summary = crc.groupby(group_keys, dropna=False)[available_metrics].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        feasibility = (
            crc.groupby(group_keys, dropna=False)["formal_crc_feasible"]
            .mean()
            .reset_index(name="formal_feasible_fraction")
        )
        summary = summary.merge(feasibility, on=group_keys, how="left")
        summary_path = output / "few_shot_crc_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "few_shot_crc_summary.md")
        manifest["files"]["few_shot_crc"] = [str(crc_path), str(summary_path)]

    atomic_json_dump(manifest, output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.53 `rc_irstd/pipelines/apply_operating_point.py`

- SHA-256：`02145a65b3fc47f17518cfb2dc603b4f22b25fcd6d3053ae15485c673d18b6fc`
- 行数：`89`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import load_score_record
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a fixed operating point to score maps.")
    parser.add_argument("--score-dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--threshold", type=float)
    group.add_argument("--threshold-json")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser


def _threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    payload = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
    if "threshold" not in payload:
        raise KeyError("threshold-json does not contain a threshold field")
    return float(payload["threshold"])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    threshold = _threshold(args)
    output_dir = ensure_dir(args.output_dir)
    mask_root = ensure_dir(output_dir / "masks")
    candidate_path = output_dir / "candidates.csv"
    records = [load_score_record(path, require_mask=False) for path in list_npz(args.score_dir)]
    records.sort(key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    selected = records[max(args.skip_first, 0) :]
    candidate_rows: list[dict[str, object]] = []
    for record in selected:
        binary = (record.probability >= threshold).astype(np.uint8) * 255
        mask_path = mask_root / f"{record.image_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(mask_path)
        scores, ys, xs = extract_fixed_peaks(
            record.probability,
            min_distance=args.peak_min_distance,
            min_score=max(0.0, threshold),
        )
        for score, y, x in zip(scores, ys, xs, strict=True):
            if score < threshold:
                continue
            candidate_rows.append(
                {
                    "image_id": record.image_id,
                    "sequence_id": record.sequence_id,
                    "frame_index": record.frame_index,
                    "y": int(y),
                    "x": int(x),
                    "score": float(score),
                    "threshold": threshold,
                }
            )
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["image_id", "sequence_id", "frame_index", "y", "x", "score", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    summary = {
        "threshold": threshold,
        "skip_first": args.skip_first,
        "num_processed_images": len(selected),
        "num_candidates": len(candidate_rows),
        "mask_directory": str(mask_root.resolve()),
        "candidate_csv": str(candidate_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.54 `rc_irstd/pipelines/build_episodes.py`

- SHA-256：`b7ad14f582dc65f64d93bcded9c244798170ea4c422b5ccb46045da6f623a786`
- 行数：`72`

````python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from rc_irstd.episodes.builder import EpisodeBuildConfig, build_episode_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build disjoint support-to-query risk-curve episodes."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-grid", default=None)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--protocol",
        choices=["auto", "iid", "temporal"],
        default="auto",
        help="IID is for unordered static images; temporal is prefix-to-future.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--peak-min-score", type=float, default=0.0)
    parser.add_argument("--peak-border", type=int, default=0)
    parser.add_argument("--peak-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum fixed peaks per image; 0 disables truncation.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_candidates < 0:
        raise ValueError("max-candidates must be non-negative")
    thresholds = np.load(args.threshold_grid) if args.threshold_grid else None
    config = EpisodeBuildConfig(
        context_size=args.context_size,
        horizon=args.horizon,
        stride=args.stride,
        protocol=args.protocol,
        seed=args.seed,
        peak_min_distance=args.peak_min_distance,
        peak_min_score=args.peak_min_score,
        peak_border=args.peak_border,
        peak_tolerance=args.peak_tolerance,
        max_candidates_per_image=(None if args.max_candidates <= 0 else args.max_candidates),
    )
    path = build_episode_file(
        args.score_dir, args.output, thresholds=thresholds, config=config
    )
    print(
        json.dumps(
            {"output": str(path.resolve()), "config": asdict(config)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
````

### 3.55 `rc_irstd/pipelines/build_supplement.py`

- SHA-256：`7acee767190fff6f06432b7d78f871bdfac946e6401c10434c9373526ae98d6a`
- 行数：`190`

````python
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
````

### 3.56 `rc_irstd/pipelines/calibrate_and_evaluate.py`

- SHA-256：`ba0fad21851d046d3d754629f3516d635292d05fcc4b7ac135f4e27b21e493bb`
- 行数：`378`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rc_irstd.calibration.crc import (
    adaptive_offset_loss_matrix,
    minimum_calibration_size,
    raw_global_threshold_loss_matrix,
    select_crc_parameter,
)
from rc_irstd.calibration.samples import (
    CalibrationSamples,
    episode_calibration_samples,
    image_calibration_samples,
    split_calibration_samples,
)
from rc_irstd.episodes.dataset import concatenate_episode_files
from rc_irstd.episodes.splits import grouped_calibration_test_split
from rc_irstd.evaluation.budget import summarise_selected_points
from rc_irstd.evaluation.risk_curve_metrics import select_indices_from_predictions
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate nested threshold offsets with explicit statistical units."
    )
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--calibration-sizes", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--calibration-unit",
        choices=["episode", "image"],
        default="episode",
        help="image means the requested size is exactly the number of labelled images.",
    )
    parser.add_argument("--offset-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Debug only: allow episode calibration/test image overlap.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def _episode_ids(arrays, indices: np.ndarray) -> set[str]:
    ids: set[str] = set()
    for index in indices:
        for field in (arrays.context_ids[int(index)], arrays.future_ids[int(index)]):
            values = json.loads(str(field))
            ids.update(str(value) for value in values)
    return ids


def _summary_at_indices(
    samples: CalibrationSamples,
    sample_indices: np.ndarray,
    threshold_indices: np.ndarray,
    rejected: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
):
    rows = np.arange(len(sample_indices))
    return summarise_selected_points(
        samples.pixel_risk[sample_indices][rows, threshold_indices],
        samples.peak_risk[sample_indices][rows, threshold_indices],
        samples.pd[sample_indices][rows, threshold_indices],
        rejected,
        samples.domains[sample_indices],
        pixel_budget,
        peak_budget,
    )


def _empirical_offset(
    losses: np.ndarray, offsets: np.ndarray, alpha: float
) -> tuple[int, bool]:
    empirical = losses.mean(axis=0)
    feasible = np.flatnonzero(empirical <= alpha)
    if len(feasible):
        return int(offsets[int(feasible[0])]), True
    return int(offsets[-1]), False


def _split(
    args: argparse.Namespace,
    arrays,
    samples: CalibrationSamples,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.calibration_unit == "episode":
        calibration, test = grouped_calibration_test_split(
            arrays, calibration_size, seed
        )
        overlap = _episode_ids(arrays, calibration).intersection(
            _episode_ids(arrays, test)
        )
        if overlap and not args.allow_overlap:
            raise ValueError(
                f"Episode split shares {len(overlap)} image IDs. Build evaluation "
                "episodes with non-overlapping windows or use --allow-overlap only for smoke/debug."
            )
        return calibration, test, {
            "strategy": "independent_episode_groups",
            "overlapping_image_ids": sorted(overlap),
            "allow_overlap": bool(args.allow_overlap),
        }
    calibration, test, metadata = split_calibration_samples(
        samples, calibration_size, seed
    )
    return calibration, test, metadata


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.offset_step <= 0:
        raise ValueError("offset-step must be positive")
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("budgets must be positive")
    device = resolve_device(args.device)
    arrays = concatenate_episode_files(args.episode)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    if not np.array_equal(arrays.thresholds, loaded.thresholds):
        raise ValueError("Episode threshold grid differs from risk-model checkpoint")
    if arrays.feature_names != loaded.feature_names:
        raise ValueError("Episode feature schema differs from risk-model checkpoint")

    predictions = predict_risk_curves(
        loaded, arrays.features, device=device, batch_size=args.batch_size
    )
    episode_base_indices, episode_base_rejected = select_indices_from_predictions(
        arrays.thresholds,
        predictions["pixel_log_risk"],
        predictions["peak_log_risk"],
        args.pixel_budget,
        args.peak_budget,
    )
    if args.calibration_unit == "image":
        samples = image_calibration_samples(
            arrays, episode_base_indices, episode_base_rejected
        )
    else:
        samples = episode_calibration_samples(
            arrays, episode_base_indices, episode_base_rejected
        )

    num_thresholds = len(arrays.thresholds)
    offsets = np.arange(0, num_thresholds, args.offset_step, dtype=np.int64)
    if offsets[-1] != num_thresholds - 1:
        offsets = np.append(offsets, num_thresholds - 1)
    threshold_parameters = np.arange(num_thresholds, dtype=np.int64)

    output_dir = ensure_dir(args.output_dir)
    all_results: list[dict[str, Any]] = []
    split_records: dict[str, Any] = {}

    for calibration_size in args.calibration_sizes:
        for seed in args.seeds:
            calibration, test, split_metadata = _split(
                args, arrays, samples, calibration_size, seed
            )
            num_labeled_images = int(samples.label_count_per_sample[calibration].sum())
            if args.calibration_unit == "image" and num_labeled_images != calibration_size:
                raise RuntimeError(
                    "Image calibration count mismatch: requested "
                    f"{calibration_size}, obtained {num_labeled_images}"
                )
            split_key = f"{samples.unit}_m{calibration_size}_seed{seed}"
            split_records[split_key] = {
                "calibration_indices": calibration.tolist(),
                "test_indices": test.tolist(),
                "calibration_sample_ids": samples.sample_ids[calibration].tolist(),
                "test_sample_ids": samples.sample_ids[test].tolist(),
                "num_calibration_samples": int(len(calibration)),
                "num_labeled_images": num_labeled_images,
                **split_metadata,
            }

            adaptive_losses, _ = adaptive_offset_loss_matrix(
                samples.pixel_risk[calibration],
                samples.peak_risk[calibration],
                samples.base_indices[calibration],
                offsets,
                args.pixel_budget,
                args.peak_budget,
            )
            adaptive_crc = select_crc_parameter(adaptive_losses, offsets, args.alpha)
            adaptive_test_indices = np.minimum(
                samples.base_indices[test] + adaptive_crc.selected_parameter,
                num_thresholds - 1,
            )
            adaptive_rejected = samples.base_rejected[test] | (
                arrays.thresholds[adaptive_test_indices] > 1.0
            )
            adaptive_summary = _summary_at_indices(
                samples,
                test,
                adaptive_test_indices,
                adaptive_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            common = {
                "calibration_unit": samples.unit,
                "calibration_size": calibration_size,
                "num_calibration_samples": int(len(calibration)),
                "num_labeled_images": num_labeled_images,
                "num_test_samples": int(len(test)),
                "seed": seed,
            }
            all_results.append(
                {
                    "method": "adaptive_risk_curve_crc_offset",
                    **common,
                    "formal_crc_feasible": adaptive_crc.feasible,
                    "selected_parameter": adaptive_crc.selected_parameter,
                    "selected_threshold": None,
                    "crc": adaptive_crc.to_dict(),
                    **adaptive_summary.to_dict(),
                }
            )

            empirical_offset, empirical_feasible = _empirical_offset(
                adaptive_losses, offsets, args.alpha
            )
            empirical_test_indices = np.minimum(
                samples.base_indices[test] + empirical_offset, num_thresholds - 1
            )
            empirical_rejected = samples.base_rejected[test] | (
                arrays.thresholds[empirical_test_indices] > 1.0
            )
            empirical_summary = _summary_at_indices(
                samples,
                test,
                empirical_test_indices,
                empirical_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "adaptive_risk_curve_empirical_offset",
                    **common,
                    "formal_crc_feasible": False,
                    "empirical_feasible": empirical_feasible,
                    "selected_parameter": empirical_offset,
                    "selected_threshold": None,
                    **empirical_summary.to_dict(),
                }
            )

            raw_losses = raw_global_threshold_loss_matrix(
                samples.pixel_risk[calibration],
                samples.peak_risk[calibration],
                args.pixel_budget,
                args.peak_budget,
            )
            raw_crc = select_crc_parameter(
                raw_losses, threshold_parameters, args.alpha
            )
            raw_test_indices = np.full(
                len(test), raw_crc.selected_parameter, dtype=np.int64
            )
            raw_rejected = arrays.thresholds[raw_test_indices] > 1.0
            raw_summary = _summary_at_indices(
                samples,
                test,
                raw_test_indices,
                raw_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "raw_global_threshold_crc",
                    **common,
                    "formal_crc_feasible": raw_crc.feasible,
                    "selected_parameter": raw_crc.selected_parameter,
                    "selected_threshold": float(
                        arrays.thresholds[raw_crc.selected_parameter]
                    ),
                    "crc": raw_crc.to_dict(),
                    **raw_summary.to_dict(),
                }
            )

            zero_summary = _summary_at_indices(
                samples,
                test,
                samples.base_indices[test],
                samples.base_rejected[test],
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "zero_label_no_calibration",
                    **common,
                    "formal_crc_feasible": False,
                    "selected_parameter": 0,
                    "selected_threshold": None,
                    **zero_summary.to_dict(),
                }
            )

    result_path = output_dir / "results.csv"
    flat_keys = [
        "method",
        "calibration_unit",
        "calibration_size",
        "num_calibration_samples",
        "num_labeled_images",
        "num_test_samples",
        "seed",
        "formal_crc_feasible",
        "selected_parameter",
        "selected_threshold",
        "joint_bsr",
        "pixel_bsr",
        "peak_bsr",
        "pixel_excess",
        "peak_excess",
        "mean_pd_selected",
        "effective_pd_with_rejects",
        "conditional_pd_non_rejected",
        "worst_domain_pd_selected",
        "rejection_rate",
        "count",
    ]
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        for result in all_results:
            writer.writerow({key: result.get(key) for key in flat_keys})

    summary = {
        "risk_control_target": "marginal probability of violating pixel or fixed-peak budget",
        "loss": "binary joint budget violation",
        "alpha": args.alpha,
        "minimum_calibration_samples_for_any_crc_solution": minimum_calibration_size(args.alpha),
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "calibration_unit": samples.unit,
        "counting_rule": (
            "calibration_size is exactly labelled images"
            if samples.unit == "image"
            else "calibration_size is labelled future blocks; num_labeled_images is also reported"
        ),
        "exchangeability_requirement": (
            "CRC requires exchangeability at the declared statistical unit. IID image mode "
            "uses unique images; temporal/block mode must be interpreted at the blocked unit."
        ),
        "results": all_results,
        "results_csv": str(result_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    atomic_json_dump(split_records, output_dir / "splits.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.57 `rc_irstd/pipelines/evaluate_baselines.py`

- SHA-256：`55cb41bc0a03c32f89cefc3be6d330958eb27fe7724a24817152c7f35e56bbda`
- 行数：`325`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays, concatenate_episode_files
from rc_irstd.evaluation.budget import summarise_selected_points
from rc_irstd.models.risk_curve import FeatureNormaliser
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed, source-derived, nearest-source, all-detection "
            "upper-bound and oracle operating-point baselines."
        )
    )
    parser.add_argument("--target-episode", action="append", required=True)
    parser.add_argument(
        "--source-episode",
        action="append",
        default=None,
        help="Episodes produced by the same final detector on source domains.",
    )
    parser.add_argument("--fixed-threshold", action="append", type=float, default=[0.5])
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    return parser


def _first_feasible_indices(
    pixel_curves: np.ndarray,
    peak_curves: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_curves, dtype=np.float64)
    peak = np.asarray(peak_curves, dtype=np.float64)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Risk curves must share shape [episodes, thresholds]")
    feasible = (pixel <= pixel_budget) & (peak <= peak_budget)
    any_feasible = feasible.any(axis=1)
    indices = np.argmax(feasible, axis=1).astype(np.int64)
    indices[~any_feasible] = pixel.shape[1] - 1
    return indices, ~any_feasible


def _mark_empty_action(
    thresholds: np.ndarray,
    indices: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    result = np.asarray(rejected, dtype=bool).copy()
    result |= np.asarray(thresholds)[np.asarray(indices, dtype=np.int64)] > 1.0
    return result


def _constant_curve_index(
    pixel_curve: np.ndarray,
    peak_curve: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[int, bool]:
    indices, rejected = _first_feasible_indices(
        np.asarray(pixel_curve)[None],
        np.asarray(peak_curve)[None],
        pixel_budget,
        peak_budget,
    )
    return int(indices[0]), bool(rejected[0])


def _summarise(
    arrays: EpisodeArrays,
    indices: np.ndarray,
    rejected: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> dict[str, Any]:
    rows = np.arange(len(indices))
    summary = summarise_selected_points(
        arrays.pixel_risk[rows, indices],
        arrays.peak_risk[rows, indices],
        arrays.pd[rows, indices],
        rejected,
        arrays.domains,
        pixel_budget,
        peak_budget,
    )
    return summary.to_dict()


def _nearest_source_indices(
    source: EpisodeArrays,
    target: EpisodeArrays,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normaliser = FeatureNormaliser.fit(source.features)
    source_features = normaliser.transform(source.features)
    target_features = normaliser.transform(target.features)
    domains = np.unique(source.domains)
    centroids = np.stack(
        [source_features[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_pixel = np.stack(
        [source.pixel_risk[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_peak = np.stack(
        [source.peak_risk[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_indices: list[int] = []
    domain_rejected: list[bool] = []
    for pixel_curve, peak_curve in zip(domain_pixel, domain_peak, strict=True):
        index, rejected = _constant_curve_index(
            pixel_curve,
            peak_curve,
            pixel_budget,
            peak_budget,
        )
        domain_indices.append(index)
        domain_rejected.append(rejected)
    squared = ((target_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(squared, axis=1)
    return (
        np.asarray(domain_indices, dtype=np.int64)[nearest],
        np.asarray(domain_rejected, dtype=bool)[nearest],
        domains[nearest],
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    target = concatenate_episode_files(args.target_episode)
    source = (
        concatenate_episode_files(args.source_episode)
        if args.source_episode
        else None
    )
    if source is not None:
        if not np.array_equal(source.thresholds, target.thresholds):
            raise ValueError("Source and target episodes use different threshold grids")
        if source.feature_names != target.feature_names:
            raise ValueError("Source and target feature schemas differ")

    methods: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for threshold in args.fixed_threshold:
        index = int(np.searchsorted(target.thresholds, threshold, side="left"))
        index = min(index, len(target.thresholds) - 1)
        methods[f"fixed_{threshold:g}"] = (
            np.full(len(target.features), index, dtype=np.int64),
            np.full(
                len(target.features),
                bool(target.thresholds[index] > 1.0),
                dtype=bool,
            ),
            None,
        )

    # Strong label-free baseline: every context detection is provisionally
    # counted as false. It is an exact upper bound on context false detections,
    # but only an empirical predictor of the disjoint future block.
    if np.isfinite(target.context_pixel_upper).all() and np.isfinite(
        target.context_peak_upper
    ).all():
        index, rejected = _first_feasible_indices(
            target.context_pixel_upper,
            target.context_peak_upper,
            args.pixel_budget,
            args.peak_budget,
        )
        rejected = _mark_empty_action(target.thresholds, index, rejected)
        methods["context_all_detection_upper"] = (index, rejected, None)

    # Target-label oracle is reported only as an upper benchmark.
    index, rejected = _first_feasible_indices(
        target.pixel_risk,
        target.peak_risk,
        args.pixel_budget,
        args.peak_budget,
    )
    rejected = _mark_empty_action(target.thresholds, index, rejected)
    methods["target_future_oracle"] = (index, rejected, None)

    if source is not None:
        pooled_index, pooled_rejected = _constant_curve_index(
            source.pixel_risk.mean(axis=0),
            source.peak_risk.mean(axis=0),
            args.pixel_budget,
            args.peak_budget,
        )
        pooled_rejected = pooled_rejected or bool(
            target.thresholds[pooled_index] > 1.0
        )
        methods["source_pooled"] = (
            np.full(len(target.features), pooled_index, dtype=np.int64),
            np.full(len(target.features), pooled_rejected, dtype=bool),
            None,
        )

        domain_pixel = np.stack(
            [
                source.pixel_risk[source.domains == domain].mean(axis=0)
                for domain in np.unique(source.domains)
            ]
        )
        domain_peak = np.stack(
            [
                source.peak_risk[source.domains == domain].mean(axis=0)
                for domain in np.unique(source.domains)
            ]
        )
        worst_index, worst_rejected = _constant_curve_index(
            domain_pixel.max(axis=0),
            domain_peak.max(axis=0),
            args.pixel_budget,
            args.peak_budget,
        )
        worst_rejected = worst_rejected or bool(
            target.thresholds[worst_index] > 1.0
        )
        methods["source_worst_domain"] = (
            np.full(len(target.features), worst_index, dtype=np.int64),
            np.full(len(target.features), worst_rejected, dtype=bool),
            None,
        )

        nearest_index, nearest_rejected, nearest_domain = _nearest_source_indices(
            source,
            target,
            args.pixel_budget,
            args.peak_budget,
        )
        nearest_rejected = _mark_empty_action(
            target.thresholds,
            nearest_index,
            nearest_rejected,
        )
        methods["nearest_source_curve"] = (
            nearest_index,
            nearest_rejected,
            nearest_domain,
        )

    output_dir = ensure_dir(args.output_dir)
    results: dict[str, Any] = {}
    selected_path = output_dir / "selected_points.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "episode_index",
                "domain",
                "sequence",
                "threshold_index",
                "threshold",
                "rejected",
                "reference_source_domain",
                "true_pixel_risk",
                "true_peak_risk",
                "pd",
                "joint_budget_satisfied",
            ]
        )
        rows = np.arange(len(target.features))
        for name, (indices, rejected, references) in methods.items():
            summary = _summarise(
                target,
                indices,
                rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            results[name] = summary
            for row, index in enumerate(indices):
                reference = "" if references is None else str(references[row])
                pixel = float(target.pixel_risk[row, index])
                peak = float(target.peak_risk[row, index])
                writer.writerow(
                    [
                        name,
                        row,
                        target.domains[row],
                        target.sequences[row],
                        int(index),
                        float(target.thresholds[index]),
                        bool(rejected[row]),
                        reference,
                        pixel,
                        peak,
                        float(target.pd[row, index]),
                        bool(pixel <= args.pixel_budget and peak <= args.peak_budget),
                    ]
                )

    summary = {
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "num_target_episodes": len(target.features),
        "source_episodes_provided": source is not None,
        "methods": results,
        "selected_points_csv": str(selected_path.resolve()),
        "interpretation": {
            "context_all_detection_upper": (
                "Label-free and deterministic on the warm-up context; all predicted "
                "pixels/peaks are counted as false. Its use for a future block still "
                "requires temporal stability and has no distribution-free guarantee."
            ),
            "target_future_oracle": "Uses target future labels and is not deployable.",
        },
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.58 `rc_irstd/pipelines/evaluate_scores.py`

- SHA-256：`b0a40ea874a47a86ea289306f04badded897a9df0880cda45c75622ef9cf15a6`
- 行数：`187`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.episodes.builder import default_threshold_grid
from rc_irstd.evaluation.component_curves import compute_component_curve
from rc_irstd.evaluation.curves import (
    aggregate_curve_counts,
    compute_image_curves,
    rates_from_counts,
)
from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal monotone risks and standard component metrics."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--threshold-grid", default=None)
    parser.add_argument("--component-grid-points", type=int, default=101)
    parser.add_argument("--pixel-budget", type=float, action="append", default=None)
    parser.add_argument("--peak-budget", type=float, action="append", default=None)
    parser.add_argument("--component-budget", type=float, action="append", default=None)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--peak-min-score", type=float, default=0.0)
    parser.add_argument("--peak-tolerance", type=float, default=2.0)
    parser.add_argument("--object-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--max-candidates", type=int, default=0,
        help="Fixed peaks/image cap; 0 means no truncation."
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _best_feasible(pd: np.ndarray, risk: np.ndarray, budget: float) -> int | None:
    feasible = np.flatnonzero(risk <= budget)
    if len(feasible) == 0:
        return None
    best_pd = np.max(pd[feasible])
    # Earliest threshold among equal-Pd feasible points preserves recall.
    return int(feasible[np.flatnonzero(pd[feasible] == best_pd)[0]])


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_candidates < 0:
        raise ValueError("max-candidates must be non-negative")
    pixel_budgets = args.pixel_budget or [1e-6, 1e-5]
    peak_budgets = args.peak_budget or [1.0, 5.0]
    component_budgets = args.component_budget or [1.0, 5.0]
    thresholds = (
        np.load(args.threshold_grid).astype(np.float32)
        if args.threshold_grid
        else default_threshold_grid()
    )
    records = [load_score_record(path, require_mask=True) for path in list_npz(args.score_dir)]
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    formal_curves = []
    for record in records:
        assert record.mask is not None
        probabilities.append(record.probability)
        masks.append(record.mask)
        formal_curves.append(
            compute_image_curves(
                record.probability,
                record.mask,
                thresholds,
                peak_min_distance=args.peak_min_distance,
                peak_min_score=args.peak_min_score,
                peak_tolerance=args.peak_tolerance,
                max_candidates=(None if args.max_candidates <= 0 else args.max_candidates),
            )
        )
    counts = aggregate_curve_counts(formal_curves)
    rates = rates_from_counts(counts)
    output_dir = ensure_dir(args.output_dir)

    formal_rows: list[dict[str, object]] = []
    for index, threshold in enumerate(thresholds):
        formal_rows.append(
            {
                "threshold": float(threshold),
                "pd": float(rates["pd"][index]),
                "fa_pixel": float(rates["pixel_false_rate"][index]),
                "false_peak_per_mp": float(rates["peak_false_per_mp"][index]),
                "false_pixels": int(counts.pixel_false[index]),
                "false_peaks": int(counts.peak_false[index]),
                "matched_gt": int(counts.matched_gt[index]),
                "total_gt": int(counts.total_gt),
                "total_pixels": int(counts.total_pixels),
            }
        )
    formal_path = output_dir / "formal_curve.csv"
    _write_dict_rows(formal_path, formal_rows)
    # Backward-compatible filename.
    _write_dict_rows(output_dir / "curve.csv", formal_rows)

    component_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.95, max(args.component_grid_points // 2, 2), endpoint=False),
                np.linspace(0.95, 1.0, max(args.component_grid_points // 2, 2)),
                np.asarray([np.nextafter(np.float32(1.0), np.float32(2.0))]),
            ]
        )
    ).astype(np.float32)
    component_rows_obj = compute_component_curve(
        probabilities, masks, component_grid, args.object_tolerance
    )
    component_rows = [row.to_dict() for row in component_rows_obj]
    component_path = output_dir / "component_curve.csv"
    _write_dict_rows(component_path, component_rows)

    operating_points: dict[str, object] = {}
    for budget in pixel_budgets:
        index = _best_feasible(rates["pd"], rates["pixel_false_rate"], budget)
        operating_points[f"formal_pixel_{budget:g}"] = None if index is None else {
            "index": index,
            "threshold": float(thresholds[index]),
            "pd": float(rates["pd"][index]),
            "risk": float(rates["pixel_false_rate"][index]),
            "rejected": bool(thresholds[index] > 1.0),
            "standard_metrics": evaluate_irstd_at_threshold(
                probabilities, masks, float(thresholds[index]), args.object_tolerance
            ).to_dict(),
        }
    for budget in peak_budgets:
        index = _best_feasible(rates["pd"], rates["peak_false_per_mp"], budget)
        operating_points[f"formal_peak_{budget:g}"] = None if index is None else {
            "index": index,
            "threshold": float(thresholds[index]),
            "pd": float(rates["pd"][index]),
            "risk": float(rates["peak_false_per_mp"][index]),
            "rejected": bool(thresholds[index] > 1.0),
        }
    component_pd = np.asarray([row.pd for row in component_rows_obj])
    component_fa = np.asarray([row.false_components_per_mp for row in component_rows_obj])
    for budget in component_budgets:
        index = _best_feasible(component_pd, component_fa, budget)
        operating_points[f"component_{budget:g}"] = None if index is None else component_rows_obj[index].to_dict()

    fixed_metrics = evaluate_irstd_at_threshold(
        probabilities, masks, 0.5, args.object_tolerance
    )
    summary = {
        "score_dir": str(Path(args.score_dir).resolve()),
        "num_images": len(records),
        "num_formal_thresholds": len(thresholds),
        "num_component_thresholds": len(component_grid),
        "total_pixels": counts.total_pixels,
        "total_gt": counts.total_gt,
        "fixed_0p5_metrics": fixed_metrics.to_dict(),
        "operating_points": operating_points,
        "formal_curve_csv": str(formal_path.resolve()),
        "component_curve_csv": str(component_path.resolve()),
        "metric_note": {
            "formal": "pixel false rate + threshold-independent fixed false local peaks/MP",
            "standard": "thresholded connected components with overlap/centroid-tolerance matching",
            "niou": "mean image IoU over images with non-empty prediction/GT union",
            "hiou": "harmonic mean of foreground and background IoU",
        },
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.59 `rc_irstd/pipelines/evaluate_zero_label.py`

- SHA-256：`15461c178d31873812bf0da1fe5b6f49429b2756da342a2c197b18e7ddf7da48`
- 行数：`124`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rc_irstd.episodes.dataset import concatenate_episode_files
from rc_irstd.evaluation.risk_curve_metrics import evaluate_risk_curve_predictions
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate zero-label risk-curve operating points.")
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    arrays = concatenate_episode_files(args.episode)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    if not np.array_equal(arrays.thresholds, loaded.thresholds):
        raise ValueError("Episode threshold grid differs from the risk-model checkpoint")
    if arrays.feature_names != loaded.feature_names:
        raise ValueError("Episode feature schema differs from the risk-model checkpoint")

    predictions = predict_risk_curves(
        loaded,
        arrays.features,
        device=device,
        batch_size=args.batch_size,
    )
    metrics, indices, rejected = evaluate_risk_curve_predictions(
        arrays.thresholds,
        predictions["pixel_log_risk"],
        predictions["peak_log_risk"],
        arrays.pixel_log_risk,
        arrays.peak_log_risk,
        arrays.pixel_risk,
        arrays.peak_risk,
        arrays.pd,
        arrays.domains,
        args.pixel_budget,
        args.peak_budget,
    )

    output_dir = ensure_dir(args.output_dir)
    rows = np.arange(len(indices))
    result_path = output_dir / "selected_points.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode_index",
                "domain",
                "sequence",
                "threshold_index",
                "threshold",
                "rejected",
                "predicted_pixel_risk",
                "predicted_peak_risk",
                "true_pixel_risk",
                "true_peak_risk",
                "pd",
                "joint_budget_satisfied",
            ]
        )
        for row, index in enumerate(indices):
            true_pixel = float(arrays.pixel_risk[row, index])
            true_peak = float(arrays.peak_risk[row, index])
            writer.writerow(
                [
                    row,
                    arrays.domains[row],
                    arrays.sequences[row],
                    int(index),
                    float(arrays.thresholds[index]),
                    bool(rejected[row]),
                    float(10.0 ** predictions["pixel_log_risk"][row, index]),
                    float(10.0 ** predictions["peak_log_risk"][row, index]),
                    true_pixel,
                    true_peak,
                    float(arrays.pd[row, index]),
                    bool(true_pixel <= args.pixel_budget and true_peak <= args.peak_budget),
                ]
            )

    np.savez_compressed(
        output_dir / "zero_label_outputs.npz",
        base_indices=indices,
        rejected=rejected,
        predicted_pixel_log_risk=predictions["pixel_log_risk"],
        predicted_peak_log_risk=predictions["peak_log_risk"],
        thresholds=arrays.thresholds,
        domains=arrays.domains,
        sequences=arrays.sequences,
    )
    summary = {
        "mode": "zero_label_empirical_adaptation",
        "formal_guarantee": False,
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "num_episodes": len(indices),
        "metrics": metrics.to_dict(),
        "selected_points_csv": str(result_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.60 `rc_irstd/pipelines/export_scores.py`

- SHA-256：`9a4c38f381b110761b21cf9bf8f04a850215ebe7068e6b3b9e327c243f16e999`
- 行数：`237`

````python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rc_irstd.data.dataset import IRSTDDataset, collate_samples
from rc_irstd.data.score_records import ScoreRecord, save_score_record
from rc_irstd.data.transforms import load_image_preserve_depth, pad_tensor_to_stride
from rc_irstd.engine.worker_seed import make_generator, seed_worker
from rc_irstd.features.image_stats import compute_image_statistics
from rc_irstd.models.detector_adapter import build_detector, resize_logits
from rc_irstd.utils.arguments import parse_hw
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export continuous detector score maps.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--detector", default="mshnet", choices=["mshnet", "mshnet_external", "tiny"]
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--inference-mode",
        choices=["resize", "native_pad", "tiled"],
        default="resize",
    )
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--stride-multiple", type=int, default=32)
    parser.add_argument("--tile-size", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument(
        "--restore-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For resize mode, interpolate probabilities back to original size.",
    )
    parser.add_argument(
        "--include-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalization",
        choices=["imagenet", "minmax", "percentile", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--dataset-type", choices=["iid_images", "temporal"], default="iid_images"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser


def _native_logits(model, image: torch.Tensor, stride: int) -> torch.Tensor:
    padded, original_hw = pad_tensor_to_stride(image, stride=stride)
    output = model(padded, training_tag=True)
    logits = resize_logits(output.logits, tuple(padded.shape[-2:]))
    return logits[..., : original_hw[0], : original_hw[1]]


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap in [0, tile)")
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(length - tile + 1, 1), step))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _tiled_logits(
    model,
    image: torch.Tensor,
    tile_hw: tuple[int, int],
    overlap: int,
    stride: int,
) -> torch.Tensor:
    if image.shape[0] != 1:
        raise ValueError("tiled inference requires batch_size=1")
    height, width = image.shape[-2:]
    tile_h, tile_w = tile_hw
    sum_logits = image.new_zeros((1, 1, height, width))
    counts = image.new_zeros((1, 1, height, width))
    for y in _tile_starts(height, tile_h, overlap):
        for x in _tile_starts(width, tile_w, overlap):
            patch = image[..., y : min(y + tile_h, height), x : min(x + tile_w, width)]
            patch_logits = _native_logits(model, patch, stride)
            patch_h, patch_w = patch_logits.shape[-2:]
            sum_logits[..., y : y + patch_h, x : x + patch_w] += patch_logits
            counts[..., y : y + patch_h, x : x + patch_w] += 1.0
    return sum_logits / counts.clamp_min(1.0)


def _infer_logits(model, images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.inference_mode == "resize":
        output = model(images, training_tag=True)
        return resize_logits(output.logits, tuple(images.shape[-2:]))
    if args.inference_mode == "native_pad":
        return _native_logits(model, images, args.stride_multiple)
    return _tiled_logits(
        model,
        images,
        tuple(int(value) for value in args.tile_size),
        args.tile_overlap,
        args.stride_multiple,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    resize_hw = parse_hw(args.resize) if args.inference_mode == "resize" else None
    output_dir = ensure_dir(args.output_dir)

    dataset = IRSTDDataset(
        args.dataset_dir,
        split=args.split,
        resize_hw=resize_hw,
        augment=False,
        require_mask=args.include_mask,
        normalization=args.normalization,
        dataset_type=args.dataset_type,
        include_component_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    model = build_detector(
        args.detector,
        checkpoint=args.checkpoint,
        device=device,
        strict=True,
    )
    model.eval()

    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="export scores"):
            images = batch["image"].to(device, non_blocking=True)
            logits = _infer_logits(model, images, args)
            probability = torch.sigmoid(logits)
            meta = batch["meta"][0]
            if (
                args.inference_mode == "resize"
                and args.restore_original
                and tuple(probability.shape[-2:]) != meta.original_hw
            ):
                probability = F.interpolate(
                    probability,
                    size=meta.original_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            probability_np = np.clip(
                probability[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0
            )

            loaded_image = load_image_preserve_depth(meta.image_path)
            image_stats, image_stat_names = compute_image_statistics(loaded_image.array)
            mask = None
            if args.include_mask:
                mask_batch = batch["mask"]
                if mask_batch is None:
                    raise RuntimeError("--include-mask was set but no mask was loaded")
                mask_tensor = mask_batch
                if tuple(mask_tensor.shape[-2:]) != tuple(probability_np.shape):
                    mask_tensor = F.interpolate(
                        mask_tensor.float(), size=probability_np.shape, mode="nearest"
                    )
                mask = (mask_tensor[0, 0].numpy() > 0.5).astype(np.uint8)

            record = ScoreRecord(
                probability=probability_np,
                mask=mask,
                image_stats=image_stats,
                image_stat_names=image_stat_names,
                image_id=meta.image_id,
                dataset_name=meta.dataset_name,
                sequence_id=meta.sequence_id,
                frame_index=meta.frame_index,
                original_hw=meta.original_hw,
                source_checkpoint=str(Path(args.checkpoint).resolve()),
                dataset_type=meta.dataset_type,
                inference_mode=args.inference_mode,
            )
            save_score_record(record, output_dir / f"{count:08d}.npz")
            count += 1

    manifest = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": dataset.dataset_name,
        "dataset_type": args.dataset_type,
        "split": args.split,
        "detector": args.detector,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "inference_mode": args.inference_mode,
        "resize_hw": resize_hw,
        "restore_original": args.restore_original,
        "stride_multiple": args.stride_multiple,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "normalization": args.normalization,
        "include_mask": args.include_mask,
        "score_type": "sigmoid_probability",
        "num_images": count,
        "risk_candidate_definition": "threshold-independent deterministic local maxima",
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.61 `rc_irstd/pipelines/make_synthetic_data.py`

- SHA-256：`a471c95b043bd90fd273851d1517123c64ed85540ebfb4f7e141192fc969772a`
- 行数：`113`

````python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create small BasicIRSTD-style synthetic domains.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--domains", nargs="+", default=["DomainA", "DomainB", "DomainC"])
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--frames-per-sequence", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def _background(rng: np.random.Generator, height: int, width: int, domain_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    base = 0.20 + 0.03 * domain_index
    image = np.full((height, width), base, dtype=np.float32)
    if domain_index % 3 == 0:
        image += 0.05 * (xx / max(width - 1, 1))
        image += rng.normal(0.0, 0.025, image.shape)
    elif domain_index % 3 == 1:
        image += 0.04 * np.sin(2.0 * np.pi * yy / 13.0)
        image += 0.03 * np.cos(2.0 * np.pi * xx / 19.0)
        image += rng.normal(0.0, 0.035, image.shape)
    else:
        image += 0.05 * np.sin(2.0 * np.pi * (xx + yy) / 17.0)
        image += rng.normal(0.0, 0.045, image.shape)
        image = ndimage.gaussian_filter(image, sigma=0.7)
    # Sparse hot clutter creates false-peak pressure.
    clutter_count = 4 + 2 * domain_index
    for _ in range(clutter_count):
        y = int(rng.integers(2, height - 2))
        x = int(rng.integers(2, width - 2))
        image[y, x] += float(rng.uniform(0.10, 0.30))
    return image


def _add_targets(
    rng: np.random.Generator,
    image: np.ndarray,
    domain_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    target_count = int(rng.integers(1, 4))
    for _ in range(target_count):
        radius = int(rng.integers(1, 3))
        y = int(rng.integers(4, height - 4))
        x = int(rng.integers(4, width - 4))
        yy, xx = np.ogrid[:height, :width]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        mask[disk] = 1
        contrast = float(rng.uniform(0.45, 0.70) - 0.06 * domain_index)
        sigma = 0.7 + 0.25 * domain_index
        impulse = np.zeros_like(image)
        impulse[y, x] = contrast
        target = ndimage.gaussian_filter(impulse, sigma=sigma)
        target /= max(float(target.max()), 1e-8)
        image += contrast * target
    return np.clip(image, 0.0, 1.0), mask


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.height < 16 or args.width < 16:
        raise ValueError("Synthetic images must be at least 16x16")
    root = ensure_dir(args.output_root)
    manifest: dict[str, object] = {"domains": {}}
    for domain_index, domain_name in enumerate(args.domains):
        domain_root = ensure_dir(root / domain_name)
        images_dir = ensure_dir(domain_root / "images")
        masks_dir = ensure_dir(domain_root / "masks")
        split_dir = ensure_dir(domain_root / "img_idx")
        train_names: list[str] = []
        test_names: list[str] = []
        rng = np.random.default_rng(args.seed + 1000 * domain_index)
        for sequence in range(args.sequences):
            for frame in range(args.frames_per_sequence):
                image = _background(rng, args.height, args.width, domain_index)
                image, mask = _add_targets(rng, image, domain_index)
                image_id = f"seq{sequence:02d}_{frame:04d}"
                rgb = np.repeat((image * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
                Image.fromarray(rgb).save(images_dir / f"{image_id}.png")
                Image.fromarray(mask * 255).save(masks_dir / f"{image_id}.png")
                if sequence < max(1, args.sequences // 2):
                    train_names.append(image_id)
                else:
                    test_names.append(image_id)
        (split_dir / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
        (split_dir / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
        manifest["domains"][domain_name] = {
            "path": str(domain_root.resolve()),
            "train_images": len(train_names),
            "test_images": len(test_names),
        }
    atomic_json_dump(manifest, root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.62 `rc_irstd/pipelines/predict_unlabeled.py`

- SHA-256：`62e2b36bb18dbbef47538a6e60370848ee2dafe7324b0ed95ea717578f3693ce`
- 行数：`76`

````python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a deployment threshold from a genuinely unlabeled score-map window."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--last-n", type=int, default=32)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    paths = list_npz(args.score_dir)
    records = [load_score_record(path, require_mask=False) for path in paths]
    records = sorted(records, key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    if args.last_n > 0:
        records = records[-args.last_n :]
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    feature_config = WindowFeatureConfig.from_dict(
        loaded.metadata.get("feature_config")
    )
    extractor = WindowFeatureExtractor(feature_config)
    features, names = extractor.extract(records)
    if names != loaded.feature_names:
        raise ValueError(
            "Unlabeled-window feature schema differs from the risk-model checkpoint. "
            "Use the same peak and statistics configuration used to build training episodes."
        )
    predictions = predict_risk_curves(loaded, features, device=device)
    point = select_dual_budget_threshold(
        loaded.thresholds,
        predictions["pixel_log_risk"][0],
        predictions["peak_log_risk"][0],
        args.pixel_budget,
        args.peak_budget,
    )
    result = {
        "mode": "zero_label_deployment",
        "num_warmup_images": len(records),
        "threshold_index": point.index,
        "threshold": point.threshold,
        "rejected": point.rejected,
        "predicted_pixel_risk": point.predicted_pixel_risk,
        "predicted_peak_risk_per_mp": point.predicted_peak_risk,
        "pixel_budget": args.pixel_budget,
        "peak_budget_per_mp": args.peak_budget,
        "formal_guarantee": False,
        "source_score_directory": str(Path(args.score_dir).resolve()),
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.63 `rc_irstd/pipelines/run_deployment.py`

- SHA-256：`c4a3686f372c4da4a0ef895a8007186900a3cb9d487b0c4152b75e3ab167bd3e`
- 行数：`215`

````python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import ScoreRecord, load_score_record
from rc_irstd.deployment.monitor import feature_ood_score
from rc_irstd.deployment.session import DeploymentState, ThresholdUpdate
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run causal zero-label threshold adaptation and apply it to future scores."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--warmup-size", type=int, default=32)
    parser.add_argument(
        "--update-every",
        type=int,
        default=0,
        help="0 freezes one threshold; positive values use a past-only rolling window.",
    )
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--offset-index", type=int, default=0)
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=8.0,
        help="Reject a window if RMS feature z-score exceeds this value; <=0 disables.",
    )
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def _group(records: list[ScoreRecord]) -> dict[str, list[ScoreRecord]]:
    groups: dict[str, list[ScoreRecord]] = {}
    for record in records:
        groups.setdefault(record.sequence_id, []).append(record)
    for values in groups.values():
        values.sort(key=lambda item: (item.frame_index, item.image_id))
    return dict(sorted(groups.items()))


def _select_update(
    sequence: str,
    update_index: int,
    warmup: list[ScoreRecord],
    loaded,
    extractor: WindowFeatureExtractor,
    args: argparse.Namespace,
    device,
) -> ThresholdUpdate:
    feature, names = extractor.extract(warmup)
    if names != loaded.feature_names:
        raise ValueError("Deployment feature schema differs from risk-model checkpoint")
    predictions = predict_risk_curves(loaded, feature, device=device)
    point = select_dual_budget_threshold(
        loaded.thresholds,
        predictions["pixel_log_risk"][0],
        predictions["peak_log_risk"][0],
        args.pixel_budget,
        args.peak_budget,
    )
    final_index = min(point.index + max(args.offset_index, 0), len(loaded.thresholds) - 1)
    ood = feature_ood_score(feature, loaded.normaliser)
    rejected = point.rejected or bool(loaded.thresholds[final_index] > 1.0)
    if args.ood_threshold > 0 and ood > args.ood_threshold:
        final_index = len(loaded.thresholds) - 1
        rejected = True
    return ThresholdUpdate(
        sequence_id=sequence,
        update_index=update_index,
        warmup_ids=tuple(item.image_id for item in warmup),
        base_threshold_index=point.index,
        offset_index=max(args.offset_index, 0),
        final_threshold_index=final_index,
        threshold=float(loaded.thresholds[final_index]),
        predicted_pixel_risk=float(10 ** predictions["pixel_log_risk"][0, final_index]),
        predicted_peak_risk_per_mp=float(10 ** predictions["peak_log_risk"][0, final_index]),
        rejected=rejected,
        feature_ood_score=ood,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.warmup_size <= 0 or args.offset_index < 0:
        raise ValueError("warmup-size must be positive and offset-index non-negative")
    device = resolve_device(args.device)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    extractor = WindowFeatureExtractor(
        WindowFeatureConfig.from_dict(loaded.metadata.get("feature_config"))
    )
    records = [load_score_record(path, require_mask=False) for path in list_npz(args.score_dir)]
    output_dir = ensure_dir(args.output_dir)
    mask_root = ensure_dir(output_dir / "masks")
    state = DeploymentState(
        detector_checkpoint=(records[0].source_checkpoint if records else ""),
        curve_checkpoint=str(Path(args.curve_checkpoint).resolve()),
        score_directory=str(Path(args.score_dir).resolve()),
        pixel_budget=args.pixel_budget,
        peak_budget_per_mp=args.peak_budget,
        warmup_size=args.warmup_size,
        offset_index=args.offset_index,
    )
    candidate_rows: list[dict[str, object]] = []
    processed = rejected_images = 0

    for sequence, sequence_records in _group(records).items():
        if len(sequence_records) <= args.warmup_size:
            continue
        update = _select_update(
            sequence,
            args.warmup_size,
            sequence_records[: args.warmup_size],
            loaded,
            extractor,
            args,
            device,
        )
        state.add(update)
        active = update
        for index in range(args.warmup_size, len(sequence_records)):
            if (
                args.update_every > 0
                and index > args.warmup_size
                and (index - args.warmup_size) % args.update_every == 0
            ):
                start = max(0, index - args.warmup_size)
                active = _select_update(
                    sequence,
                    index,
                    sequence_records[start:index],
                    loaded,
                    extractor,
                    args,
                    device,
                )
                state.add(active)
            record = sequence_records[index]
            threshold = active.threshold
            binary = np.zeros_like(record.probability, dtype=np.uint8)
            if not active.rejected:
                binary = (record.probability >= threshold).astype(np.uint8)
            else:
                rejected_images += 1
            mask_path = mask_root / f"{record.image_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary * 255).save(mask_path)
            if not active.rejected:
                scores, ys, xs = extract_fixed_peaks(
                    record.probability,
                    min_distance=args.peak_min_distance,
                    min_score=max(threshold, 0.0),
                )
                for score, y, x in zip(scores, ys, xs, strict=True):
                    if score < threshold:
                        continue
                    candidate_rows.append(
                        {
                            "image_id": record.image_id,
                            "sequence_id": sequence,
                            "frame_index": record.frame_index,
                            "y": int(y),
                            "x": int(x),
                            "score": float(score),
                            "threshold": threshold,
                            "update_index": active.update_index,
                        }
                    )
            processed += 1

    candidate_path = output_dir / "candidates.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "image_id", "sequence_id", "frame_index", "y", "x", "score",
            "threshold", "update_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    atomic_json_dump(state.to_dict(), output_dir / "deployment_state.json")
    summary = {
        "mode": "causal_zero_label_deployment",
        "formal_guarantee": False,
        "num_input_images": len(records),
        "num_processed_future_images": processed,
        "num_rejected_future_images": rejected_images,
        "num_threshold_updates": len(state.updates),
        "num_candidates": len(candidate_rows),
        "state": str((output_dir / "deployment_state.json").resolve()),
        "masks": str(mask_root.resolve()),
        "candidates": str(candidate_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.64 `rc_irstd/pipelines/run_lodo.py`

- SHA-256：`18eb38cf054b49b50e7073704f2aca752be1bd081fe054e3a15b1dd812c9cba5`
- 行数：`753`

````python
from __future__ import annotations

import argparse
import hashlib
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rc_irstd.provenance.fingerprint import command_fingerprint
from rc_irstd.provenance.manifest import load_run_manifest, write_run_manifest
from rc_irstd.utils.config import load_yaml
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


ALL_STAGES = (
    "detector",
    "export",
    "episodes",
    "curve",
    "zero",
    "calibrate",
    "baselines",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nested leave-one-domain-out RC-IRSTD protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--outer-target",
        action="append",
        default=None,
        help="Run only the named outer target; repeat for several targets.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a command when its declared output artifact already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


class CommandRunner:
    def __init__(
        self,
        log_path: Path,
        dry_run: bool,
        resume_existing: bool,
        working_directory: Path,
        environment: dict[str, str],
    ) -> None:
        self.log_path = log_path
        self.dry_run = dry_run
        self.resume_existing = resume_existing
        self.working_directory = working_directory
        self.environment = environment
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, command: list[str], expected: Path | None = None) -> None:
        printable = shlex.join(command)
        source_root = Path(__file__).resolve().parents[1]
        fingerprint, provenance = command_fingerprint(
            command, self.working_directory, source_root
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# cwd: {self.working_directory}\n")
            handle.write(f"# fingerprint: {fingerprint}\n")
            handle.write(printable + "\n")
        if expected is not None and expected.exists() and self.resume_existing:
            manifest = load_run_manifest(expected)
            if manifest is not None and manifest.get("fingerprint") == fingerprint:
                print(f"[skip:fingerprint-match] {expected}")
                return
            print(f"[rerun:stale-or-untracked] {expected}")
        print(f"[run] (cd {self.working_directory} && {printable})")
        if self.dry_run:
            return
        subprocess.run(
            command,
            check=True,
            cwd=self.working_directory,
            env=self.environment,
        )
        if expected is not None:
            if not expected.exists():
                raise RuntimeError(
                    f"Command completed but expected artifact is missing: {expected}"
                )
            write_run_manifest(expected, fingerprint, provenance)


def _require(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing configuration key: {key}")
    return config[key]


def _resolve_path(value: str | Path, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _normalise_config(
    raw: dict[str, Any],
    config_directory: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    raw_datasets = _require(config, "datasets")
    if not isinstance(raw_datasets, dict):
        raise TypeError("datasets must be a mapping")
    datasets: dict[str, dict[str, Any]] = {}
    for name, raw_item in raw_datasets.items():
        if isinstance(raw_item, (str, Path)):
            item: dict[str, Any] = {"path": str(raw_item)}
        elif isinstance(raw_item, dict):
            item = dict(raw_item)
        else:
            raise TypeError(
                f"Dataset '{name}' must be a path string or mapping, got "
                f"{type(raw_item).__name__}"
            )
        if "path" not in item:
            raise KeyError(f"Dataset '{name}' is missing path")
        item["path"] = str(_resolve_path(item["path"], config_directory))
        item.setdefault("train_split", "train")
        item.setdefault("eval_split", "test")
        datasets[str(name)] = item
    config["datasets"] = datasets

    config["output_root"] = str(
        _resolve_path(config.get("output_root", "outputs/lodo"), config_directory)
    )
    config["working_directory"] = str(
        _resolve_path(config.get("working_directory", "."), config_directory)
    )
    return config


def _dataset(config: dict[str, Any], name: str) -> dict[str, Any]:
    datasets = _require(config, "datasets")
    if name not in datasets:
        raise KeyError(f"Unknown dataset '{name}'")
    return dict(datasets[name])


def _extend_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _detector_cache_directory(
    output_root: Path,
    source_names: list[str],
    detector_cfg: dict[str, Any],
) -> Path:
    payload = json.dumps(
        {
            "sources": sorted(source_names),
            "detector": detector_cfg,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()[:16]
    directory = ensure_dir(output_root / "detector_cache" / key)
    atomic_json_dump(
        {"key": key, "sources": sorted(source_names), "detector": detector_cfg},
        directory / "source_set.json",
    )
    return directory


def _train_detector_command(
    python: str,
    detector_cfg: dict[str, Any],
    datasets_cfg: dict[str, Any],
    source_names: list[str],
    output_dir: Path,
) -> list[str]:
    if not source_names:
        raise ValueError("Detector training requires at least one source domain")
    command = [python, "-m", "rc_irstd.pipelines.train_detector"]
    for name in source_names:
        dataset_cfg = datasets_cfg[name]
        command.extend(
            [
                "--source-dataset",
                str(dataset_cfg["path"]),
                "--source-train-split",
                str(dataset_cfg.get("train_split", detector_cfg.get("train_split", "train"))),
                "--source-val-split",
                str(dataset_cfg.get("eval_split", detector_cfg.get("val_split", "test"))),
            ]
        )
    resize = detector_cfg.get("resize", [256, 256])
    command.extend(
        [
            "--detector",
            str(detector_cfg.get("name", "mshnet")),
            "--base-loss",
            str(detector_cfg.get("base_loss", "auto")),
            "--resize",
            str(resize[0]),
            str(resize[1]),
            "--batch-size",
            str(int(detector_cfg.get("per_domain_batch", 2)) * len(source_names)),
            "--epochs",
            str(detector_cfg.get("epochs", 400)),
            "--warm-epoch",
            str(detector_cfg.get("warm_epoch", 5)),
            "--optimizer",
            str(detector_cfg.get("optimizer", "adagrad")),
            "--lr",
            str(detector_cfg.get("lr", 0.05)),
            "--weight-decay",
            str(detector_cfg.get("weight_decay", 0.0)),
            "--lambda-tail",
            str(detector_cfg.get("lambda_tail", 0.1)),
            "--lambda-miss",
            str(detector_cfg.get("lambda_miss", 0.1)),
            "--tail-quantile",
            str(detector_cfg.get("tail_quantile", 0.95)),
            "--miss-quantile",
            str(detector_cfg.get("miss_quantile", 0.8)),
            "--peak-kernel",
            str(detector_cfg.get("peak_kernel", 5)),
            "--exclusion-radius",
            str(detector_cfg.get("exclusion_radius", 2)),
            "--worst-gamma",
            str(detector_cfg.get("worst_gamma", 10.0)),
            "--auxiliary-weight",
            str(detector_cfg.get("auxiliary_weight", 1.0)),
            "--pixel-budget",
            str(detector_cfg.get("pixel_budget", 1e-5)),
            "--peak-budget",
            str(detector_cfg.get("peak_budget", 5.0)),
            "--normalization",
            str(detector_cfg.get("normalization", "imagenet")),
            "--dataset-type",
            str(detector_cfg.get("dataset_type", "iid_images")),
            "--num-workers",
            str(detector_cfg.get("num_workers", 4)),
            "--device",
            str(detector_cfg.get("device", "auto")),
            "--seed",
            str(detector_cfg.get("seed", 42)),
            "--grad-clip",
            str(detector_cfg.get("grad_clip", 5.0)),
            "--val-every",
            str(detector_cfg.get("val_every", 1)),
            "--save-every",
            str(detector_cfg.get("save_every", 20)),
            "--output-dir",
            str(output_dir),
        ]
    )
    command.append("--amp" if bool(detector_cfg.get("amp", True)) else "--no-amp")
    command.append(
        "--deterministic"
        if bool(detector_cfg.get("deterministic", True))
        else "--no-deterministic"
    )
    return command


def _export_command(
    python: str,
    detector_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    resize = detector_cfg.get("resize", [256, 256])
    tile_size = detector_cfg.get("tile_size", [512, 512])
    command = [
        python,
        "-m",
        "rc_irstd.pipelines.export_scores",
        "--dataset-dir",
        str(dataset_cfg["path"]),
        "--split",
        str(dataset_cfg.get("eval_split", "test")),
        "--detector",
        str(detector_cfg.get("name", "mshnet")),
        "--checkpoint",
        str(checkpoint),
        "--inference-mode",
        str(detector_cfg.get("inference_mode", "resize")),
        "--normalization",
        str(detector_cfg.get("normalization", "imagenet")),
        "--dataset-type",
        str(dataset_cfg.get("dataset_type", detector_cfg.get("dataset_type", "iid_images"))),
        "--resize",
        str(resize[0]),
        str(resize[1]),
        "--stride-multiple",
        str(detector_cfg.get("stride_multiple", 32)),
        "--tile-size",
        str(tile_size[0]),
        str(tile_size[1]),
        "--tile-overlap",
        str(detector_cfg.get("tile_overlap", 64)),
        "--include-mask",
        "--num-workers",
        str(detector_cfg.get("num_workers", 4)),
        "--device",
        str(detector_cfg.get("device", "auto")),
        "--seed",
        str(detector_cfg.get("seed", 42)),
        "--output-dir",
        str(output_dir),
    ]
    command.append(
        "--restore-original"
        if bool(detector_cfg.get("restore_original", True))
        else "--no-restore-original"
    )
    return command


def _episode_command(
    python: str,
    episode_cfg: dict[str, Any],
    score_dir: Path,
    output: Path,
) -> list[str]:
    command = [
        python,
        "-m",
        "rc_irstd.pipelines.build_episodes",
        "--score-dir",
        str(score_dir),
        "--output",
        str(output),
        "--context-size",
        str(episode_cfg.get("context_size", 32)),
        "--horizon",
        str(episode_cfg.get("horizon", 16)),
        "--stride",
        str(episode_cfg.get("stride", 48)),
        "--protocol",
        str(episode_cfg.get("protocol", "auto")),
        "--seed",
        str(episode_cfg.get("seed", 0)),
        "--peak-min-distance",
        str(episode_cfg.get("peak_min_distance", 2)),
        "--peak-min-score",
        str(episode_cfg.get("peak_min_score", 0.0)),
        "--peak-border",
        str(episode_cfg.get("peak_border", 0)),
        "--peak-tolerance",
        str(episode_cfg.get("peak_tolerance", 2.0)),
        "--max-candidates",
        str(episode_cfg.get("max_candidates", 0)),
    ]
    _extend_optional(command, "--threshold-grid", episode_cfg.get("threshold_grid"))
    return command


def _curve_command(
    python: str,
    curve_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    episode_files: list[Path],
    output_dir: Path,
) -> list[str]:
    command = [python, "-m", "rc_irstd.pipelines.train_curve"]
    for episode in episode_files:
        command.extend(["--train-episode", str(episode)])
    command.extend(
        [
            "--val-fraction",
            str(curve_cfg.get("val_fraction", 0.2)),
            "--quantile",
            str(curve_cfg.get("quantile", 0.9)),
            "--hidden-dim",
            str(curve_cfg.get("hidden_dim", 256)),
            "--dropout",
            str(curve_cfg.get("dropout", 0.1)),
            "--lambda-peak",
            str(curve_cfg.get("lambda_peak", 1.0)),
            "--lambda-crossing",
            str(curve_cfg.get("lambda_crossing", 0.25)),
            "--crossing-temperature",
            str(curve_cfg.get("crossing_temperature", 0.25)),
            "--focus-base-weight",
            str(curve_cfg.get("focus_base_weight", 1.0)),
            "--focus-weight",
            str(curve_cfg.get("focus_weight", 4.0)),
            "--focus-log-scale",
            str(curve_cfg.get("focus_log_scale", 1.0)),
            "--empty-action-weight",
            str(curve_cfg.get("empty_action_weight", 0.1)),
            "--batch-size",
            str(curve_cfg.get("batch_size", 64)),
            "--epochs",
            str(curve_cfg.get("epochs", 300)),
            "--lr",
            str(curve_cfg.get("lr", 1e-3)),
            "--weight-decay",
            str(curve_cfg.get("weight_decay", 1e-4)),
            "--patience",
            str(curve_cfg.get("patience", 40)),
            "--pixel-budget",
            str(budget_cfg.get("pixel", 1e-6)),
            "--peak-budget",
            str(budget_cfg.get("peak_per_mp", 1.0)),
            "--num-workers",
            str(curve_cfg.get("num_workers", 0)),
            "--device",
            str(curve_cfg.get("device", "auto")),
            "--seed",
            str(curve_cfg.get("seed", 42)),
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = _normalise_config(load_yaml(config_path), config_path.parent)

    python = str(config.get("python", sys.executable))
    output_root = ensure_dir(config["output_root"])
    working_directory = Path(config["working_directory"])
    if not working_directory.is_dir():
        raise FileNotFoundError(
            f"working_directory does not exist: {working_directory}. Point it "
            "to the RC-IRSTD project root."
        )

    package_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path_entries = [str(package_root), str(working_directory)]
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    datasets_cfg = _require(config, "datasets")
    all_domains = list(datasets_cfg)
    if len(all_domains) < 3:
        raise ValueError("Nested LODO requires at least three domains")
    outer_targets = args.outer_target or config.get("outer_targets", all_domains)
    unknown = set(outer_targets).difference(all_domains)
    if unknown:
        raise KeyError(f"Unknown outer targets: {sorted(unknown)}")

    detector_cfg = dict(config.get("detector", {}))
    episode_cfg = dict(config.get("episodes", {}))
    episode_total = int(episode_cfg.get("context_size", 32)) + int(
        episode_cfg.get("horizon", 16)
    )
    pseudo_episode_cfg = dict(episode_cfg)
    pseudo_episode_cfg["stride"] = int(
        episode_cfg.get("train_stride", episode_cfg.get("stride", 16))
    )
    target_episode_cfg = dict(episode_cfg)
    target_episode_cfg["stride"] = int(
        episode_cfg.get(
            "eval_stride",
            episode_cfg.get("target_stride", max(episode_total, int(episode_cfg.get("stride", episode_total)))),
        )
    )
    if target_episode_cfg["stride"] < episode_total:
        raise ValueError(
            "Formal target calibration/test episodes require eval_stride >= "
            "context_size + horizon so windows do not share images."
        )
    curve_cfg = dict(config.get("curve", {}))
    curve_cfg.setdefault("device", detector_cfg.get("device", "auto"))
    budget_cfg = dict(config.get("budgets", {}))
    detector_cfg.setdefault("pixel_budget", budget_cfg.get("pixel", 1e-5))
    detector_cfg.setdefault("peak_budget", budget_cfg.get("peak_per_mp", 5.0))
    calibration_cfg = dict(config.get("calibration", {}))
    stages = set(args.stages)

    protocol_manifest = {
        "protocol": "nested_leave_one_domain_out",
        "config_path": str(config_path),
        "working_directory": str(working_directory),
        "outer_targets": outer_targets,
        "all_domains": all_domains,
        "stages": list(args.stages),
        "leakage_rule": (
            "For outer target t and pseudo-target p, the episode detector is "
            "trained only on domains excluding both t and p. Warm-up context and "
            "future risk windows are disjoint. Calibration and test sequences are "
            "disjoint."
        ),
        "risk_definition": (
            "pixel false rate and threshold-independent fixed false local peaks "
            "per megapixel"
        ),
        "resolved_episode_protocol": {
            "pseudo_train_stride": pseudo_episode_cfg["stride"],
            "target_eval_stride": target_episode_cfg["stride"],
            "context_size": episode_cfg.get("context_size", 32),
            "horizon": episode_cfg.get("horizon", 16),
        },
        "config": config,
    }
    atomic_json_dump(protocol_manifest, output_root / "protocol.json")

    for outer in outer_targets:
        outer_dir = ensure_dir(output_root / f"outer_{outer}")
        runner = CommandRunner(
            outer_dir / "commands.log",
            dry_run=args.dry_run,
            resume_existing=args.resume_existing,
            working_directory=working_directory,
            environment=environment,
        )
        sources = [name for name in all_domains if name != outer]

        # Final detector: trained on every non-target source domain.
        final_detector_dir = _detector_cache_directory(
            output_root, sources, detector_cfg
        )
        final_checkpoint = final_detector_dir / "best.pt"
        if "detector" in stages:
            runner.run(
                _train_detector_command(
                    python,
                    detector_cfg,
                    datasets_cfg,
                    sources,
                    final_detector_dir,
                ),
                expected=final_checkpoint,
            )

        # Target score records and labelled episodes are used only for offline
        # evaluation. True deployment uses predict_unlabeled.py without masks.
        target_scores = outer_dir / "target_scores"
        if "export" in stages or "baselines" in stages:
            runner.run(
                _export_command(
                    python,
                    detector_cfg,
                    _dataset(config, outer),
                    final_checkpoint,
                    target_scores,
                ),
                expected=target_scores / "manifest.json",
            )
        target_episode = outer_dir / "target_episodes.npz"
        if "episodes" in stages or "baselines" in stages:
            runner.run(
                _episode_command(python, target_episode_cfg, target_scores, target_episode),
                expected=target_episode,
            )

        # Inner pseudo-target episodes. Excluding both outer and pseudo domains
        # prevents target identity or pseudo-target labels from entering detector
        # training for that episode family.
        pseudo_episode_files: list[Path] = []
        for pseudo in sources:
            pseudo_dir = ensure_dir(outer_dir / "pseudo" / pseudo)
            inner_sources = [
                name for name in all_domains if name not in {outer, pseudo}
            ]
            inner_detector_dir = _detector_cache_directory(
                output_root, inner_sources, detector_cfg
            )
            inner_checkpoint = inner_detector_dir / "best.pt"
            if "detector" in stages:
                runner.run(
                    _train_detector_command(
                        python,
                        detector_cfg,
                        datasets_cfg,
                        inner_sources,
                        inner_detector_dir,
                    ),
                    expected=inner_checkpoint,
                )
            pseudo_scores = pseudo_dir / "scores"
            if "export" in stages:
                runner.run(
                    _export_command(
                        python,
                        detector_cfg,
                        _dataset(config, pseudo),
                        inner_checkpoint,
                        pseudo_scores,
                    ),
                    expected=pseudo_scores / "manifest.json",
                )
            pseudo_episode = pseudo_dir / "episodes.npz"
            pseudo_episode_files.append(pseudo_episode)
            if "episodes" in stages:
                runner.run(
                    _episode_command(
                        python,
                        pseudo_episode_cfg,
                        pseudo_scores,
                        pseudo_episode,
                    ),
                    expected=pseudo_episode,
                )

        if "baselines" in stages:
            final_source_episodes: list[Path] = []
            for source in sources:
                source_root = ensure_dir(outer_dir / "final_source" / source)
                source_scores = source_root / "scores"
                runner.run(
                    _export_command(
                        python,
                        detector_cfg,
                        _dataset(config, source),
                        final_checkpoint,
                        source_scores,
                    ),
                    expected=source_scores / "manifest.json",
                )
                source_episode = source_root / "episodes.npz"
                runner.run(
                    _episode_command(
                        python,
                        target_episode_cfg,
                        source_scores,
                        source_episode,
                    ),
                    expected=source_episode,
                )
                final_source_episodes.append(source_episode)

            baseline_command = [
                python,
                "-m",
                "rc_irstd.pipelines.evaluate_baselines",
                "--target-episode",
                str(target_episode),
            ]
            for source_episode in final_source_episodes:
                baseline_command.extend(["--source-episode", str(source_episode)])
            baseline_command.extend(
                [
                    "--pixel-budget",
                    str(budget_cfg.get("pixel", 1e-6)),
                    "--peak-budget",
                    str(budget_cfg.get("peak_per_mp", 1.0)),
                    "--output-dir",
                    str(outer_dir / "baselines"),
                ]
            )
            runner.run(
                baseline_command,
                expected=outer_dir / "baselines" / "summary.json",
            )

        curve_dir = outer_dir / "risk_curve"
        curve_checkpoint = curve_dir / "best.pt"
        if "curve" in stages:
            runner.run(
                _curve_command(
                    python,
                    curve_cfg,
                    budget_cfg,
                    pseudo_episode_files,
                    curve_dir,
                ),
                expected=curve_checkpoint,
            )

        if "zero" in stages:
            runner.run(
                [
                    python,
                    "-m",
                    "rc_irstd.pipelines.evaluate_zero_label",
                    "--episode",
                    str(target_episode),
                    "--curve-checkpoint",
                    str(curve_checkpoint),
                    "--pixel-budget",
                    str(budget_cfg.get("pixel", 1e-6)),
                    "--peak-budget",
                    str(budget_cfg.get("peak_per_mp", 1.0)),
                    "--device",
                    str(curve_cfg.get("device", "auto")),
                    "--output-dir",
                    str(outer_dir / "zero_label"),
                ],
                expected=outer_dir / "zero_label" / "summary.json",
            )

        if "calibrate" in stages:
            command = [
                python,
                "-m",
                "rc_irstd.pipelines.calibrate_and_evaluate",
                "--episode",
                str(target_episode),
                "--curve-checkpoint",
                str(curve_checkpoint),
                "--pixel-budget",
                str(budget_cfg.get("pixel", 1e-6)),
                "--peak-budget",
                str(budget_cfg.get("peak_per_mp", 1.0)),
                "--alpha",
                str(calibration_cfg.get("alpha", 0.1)),
                "--calibration-sizes",
                *[
                    str(value)
                    for value in calibration_cfg.get("sizes", [10, 20, 50])
                ],
                "--seeds",
                *[
                    str(value)
                    for value in calibration_cfg.get("seeds", [0, 1, 2, 3, 4])
                ],
                "--calibration-unit",
                str(calibration_cfg.get("unit", "image")),
                "--offset-step",
                str(calibration_cfg.get("offset_step", 1)),
                "--device",
                str(curve_cfg.get("device", "auto")),
                "--output-dir",
                str(outer_dir / "few_shot_crc"),
            ]
            runner.run(
                command,
                expected=outer_dir / "few_shot_crc" / "summary.json",
            )

    print(json.dumps(protocol_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.65 `rc_irstd/pipelines/smoke.py`

- SHA-256：`a2bbf0429eed3edf36f66756916e94305b1aa12ca52beea35b7395513e74e857`
- 行数：`250`

````python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rc_irstd.pipelines import (
    build_episodes,
    calibrate_and_evaluate,
    evaluate_zero_label,
    export_scores,
    make_synthetic_data,
    train_curve,
    train_detector,
)
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an in-process synthetic end-to-end RC-IRSTD smoke test."
    )
    parser.add_argument("--work-dir", default="outputs/smoke")
    parser.add_argument("--clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    work = Path(args.work_dir).expanduser().resolve()
    if args.clean and work.exists():
        shutil.rmtree(work)
    ensure_dir(work)
    data = work / "data"
    run = work / "run"

    make_synthetic_data.main(
        [
            "--output-root",
            str(data),
            "--domains",
            "DomainA",
            "DomainB",
            "DomainC",
            "--height",
            "16",
            "--width",
            "16",
            "--sequences",
            "4",
            "--frames-per-sequence",
            "6",
            "--seed",
            "7",
        ]
    )

    detector_dir = run / "detector"
    train_detector.main(
        [
            "--source-dataset",
            str(data / "DomainA"),
            "--source-dataset",
            str(data / "DomainB"),
            "--train-split",
            "train",
            "--val-split",
            "test",
            "--detector",
            "tiny",
            "--base-loss",
            "bce_dice",
            "--resize",
            "16",
            "16",
            "--batch-size",
            "24",
            "--epochs",
            "1",
            "--warm-epoch",
            "0",
            "--optimizer",
            "adamw",
            "--lr",
            "0.001",
            "--lambda-tail",
            "0.05",
            "--lambda-miss",
            "0.05",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--no-amp",
            "--output-dir",
            str(detector_dir),
        ]
    )

    episode_paths: dict[str, Path] = {}
    for domain in ("DomainA", "DomainB", "DomainC"):
        score_dir = run / "scores" / domain
        export_scores.main(
            [
                "--dataset-dir",
                str(data / domain),
                "--split",
                "test",
                "--detector",
                "tiny",
                "--checkpoint",
                str(detector_dir / "best.pt"),
                "--resize",
                "16",
                "16",
                "--restore-original",
                "--include-mask",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--output-dir",
                str(score_dir),
            ]
        )
        episode_path = run / "episodes" / f"{domain}.npz"
        build_episodes.main(
            [
                "--score-dir",
                str(score_dir),
                "--output",
                str(episode_path),
                "--context-size",
                "2",
                "--horizon",
                "1",
                "--stride",
                "3",
                "--peak-min-distance",
                "2",
                "--max-candidates",
                "1024",
            ]
        )
        episode_paths[domain] = episode_path

    curve_dir = run / "curve"
    train_curve.main(
        [
            "--train-episode",
            str(episode_paths["DomainA"]),
            "--train-episode",
            str(episode_paths["DomainB"]),
            "--quantile",
            "0.90",
            "--hidden-dim",
            "32",
            "--dropout",
            "0.0",
            "--batch-size",
            "4",
            "--epochs",
            "3",
            "--patience",
            "3",
            "--lr",
            "0.001",
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(curve_dir),
        ]
    )

    zero_dir = run / "zero"
    evaluate_zero_label.main(
        [
            "--episode",
            str(episode_paths["DomainC"]),
            "--curve-checkpoint",
            str(curve_dir / "best.pt"),
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--device",
            "cpu",
            "--output-dir",
            str(zero_dir),
        ]
    )

    crc_dir = run / "crc"
    calibrate_and_evaluate.main(
        [
            "--episode",
            str(episode_paths["DomainC"]),
            "--curve-checkpoint",
            str(curve_dir / "best.pt"),
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--alpha",
            "0.50",
            "--calibration-sizes",
            "2",
            "--seeds",
            "0",
            "--offset-step",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(crc_dir),
        ]
    )

    required = [
        detector_dir / "best.pt",
        curve_dir / "best.pt",
        zero_dir / "summary.json",
        crc_dir / "summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Smoke test is missing artifacts: {missing}")
    summary = {
        "status": "passed",
        "work_dir": str(work),
        "artifacts": [str(path) for path in required],
        "notes": (
            "This validates the complete training/export/episode/risk-curve/"
            "zero-label/CRC software path with TinyUNet and synthetic data. It "
            "does not substitute for real MSHNet or benchmark experiments."
        ),
    }
    atomic_json_dump(summary, work / "smoke_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

### 3.66 `rc_irstd/pipelines/train_curve.py`

- SHA-256：`e34923cce3a27b8868be25b6e9e2baf68de84dc8400542844f7130d541068af9`
- 行数：`367`

````python
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from rc_irstd.engine.worker_seed import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
)
from rc_irstd.episodes.dataset import (
    EpisodeArrays,
    RiskCurveDataset,
    concatenate_episode_files,
)
from rc_irstd.episodes.splits import grouped_train_val_split
from rc_irstd.evaluation.risk_curve_metrics import evaluate_risk_curve_predictions
from rc_irstd.losses.quantile import budget_focused_weight, crossing_loss, pinball_loss
from rc_irstd.models.risk_curve import FeatureNormaliser, RiskCurvePredictor
from rc_irstd.utils.checkpoint import atomic_torch_save
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir
from rc_irstd.utils.logging import JsonlLogger
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a budget-focused monotone upper-quantile risk curve."
    )
    parser.add_argument("--train-episode", action="append", required=True)
    parser.add_argument("--val-episode", action="append", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lambda-peak", type=float, default=1.0)
    parser.add_argument("--lambda-crossing", type=float, default=0.25)
    parser.add_argument("--crossing-temperature", type=float, default=0.25)
    parser.add_argument("--focus-base-weight", type=float, default=1.0)
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--focus-log-scale", type=float, default=1.0)
    parser.add_argument("--empty-action-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _split_arrays(args: argparse.Namespace) -> tuple[EpisodeArrays, EpisodeArrays]:
    train_all = concatenate_episode_files(args.train_episode)
    if args.val_episode:
        validation = concatenate_episode_files(args.val_episode)
        if not np.array_equal(train_all.thresholds, validation.thresholds):
            raise ValueError("Training and validation threshold grids differ")
        if train_all.feature_names != validation.feature_names:
            raise ValueError("Training and validation feature schemas differ")
        if train_all.feature_config != validation.feature_config:
            raise ValueError("Training and validation feature configurations differ")
        return train_all, validation
    train_indices, val_indices = grouped_train_val_split(
        train_all, val_fraction=args.val_fraction, seed=args.seed
    )
    return train_all.subset(train_indices), train_all.subset(val_indices)


def _predict(
    model,
    dataset: RiskCurveDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: dict[str, list[np.ndarray]] = {
        "pixel_log_risk": [],
        "peak_log_risk": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch["features"].to(device))
            for key in outputs:
                outputs[key].append(prediction[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _loss_terms(
    prediction: dict[str, torch.Tensor],
    target_pixel: torch.Tensor,
    target_peak: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pixel_weight = budget_focused_weight(
        target_pixel,
        args.pixel_budget,
        args.focus_base_weight,
        args.focus_weight,
        args.focus_log_scale,
        args.empty_action_weight,
    )
    peak_weight = budget_focused_weight(
        target_peak,
        args.peak_budget,
        args.focus_base_weight,
        args.focus_weight,
        args.focus_log_scale,
        args.empty_action_weight,
    )
    pixel_pinball = pinball_loss(
        prediction["pixel_log_risk"], target_pixel, args.quantile, pixel_weight
    )
    peak_pinball = pinball_loss(
        prediction["peak_log_risk"], target_peak, args.quantile, peak_weight
    )
    pixel_crossing = crossing_loss(
        prediction["pixel_log_risk"],
        target_pixel,
        args.pixel_budget,
        args.crossing_temperature,
        pixel_weight,
    )
    peak_crossing = crossing_loss(
        prediction["peak_log_risk"],
        target_peak,
        args.peak_budget,
        args.crossing_temperature,
        peak_weight,
    )
    pinball = pixel_pinball + args.lambda_peak * peak_pinball
    crossing = pixel_crossing + args.lambda_peak * peak_crossing
    total = pinball + args.lambda_crossing * crossing
    return {
        "total": total,
        "pinball": pinball,
        "crossing": crossing,
        "pixel_pinball": pixel_pinball,
        "peak_pinball": peak_pinball,
    }


def _checkpoint_payload(
    model: RiskCurvePredictor,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_selected_key: tuple[float, float, float, float],
    best_selected_epoch: int,
    best_pinball: float,
    normaliser: FeatureNormaliser,
    arrays: EpisodeArrays,
    args: argparse.Namespace,
    validation_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "epoch": epoch,
        "best_selected_key": list(best_selected_key),
        "best_selected_epoch": best_selected_epoch,
        "best_pinball": best_pinball,
        "normaliser": normaliser.to_dict(),
        "thresholds": arrays.thresholds.tolist(),
        "feature_names": list(arrays.feature_names),
        "feature_config": dict(arrays.feature_config),
        "model_config": {
            "input_dim": int(arrays.features.shape[1]),
            "num_thresholds": int(arrays.thresholds.shape[0]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "training_config": vars(args),
        "validation_metrics": validation_metrics,
        "risk_definition": "pixel false rate and fixed false local peaks per megapixel",
        "guarantee_note": (
            "The model is an empirical conditional upper-quantile estimator. "
            "Any finite-sample marginal statement belongs to the optional CRC stage."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("budgets must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    atomic_json_dump(vars(args), output_dir / "arguments.json")
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    train_arrays, val_arrays = _split_arrays(args)
    normaliser = FeatureNormaliser.fit(train_arrays.features)
    train_dataset = RiskCurveDataset(train_arrays, normaliser.mean, normaliser.std)
    val_dataset = RiskCurveDataset(val_arrays, normaliser.mean, normaliser.std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=make_generator(args.seed),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    model = RiskCurvePredictor(
        input_dim=train_arrays.features.shape[1],
        num_thresholds=train_arrays.thresholds.shape[0],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, args.patience // 4)
    )

    start_epoch = 0
    best_selected_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_selected_epoch = -1
    best_pinball = math.inf
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        loaded_key = payload.get("best_selected_key")
        if loaded_key is not None:
            best_selected_key = tuple(float(value) for value in loaded_key)  # type: ignore[assignment]
        best_selected_epoch = int(payload.get("best_selected_epoch", -1))
        best_pinball = float(payload.get("best_pinball", math.inf))
        restore_rng_state(payload.get("rng_state"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"total": 0.0, "pinball": 0.0, "crossing": 0.0}
        count = 0
        for batch in tqdm(train_loader, desc=f"curve {epoch + 1}/{args.epochs}", leave=False):
            features = batch["features"].to(device)
            target_pixel = batch["pixel_log_risk"].to(device)
            target_peak = batch["peak_log_risk"].to(device)
            prediction = model(features)
            terms = _loss_terms(prediction, target_pixel, target_peak, args)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            current = len(features)
            count += current
            for key in running:
                running[key] += float(terms[key].detach()) * current

        predictions = _predict(model, val_dataset, device, args.batch_size)
        val_prediction = {
            "pixel_log_risk": torch.from_numpy(predictions["pixel_log_risk"]),
            "peak_log_risk": torch.from_numpy(predictions["peak_log_risk"]),
        }
        val_terms = _loss_terms(
            val_prediction,
            torch.from_numpy(val_arrays.pixel_log_risk),
            torch.from_numpy(val_arrays.peak_log_risk),
            args,
        )
        val_objective = float(val_terms["total"])
        val_pinball = float(val_terms["pinball"])
        scheduler.step(val_objective)
        metrics, _, _ = evaluate_risk_curve_predictions(
            val_arrays.thresholds,
            predictions["pixel_log_risk"],
            predictions["peak_log_risk"],
            val_arrays.pixel_log_risk,
            val_arrays.peak_log_risk,
            val_arrays.pixel_risk,
            val_arrays.peak_risk,
            val_arrays.pd,
            val_arrays.domains,
            args.pixel_budget,
            args.peak_budget,
        )
        selected = metrics.selected
        normalised_excess = (
            selected.pixel_excess / args.pixel_budget
            + selected.peak_excess / args.peak_budget
        )
        current_key = (
            -float(normalised_excess),
            float(selected.effective_pd_with_rejects),
            float(selected.joint_bsr),
            -val_objective,
        )
        is_best_selected = current_key > best_selected_key
        if is_best_selected:
            best_selected_key = current_key
            best_selected_epoch = epoch
        is_best_pinball = val_pinball < best_pinball
        if is_best_pinball:
            best_pinball = val_pinball

        record = {
            "epoch": epoch,
            "train_total": running["total"] / max(count, 1),
            "train_pinball": running["pinball"] / max(count, 1),
            "train_crossing": running["crossing"] / max(count, 1),
            "val_objective": val_objective,
            "val_pinball": val_pinball,
            "val_crossing": float(val_terms["crossing"]),
            "normalised_selected_excess": normalised_excess,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics.to_dict(),
            "is_best_selected": is_best_selected,
            "is_best_pinball": is_best_pinball,
        }
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_selected_key,
            best_selected_epoch,
            best_pinball,
            normaliser,
            train_arrays,
            args,
            metrics.to_dict(),
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best_selected:
            atomic_torch_save(payload, output_dir / "best_selected.pt")
            atomic_torch_save(payload, output_dir / "best.pt")
        if is_best_pinball:
            atomic_torch_save(payload, output_dir / "best_pinball.pt")
        logger.log(record)
        print(json.dumps(record, ensure_ascii=False))

        if epoch - best_selected_epoch >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; selected-point best was "
                f"epoch {best_selected_epoch}."
            )
            break


if __name__ == "__main__":
    main()
````

### 3.67 `rc_irstd/pipelines/train_detector.py`

- SHA-256：`8561df2033057ad54efe1c460f8c82cdd9e8fcef0830d7ee21a72ef2512b408f`
- 行数：`492`

````python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adagrad, AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from rc_irstd.data.dataset import IRSTDDataset, collate_samples
from rc_irstd.data.sampler import DomainBalancedBatchSampler
from rc_irstd.engine.worker_seed import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
)
from rc_irstd.evaluation.curves import compute_image_curves
from rc_irstd.evaluation.detector_selection import (
    DetectorBudgetSelection,
    summarise_detector_budget,
    validation_threshold_grid,
)
from rc_irstd.evaluation.segmentation import evaluate_binary_segmentation
from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss, fallback_segmentation_loss
from rc_irstd.losses.sls import SLSIoULoss
from rc_irstd.models.detector_adapter import build_detector, resize_logits
from rc_irstd.utils.arguments import parse_hw
from rc_irstd.utils.checkpoint import atomic_torch_save
from rc_irstd.utils.device import autocast_context, create_grad_scaler, resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, normalise_state_dict
from rc_irstd.utils.logging import JsonlLogger
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a domain-balanced, risk-aware IRSTD detector."
    )
    parser.add_argument(
        "--source-dataset",
        action="append",
        required=True,
        help="BasicIRSTD-style source directory; repeat for multiple domains.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--source-train-split", action="append", default=None)
    parser.add_argument("--source-val-split", action="append", default=None)
    parser.add_argument(
        "--detector",
        default="mshnet",
        choices=["mshnet", "mshnet_external", "tiny"],
        help="mshnet is the self-contained implementation bundled in this package.",
    )
    parser.add_argument(
        "--base-loss",
        default="auto",
        choices=["auto", "sls", "bce_dice"],
    )
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument(
        "--normalization",
        choices=["imagenet", "minmax", "percentile", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["iid_images", "temporal"],
        default="iid_images",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adagrad", "adamw"], default="adagrad")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-tail", type=float, default=0.10)
    parser.add_argument("--lambda-miss", type=float, default=0.10)
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--miss-quantile", type=float, default=0.80)
    parser.add_argument("--peak-kernel", type=int, default=5)
    parser.add_argument("--exclusion-radius", type=int, default=2)
    parser.add_argument("--worst-gamma", type=float, default=10.0)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument(
        "--pixel-budget",
        type=float,
        default=1e-5,
        help="Source-validation budget used to select best_budget.pt.",
    )
    parser.add_argument(
        "--peak-budget",
        type=float,
        default=5.0,
        help="Fixed false local peaks/MP budget used for checkpoint selection.",
    )
    parser.add_argument("--selection-grid-points", type=int, default=96)
    parser.add_argument("--selection-peak-min-distance", type=int, default=2)
    parser.add_argument("--selection-peak-tolerance", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _build_base_loss(name: str, detector_name: str):
    requested = name
    if requested == "auto":
        requested = "sls" if detector_name.startswith("mshnet") else "bce_dice"
    if requested == "bce_dice":
        return fallback_segmentation_loss, "bce_dice"
    if detector_name == "mshnet_external":
        try:
            from model.loss import SLSIoULoss as ExternalSLSIoULoss  # type: ignore

            return ExternalSLSIoULoss(), "sls_external"
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "mshnet_external with SLS requires model.loss.SLSIoULoss on PYTHONPATH"
            ) from exc
    return SLSIoULoss(), "sls_internal"


def _source_splits(args: argparse.Namespace, kind: str) -> list[str]:
    values = args.source_train_split if kind == "train" else args.source_val_split
    fallback = args.train_split if kind == "train" else args.val_split
    if values is None:
        return [str(fallback)] * len(args.source_dataset)
    if len(values) != len(args.source_dataset):
        raise ValueError(
            f"--source-{kind}-split must occur once per --source-dataset "
            f"({len(args.source_dataset)} required, got {len(values)})"
        )
    return [str(value) for value in values]


def _make_datasets(args: argparse.Namespace, split_kind: str, augment: bool):
    resize_hw = parse_hw(args.resize)
    splits = _source_splits(args, split_kind)
    return [
        IRSTDDataset(
            path,
            split=split,
            resize_hw=resize_hw,
            augment=augment,
            domain_id=domain_id,
            require_mask=True,
            normalization=args.normalization,
            dataset_type=args.dataset_type,
            include_component_labels=True,
        )
        for domain_id, (path, split) in enumerate(
            zip(args.source_dataset, splits, strict=True)
        )
    ]


def _make_train_loader(args: argparse.Namespace) -> tuple[DataLoader, DomainBalancedBatchSampler]:
    datasets = _make_datasets(args, "train", augment=True)
    concatenated = ConcatDataset(datasets)
    domain_ids: list[int] = []
    for domain_id, dataset in enumerate(datasets):
        domain_ids.extend([domain_id] * len(dataset))
    sampler = DomainBalancedBatchSampler(
        domain_ids,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    loader = DataLoader(
        concatenated,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    return loader, sampler


def _make_val_loader(args: argparse.Namespace) -> DataLoader:
    datasets = _make_datasets(args, "val", augment=False)
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1),
    )


def _build_optimizer(args: argparse.Namespace, parameters):
    if args.optimizer == "adagrad":
        return Adagrad(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)


def _validate(
    model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    warm_epoch: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], DetectorBudgetSelection]:
    model.eval()
    intersection = union = false_pixels = total_pixels = 0
    detected_objects = gt_objects = false_components = 0
    thresholds = validation_threshold_grid(args.selection_grid_points)
    domain_curves: dict[str, list[Any]] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="validate", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                raise RuntimeError("Validation requires masks")
            masks = masks.to(device, non_blocking=True)
            output = model(images, training_tag=epoch >= warm_epoch)
            logits = resize_logits(output.logits, tuple(masks.shape[-2:]))
            probabilities = torch.sigmoid(logits)
            prediction = probabilities >= 0.5
            for pred, probability, target, meta in zip(
                prediction, probabilities, masks, batch["meta"], strict=True
            ):
                target_np = target.detach().cpu().numpy()
                metrics = evaluate_binary_segmentation(
                    pred.detach().cpu().numpy(), target_np
                )
                intersection += metrics.intersection
                union += metrics.union
                false_pixels += metrics.false_positive_pixels
                total_pixels += int(target.numel())
                detected_objects += metrics.detected_objects
                gt_objects += metrics.gt_objects
                false_components += metrics.false_components
                curve = compute_image_curves(
                    probability[0].detach().cpu().numpy(),
                    target_np,
                    thresholds,
                    peak_min_distance=args.selection_peak_min_distance,
                    peak_tolerance=args.selection_peak_tolerance,
                )
                domain_curves.setdefault(meta.dataset_name, []).append(curve)

    metrics = {
        "iou": float(intersection / max(union, 1)),
        "pd_0p5": float(detected_objects / max(gt_objects, 1)),
        "fa_pixel_0p5": float(false_pixels / max(total_pixels, 1)),
        "fa_component_per_mp_0p5": float(
            false_components / max(total_pixels / 1_000_000.0, 1e-12)
        ),
    }
    selection = summarise_detector_budget(
        domain_curves,
        thresholds,
        args.pixel_budget,
        args.peak_budget,
    )
    metrics.update(
        {
            "budget_mean_domain_pd": selection.mean_domain_pd,
            "budget_worst_domain_pd": selection.worst_domain_pd,
            "budget_rejection_rate": selection.rejection_rate,
            "budget_mean_threshold": selection.mean_threshold,
        }
    )
    return metrics, selection


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_iou: float,
    best_budget_key: tuple[float, float, float, float],
    resolved_base_loss: str,
    args: argparse.Namespace,
    validation: dict[str, float] | None,
    budget_selection: DetectorBudgetSelection | None,
) -> dict[str, Any]:
    return {
        "model": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_state": capture_rng_state(),
        "epoch": epoch,
        "best_iou": best_iou,
        "best_budget_key": list(best_budget_key),
        "detector": args.detector,
        "base_loss": resolved_base_loss,
        "validation": validation,
        "budget_selection": (
            budget_selection.to_dict() if budget_selection is not None else None
        ),
        "arguments": vars(args),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.batch_size % len(args.source_dataset) != 0:
        raise ValueError(
            "batch-size must be divisible by the number of source domains"
        )
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("selection budgets must be positive")
    _source_splits(args, "train")
    _source_splits(args, "val")

    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    atomic_json_dump(vars(args), output_dir / "arguments.json")
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    train_loader, batch_sampler = _make_train_loader(args)
    val_loader = _make_val_loader(args)
    model = build_detector(args.detector, device=device)
    base_loss, resolved_base_loss = _build_base_loss(args.base_loss, args.detector)
    criterion = RiskAwareDetectorLoss(
        base_loss=base_loss,
        lambda_tail=args.lambda_tail,
        lambda_miss=args.lambda_miss,
        tail_quantile=args.tail_quantile,
        miss_quantile=args.miss_quantile,
        peak_kernel=args.peak_kernel,
        exclusion_radius=args.exclusion_radius,
        worst_gamma=args.worst_gamma,
        auxiliary_weight=args.auxiliary_weight,
    )
    optimizer = _build_optimizer(args, (p for p in model.parameters() if p.requires_grad))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = create_grad_scaler(device, args.amp)

    start_epoch = 0
    best_iou = -math.inf
    best_budget_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.model.load_state_dict(normalise_state_dict(payload), strict=True)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            if "scaler" in payload and payload["scaler"] is not None:
                scaler.load_state_dict(payload["scaler"])
            start_epoch = int(payload.get("epoch", -1)) + 1
            best_iou = float(payload.get("best_iou", best_iou))
            loaded_key = payload.get("best_budget_key")
            if loaded_key is not None:
                best_budget_key = tuple(float(value) for value in loaded_key)  # type: ignore[assignment]
            restore_rng_state(payload.get("rng_state"))

    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        model.train()
        totals = {"total": 0.0, "base": 0.0, "tail": 0.0, "miss": 0.0}
        sample_count = 0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                raise RuntimeError("Detector training requires masks")
            masks = masks.to(device, non_blocking=True)
            component_labels = batch.get("component_labels")
            if component_labels is not None:
                component_labels = component_labels.to(device, non_blocking=True)
            domain_ids = batch["domain_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                output = model(images, training_tag=epoch >= args.warm_epoch)
                logits = resize_logits(output.logits, tuple(masks.shape[-2:]))
                loss_terms = criterion(
                    logits,
                    masks,
                    domain_ids,
                    auxiliary_logits=output.auxiliary_logits,
                    component_labels=component_labels,
                    warm_epoch=args.warm_epoch,
                    epoch=epoch,
                )
            scaler.scale(loss_terms["total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            current_batch = images.shape[0]
            sample_count += current_batch
            for key in totals:
                totals[key] += float(loss_terms[key].detach()) * current_batch
            progress.set_postfix(loss=f"{totals['total'] / sample_count:.4f}")

        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "base_loss": resolved_base_loss,
            **{
                f"train_{key}": value / max(sample_count, 1)
                for key, value in totals.items()
            },
        }
        validation: dict[str, float] | None = None
        budget_selection: DetectorBudgetSelection | None = None
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            validation, budget_selection = _validate(
                model, val_loader, device, epoch, args.warm_epoch, args
            )
            record.update({f"val_{key}": value for key, value in validation.items()})
            record["val_budget_domains"] = [
                point.__dict__ for point in budget_selection.domain_points
            ]

        is_best_iou = validation is not None and validation["iou"] > best_iou
        if is_best_iou:
            best_iou = validation["iou"]
        current_budget_key = (
            budget_selection.rank_key(validation["iou"])
            if budget_selection is not None and validation is not None
            else (-math.inf, -math.inf, -math.inf, -math.inf)
        )
        is_best_budget = current_budget_key > best_budget_key
        if is_best_budget:
            best_budget_key = current_budget_key

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_iou,
            best_budget_key,
            resolved_base_loss,
            args,
            validation,
            budget_selection,
        )
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if (epoch + 1) % args.save_every == 0:
            atomic_torch_save(checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")
        if is_best_iou:
            atomic_torch_save(checkpoint, output_dir / "best_iou.pt")
        if is_best_budget:
            atomic_torch_save(checkpoint, output_dir / "best_budget.pt")
            # Backward-compatible default consumed by existing export/smoke code.
            atomic_torch_save(checkpoint, output_dir / "best.pt")
            if budget_selection is not None:
                atomic_json_dump(
                    budget_selection.to_dict(), output_dir / "best_budget_metrics.json"
                )

        record["is_best_iou"] = is_best_iou
        record["is_best_budget"] = is_best_budget
        logger.log(record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
````

### 3.68 `rc_irstd/provenance/__init__.py`

- SHA-256：`5d41c776f9f6dc60188162960974541148fb4295b3e17320c905f72566f0fef0`
- 行数：`5`

````python
"""Artifact provenance and cache invalidation."""

from rc_irstd.provenance.fingerprint import command_fingerprint

__all__ = ["command_fingerprint"]
````

### 3.69 `rc_irstd/provenance/fingerprint.py`

- SHA-256：`f0a2f857ea4d28a8bbd43a335fe7b9f13a98cca4e5db70591e4f05547c5c748e`
- 行数：`86`

````python
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
````

### 3.70 `rc_irstd/provenance/manifest.py`

- SHA-256：`83e91c6c945ad61e5fffeb5111eeef74f799922ec45c2f3f3266321f72e07376`
- 行数：`29`

````python
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
````

### 3.71 `rc_irstd/utils/__init__.py`

- SHA-256：`8f6873342e6b0804d92aeb4c6861f0706ea857f3b3b542ade84310e181a46045`
- 行数：`4`

````python
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.seed import seed_everything

__all__ = ["resolve_device", "seed_everything"]
````

### 3.72 `rc_irstd/utils/arguments.py`

- SHA-256：`3ceb74960de74d62b186532bdd83914ce4176511d35a821afe4c224e6f4c221a`
- 行数：`24`

````python
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def parse_hw(values: Iterable[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    items = list(values)
    if len(items) != 2:
        raise ValueError("Image size must contain exactly two integers: H W")
    height, width = int(items[0]), int(items[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    return height, width


def existing_paths(values: Iterable[str]) -> list[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing paths: {missing}")
    return paths
````

### 3.73 `rc_irstd/utils/checkpoint.py`

- SHA-256：`219ad5ceb4fb7f014d7a20447de485e4d7568cacafb5a5b4c6c9528c7b9a4353`
- 行数：`21`

````python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
````

### 3.74 `rc_irstd/utils/config.py`

- SHA-256：`4ba934ccde96969507fe4cb32bda426ddfa392d2a1b8ec03fadc26259615bd45`
- 行数：`19`

````python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data
````

### 3.75 `rc_irstd/utils/device.py`

- SHA-256：`5e27b9558dca13575c554212dd3f3c21d24cc48ba7ebf9f6b90269b49a8cbbc0`
- 行数：`26`

````python
from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def create_grad_scaler(device: torch.device, enabled: bool):
    """Construct a GradScaler across PyTorch 2.x API variants."""
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch 2.0/2.1 compatibility
        return torch.cuda.amp.GradScaler(enabled=use_amp)
````

### 3.76 `rc_irstd/utils/io.py`

- SHA-256：`9159c2c8038571772d4362f7de8e3729bbbd59bd74d7703efeae3ddfb81d75a4`
- 行数：`94`

````python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def atomic_json_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_torch_save(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(data, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_npz(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz records found under {directory}")
    return files


def normalise_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        elif "net" in checkpoint:
            state = checkpoint["net"]
        else:
            state = checkpoint
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def as_jsonable_array(values: Iterable[Any]) -> list[Any]:
    return np.asarray(list(values)).tolist()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
````

### 3.77 `rc_irstd/utils/logging.py`

- SHA-256：`3891d93ea64aa5dcad074a650d48cd4a52a98875654eace86a831312f13fd659`
- 行数：`16`

````python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
````

### 3.78 `rc_irstd/utils/seed.py`

- SHA-256：`80a8b74929682408212d62f928f22913b2472fb0dbec2e6b9a53ad40ca0f066d`
- 行数：`20`

````python
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch with optional deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
````

### 3.79 `requirements.txt`

- SHA-256：`6d5338d4b76fc91bf122fe37249e2605b9d12cf11c952176c51f9062f3bf5861`
- 行数：`10`

````text
numpy>=1.24
scipy>=1.10
scikit-image>=0.20
pandas>=2.0
Pillow>=9.0
PyYAML>=6.0
torch>=2.0
tqdm>=4.65
tabulate>=0.9
pytest>=8.0
````

### 3.80 `scripts/aggregate_paper_results.sh`

- SHA-256：`bdf5638275752f198a20648d4b4df3998a3f2656a6c90c6895db9a88f3e33ed3`
- 行数：`7`

````bash
#!/usr/bin/env bash
set -euo pipefail
LODO_ROOT="${1:?Usage: $0 /path/to/lodo/output [output_dir]}"
OUTPUT_DIR="${2:-$LODO_ROOT/paper_tables}"
python -m rc_irstd.pipelines.aggregate_results \
  --lodo-root "$LODO_ROOT" \
  --output-dir "$OUTPUT_DIR"
````

### 3.81 `scripts/build_anonymous_supplement.sh`

- SHA-256：`33180cbd1db9dc8d627bdae6f77a5491c5eb6efc45917aeba535567420172f31`
- 行数：`8`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/dist/RC_IRSTD_Anonymous_Supplement.zip}"
mkdir -p "$(dirname "$OUTPUT")"
python -m rc_irstd.pipelines.build_supplement \
  --source-root "$ROOT" \
  --output "$OUTPUT"
````

### 3.82 `scripts/deploy_target.sh`

- SHA-256：`f87c07666d0511280aeca2521d2c6d0421fe6f5ed9e7f461c524159be33a1820`
- 行数：`48`

````bash
#!/usr/bin/env bash
set -euo pipefail

# Export unlabeled target scores, estimate an operating point from a past-only
# warm-up window, and apply it to future images.
#
# Usage:
#   ./scripts/deploy_target.sh DATASET SPLIT DETECTOR_PT CURVE_PT OUTPUT_DIR

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 DATASET_DIR SPLIT DETECTOR_CHECKPOINT CURVE_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi
DATASET_DIR="$1"
SPLIT="$2"
DETECTOR_CHECKPOINT="$3"
CURVE_CHECKPOINT="$4"
OUTPUT_DIR="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE_DIR="$OUTPUT_DIR/scores"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

python -m rc_irstd.pipelines.export_scores \
  --dataset-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --detector mshnet \
  --checkpoint "$DETECTOR_CHECKPOINT" \
  --inference-mode "${INFERENCE_MODE:-native_pad}" \
  --stride-multiple "${STRIDE_MULTIPLE:-32}" \
  --normalization "${NORMALIZATION:-imagenet}" \
  --dataset-type "${DATASET_TYPE:-iid_images}" \
  --no-include-mask \
  --device "${DEVICE:-cuda}" \
  --output-dir "$SCORE_DIR"

python -m rc_irstd.pipelines.run_deployment \
  --score-dir "$SCORE_DIR" \
  --curve-checkpoint "$CURVE_CHECKPOINT" \
  --warmup-size "${WARMUP_SIZE:-32}" \
  --update-every "${UPDATE_EVERY:-0}" \
  --pixel-budget "${PIXEL_BUDGET:-1e-6}" \
  --peak-budget "${PEAK_BUDGET:-1.0}" \
  --offset-index "${OFFSET_INDEX:-0}" \
  --ood-threshold "${OOD_THRESHOLD:-8.0}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "$OUTPUT_DIR/deployment"
````

### 3.83 `scripts/full_pipeline_start.sh`

- SHA-256：`917ea33d84ab7064a0f7f127c2274aaea22539b0f3326cd7873f9c13ad6e0575`
- 行数：`7`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/lodo_example.yaml}"
shift || true
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec python -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
````

### 3.84 `scripts/launch_lodo_fold.sh`

- SHA-256：`5804a941b2f0f41a8ae739f5a9ae1c3432724cf907151d1593e42fd287dd2d94`
- 行数：`11`

````bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/lodo_example.yaml}"
shift || true

# Examples:
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml --outer-target RealScene-ISTD
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml --stages detector export episodes
python -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
````

### 3.85 `scripts/mshnet_integration_test.sh`

- SHA-256：`14852529d6a34c6511a3a4f119ed01b4fdf10fe3f88c120d135280c70b3717d4`
- 行数：`8`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m pytest -q tests/test_mshnet_integration.py
````

### 3.86 `scripts/run_lodo.sh`

- SHA-256：`4954a41ab2fa22e763a71af331a4fa1e08edc285a86ca9057dd97a00c095b3ef`
- 行数：`5`

````bash
#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/lodo_example.yaml}"
shift || true
python -m rc_irstd.pipelines.run_lodo --config "${CONFIG}" "$@"
````

### 3.87 `scripts/setup.sh`

- SHA-256：`1f6a5efdfbb65b5eb752e9dca0e75cc6de6d5e2248c3661485bdf04b8c809d72`
- 行数：`6`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --upgrade pip
python -m pip install -e "${ROOT}[dev]"
echo "Installed rc-irstd from ${ROOT}"
````

### 3.88 `scripts/smoke_pipeline.sh`

- SHA-256：`224b92179d1878879823d2c1dc26aa2cba2613469d03b985ae293a5a8ccadec4`
- 行数：`8`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$ROOT/outputs/smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m rc_irstd.pipelines.smoke --work-dir "$WORK" --clean
````

### 3.89 `scripts/smoke_test.sh`

- SHA-256：`959e26b03a8eb8d2944ef8c5c9834c2a581ee182582f096c6de01b833e3cab05`
- 行数：`11`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$ROOT/outputs/smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m rc_irstd.pipelines.smoke --work-dir "$WORK" --clean
python -m pytest -q
echo "Smoke test completed: $WORK"
````

### 3.90 `scripts/start_training.sh`

- SHA-256：`6a7b177408c0f9aaba17190fce73fc27cca51324901598fe2e28e74c5b305fd7`
- 行数：`29`

````bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
shift || true

case "$MODE" in
  detector)
    exec "$ROOT/scripts/train_detector_mshnet.sh" "$@"
    ;;
  lodo)
    CONFIG="${1:-$ROOT/configs/lodo_example.yaml}"
    shift || true
    exec python -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
    ;;
  smoke)
    exec "$ROOT/scripts/smoke_test.sh" "$@"
    ;;
  *)
    cat >&2 <<EOF
Usage:
  $0 detector /data/sourceA /data/sourceB [...] [-- extra detector options]
  $0 lodo configs/lodo_example.yaml [--outer-target TARGET] [--dry-run]
  $0 smoke [work-directory]
EOF
    exit 2
    ;;
esac
````

### 3.91 `scripts/train_detector.sh`

- SHA-256：`279210ca6b68a991b0cb417da26fbb0a594ac730dc5ba7260b9f8241f3696c60`
- 行数：`4`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train_detector_mshnet.sh" "$@"
````

### 3.92 `scripts/train_detector_mshnet.sh`

- SHA-256：`91aff127709d1bfdb1c2a28a89ba281c3bd88c59b3587d2d1c7d71a3bf9abe4d`
- 行数：`90`

````bash
#!/usr/bin/env bash
set -euo pipefail

# Train the bundled risk-aware MSHNet on balanced source-domain batches.
#
# Usage:
#   ./scripts/train_detector_mshnet.sh /data/NUAA-SIRST /data/NUDT-SIRST /data/IRSTD-1K
#   SOURCE_DATASETS=/data/A:/data/B ./scripts/train_detector_mshnet.sh
#
# Extra Python arguments follow "--":
#   ./scripts/train_detector_mshnet.sh /data/A /data/B -- --epochs 40 --lr 0.02

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/detector_mshnet}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

DATASETS=()
EXTRA_ARGS=()
SEEN_SEPARATOR=0
for value in "$@"; do
  if [[ "$value" == "--" && "$SEEN_SEPARATOR" -eq 0 ]]; then
    SEEN_SEPARATOR=1
    continue
  fi
  if [[ "$SEEN_SEPARATOR" -eq 0 ]]; then
    DATASETS+=("$value")
  else
    EXTRA_ARGS+=("$value")
  fi
done

if [[ ${#DATASETS[@]} -eq 0 && -n "${SOURCE_DATASETS:-}" ]]; then
  IFS=':' read -r -a DATASETS <<< "$SOURCE_DATASETS"
fi
if [[ ${#DATASETS[@]} -lt 1 ]]; then
  echo "Usage: $0 /data/sourceA [/data/sourceB ...] [-- extra-options]" >&2
  exit 2
fi
for dataset in "${DATASETS[@]}"; do
  if [[ ! -d "$dataset" ]]; then
    echo "ERROR: dataset directory does not exist: $dataset" >&2
    exit 2
  fi
done

PER_DOMAIN_BATCH="${PER_DOMAIN_BATCH:-2}"
BATCH_SIZE="$((PER_DOMAIN_BATCH * ${#DATASETS[@]}))"
COMMAND=(
  python -m rc_irstd.pipelines.train_detector
  --train-split "${TRAIN_SPLIT:-train}"
  --val-split "${VAL_SPLIT:-test}"
  --detector mshnet
  --base-loss auto
  --resize "${RESIZE_H:-256}" "${RESIZE_W:-256}"
  --normalization "${NORMALIZATION:-imagenet}"
  --dataset-type "${DATASET_TYPE:-iid_images}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS:-400}"
  --warm-epoch "${WARM_EPOCH:-5}"
  --optimizer "${OPTIMIZER:-adagrad}"
  --lr "${LR:-0.05}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --lambda-tail "${LAMBDA_TAIL:-0.10}"
  --lambda-miss "${LAMBDA_MISS:-0.10}"
  --tail-quantile "${TAIL_QUANTILE:-0.95}"
  --miss-quantile "${MISS_QUANTILE:-0.80}"
  --peak-kernel "${PEAK_KERNEL:-5}"
  --exclusion-radius "${EXCLUSION_RADIUS:-2}"
  --worst-gamma "${WORST_GAMMA:-10.0}"
  --auxiliary-weight "${AUXILIARY_WEIGHT:-1.0}"
  --pixel-budget "${SELECTION_PIXEL_BUDGET:-1e-5}"
  --peak-budget "${SELECTION_PEAK_BUDGET:-5.0}"
  --num-workers "${NUM_WORKERS:-4}"
  --device "${DEVICE:-cuda}"
  --amp
  --deterministic
  --seed "${SEED:-42}"
  --output-dir "$RUN_ROOT"
)
for dataset in "${DATASETS[@]}"; do
  COMMAND+=(--source-dataset "$dataset")
done
COMMAND+=("${EXTRA_ARGS[@]}")

cd "$ROOT"
printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
````

### 3.93 `scripts/validate_release.sh`

- SHA-256：`309eda7e2da6efb80718cb935c88bf0b7617133632ecab4a3af470c9a770ab7b`
- 行数：`13`

````bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-/tmp/rc_irstd_release_validation}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m compileall -q rc_irstd tests
for script in scripts/*.sh; do bash -n "$script"; done
python -m pytest -q
python -m rc_irstd.pipelines.smoke --work-dir "$WORK" --clean
printf 'Release validation passed. Smoke artifacts: %s\n' "$WORK"
````

### 3.94 `tests/conftest.py`

- SHA-256：`7b436c0f7d1338a699da3e769cc7fabfd29ac2d3ffe979ba0e2545a9b784b2f5`
- 行数：`25`

````python
from __future__ import annotations

"""Test-suite resource policy.

Small convolutional smoke tests can become dramatically slower when a CI host
exposes a very large CPU thread pool.  Pinning PyTorch to one thread makes the
software validation deterministic and does not change training defaults.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only permits setting this before parallel work starts.  A plugin
    # may already have initialized it, in which case the intra-op limit above
    # is still sufficient for the tests.
    pass
````

### 3.95 `tests/test_crc.py`

- SHA-256：`9332c491f4b6e78963653547f6a0ad6bd392f9279214e5f365ecec00ca1ea575`
- 行数：`28`

````python
import numpy as np

from rc_irstd.calibration.crc import (
    adaptive_offset_loss_matrix,
    select_crc_parameter,
)


def test_crc_detects_small_sample_infeasibility() -> None:
    losses = np.zeros((5, 3), dtype=np.float64)
    result = select_crc_parameter(losses, np.asarray([0, 1, 2]), alpha=0.1)
    assert not result.feasible
    assert np.isclose(result.minimum_possible_corrected_risk, 1.0 / 6.0)


def test_adaptive_joint_loss_is_nested() -> None:
    pixel = np.asarray([[0.2, 0.1, 0.01], [0.3, 0.05, 0.0]])
    peak = np.asarray([[3.0, 1.0, 0.0], [4.0, 2.0, 0.0]])
    losses, selected = adaptive_offset_loss_matrix(
        pixel,
        peak,
        base_indices=np.asarray([0, 0]),
        offsets=np.asarray([0, 1, 2]),
        pixel_budget=0.1,
        peak_budget=1.5,
    )
    assert selected.shape == losses.shape
    assert np.all(np.diff(losses, axis=1) <= 0)
````

### 3.96 `tests/test_data_protocols.py`

- SHA-256：`559aab2447c578fddc260aa2a2dea0d4a86f836d9892424fc0bad61bc58ce03d`
- 行数：`64`

````python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.data.dataset import IRSTDDataset
from rc_irstd.data.transforms import load_image_preserve_depth, target_preserving_resize_mask
from rc_irstd.data.windows import build_iid_windows


def _make_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "img_idx").mkdir(parents=True)
    image = np.linspace(0, 65535, 8 * 12, dtype=np.uint16).reshape(8, 12)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[7, 11] = 255
    Image.fromarray(image).save(root / "images" / "0001.png")
    Image.fromarray(mask).save(root / "masks" / "0001.png")
    (root / "img_idx" / "train.txt").write_text("0001\n", encoding="utf-8")


def test_16bit_loader_and_target_preserving_resize(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    loaded = load_image_preserve_depth(tmp_path / "images" / "0001.png")
    assert loaded.bit_depth == 16
    assert loaded.array.dtype == np.uint16

    dataset = IRSTDDataset(
        tmp_path,
        split="train",
        resize_hw=(4, 18),  # mixed down/up resize
        normalization="percentile",
        include_component_labels=True,
    )
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 4, 18)
    assert sample["meta"].bit_depth == 16
    assert int(sample["mask"].sum()) >= 1
    assert int(sample["component_labels"].max()) == 1


def test_target_preserving_resize_never_drops_single_pixel() -> None:
    mask = np.zeros((17, 23), dtype=np.uint8)
    mask[16, 22] = 1
    for target in ((3, 5), (3, 40), (40, 5), (40, 50)):
        resized = target_preserving_resize_mask(mask, target)
        assert resized.shape == target
        assert resized.max() == 1


def test_iid_windows_are_deterministic_and_disjoint() -> None:
    first = build_iid_windows(30, context_size=5, horizon=3, stride=8, seed=11)
    second = build_iid_windows(30, context_size=5, horizon=3, stride=8, seed=11)
    assert first == second
    used: set[int] = set()
    for window in first:
        assert window.protocol == "iid"
        block = set(window.context_indices) | set(window.future_indices)
        assert not set(window.context_indices) & set(window.future_indices)
        assert not used & block
        used |= block
````

### 3.97 `tests/test_dataset_paths.py`

- SHA-256：`b85f1eb0ef3dc4cf6cc7d25feb9f16dc49a7ddbe0a808500ca7336e8a313f275`
- 行数：`39`

````python
from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.data.dataset import IRSTDDataset


def _write_sample(root: Path, relative: str, value: int) -> None:
    image_path = root / "images" / relative
    mask_path = root / "masks" / relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 9, 3), value, dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[3:5, 4:6] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)


def test_nested_split_paths_preserve_unique_image_ids(tmp_path: Path) -> None:
    root = tmp_path / "NestedDomain"
    _write_sample(root, "scene_a/frame_0001.png", 32)
    _write_sample(root, "scene_b/frame_0001.png", 64)
    split = root / "img_idx" / "test.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(
        "scene_a/frame_0001.png\nscene_b/frame_0001.png\n",
        encoding="utf-8",
    )

    dataset = IRSTDDataset(root, split="test", resize_hw=(8, 9))
    first = dataset[0]["meta"]
    second = dataset[1]["meta"]

    assert first.image_id == "scene_a/frame_0001"
    assert second.image_id == "scene_b/frame_0001"
    assert first.image_id != second.image_id
    assert first.sequence_id != second.sequence_id
````

### 3.98 `tests/test_deployment_and_calibration_units.py`

- SHA-256：`723bb041447c4bfe26a0d8614269641372db97cc4622d876ae5e846b7f38cd85`
- 行数：`137`

````python
from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_irstd.calibration.samples import image_calibration_samples
from rc_irstd.deployment.monitor import feature_ood_score, score_drift
from rc_irstd.deployment.session import DeploymentState, ThresholdUpdate
from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.models.risk_curve import FeatureNormaliser


def _episode_arrays() -> EpisodeArrays:
    thresholds = np.asarray([0.0, 0.5, 1.000001], dtype=np.float32)
    pixel_risk = np.asarray([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    peak_risk = np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    pd = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    return EpisodeArrays(
        features=np.zeros((2, 4), dtype=np.float32),
        pixel_log_risk=np.log10(np.maximum(pixel_risk, 1e-12)),
        peak_log_risk=np.log10(np.maximum(peak_risk, 1e-6)),
        pixel_risk=pixel_risk,
        peak_risk=peak_risk,
        pd=pd,
        context_pixel_upper=np.zeros_like(pixel_risk),
        context_peak_upper=np.zeros_like(peak_risk),
        thresholds=thresholds,
        domains=np.asarray(["A", "A"]),
        sequences=np.asarray(["iid0", "iid1"]),
        context_ids=np.asarray(['["c0"]', '["c1"]']),
        future_ids=np.asarray(['["f0"]', '["f1"]']),
        feature_names=("a", "b", "c", "d"),
        protocols=np.asarray(["iid", "iid"]),
        future_pixel_risk=np.asarray(
            [[[0.2, 0.0, 0.0]], [[0.3, 0.0, 0.0]]], dtype=np.float32
        ),
        future_peak_risk=np.asarray(
            [[[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]], dtype=np.float32
        ),
        future_pd=np.asarray(
            [[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]], dtype=np.float32
        ),
        future_gt_count=np.ones((2, 1), dtype=np.int32),
    )


def test_image_calibration_samples_count_exact_images() -> None:
    arrays = _episode_arrays()
    samples = image_calibration_samples(
        arrays,
        base_indices=np.asarray([1, 1]),
        base_rejected=np.asarray([False, False]),
    )
    assert samples.unit == "image"
    assert samples.num_samples == 2
    assert samples.label_count_per_sample.tolist() == [1, 1]
    assert samples.sample_ids.tolist() == ["f0", "f1"]


def test_deployment_state_and_monitor() -> None:
    normaliser = FeatureNormaliser(
        mean=np.asarray([1.0, 2.0], dtype=np.float32),
        std=np.asarray([2.0, 4.0], dtype=np.float32),
    )
    assert feature_ood_score(np.asarray([1.0, 2.0]), normaliser) == 0.0
    assert score_drift(np.asarray([1.0, 1.0]), np.asarray([2.0, 1.0])) > 0.0

    state = DeploymentState(
        detector_checkpoint="det.pt",
        curve_checkpoint="curve.pt",
        score_directory="scores",
        pixel_budget=1e-6,
        peak_budget_per_mp=1.0,
        warmup_size=32,
    )
    state.add(
        ThresholdUpdate(
            sequence_id="default",
            update_index=32,
            warmup_ids=("a", "b"),
            base_threshold_index=3,
            offset_index=1,
            final_threshold_index=4,
            threshold=0.8,
            predicted_pixel_risk=1e-7,
            predicted_peak_risk_per_mp=0.5,
            rejected=False,
            feature_ood_score=1.2,
        )
    )
    payload = state.to_dict()
    assert payload["updates"][0]["threshold"] == 0.8
    assert payload["updates"][0]["warmup_ids"] == ["a", "b"]


def test_apply_operating_point_serialises_single_peak_coordinates(tmp_path: Path) -> None:
    import csv

    from rc_irstd.data.score_records import ScoreRecord, save_score_record
    from rc_irstd.pipelines.apply_operating_point import main as apply_main

    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    probability = np.zeros((7, 9), dtype=np.float32)
    probability[3, 5] = 0.9
    save_score_record(
        ScoreRecord(
            probability=probability,
            mask=None,
            image_stats=np.zeros(2, dtype=np.float32),
            image_stat_names=("mean", "std"),
            image_id="single",
            dataset_name="D",
            sequence_id="default",
            frame_index=0,
            original_hw=(7, 9),
        ),
        score_dir / "00000000.npz",
    )
    output_dir = tmp_path / "applied"
    apply_main(
        [
            "--score-dir",
            str(score_dir),
            "--threshold",
            "0.5",
            "--peak-min-distance",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    with (output_dir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert (int(rows[0]["y"]), int(rows[0]["x"])) == (3, 5)
````

### 3.99 `tests/test_episode_metrics.py`

- SHA-256：`33978437d5511808b537be20814d0229585b319899af14f07531ede18a82eeae`
- 行数：`17`

````python
import numpy as np

from rc_irstd.evaluation.curves import compute_image_curves, rates_from_counts


def test_pixel_and_fixed_peak_risks_are_monotone():
    score = np.zeros((20, 20), dtype=np.float32)
    score[2, 2] = 0.9
    score[10, 10] = 0.7
    score[15, 15] = 0.5
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[1:4, 1:4] = 1
    thresholds = np.linspace(0, 1, 101, dtype=np.float32)
    counts = compute_image_curves(score, mask, thresholds, peak_min_distance=1)
    rates = rates_from_counts(counts)
    assert np.all(np.diff(rates["pixel_false_rate"]) <= 0)
    assert np.all(np.diff(rates["peak_false_per_mp"]) <= 0)
````

### 3.100 `tests/test_feature_config.py`

- SHA-256：`129a70a95065b910873c0a34d4482b3a01bd347fa8e0787089e3c64745df8f86`
- 行数：`22`

````python
import numpy as np

from rc_irstd.features.window_stats import WindowFeatureConfig


def test_window_feature_config_round_trip() -> None:
    config = WindowFeatureConfig(
        survival_thresholds=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        quantiles=np.asarray([0.5, 0.95], dtype=np.float32),
        peak_min_distance=3,
        peak_min_score=0.02,
        peak_border=1,
        max_candidates_per_image=None,
    )
    restored = WindowFeatureConfig.from_dict(config.to_dict())

    assert np.array_equal(restored.survival_thresholds, config.survival_thresholds)
    assert np.array_equal(restored.quantiles, config.quantiles)
    assert restored.peak_min_distance == config.peak_min_distance
    assert restored.peak_min_score == config.peak_min_score
    assert restored.peak_border == config.peak_border
    assert restored.max_candidates_per_image is None
````

### 3.101 `tests/test_lodo_protocol.py`

- SHA-256：`de55494a9805ab55d6cd5f7bbd2efd827ffc0b049fa27d8ee96b4893e8e35847`
- 行数：`53`

````python
import json
from pathlib import Path

import yaml

from rc_irstd.pipelines.run_lodo import main


def test_lodo_separates_training_and_evaluation_strides(tmp_path: Path) -> None:
    config = {
        "python": "python",
        "working_directory": ".",
        "output_root": "outputs",
        "datasets": {
            "A": {"path": "data/A"},
            "B": {"path": "data/B"},
            "C": {"path": "data/C"},
        },
        "outer_targets": ["C"],
        "detector": {"name": "tiny", "device": "cpu", "amp": False},
        "episodes": {
            "context_size": 2,
            "horizon": 1,
            "train_stride": 1,
            "eval_stride": 3,
        },
    }
    config_path = tmp_path / "lodo.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main([
        "--config",
        str(config_path),
        "--outer-target",
        "C",
        "--stages",
        "detector",
        "episodes",
        "--dry-run",
    ])

    protocol = json.loads((tmp_path / "outputs" / "protocol.json").read_text())
    resolved = protocol["resolved_episode_protocol"]
    assert resolved["pseudo_train_stride"] == 1
    assert resolved["target_eval_stride"] == 3

    command_log = (
        tmp_path / "outputs" / "outer_C" / "commands.log"
    ).read_text(encoding="utf-8")
    assert "target_episodes.npz --context-size 2 --horizon 1 --stride 3" in command_log
    assert command_log.count("--context-size 2 --horizon 1 --stride 1") == 2
    assert command_log.count("--source-train-split train") >= 3
    assert command_log.count("--source-val-split test") >= 3
````

### 3.102 `tests/test_mshnet_integration.py`

- SHA-256：`e5f47b292e35e5fa0ad11ef8a984c3c4191bdc94b5299221186d40ae56e6b1fa`
- 行数：`70`

````python
from __future__ import annotations

from pathlib import Path

import torch

from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss
from rc_irstd.losses.sls import SLSIoULoss
from rc_irstd.models.detector_adapter import DetectorAdapter, build_detector
from rc_irstd.models.mshnet import MSHNet


def test_bundled_mshnet_forward_backward_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = MSHNet(
        input_channels=3,
        channels=(2, 4, 8, 16, 32),
        blocks=(1, 1, 1, 1),
    )
    adapter = DetectorAdapter(model, "mshnet")
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, 0, 8:10, 9:11] = 1.0
    masks[1, 0, 20:23, 21:24] = 1.0
    labels = torch.zeros_like(masks, dtype=torch.long)
    labels[0, 0, 8:10, 9:11] = 1
    labels[1, 0, 20:23, 21:24] = 1
    domains = torch.tensor([0, 1], dtype=torch.long)

    output = adapter(images, training_tag=True)
    assert output.logits.shape == (2, 1, 32, 32)
    assert len(output.auxiliary_logits) == 4
    assert [tuple(x.shape[-2:]) for x in output.auxiliary_logits] == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
    ]

    criterion = RiskAwareDetectorLoss(
        base_loss=SLSIoULoss(),
        lambda_tail=0.05,
        lambda_miss=0.05,
        auxiliary_weight=1.0,
    )
    losses = criterion(
        output.logits,
        masks,
        domains,
        auxiliary_logits=output.auxiliary_logits,
        component_labels=labels,
        warm_epoch=0,
        epoch=1,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint = tmp_path / "mshnet.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    # Verify the public/default architecture's checkpoint path separately using
    # its own state dict.  This catches adapter prefix and round-trip issues.
    default = build_detector("mshnet", device="cpu")
    default_checkpoint = tmp_path / "mshnet_default.pt"
    torch.save({"model_state": default.model.state_dict()}, default_checkpoint)
    reloaded = build_detector("mshnet", checkpoint=default_checkpoint, device="cpu")
    assert set(default.model.state_dict()) == set(reloaded.model.state_dict())
````

### 3.103 `tests/test_new_evaluation_and_provenance.py`

- SHA-256：`f5c44f29dd70522784f9fe9bfa52cc3989522ab70322ae32252370efedd95db0`
- 行数：`47`

````python
from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold
from rc_irstd.provenance.fingerprint import command_fingerprint, source_tree_fingerprint


def test_irstd_metrics_perfect_and_false_component() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:6, 5:7] = 1
    probability = mask.astype(np.float32)
    perfect = evaluate_irstd_at_threshold([probability], [mask], threshold=0.5)
    assert perfect.iou == 1.0
    assert perfect.niou == 1.0
    assert perfect.pd == 1.0
    assert perfect.false_components == 0

    noisy = probability.copy()
    noisy[13, 13] = 1.0
    result = evaluate_irstd_at_threshold([noisy], [mask], threshold=0.5)
    assert result.pd == 1.0
    assert result.false_components == 1
    assert result.false_components_per_mp > 0
    assert 0 < result.iou < 1


def test_fingerprint_changes_with_source_and_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    module = source / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"a": 1}\n', encoding="utf-8")
    command = ["python", "run.py", "--config", str(config)]
    first, _ = command_fingerprint(command, tmp_path, source)
    assert first == command_fingerprint(command, tmp_path, source)[0]

    config.write_text('{"a": 2}\n', encoding="utf-8")
    second, _ = command_fingerprint(command, tmp_path, source)
    assert second != first

    before_source = source_tree_fingerprint(source)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_fingerprint(source) != before_source
````

### 3.104 `tests/test_operating_point.py`

- SHA-256：`2a4b62b322b2c8399713b4cde2edd8ada37db0cdd5b5436593a498d6fb88efe8`
- 行数：`24`

````python
import numpy as np

from rc_irstd.episodes.builder import default_threshold_grid
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold


def test_threshold_grid_contains_empty_prediction_action() -> None:
    thresholds = default_threshold_grid()
    assert thresholds[-1] > 1.0


def test_empty_prediction_action_is_reported_as_rejection() -> None:
    thresholds = np.asarray([0.0, 0.5, 1.000001], dtype=np.float32)
    pixel_log = np.log10(np.asarray([1.0, 0.2, 1e-12]))
    peak_log = np.log10(np.asarray([10.0, 2.0, 1e-6]))
    point = select_dual_budget_threshold(
        thresholds,
        pixel_log,
        peak_log,
        pixel_budget=0.1,
        peak_budget_per_mp=1.0,
    )
    assert point.index == 2
    assert point.rejected
````

### 3.105 `tests/test_peaks.py`

- SHA-256：`38dd90962253a6e58a59a0f895d260018906154f30e80f11f4df8a3b541f0277`
- 行数：`41`

````python
import numpy as np

from rc_irstd.candidates.peaks import build_fixed_peak_set, fixed_peak_curves


def test_fixed_peak_false_count_is_monotone():
    score = np.zeros((16, 16), dtype=np.float32)
    score[3, 3] = 0.9
    score[7, 7] = 0.8
    score[12, 12] = 0.7
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[2:5, 2:5] = 1
    peaks = build_fixed_peak_set(score, mask, min_distance=1, min_score=0.1)
    thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    total, false, matched = fixed_peak_curves(peaks, thresholds)
    assert np.all(np.diff(total) <= 0)
    assert np.all(np.diff(false) <= 0)
    assert np.all(np.diff(matched) <= 0)


def test_duplicate_candidates_near_one_target_count_as_false() -> None:
    score = np.zeros((20, 20), dtype=np.float32)
    score[8, 8] = 0.90
    score[8, 12] = 0.80
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[8, 10] = 1

    peaks = build_fixed_peak_set(
        score,
        mask,
        min_distance=1,
        min_score=0.1,
        tolerance=3.0,
    )
    assert int((peaks.gt_ids > 0).sum()) == 1
    assert int((peaks.gt_ids == 0).sum()) == 1

    thresholds = np.asarray([0.0, 0.85, 0.95], dtype=np.float32)
    _, false, matched = fixed_peak_curves(peaks, thresholds)
    assert false.tolist() == [1, 0, 0]
    assert matched.tolist() == [1, 1, 0]
````

### 3.106 `tests/test_risk_aware_loss.py`

- SHA-256：`a39b366073d73cce1f874f175f9e5ae93e279e45dbf37de303035b728f1324df`
- 行数：`51`

````python
from __future__ import annotations

import torch

from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss


def _mean_logit_loss(logits, target):
    del target
    return logits.mean()


def test_auxiliary_base_loss_matches_weighted_reference_average() -> None:
    criterion = RiskAwareDetectorLoss(
        base_loss=_mean_logit_loss,
        lambda_tail=0.0,
        lambda_miss=0.0,
        auxiliary_weight=1.0,
    )
    final = torch.full((1, 1, 8, 8), 1.0)
    auxiliary = [
        torch.full((1, 1, 4, 4), 2.0),
        torch.full((1, 1, 2, 2), 4.0),
    ]
    target = torch.zeros_like(final)
    result = criterion(
        final,
        target,
        torch.zeros(1, dtype=torch.long),
        auxiliary_logits=auxiliary,
    )
    assert torch.isclose(result["base"], torch.tensor((1.0 + 2.0 + 4.0) / 3.0))
    assert torch.isclose(result["total"], result["base"])


def test_auxiliary_weight_zero_uses_final_map_only() -> None:
    criterion = RiskAwareDetectorLoss(
        base_loss=_mean_logit_loss,
        lambda_tail=0.0,
        lambda_miss=0.0,
        auxiliary_weight=0.0,
    )
    final = torch.full((1, 1, 8, 8), 1.5)
    target = torch.zeros_like(final)
    result = criterion(
        final,
        target,
        torch.zeros(1, dtype=torch.long),
        auxiliary_logits=[torch.full((1, 1, 4, 4), 10.0)],
    )
    assert torch.isclose(result["base"], torch.tensor(1.5))
````

### 3.107 `tests/test_risk_curve.py`

- SHA-256：`e7a7d7378fbf99fff6a240c8a5d37a3097c204af87029a6ebf51f6318074922c`
- 行数：`11`

````python
import torch

from rc_irstd.models.risk_curve import RiskCurvePredictor


def test_risk_curve_is_structurally_monotone():
    torch.manual_seed(0)
    model = RiskCurvePredictor(input_dim=12, num_thresholds=32, hidden_dim=16, dropout=0.0)
    output = model(torch.randn(7, 12))
    for curve in output.values():
        assert torch.all(curve[:, 1:] <= curve[:, :-1] + 1e-7)
````

### 3.108 `tests/test_sampler.py`

- SHA-256：`d1acc50b924f46dc87fd501c613bc4a6e5b91ef7ad0422e5dc26fc7e0390e5f9`
- 行数：`14`

````python
from rc_irstd.data.sampler import DomainBalancedBatchSampler


def test_balanced_sampler_yields_a_batch_when_per_domain_exceeds_domain_size() -> None:
    sampler = DomainBalancedBatchSampler(
        domain_ids=[0, 1],
        batch_size=8,
        shuffle=False,
        drop_last=True,
        seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 8
````

### 3.109 `tests/test_splits.py`

- SHA-256：`e518164802a635c7e2da0e5059669cceb0fc0446b86d2c3082db640bd267d4dc`
- 行数：`33`

````python
import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.episodes.splits import grouped_calibration_test_split


def _arrays() -> EpisodeArrays:
    n = 12
    t = 4
    return EpisodeArrays(
        features=np.zeros((n, 3), dtype=np.float32),
        pixel_log_risk=np.zeros((n, t), dtype=np.float32),
        peak_log_risk=np.zeros((n, t), dtype=np.float32),
        pixel_risk=np.zeros((n, t), dtype=np.float32),
        peak_risk=np.zeros((n, t), dtype=np.float32),
        pd=np.zeros((n, t), dtype=np.float32),
        context_pixel_upper=np.zeros((n, t), dtype=np.float32),
        context_peak_upper=np.zeros((n, t), dtype=np.float32),
        thresholds=np.linspace(0, 1, t, dtype=np.float32),
        domains=np.asarray(["target"] * n),
        sequences=np.asarray(["s0"] * 4 + ["s1"] * 4 + ["s2"] * 4),
        context_ids=np.asarray(["[]"] * n),
        future_ids=np.asarray(["[]"] * n),
        feature_names=("a", "b", "c"),
    )


def test_calibration_test_are_sequence_disjoint() -> None:
    arrays = _arrays()
    calibration, test = grouped_calibration_test_split(arrays, calibration_size=3, seed=7)
    assert len(calibration) == 3
    assert len(test) > 0
    assert set(arrays.sequences[calibration]).isdisjoint(set(arrays.sequences[test]))
````

### 3.110 `tests/test_windows.py`

- SHA-256：`5c3fb67c789acefa8bcd303f44727c500b22a007d2244fdbafe2dc89a0ba15b3`
- 行数：`12`

````python
from rc_irstd.data.windows import build_causal_windows


def test_causal_windows_are_disjoint_and_sequence_local():
    sequences = ["a"] * 10 + ["b"] * 10
    frames = list(range(10)) + list(range(10))
    windows = build_causal_windows(sequences, frames, context_size=4, horizon=2, stride=2)
    assert windows
    for window in windows:
        assert set(window.context_indices).isdisjoint(window.future_indices)
        expected_range = range(0, 10) if window.sequence_id == "a" else range(10, 20)
        assert all(index in expected_range for index in window.context_indices + window.future_indices)
````
