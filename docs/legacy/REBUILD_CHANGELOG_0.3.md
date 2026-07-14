# RC-IRSTD Rebuild Changelog

## 0.3.0 — 2026-07-14

### Model and losses

- Bundled a self-contained MSHNet implementation with public checkpoint-style module names.
- Bundled a numerically guarded SLS-IoU implementation.
- Reworked background risk as per-image local-peak CVaR followed by domain aggregation.
- Added precomputed component-label support for hard-target Miss-CVaR.
- Added an explicit TinyUNet path for software smoke tests only.

### Data and inference

- Added 8/16-bit preserving image loading and configurable normalisation.
- Added max-preserving target-mask downsampling.
- Added `resize`, `native_pad`, and `tiled` inference protocols.
- Added explicit `iid_images` and `temporal` dataset semantics.

### Training and model selection

- Added deterministic worker seeding and RNG checkpoint/resume.
- Added source-validation budget checkpoint selection.
- Training now emits `best_budget.pt`, `best_iou.pt`, `best.pt`, and `last.pt`.
- Risk-curve training now uses budget-focused weights and crossing loss.
- Risk-curve checkpoints are selected by selected-point excess and effective Pd.

### Calibration and evaluation

- Split image-shot calibration from block/episode calibration.
- Added formal fixed-peak/pixel curves and literature-compatible component curves.
- Added IoU, nIoU, hIoU, precision, recall, F1, object Pd, and component FA/MP.
- Preserved structural monotonicity of the dual risk-curve predictor.

### Deployment and orchestration

- Added threshold application, mask/candidate output, deployment state, rolling updates, and OOD diagnostics.
- Added source-set detector deduplication and artifact fingerprints to Nested LODO.
- Added complete static-IID and temporal example configurations.
- Added direct training, full-pipeline, deployment, integration, and release-validation launchers.

### Validation

- Added MSHNet forward/backward/checkpoint integration tests.
- Added 16-bit, target-preserving resize, IID protocol, calibration-unit, metrics, provenance, deployment, and candidate-coordinate tests.
- Current test suite: 25 tests.
- Synthetic detector → scores → episodes → risk curve → zero-label → CRC path passes.
