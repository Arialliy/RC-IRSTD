# Infrared Small Target Detection with Scale and Location Sensitivity

## Notice! 📰
First of all, thank you to all relevant workers for your attention. Recently, many people have discovered some obvious errors in the code, so we re-checked, modified and debugged the code. Surprisingly, we unexpectedly obtained a pretty good result on the IRSTD-1k data set. The results are published below for your reference.
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.87 | 92.86 | 8.88 | [new_weights](https://drive.google.com/file/d/1CSDwQG8xg7hv0_oGKa4NCEWUiMRU7eIs/view?usp=sharing) |

## Overview
![](assert/overview.png)

## Introduction
This repository is the official implementation of our CVPR 2024 paper [Infrared Small Target Detection with Scale and Location Sensitivity](https://arxiv.org/abs/2403.19366).

In this paper, we first propose a novel Scale and Location Sensitive (SLS) loss to handle the limitations of existing losses: 1) for scale sensitivity, we compute a weight for the IoU loss based on target scales to help the detector distinguish targets with different scales: 2) for location sensitivity, we introduce a penalty term based on the center points of targets to help the detector localize targets more precisely. Then, we design a simple Multi-Scale Head to the plain U-Net (MSHNet). By applying SLS loss to each scale of the predictions, our MSHNet outperforms existing state-of-the-art methods by a large margin. In addition, the detection performance of existing detectors can be further improved when trained with our SLS loss, demonstrating the effectiveness and generalization of our SLS loss. The contribution of this paper are as follows:

1. We propose a novel scale and location sensitive loss for infrared small target detection, which helps detectors distinguish objects with different scales and locations.
   
2. We propose a simple but effective detector which achieves SOTA performance without bells and whistles.
   
3. We apply our loss to existing detectors and show that the detection performance can be further boosted.

## Training
The training command is very simple like this:

```
python main.py --dataset-dir <DATASET> --batch-size 4 --epochs 400 --lr 0.05 --mode train
```

For example:
```
python main.py --dataset-dir '/dataset/IRSTD-1k' --batch-size 4 --epochs 400 --lr 0.05 --mode 'train'
```

This repo also provides separate entrypoints:
```
python3 train.py --dataset-dir datasets/IRSTD-1K --batch-size 4 --epochs 400 --lr 0.05
python3 train.py --dataset-dir datasets/NUDT-SIRST --batch-size 4 --epochs 400 --lr 0.05
```

Training now defaults to a fixed-last policy: it never constructs the official
test loader and saves `weight-last.pkl` plus `checkpoint.pkl` under
`repro_runs/`. The historical behavior that evaluates test every epoch and
selects `weight.pkl` is available only with `--allow-test-selection`; such a
run is explicitly legacy/non-claim-bearing and must not be used in RC/AAAI
results.

## Testing
You can test the model with the following command:
```
python main.py --dataset-dir '/dataset/IRSTD-1k' --batch-size 4 --mode 'test' --weight-path '/weight/MSHNet_weight.tar'
```

Or use the separate testing entrypoint:
```
python3 test.py --dataset-dir datasets/IRSTD-1K --weight-path repro_runs/MSHNet-YYYY-MM-DD-HH-MM-SS/weight-last.pkl
```

The dataset loader supports both the original `trainval.txt`/`test.txt` layout
and the local `img_idx/train_*.txt`/`img_idx/test_*.txt` layout. `val` and
`test` are distinct roles: a missing validation split never falls back to the
official test split.

## RC-IRSTD research extension (experimental)

This worktree also contains an experimental, budget-aware cross-domain deployment pipeline. It is not part of the original CVPR 2024 release. The direct threshold calibrator remains a baseline. An end-to-end `monotone_pixel` calibrator is available as a method candidate, but its current objective is still asymmetric oracle-threshold regression plus reject BCE; the query risk-aligned loss remains a separate evidence gate and must not be claimed as completed.

Install the local requirements and run the regression suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Audit the exact local train/test splits, image-byte separation, mask geometry,
and nested-LODO eligibility before launching an experiment:

```bash
python -m scripts.audit_aaai_protocol \
  --dataset-dirs datasets/IRSTD-1K datasets/NUAA-SIRST datasets/NUDT-SIRST \
  --outer-target NUAA-SIRST \
  --pseudo-target NUDT-SIRST \
  --output repro_runs/validation/aaai_protocol_audit.json
```

Train a balanced multi-source detector without constructing any target/test
loader or selecting a checkpoint on target labels. The explicit `margin`
candidate compares hard target-object logits against the background local-peak
tail in logit-difference space; a common logit shift therefore cancels. Its
reduction order is fixed as image first, equal-image domain mean, then the
normalized smooth worst-domain aggregation. `--risk-objective separate`
remains the default compatibility baseline and keeps the original probability-
space background-tail and miss losses separate.

```bash
python -m scripts.train_multisource_tail \
  --source-dirs datasets/IRSTD-1K datasets/NUDT-SIRST \
  --source-names IRSTD-1K NUDT-SIRST \
  --outer-fold-id outer-nuaa \
  --outer-target NUAA-SIRST \
  --held-out-domains NUAA-SIRST \
  --risk-objective margin \
  --lambda-margin 0.1 \
  --target-background-margin 1.0 \
  --batch-per-domain 2 \
  --epochs 400 \
  --device cuda \
  --save-dir outputs/detectors \
  --run-name outer-nuaa
```

For the designated physical GPUs 0, 1, and 2, use the checked wrapper. It
enables `DataParallel` and defaults to a per-domain batch of three so every
replica receives the same interleaved domain mixture:

```bash
PYTHON_BIN=python ./scripts/train_rc_3gpu.sh \
  --source-dirs datasets/IRSTD-1K datasets/NUDT-SIRST \
  --source-names IRSTD-1K NUDT-SIRST \
  --outer-fold-id outer-nuaa \
  --outer-target NUAA-SIRST \
  --held-out-domains NUAA-SIRST \
  --risk-objective margin \
  --epochs 400 \
  --save-dir outputs/detectors \
  --run-name outer-nuaa-s42
```

With only the three local datasets, fixing both an outer target and an inner
pseudo-target leaves a single detector source. Such nested runs are smoke
tests only; they are not claim-bearing strict nested LODO evidence.

Score manifests now freeze an official-split contract. Calibration episodes
must be built from the pseudo-target's official training split; both the
optimisation episodes and the pseudo-target validation episodes are rejected
before fitting or best-checkpoint selection if their role is test, unknown, or
legacy. Export each pseudo-target calibration stream explicitly:

```bash
python -m evaluation.export_score_maps \
  --dataset-dir datasets/<PSEUDO_TARGET> \
  --weight-path outputs/detectors/<INNER_FOLD>/checkpoint_last.pt \
  --output-dir outputs/scores/<INNER_FOLD>-train \
  --split train \
  --split-role official_train \
  --device cuda
```

The manifest binds and replays both official split files, their hashes and
ordered IDs, and every selected raw-image hash. Any train/test ID or image-byte
overlap is fatal. Meta episodes using this proof are schema v4; legacy v3
episodes remain readable only for diagnostics and cannot train a calibrator.

The monotone method path uses the pixel false-alarm budget only. It predicts
threshold and rejection curves that cannot decrease as the pixel budget is
tightened, and prohibits extrapolation outside the frozen grid. Traditional
connected-component false alarms remain a compatibility evaluation metric:
their counts can rise when a threshold fragments one region, so they must not
be presented as a strictly monotone inverse-risk constraint.
Every context/query curve group must supervise the complete frozen budget
grid; grid coverage, ordering, interpolation policy, and policy SHA are bound
into the checkpoint and independently replayed online.

```bash
python -m rc.train_calibrator \
  --episodes outputs/episodes/outer-nuaa.jsonl \
  --val-pseudo-target <HELD_OUT_PSEUDO_TARGET> \
  --output-dir outputs/rc/outer-nuaa \
  --deployment-detector-checkpoint-sha <SHA256> \
  --deployment-detector-source-domain IRSTD-1K \
  --deployment-detector-source-domain NUDT-SIRST \
  --deployment-source-reference outputs/references/outer-nuaa.npz \
  --calibrator-model monotone_pixel \
  --pixel-budget-grid 1e-4 1e-5 1e-6
```

Domain statistics now use the bounded `rc-domain-statistics-v3` contract.
Existing v2 source references, episodes, and calibrator checkpoints must be
rebuilt; they are rejected rather than silently mixed with the new estimator.
The official-split proof likewise requires rebuilding old role-less score
manifests and meta episodes before any claim-bearing calibration run.

Export native-resolution continuous score maps. This inference stage is
structurally label-free: it neither resolves masks nor embeds them in score
NPZ files. Before opening final-target labels, freeze the calibrator,
context/query sizes, budgets, rejection rule, and the resulting online adapter
JSON:

```bash
python -m evaluation.export_score_maps \
  --dataset-dir datasets/NUAA-SIRST \
  --weight-path outputs/detectors/outer-nuaa/checkpoint_last.pt \
  --output-dir outputs/scores/outer-nuaa \
  --split test \
  --split-role official_test \
  --device cuda

python -m rc.online_adapter \
  --manifest outputs/scores/outer-nuaa/manifest.json \
  --calibrator-checkpoint outputs/rc/outer-nuaa/calibrator.pt \
  --target-domain NUAA-SIRST \
  --context-size 32 \
  --query-size 32 \
  --pixel-budget 1e-6 \
  --output outputs/rc/outer-nuaa/online.json

sha256sum outputs/rc/outer-nuaa/online.json \
  > outputs/rc/outer-nuaa/online.json.sha256
```

Claim-bearing online adaptation and replay require the `official_test` role.
Role-less schema-v2 score manifests are accepted only in explicitly diagnostic
paths and cannot enter calibration or paper-result replay.

Only after that freeze should labels be attached and query metrics replayed.
`threshold_sweep` and any operating point selected from its label-using curve
are oracle diagnostics/upper bounds, never deployed method outputs:

```bash

python -m evaluation.export_label_maps \
  --dataset-dir datasets/NUAA-SIRST \
  --score-manifest outputs/scores/outer-nuaa/manifest.json \
  --output-dir outputs/labels/outer-nuaa

python -m evaluation.threshold_sweep \
  --score-dir outputs/scores/outer-nuaa \
  --label-manifest outputs/labels/outer-nuaa/label-manifest.json \
  --image-id-file outputs/splits/outer-nuaa-query.txt \
  --threshold-mode adaptive \
  --output outputs/curves/outer-nuaa-query.csv

python -m evaluation.evaluate_adapter_output \
  --adapter-output outputs/rc/outer-nuaa/online.json \
  --score-manifest outputs/scores/outer-nuaa/manifest.json \
  --calibrator-checkpoint outputs/rc/outer-nuaa/calibrator.pt \
  --label-manifest outputs/labels/outer-nuaa/label-manifest.json \
  --output outputs/rc/outer-nuaa/evaluation.json
```

The curve sidecar records the score- and label-manifest hashes, query IDs,
detector hash, matching contract, event-threshold coverage, and whether the
sweep is globally exact. Verified episode construction rederives the threshold
plan and every curve field from the bound query scores and labels before
selecting an oracle. Hand-asserted or mismatched provenance is rejected.

Replay verifies the actual calibrator checkpoint SHA and deterministically reruns
the context-to-threshold/reject decision on CPU before opening the independent
query-label artifact. A rejected decision returns without opening that artifact.
Manifest order is reported as `prefix_holdout` by default; the optional
`--assert-temporal-order` records a user assertion, not independent temporal
verification.

The complete nested-LODO artifact contract and episode JSON example are in [02_RC-IRSTD_方案_代码_步骤.md](02_RC-IRSTD_方案_代码_步骤.md). Engineering-only smoke results and unresolved evidence gaps are recorded in [baseline_results.md](baseline_results.md).

## Visual Results
![](assert/visual_result.png)

## Original MSHNet release results (not RC-IRSTD)
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.16 | 93.88 | 15.03 | [IRSTD-1k_weights](https://drive.google.com/file/d/1q3zfzJRczodGQb0dZ3y3KmLn0zz4F8ra/view?usp=drive_link) |
| NUDT-SIRST | 80.55 | 97.99 | 11.77 | [NUDT-SIRST_weights](https://drive.google.com/file/d/1uczanUIHePZqJA79RZu25fv9FNSHSDQZ/view?usp=drive_link) |


## Citation
**Please kindly cite the papers if this code is useful and helpful for your research.**

    @inproceedings{liu2024infrared,
      title={Infrared Small Target Detection with Scale and Location Sensitivity},
      author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
      booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
      year={2024}
    }
