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

Training checkpoints and best weights are saved under `repro_runs/` by default.

## Testing
You can test the model with the following command:
```
python main.py --dataset-dir '/dataset/IRSTD-1k' --batch-size 4 --mode 'test' --weight-path '/weight/MSHNet_weight.tar'
```

Or use the separate testing entrypoint:
```
python3 test.py --dataset-dir datasets/IRSTD-1K --weight-path repro_runs/MSHNet-YYYY-MM-DD-HH-MM-SS/weight.pkl
```

The dataset loader supports both the original `trainval.txt`/`test.txt` layout and the local `img_idx/train_*.txt`/`img_idx/test_*.txt` layout.

## RC-IRSTD research extension (experimental)

This worktree also contains an experimental, budget-aware cross-domain deployment pipeline. It is not part of the original CVPR 2024 release, and the current direct threshold calibrator is a baseline rather than the proposed monotone inverse-risk upgrade discussed in the AAAI review notes.

Install the local requirements and run the regression suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Train a balanced multi-source detector without constructing any target/test loader or selecting a checkpoint on target labels:

```bash
python -m scripts.train_multisource_tail \
  --source-dirs datasets/IRSTD-1K datasets/NUDT-SIRST \
  --source-names IRSTD-1K NUDT-SIRST \
  --outer-fold-id outer-nuaa \
  --outer-target NUAA-SIRST \
  --held-out-domains NUAA-SIRST \
  --batch-per-domain 2 \
  --epochs 400 \
  --device cuda \
  --save-dir outputs/detectors \
  --run-name outer-nuaa
```

Export native-resolution continuous score maps and build a query-only, adaptive high-tail curve:

```bash
python -m evaluation.export_score_maps \
  --dataset-dir datasets/NUAA-SIRST \
  --weight-path outputs/detectors/outer-nuaa/checkpoint_last.pt \
  --output-dir outputs/scores/outer-nuaa \
  --device cuda

python -m evaluation.threshold_sweep \
  --score-dir outputs/scores/outer-nuaa \
  --image-id-file outputs/splits/outer-nuaa-query.txt \
  --threshold-mode adaptive \
  --output outputs/curves/outer-nuaa-query.csv
```

The curve sidecar records the score-manifest hash, query IDs, detector hash, event-threshold coverage, and whether the sweep is globally exact. RC meta-training rejects hand-asserted or mismatched provenance.

Online adaptation consumes only a manifest prefix. Label-using metrics are produced afterward by a separate, hash-bound query replay:

```bash
python -m rc.online_adapter \
  --manifest outputs/scores/outer-nuaa/manifest.json \
  --calibrator-checkpoint outputs/rc/outer-nuaa/calibrator.pt \
  --target-domain NUAA-SIRST \
  --context-size 32 \
  --pixel-budget 1e-6 \
  --component-budget 1.0 \
  --output outputs/rc/outer-nuaa/online.json

python -m evaluation.evaluate_adapter_output \
  --adapter-output outputs/rc/outer-nuaa/online.json \
  --score-manifest outputs/scores/outer-nuaa/manifest.json \
  --calibrator-checkpoint outputs/rc/outer-nuaa/calibrator.pt \
  --output outputs/rc/outer-nuaa/evaluation.json
```

Replay verifies the actual calibrator checkpoint SHA and deterministically reruns
the context-to-threshold/reject decision on CPU before reading query labels.

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
