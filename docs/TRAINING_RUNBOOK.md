# RC-IRSTD 分阶段训练手册

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

目标是先证伪/证实关键假设，再扩展计算量。不要从零直接启动所有 outer folds、400 epochs 和 3 seeds。

## 0. 环境与仓库指纹

```bash
cd RC-IRSTD_AAAI27_TwoStage_NoReject
python -m pip install -e ".[dev]"

python --version | tee environment.python.txt
python - <<'PY' | tee environment.torch.txt
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
python -m pip freeze > environment.freeze.txt
sha256sum pyproject.toml VERSION > code_fingerprint.txt
```

若从 GitHub 工作树启动，额外保存：

```bash
git rev-parse HEAD | tee TRAINING_COMMIT.txt
git branch --show-current | tee TRAINING_BRANCH.txt
git status --porcelain | tee TRAINING_DIRTY.txt
git diff --check
```

## 1. 软件自检

```bash
bash scripts/validate_two_stage_release.sh /tmp/rc_irstd_two_stage_validation
```

必须全部通过：compileall、Shell syntax、pytest、3-epoch synthetic two-stage smoke、MSHNet integration。

## 2. Gate 0：数据审计

为每个数据集建立：

```text
images/
masks/
img_idx/
  detector_train.txt
  detector_val.txt
  meta_train.txt
  meta_val.txt
  test.txt
```

执行检查：

- image/mask 一一对应；
- mask canvas 与 image canvas；
- 8/16-bit 读取；
- 重复 ID 与跨数据集近重复；
- 序列边界；
- 每个 split 的图像数、目标数、空目标图数、分辨率分布；
- 各角色 ID 交集为空。

记录到 `outputs/audit/<dataset>/`。任何静默选择多个 mask 候选的 loader 都应 fail closed。

## 3. Gate 1：一轮集成训练

先在两个或三个源域运行 1 epoch，验证真实数据接口：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT=outputs/integration/detector_1ep \
EPOCHS=1 PER_DOMAIN_BATCH=1 \
LAMBDA_SEP=0.20 RISK_START_EPOCH=0 RISK_RAMP_EPOCHS=1 \
bash scripts/start_training.sh detector \
  /data/NUAA-SIRST \
  /data/NUDT-SIRST \
  /data/IRSTD-1K
```

验收：

- loss、梯度、checkpoint 都是有限值；
- 每个 batch 域样本数一致；
- 无目标图像不报错；
- source validation 不读取 test；
- `last.pt` 可恢复；
- score export 恢复到原图尺寸。

## 4. Gate 2：Detector 配对试训

对同一 fold、同一 seed 运行：

### Baseline

```bash
python -m rc_irstd.pipelines.train_detector \
  --source-dataset /data/A \
  --source-dataset /data/B \
  --source-dataset /data/C \
  --source-train-split detector_train \
  --source-val-split detector_val \
  --detector mshnet \
  --detector-objective baseline \
  --epochs 40 \
  --batch-size 2 \
  --pixel-budget 1e-5 \
  --no-selection-use-peak-constraint \
  --device cuda \
  --seed 42 \
  --output-dir outputs/pilot/baseline_seed42
```

### Proposed detector

```bash
python -m rc_irstd.pipelines.train_detector \
  --source-dataset /data/A \
  --source-dataset /data/B \
  --source-dataset /data/C \
  --source-train-split detector_train \
  --source-val-split detector_val \
  --detector mshnet \
  --detector-objective domain_tail_separation \
  --lambda-sep 0.20 \
  --separation-margin 1.0 \
  --background-tail-fraction 0.05 \
  --object-top-fraction 0.25 \
  --hard-object-fraction 0.25 \
  --peak-kernel 5 \
  --exclusion-radius 2 \
  --risk-start-epoch 5 \
  --risk-ramp-epochs 10 \
  --epochs 40 \
  --batch-size 2 \
  --pixel-budget 1e-5 \
  --no-selection-use-peak-constraint \
  --device cuda \
  --seed 42 \
  --output-dir outputs/pilot/tail_sep_seed42
```

先看：

- source-val IoU 下降是否可接受；
- background peak upper tail 是否下降；
- hard target lower tail 是否上升；
- unseen-domain oracle Pd 是否改善；
- logits 是否整体爆炸；
- 低虚警曲线是否出现真实排序改善。

若 oracle 也无法恢复 Pd，不进入 calibrator 长训练。

## 5. Gate 3：连续 score maps 与问题诊断

冻结 detector，分别导出 meta-source 与候选 outer target 的连续分数：

```bash
python -m rc_irstd.pipelines.export_scores \
  --dataset-dir /data/A \
  --split meta_train \
  --detector mshnet \
  --checkpoint outputs/pilot/tail_sep_seed42/best_budget.pt \
  --inference-mode native_pad \
  --normalization imagenet \
  --include-mask \
  --device cuda \
  --output-dir outputs/scores/A_meta_train
```

诊断表必须包含：

```text
source-domain threshold
fixed 0.5
worst-source threshold
target oracle threshold
Pd@1e-4 / 1e-5 / 1e-6
exact false pixels / total original pixels
```

继续条件：至少多个目标域存在明显阈值漂移，且 target oracle 能恢复低虚警 Pd。

## 6. Gate 4：构建严格 inner pseudo-target folds

以 outer target `D` 为例，meta-source 为 `A/B/C/E`。当 pseudo-target 为 `A`：

```text
inner detector sources = B/C/E
held out = A/D
A-meta-train → calibrator train episodes
A-meta-val   → calibrator validation episodes
```

每个 pseudo-target 使用排除自身的 inner detector。不能用同一个含 pseudo-target 标签训练过的 detector 生成其元 episode。

## 7. 构建 source reference

```bash
python -m rc_irstd.pipelines.build_source_reference \
  --score-directory outputs/scores/B_meta_train \
  --score-directory outputs/scores/C_meta_train \
  --score-directory outputs/scores/E_meta_train \
  --context-size 32 \
  --stride 32 \
  --output outputs/meta/outer-D/pseudo-A/source_reference.npz
```

输入只包含 label-free score/statistics。

## 8. 构建 grouped-budget meta data

训练：

```bash
BUDGET_LOOSE=1e-4 BUDGET_MID=1e-5 BUDGET_STRICT=1e-6 \
CONTEXT_SIZE=32 QUERY_SIZE=64 STRIDE=96 \
SPLIT_ROLE=official_train_meta_train \
bash scripts/build_meta_fold.sh \
  outputs/scores/A_meta_train \
  outputs/meta/outer-D/pseudo-A/train.npz \
  outputs/meta/outer-D/pseudo-A/source_reference.npz
```

验证：

```bash
BUDGET_LOOSE=1e-4 BUDGET_MID=1e-5 BUDGET_STRICT=1e-6 \
CONTEXT_SIZE=32 QUERY_SIZE=64 STRIDE=96 \
SPLIT_ROLE=official_train_meta_val \
bash scripts/build_meta_fold.sh \
  outputs/scores/A_meta_val \
  outputs/meta/outer-D/pseudo-A/val.npz \
  outputs/meta/outer-D/pseudo-A/source_reference.npz
```

将多个 pseudo-target 的 train NPZ 合并为 outer-D train meta；选定的 pseudo-target validation 单独保留。不得把相同图像 ID 放入 train/val。

## 9. Gate 5：Calibrator pilot

```bash
DEVICE=cuda EPOCHS=100 BATCH_SIZE=32 \
LAMBDA_VIOLATION=4.0 \
LAMBDA_UTILITY=1.0 \
LAMBDA_ORACLE=0.10 \
LAMBDA_SMOOTHNESS=0.01 \
PIXEL_TEMPERATURE=0.10 \
OBJECT_TEMPERATURE=0.20 \
bash scripts/train_calibrator_risk_aligned.sh \
  outputs/meta/outer-D/train_all_pseudo_targets.npz \
  outputs/meta/outer-D/val_pseudo_target.npz \
  outputs/calibrator/outer-D/seed42
```

每个 epoch 对独立 validation 做 exact replay。主要查看：

- BSR；
- LogExcess；
- mean/worst-domain Pd；
- structural monotonicity；
- 与 rolling quantile、EVT、direct MLP、monotone regression 的配对差异；
- 严格预算下是否通过过高阈值牺牲全部 Pd。

## 10. Gate 6：单 outer、单 seed 完整 pilot

只在前述门通过后：

1. 用所有 meta-source 域训练 outer detector；
2. 用严格 inner pseudo-target episodes 训练 calibrator；
3. 在 outer target official test 的无标签前缀上预测阈值；
4. 固定阈值处理未来 query；
5. 最后才打开 query 标签 exact replay。

部署命令：

```bash
PIXEL_BUDGET=1e-5 CONTEXT_SIZE=32 DEVICE=cuda \
bash scripts/deploy_no_reject.sh \
  outputs/scores/D_official_test_unlabeled \
  outputs/calibrator/outer-D/seed42/best.pt \
  outputs/deployment/outer-D/seed42/B1e-5.json \
  outputs/meta/outer-D/deployment_source_reference.npz
```

输出没有 Reject 字段。

## 11. Gate 7：扩展正式实验

扩展顺序固定：

```text
single outer × single seed
→ all outer targets × single seed
→ all outer targets × three seeds
→ 2–3 external unseen targets
→ second backbone generality check
```

每次扩展前冻结配置文件和 commit/hash，不根据 external target 结果回头改方法。

## 12. 结果目录规范

每次 run：

```text
outputs/experiments/<experiment_id>/
├── config.yaml
├── command.txt
├── code_fingerprint.txt
├── environment.freeze.txt
├── data_manifest.json
├── split_hashes.json
├── checkpoint/
├── metrics.jsonl
├── exact_replay.json
├── curves/
└── notes.md
```

没有 checkpoint、split、seed、inference mode、original-size contract 和 threshold provenance 的结果不能进入主表。
