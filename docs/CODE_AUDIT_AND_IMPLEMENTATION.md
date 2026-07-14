# 代码审计与本次实现说明

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

## 1. 审计对象和边界

用户指定远端：

```text
https://github.com/Arialliy/RC-IRSTD
```

当前执行环境无法取得该远端的实时 HEAD。本次可执行代码基于用户文件库中最新的完整源码快照：

```text
RC-IRSTD_Rebuilt_Complete.zip
SHA-256: fc4051e051ad3dd4b57bc1067c0f193add881977abcce9796340211b7fc5165b
```

因此本包是对可获得源码快照的审计增强版，不是对未知远端 HEAD 的逐字节镜像。推回远端前必须做 commit/diff 核查。

## 2. 原快照可复用的工程基础

原快照已经包含：

- MSHNet 与 TinyUNet detector adapter；
- BasicIRSTD 风格数据读取；
- 多源均衡训练；
- score-record 导出；
- 局部候选、像素与组件评价；
- support/query episode 工具；
- 风险曲线、CRC 与部署模块；
- LODO orchestration；
- synthetic smoke 与单元测试。

因此本次没有重建一个新的 backbone，而是在原工程上替换论文主方法的训练语义。

## 3. 发现的关键方法错位

### 3.1 旧 margin 不是域级两尾

旧实现先对每张图做 hinge，再求域均值。它不等价于：先形成域级背景上尾和目标下尾，再做一个域级 hinge。

### 3.2 旧 calibrator 仍以 oracle threshold 回归为主

threshold MSE 与风险误差不一致，尤其在极端尾部。旧路径还包含 Reject BCE，不符合最终无 Reject 设计。

### 3.3 一个预算一个样本

同一 support/query 曲线被拆成多个 scalar-budget 样本，无法利用结构保证整条预算—阈值关系的单调一致性。

### 3.4 模型选择目标错位

按 threshold MSE 选 checkpoint 不能直接反映预算满足和检测效用。最终应使用 exact replay 的 BSR、LogExcess、Pd。

### 3.5 主风险混合

connected-component FA 与 pixel/candidate risk 的单调性不同。主路径必须只选择一个可单调控制的风险。

## 4. 本次核心改动

### 4.1 `rc_irstd/losses/target_background_margin.py`

新增/修正：

- `domain_tail_separation_loss`；
- GT dilation；
- deterministic local-maximum plateau collapse；
- 无目标图像背景尾；
- image/object/domain 分层等权；
- domain-level two-tail hinge；
- normalized smooth max；
- logit-shift invariance；
- warm-up/ramp wrapper `DomainTailSeparationDetectorLoss`。

兼容入口：`losses/target_background_margin.py`。

### 4.2 `rc_irstd/models/monotone_pixel_calibrator.py`

新增：

- 单次输出完整 `[J]` threshold-logit curve；
- 正跨度 + softmax 非负增量的结构单调参数化；
- DeepSets 风格 source-distance encoder；
- `log10(B)` 范围内插；
- 越界禁止外推；
- 无 Reject 输出。

兼容入口：`model/monotone_pixel_calibrator.py`。

### 4.3 `rc_irstd/episodes/meta_dataset.py`

新增：

- grouped multi-budget episode；
- support mask fail-closed；
- query background logit/weight 监督；
- object top-fraction logits；
- oracle threshold-logit auxiliary target；
- exact replay 所需 query paths/provenance；
- 默认全局不重叠窗口；
- train/validation ID 泄漏检查。

兼容入口：`rc/meta_dataset.py`。

### 4.4 `rc_irstd/losses/calibrator.py`

实现：

- smoothed pixel risk；
- smoothed object Pd；
- squared log budget violation；
- utility loss；
- Huber oracle auxiliary loss；
- curve smoothness；
- 总 `CalibratorRiskAlignedLoss`。

### 4.5 `rc_irstd/pipelines/train_calibrator.py`

实现：

- 独立 train/val meta inputs；
- train-only feature normalizer；
- no-Reject calibrator；
- query-risk-aligned optimization；
- structural monotonicity assertions；
- validation exact hard replay；
- `BSR → LogExcess → Pd` checkpoint selection；
- resume/checkpoint provenance。

兼容入口：`rc/train_calibrator.py`。

### 4.6 `rc_irstd/evaluation/calibrator_replay.py`

实现对 query score records 的 hard-threshold 重放，输出 pixel risk、Pd、BSR、LogExcess 和 rank key。

### 4.7 Prefix deployment

`rc_irstd/pipelines/apply_calibrator.py`：

- 只读取无标签前缀；
- 预测指定预算阈值；
- 不加载 mask；
- 不返回 Reject；
- 记录 checkpoint、budget、context IDs、threshold 与 source reference provenance。

### 4.8 Detector 训练入口

`rc_irstd/pipelines/train_detector.py` 新增：

```text
--detector-objective domain_tail_separation
--detector-objective baseline
--detector-objective legacy_tail_miss
```

final path 默认 pixel-only checkpoint selection；fixed-peak constraint 需显式开启且仅作兼容基线。

## 5. 新增测试

`tests/test_two_stage_no_reject.py` 覆盖：

- logit-shift invariance；
- 无目标图像背景尾梯度；
- 平坦峰平台只保留一个候选；
- structural monotonicity；
- no Reject output；
- source order permutation invariance；
- budget extrapolation fail-closed；
- query-risk loss 梯度；
- grouped `[J]` budget episode；
- support label-free 读取。

## 6. 实际执行的验证

打包前的实际运行包括：

```text
python -m compileall -q ...                     PASS
for script in scripts/*.sh; do bash -n ...      PASS
pytest -q                                       30 passed
MSHNet forward/backward/checkpoint roundtrip     PASS
3-epoch synthetic two-stage smoke training       PASS
exact hard replay                               PASS
```

合成 smoke 的作用仅是证明：张量、梯度、checkpoint、grouped budget、no-Reject、exact replay 和部署接口可以闭环。其 BSR/Pd 数值不代表真实数据性能。

## 7. 仍需在训练机完成

- 与远端当前 commit 做增量 diff；
- 数据去重、split 与 mask canvas 审计；
- 至少四个独立 meta-source；
- 至少 2–3 个 external unseen targets；
- real-data 1-epoch integration；
- detector paired pilot；
- single-outer calibrator pilot；
- full outer folds × 3 seeds；
- rolling quantile、EVT/GPD、direct/monotone baselines；
- prior-art 新颖性检索与第二 backbone 验证。
