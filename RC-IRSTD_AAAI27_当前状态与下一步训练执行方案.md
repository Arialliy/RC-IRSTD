# RC-IRSTD AAAI-27：当前状态、训练决策与下一步执行方案
> **2026-07-17 权威覆盖说明**：本文件保留为历史执行记录，不再代表当前 Stage-2 模型、三域协议或 GO/NO-GO 状态。当前唯一权威设计与实验合同为 [RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md](RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md)；只有其定义的 `S2_I0 PASS` 才表示完整模型设计完成，且仍不等于性能或 AAAI 成功。

> **2026-07-15 RC4 覆盖更新**：RC3 已完成 P0、P1、P2，但 P3 首个风险 epoch 暴露出“只记录首 batch 隔离梯度”的证据缺陷；official test 与 G1 性能端点均未开启。RC3 全部运行现仅保留为工程/诊断证据。RC4 已完成记录器与 P0-P3 审计语义修正，并通过项目全量 CPU、独立审计器和 GPU 0/1/2 smoke；当前正在冻结 RC4 clean release，随后从 epoch 0 重跑完整八项 Stage-1 矩阵。权威修正与重跑合同见 [RC-IRSTD_AAAI27_RC4风险梯度证据修正与重跑决定.md](RC-IRSTD_AAAI27_RC4风险梯度证据修正与重跑决定.md)。本文件后续 RC3 段落保留为当时决策记录，不再代表最新状态。

> **仓库**：`https://github.com/Arialliy/RC-IRSTD`
> **日期**：2026-07-15
> **方法版本**：Two-Stage / No-Reject / schema-v4 / v5 calibrator
> **当前结论**：**RC1 D0 因旧 SLS 实现偏差已标记 implementation-invalid；RC2 sealed preflight 因 split manifest 键顺序不可字节重放而失败；RC3 已修正并等待 clean-release sealed preflight。Stage 1 仅在 RC3 全部通过后 GO。Stage 2 仅 synthetic/engineering smoke GO；三域真实 Stage 2 训练、正式主实验与 AAAI 结论 NO-GO。**

---

## 0. 一页结论

当前工程已经证明：

- 三个真实数据域 `NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K` 能进入同一套 Stage 1 多源训练链路；
- 三域同时训练和三个两源 LODO detector 路由均可完成前向、反向、保存、严格重载和有限性检查；
- Stage 1 的域级目标下尾—背景上尾间隔、GT 邻域排除和确定性 plateau collapse 已接入；
- Stage 2 的完整预算单调曲线、无 Reject v5 模型、query-risk-aligned loss、schema-v4 grouped episodes、checkpoint 审计和原分辨率 exact replay 已形成代码闭环；
- RC1 sealed preflight 为 `PASS`：226 tests、10 subtests；但该测试集没有覆盖 strict Stage-1 入口误绑旧 SLS 的空目标语义，因此不能挽救已失效的 RC1 性能运行；
- RC2 sealed preflight 已通过 GPU 0/1/2、compile 和 245 tests/10 subtests，但在 frozen split manifest 字节重放处 fail-closed，因此 RC2 未获训练授权；
- RC3 已修正唯一的 JSON 键顺序差异；预发布 CPU 为 244 passed/10 subtests，D0/D3 的 GPU 0/1/2 DataParallel smoke 均 finite，仍须创建新 tag/archive 并从头重跑 clean-release sealed preflight；
- 历史 engineering smoke 的最小训练 → checkpoint → 重载 → 无标签前缀适配 → CPU 阈值复算 → 标签后读 exact replay 闭环通过；该 smoke 不参与本轮模型选择或性能 Gate。

当前已有一次真实训练尝试：

```text
RC1 D0 all-three：完整 epoch 0--5；epoch 6 中受控停止
```

该运行因 strict 入口使用旧 `model.loss.SLSIoULoss`、影响 17.89% 的空目标 crop，已归档为 `implementation-invalid / diagnostic-aborted`，禁止 resume、禁止进入 Gate 或论文结果。它连同历史 smoke 只证明**软件链路可运行并帮助定位实现错误**，不证明：

```text
性能提升；
跨域泛化；
预算满足优势；
统计显著性；
AAAI 主张成立。
```

RC1 失效证据和 RC3 修复合同见：

```text
docs/AAAI27_RC1_SLS_IMPLEMENTATION_INVALID.md
```

下一步按以下唯一顺序推进：

```text
完成 RC3 全量验证并冻结工作树与实验契约
→ 完成并冻结近重复隔离、数据契约与分析计划
→ RC3 Stage 1 从 epoch 0 单 seed、30 epoch 性能 Gate
→ Gate 通过后扩展 3 seeds
→ 新增至少第 4 个独立域
→ 启动合法的 Stage 2 outer-fold pilot
→ 全 outer folds、基线、消融、置信区间
```

---

## 1. 当前事实与证据边界

### 1.1 已确认的工程证据

RC1 sealed preflight（仅作历史工程证据，不能替代 RC3 最终冻结）：

```text
outputs/preflight/aaai27_20260714_stage1_pilot_sealed_rc1_final/
```

核心状态：

```text
RUN_STATUS.txt = PASS
226 tests passed
10 subtests passed
GPU 0/1/2 passed
31 formal CLI entries passed
9 scheduler routes passed
compileall passed
shell syntax passed
JSON/config validation passed
git diff --check passed
```

三域数据审计：

```text
outputs/preflight/aaai27_20260714_stage1_pilot_sealed_rc1_final/
└── data_audit_three_domains.json
```

已确认：

- official-train / official-test 无 ID 重叠；
- official-train / official-test 无精确内容重叠；
- `Misc_111` 按 BasicIRSTD 的 `NEAREST` mask resize 规则处理；
- 三个域均实际进入训练，而不是仅出现在配置中。

Stage 1 工程 smoke：

| 运行 | GPU | official-train sources | 状态 |
|---|---:|---|---:|
| 三域同时训练 | 0/1/2 | NUAA 213 + NUDT 663 + IRSTD-1K 800 | PASS |
| 留出 NUAA | 0 | IRSTD-1K + NUDT | PASS |
| 留出 NUDT | 1 | IRSTD-1K + NUAA | PASS |
| 留出 IRSTD-1K | 2 | NUAA + NUDT | PASS |

代表性工件：

```text
outputs/stage1_engineering_smoke/
└── final-all-three-dp-s42/
    └── config.json
```

三个 LODO checkpoint 也已生成并通过严格重载验证。当前日志显示：

- logits 有限；
- loss 有限；
- gradients 有限；
- parameters 有限；
- risk gradient 非零；
- 仅使用 official_train；
- checkpoint 使用 fixed-last；
- 未使用 official_test 选模。

### 1.2 尚未形成的证据

以下内容仍为空缺，不能从 smoke 结果推断：

- 真实 20–400 epoch 性能曲线；
- detector baseline 与 tail-separation 的公平配对结果；
- source threshold、rolling quantile、EVT/GPD、direct MLP 等基线；
- Stage 2 在合法 independent outer target 上的结果；
- 多随机种子均值、标准差和置信区间；
- Stage 1 D0/D3 的 30 epoch 性能结果；
- 外部完全未见域；
- 第二 detector/backbone 的可迁移性结果；
- AAAI 主表、消融表和压力测试。

### 1.3 本文证据边界

本文依据当前工作树报告、sealed preflight 和用户提供的运行结果制定下一步方案。本文件本身不是对训练机文件、GitHub 当前 HEAD 或 GPU 日志的独立第三方复验。

---

## 2. 方法身份冻结

正式方法必须统一为：

```text
Stage 1
MSHNet
  └─ Domain-level Target-Lower-Tail / Background-Upper-Tail Separation
          ↓ freeze detector

Stage 2
Unlabeled prefix S from an unseen domain
  └─ compact unlabeled statistics z(S)
          ↓
No-Reject Monotone Inverse Pixel-Risk Calibrator
          ↓
budget B ↦ threshold τ(B)
          ↓ freeze threshold
future disjoint query Q
```

### 2.1 主风险与评测语义

主文统一使用：

```text
主约束：
original-resolution pixel false-alarm rate

主检测效用：
object-level Pd

固定候选：
只用于 Stage 1 尾部间隔、无标签统计和诊断

connected-component FA：
仅作 IRSTD 文献兼容评测，不作为单调风险理论主约束
```

### 2.2 No-Reject 约束

主方法、在线协议、checkpoint schema 和主表均不得出现：

```text
reject
abstain
coverage
non-rejected Pd
reject rate
```

如保留历史 direct+reject 实现，只能作为明确命名的 legacy baseline，不能与 v5 主路径共享方法名称、checkpoint schema 或在线输出结构。

建议在冻结前执行：

```bash
cd /home/md0/ly/RC-IRSTD

rg -n \
  --glob '!outputs/**' \
  --glob '!.git/**' \
  --glob '!reference/**' \
  'Reject|reject|abstain|abstention|coverage|non-rejected'
```

所有命中项逐条分类为：

```text
必须删除的主路径陈述；
允许保留的历史说明；
允许保留的独立 baseline；
测试中用于拒绝非法工件的普通英文动词。
```

### 2.3 主张边界

论文可以使用：

- budget-aware；
- empirical risk alignment；
- unseen-domain operating-point adaptation；
- monotone inverse risk curve；
- causal prefix-to-future 或 prefix-holdout；
- external unseen-domain evaluation。

论文不得使用：

- guaranteed false-alarm control；
- certified risk control；
- distribution-free control；
- target-domain risk guarantee；
- “在任意未知域必然满足预算”。

---

## 3. GO / NO-GO 决策矩阵

| 工作项 | 当前决策 | 说明 |
|---|---:|---|
| Stage 1 三域工程 smoke | **PASS** | 已完成，不需要重复堆 smoke |
| Stage 1 20–40 epoch 单 seed 性能 pilot | **Conditional GO** | 先冻结代码、数据、分析计划和基线 |
| Stage 1 三折 detector LODO | **GO after freeze** | 两源训练 → 一域测试，能形成范围有限但有效的 detector DG 证据 |
| Stage 1 三随机种子长训练 | **Gate 后 GO** | 单 seed 先证明研究假设成立 |
| Stage 2 schema-v4 / v5 工程 smoke | **PASS / GO** | 仅作为实现证据 |
| Stage 2 三域 meta-calibration 诊断 | **GO, non-claim-bearing** | 可测梯度、单调性、exact replay，不进入主表 |
| Stage 2 三域严格 nested-LODO 主结果 | **NO-GO** | 固定 outer 后 inner detector 只剩单源 |
| Stage 2 四域最小合法 outer-fold pilot | **新增第 4 域后 GO** | 可形成两源 inner detector + pseudo-target + outer target |
| AAAI 主表、多 seed、CI | **NO-GO** | 还缺独立域、基线、消融和正式统计 |
| 重新设计复杂 backbone | **NO-GO** | 当前创新应集中在尾部可分性和预算工作点适配 |

---

## 4. Freeze Gate：正式性能训练前必须完成

当前主要阻塞不再是代码正确性，而是 claim-bearing 实验版本尚未冻结。

### 4.1 冻结 Git 工作树

在训练机执行：

```bash
cd /home/md0/ly/RC-IRSTD

git diff --check
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline -5
```

禁止直接执行：

```bash
git add .
```

应先检查变更，再显式 stage 本次方法、协议、测试和配置文件：

```bash
git diff --name-only
git add <explicit-source-files> <explicit-config-files> <explicit-doc-files>
git commit -m "freeze: AAAI27 two-stage no-reject experiment candidate"
git tag aaai27-rc-irstd-v5-rc3
```

生成代码归档：

```bash
mkdir -p outputs/release

git archive \
  --format=zip \
  --output outputs/release/RC-IRSTD_v5_rc3.zip \
  aaai27-rc-irstd-v5-rc3

sha256sum \
  outputs/release/RC-IRSTD_v5_rc3.zip \
  > outputs/release/RC-IRSTD_v5_rc3.zip.sha256
```

正式运行必须满足：

```text
git status --porcelain = empty
git diff --check = PASS
git commit recorded
git tag recorded
source archive SHA-256 recorded
```

训练后若修改以下任一项，必须产生新的 release candidate，旧结果不得与新结果混表：

- loss；
- tail definition；
- GT dilation；
- plateau collapse；
- budget grid；
- feature definition；
- episode schema；
- checkpoint selection；
- matching rule；
- threshold semantics；
- exact replay；
- dataset split；
- data normalization；
- seed handling。

### 4.2 冻结环境

保存：

```bash
python --version > outputs/release/python_version.txt
python -m pip freeze > outputs/release/pip_freeze.txt
nvidia-smi -q > outputs/release/nvidia_smi_q.txt
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv \
  > outputs/release/gpu_inventory.csv
```

PyTorch/CUDA 信息：

```bash
python - <<'PY' > outputs/release/torch_environment.txt
import torch
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.version())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

如使用容器，还需保存：

```text
container image name
container image digest
mount map
CUDA_VISIBLE_DEVICES mapping
```

### 4.3 冻结数据契约

为每个域保存：

```text
dataset root logical name
official_train split SHA-256
official_test split SHA-256
ordered image IDs SHA-256
ordered mask IDs SHA-256
image byte SHA-256 manifest
mask byte SHA-256 manifest
mask alignment policy
original-resolution policy
```

`Misc_111` 必须在 manifest 中明确记录：

```text
mask_alignment = nearest
original_mask_hw = ...
image_hw = ...
alignment_applied = true
```

---

## 5. 训练前剩余数据审计

### 5.1 已通过的部分

当前已通过：

- official_train / official_test ID overlap audit；
- official_train / official_test exact-content overlap audit；
- 三域实际样本数量核对；
- 特殊 mask 几何规则核对。

### 5.2 已完成：跨域与 train/test near-duplicate 审计

精确 SHA 不足以发现：

- resize 后副本；
- 裁剪版本；
- PNG/JPEG 重编码；
- 对比度或亮度变化；
- 同一序列的相邻帧；
- 图像重复但 mask 修订；
- 数据集收录关系。

已按三层审计：

```text
Level 1：SHA-256 exact duplicate
Level 2：pHash / dHash approximate duplicate
Level 3：对候选 pair 做 resize-normalized SSIM / correlation + 人工复核
```

冻结输出：

```text
audits/aaai27/
├── near_duplicates_original_official_splits_v2.json
├── near_duplicate_pair_previews_v1.png
├── near_duplicate_manual_review_v1.md
├── near_duplicates_effective_splits_v2.json
├── dataset_contract_v1.json
└── final_domain_independence_decision_v1.json
```

结果与处置：

```text
原始 2755 张图：59 个 pHash 候选，31 个固定相关性确认对；
31 对均为同数据集 official_train ↔ official_test 的 same_scene_related；
涉及 30 个唯一训练 ID：NUDT 27、IRSTD-1K 3、NUAA 0；
不修改原始数据或官方 split，只从所有 development roles 隔离；
v2 effective development 与 official_test 复审：确认对为 0。
```

### 5.3 独立域判定原则

新增第 4 域时不能只看数据集名称不同。必须确认：

- 不是现有数据集的合集；
- 不是现有数据集的重标注；
- 不共享大量同场景帧；
- 不存在 train/test 污染；
- 采集平台、场景或成像条件至少有实质差异；
- 可以形成独立 outer target。

---

## 6. 冻结 Statistical Analysis Plan

已在任何性能 pilot 前创建：

```text
docs/AAAI27_STATISTICAL_ANALYSIS_PLAN.md
```

必须与本轮代码一起提交到 frozen commit。

### 6.1 主预算

建议冻结：

```text
B = [1e-4, 1e-5, 1e-6]
```

解释：

- `1e-4`：较稳定的低虚警工作点；
- `1e-5`：主低虚警工作点；
- `1e-6`：严格压力设置，不作为唯一结论。

### 6.2 主指标

按以下顺序报告：

```text
1. Pd@B
2. BSR
3. LogExcess
4. worst-domain Pd
5. exact pixel FA
```

补充报告：

```text
IoU / nIoU / hIoU
connected-component FA/MP
threshold values
target oracle
tail distribution diagnostics
```

### 6.3 Checkpoint 规则

Stage 1：

```text
fixed-last
```

禁止：

```text
根据 outer target official_test 选择 epoch；
根据 target oracle 选择 epoch；
根据 target Pd 或 FA early stop。
```

Stage 2：

```text
BSR → LogExcess → Pd
```

具体排序：

1. BSR 更高；
2. BSR 相同或预注册容差内，LogExcess 更低；
3. 前两项相同或容差内，Pd 更高；
4. 仍并列时，选择更早 epoch 或固定 tie-break。

### 6.4 随机种子

冻结：

```text
[42, 123, 3407]
```

所有对照必须使用相同 seeds。

### 6.5 置信区间

优先使用：

```text
sequence-level block bootstrap
```

没有可靠 sequence ID 时：

```text
image-level block bootstrap
```

不得把像素当独立样本做置信区间。

建议输出：

```text
mean
standard deviation across seeds
95% bootstrap CI
per-domain values
worst-domain value
```

---

## 7. Stage 1：下一步性能训练方案

### 7.1 第一批只跑单 seed Gate

先使用：

```text
seed = 42
epochs = 20–40
checkpoint = fixed-last
```

目标不是立即追求最终 SOTA，而是判断核心假设是否成立。

### 7.2 最小必要 detector 对照

| ID | 模型 | 背景上尾 | 目标下尾 | 域级 hinge | 用途 |
|---|---|---:|---:|---:|---|
| D0 | MSHNet + segmentation loss |  |  |  | 主 baseline |
| D1 | D0 + background upper tail | ✓ |  | ✓ | 背景尾贡献 |
| D2 | D0 + target lower tail |  | ✓ | ✓ | 目标尾贡献 |
| D3 | D0 + full domain tail separation | ✓ | ✓ | ✓ | 主方法 |

所有对照必须保持一致：

- MSHNet 结构；
- initialization；
- domain-balanced sampling；
- augmentation；
- batch size；
- optimizer；
- LR schedule；
- epoch/step 数；
- seed；
- original-resolution evaluation；
- fixed-last；
- threshold sweep；
- matching rule。

### 7.3 首批推荐运行矩阵

最低成本 Gate：

```text
D0 baseline:
  all-three
  leave-NUAA
  leave-NUDT
  leave-IRSTD1K

D3 full:
  all-three
  leave-NUAA
  leave-NUDT
  leave-IRSTD1K
```

即 8 个性能 pilot。

在 D3 显示有效趋势后，再运行 D1、D2 消融。不要在主假设尚未成立前启动所有消融 × 三 seeds × 长周期。

### 7.4 GPU 编排

先运行三域 all-source DataParallel：

```text
physical GPU 0/1/2
```

然后并行运行三个 LODO fold：

```text
GPU 0：leave-NUAA，sources = NUDT + IRSTD-1K
GPU 1：leave-NUDT，sources = NUAA + IRSTD-1K
GPU 2：leave-IRSTD1K，sources = NUAA + NUDT
```

不要同时启动 all-three DataParallel 与三个单卡 LODO，以免出现资源争用、显存不可比和吞吐异常。

训练命令应直接复用已在以下文档及 sealed preflight 中验证过的正式入口与调度路由：

```text
docs/AAAI27_TWO_STAGE_NO_REJECT_IMPLEMENTATION.md
outputs/preflight/aaai27_20260714_final_sealed/
```

不要为性能训练临时新增未经测试的 shell wrapper。

### 7.5 Stage 1 每 epoch 必须记录

基础量：

```text
loss_total
loss_seg
loss_tail_sep
learning_rate
grad_norm
parameter_norm
elapsed_steps
```

每域尾部量：

```text
R_minus_domain_d
R_plus_domain_d
margin_gap_domain_d = R_plus_domain_d - R_minus_domain_d
hinge_domain_d
num_background_candidates_domain_d
num_target_objects_domain_d
num_empty_target_images_domain_d
```

稳定性量：

```text
logit_mean
logit_std
logit_q001
logit_q50
logit_q99
logit_q999
max_abs_logit
nonfinite_count
```

单 seed Gate 的 fixed-last 后，只在冻结的 official-train-derived
`detector_diagnostic` 上生成：

```text
original-resolution score maps
exact/adaptive threshold curves
Pd@1e-4
Pd@1e-5
Pd@1e-6
source threshold result
target oracle result
pixel FA
component FA/MP compatibility result
```

官方 test 在方法、D0–D3、预算、三 seeds 与完整 Stage 1 扩展策略冻结前
保持 sealed；不得用 20–40 epoch official-test pilot 决定是否扩 seeds、运行
D1/D2 或修改方法。

### 7.6 Stage 1 Gate 通过条件

建议同时满足：

1. 三个 held-out target 中至少两个存在明显 threshold drift；
2. target oracle 相比 source threshold，在至少两个域带来有意义的 `Pd` 恢复，预注册参考门槛为约 3 个百分点；
3. D3 在至少两个 held-out 域降低高分背景上尾，或提高 `Pd@B`；
4. D3 不只是整体降低所有 logits；
5. `R_plus - R_minus` 在训练后变大，且目标下尾不持续恶化；
6. source/held-in 普通分割性能相对 D0 不出现不可接受下降，建议预注册容忍度不超过约 1 个百分点；
7. 无 logit scale 持续膨胀；
8. 无空预测捷径；
9. 改善不能只出现在单 fold、单预算；
10. exact replay 与在线阈值语义完全一致。

### 7.7 Stage 1 停止或转向条件

停止 tail-separation 主线：

```text
D3 只造成整体 logit 下移；
oracle Pd 不提高；
目标下尾显著下降；
低虚警 Pd 在多数域下降；
D3 与 D0 在三个域均无稳定差异。
```

转向 representation 问题：

```text
target oracle 也无法恢复 Pd；
真实目标在低虚警区没有可排序分数；
目标与背景尾部严重重叠。
```

若 D1 优于 D3：

```text
删除或弱化 target-tail 部分。
```

若 D2 优于 D3：

```text
重新检查背景候选、GT exclusion 和 tail aggregation。
```

---

## 8. Stage 1 Gate 通过后的三随机种子

先扩展主对照：

```text
D0 baseline × 3 folds × 3 seeds
D3 full     × 3 folds × 3 seeds
```

D1/D2 可先单 seed；只有其结论影响主方法解释时，再扩展三 seeds。

每个运行目录至少保存：

```text
outputs/experiments/<experiment_id>/
├── config.json
├── command.txt
├── git_commit.txt
├── git_status.txt
├── source_archive_sha256.txt
├── environment.json
├── dataset_contract.json
├── checkpoint_last.pt
├── checkpoint_sha256.txt
├── history.jsonl
├── score_manifest.json
├── exact_curve.csv
├── exact_replay.json
├── metrics.json
└── notes.md
```

实验 ID 建议：

```text
stage1__D0__leave-NUAA__s42__rc1
stage1__D3__leave-NUAA__s42__rc1
stage1__D0__leave-NUDT__s123__rc1
...
```

---

## 9. 必须冻结的基线与消融

### 9.1 Detector baselines

主文最低要求：

- MSHNet baseline；
- MSHNet + background upper tail；
- MSHNet + target lower tail；
- MSHNet + full domain tail separation。

可选增强：

- pixel top-k hard-negative；
- local-peak CVaR；
- focal / OHEM；
- 第二 detector：SCTransNet 或 DNANet。

### 9.2 Threshold / calibrator baselines

必须包含：

1. Fixed threshold 0.5；
2. pooled-source threshold；
3. worst-source safe threshold；
4. nearest-source threshold；
5. rolling quantile；
6. EVT/GPD；
7. direct threshold MLP；
8. monotone oracle-threshold regression；
9. proposed no-Reject monotone query-risk-aligned calibrator；
10. target oracle。

### 9.3 Stage 2 核心消融

| ID | 单调结构 | Query risk loss | Oracle auxiliary | Support statistics |
|---|---:|---:|---:|---|
| C0 |  |  | ✓ | full |
| C1 | ✓ |  | ✓ | full |
| C2 | ✓ | ✓ |  | full |
| C3 | ✓ | ✓ | ✓ | full |
| C4 | ✓ | ✓ | ✓ | score histogram only |
| C5 | ✓ | ✓ | ✓ | + local peaks |
| C6 | ✓ | ✓ | ✓ | + noise + source-distance |

主方法为 C3。

### 9.4 不应包装成普通方法消融的正确性机制

以下机制改变了问题定义或候选确定性，不宜在主表中作为“可选模块”：

- deterministic plateau collapse；
- original-resolution replay；
- GT neighborhood exclusion 的基本合法版本；
- support/query disjointness；
- official_train / official_test role；
- no label read in support；
- CPU threshold recomputation before label access；
- checkpoint SHA verification；
- schema-v4 evidence audit。

可在补充材料做 sanity check，但关闭这些机制产生的结果不能与合法主协议等价比较。

---

## 10. Stage 2：当前三域允许做什么

当前三域可以继续做：

- v5 forward/backward；
- grouped budget tensor 检查；
- monotonicity invariant；
- in-range log-budget interpolation；
- no extrapolation fail-closed；
- query-risk loss 梯度；
- checkpoint save/load；
- CPU threshold recomputation；
- online artifact schema；
- exact replay；
- baseline implementation smoke；
- window-size和feature维度的工程测试。

这些运行必须标记：

```text
three_domain_stage2_diagnostic
non_claim_bearing
no_independent_outer_target
```

不得进入：

```text
AAAI main table
abstract claims
final conclusion
statistical significance statement
```

---

## 11. Stage 2 的硬阻塞：独立域数量

### 11.1 为什么三个域不够

设三个域为 A、B、C。

固定 C 为 outer unseen target 后，Stage 2 元训练只能使用 A、B。

若 A 为 pseudo-target，则 inner detector 只能在 B 上训练；若 B 为 pseudo-target，则 inner detector 只能在 A 上训练。

这不再是：

```text
multi-source detector → pseudo-target
```

而是：

```text
single-source detector → pseudo-target
```

因此不能支撑严格 nested-LODO 主结论。

### 11.2 四域最低合法结构

增加独立域 D 后，固定 D 为 outer target：

```text
pseudo-target A:
  detector sources = B + C

pseudo-target B:
  detector sources = A + C

pseudo-target C:
  detector sources = A + B

outer deployment detector:
  sources = A + B + C
```

这才满足：

- inner detector 为多源训练；
- pseudo-target 未进入对应 detector；
- outer D 未进入 detector/calibrator 训练与选模；
- calibrator 只在 A/B/C official_train episodes 上训练和验证；
- D official_test 仅在冻结后做无标签 prefix → future query 评测。

### 11.3 更有说服力的目标配置

最低软件闭环：

```text
4 个独立域
```

更适合 AAAI 主证据：

```text
至少 4 个 meta-source domains
+
2–3 个 external unseen targets
```

外部 target 标签必须在以下全部冻结后才打开：

- 方法；
- 特征；
-预算；
- 窗口长度；
- checkpoint；
- baselines；
- ablations；
- analysis plan；
- code release；
- source-domain hyperparameters。

---

## 12. 第一个合法 Stage 2 pilot

新增第 4 域并通过 near-duplicate audit 后，先运行：

```text
1 outer target
1 seed
2 pseudo-targets for calibrator training
1 pseudo-target for calibrator validation
medium epoch
```

不要直接启动：

```text
all outer folds × 3 seeds × all baselines × long training
```

### 12.1 必须验证的契约

- pseudo-target detector 严格排除该域；
- detector sources 至少为两个独立域；
- all meta episodes 来自 official_train；
- support/query 不重叠；
- support 路径不可读取 mask；
- query label 只进入元训练 loss 或离线 replay；
- outer official_test 不参与 checkpoint selection；
- budget curve 结构单调；
- 未见预算只在 `log10(B)` 域内插值；
- 禁止预算外推；
- v5 checkpoint schema 与 provenance 通过；
- online output 无 Reject；
- threshold 先在 CPU 确定性复算；
- 只有复算一致后才允许打开 query labels；
- 最终结果全部来自 original-resolution hard-threshold exact replay；
- checkpoint 选择严格使用 BSR → LogExcess → Pd。

### 12.2 扩展顺序

```text
single outer × single seed
→ all outer targets × single seed
→ all outer targets × 3 seeds
→ calibrator baselines
→ feature/loss ablations
→ second detector
→ contamination/noise/resolution/drift stress tests
```

---

## 13. 训练结果的 Go/No-Go 门槛

### Gate A：问题是否真实存在

继续 Stage 2 的最低证据：

- 至少两个 held-out domains 存在 threshold drift；
- source/fixed threshold 经常发生预算超限；
- target oracle 可恢复有意义的 Pd；
- oracle threshold 不是所有域近似相同。

若不满足，停止 calibrator 主线。

### Gate B：Stage 1 是否提高可校准性

继续条件：

- 背景上尾下降；
- 目标下尾保持或提高；
- gap 增大；
- oracle Pd 不下降；
- low-FA Pd 至少在多数域不差于 baseline。

若只降低 score scale 而不改善排序，删除 Stage 1 创新主张。

### Gate C：学习校准是否超过简单规则

Proposed 至少应在合法 unseen outer targets 上：

- BSR 优于 rolling quantile / source threshold；
- LogExcess 更低；
- Pd 不因过度保守而塌陷；
- 单调违反率为 0；
- 不依赖目标域标签或隐性 target identity；
- 在多个预算与多个域上趋势一致。

若 proposed 与 rolling quantile 或 EVT/GPD 无稳定差异，应降级为更简单方法或转向可解释的非学习阈值方案。

### Gate D：AAAI 主张是否成立

只有在以下条件全部满足后才进入论文主表：

- 独立域数量合法；
- 至少三个随机种子；
- baseline 完整；
- ablation 完整；
- exact replay；
- confidence interval；
- no target leakage；
- source archive、checkpoint、split、manifest 均可审计；
- prior-art 检索完成；
- 结果在至少两个 external targets 上一致。

---

## 14. 建议主表与消融表

### 14.1 主表

| Method | Target labels | Causal/prefix | Monotone | Pd@1e-4 | Pd@1e-5 | Pd@1e-6 | BSR ↑ | LogExcess ↓ | Worst-domain Pd ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0 | ✓ | — |  |  |  |  |  |  |
| Pooled source threshold | 0 | ✓ | — |  |  |  |  |  |  |
| Worst-source threshold | 0 | ✓ | — |  |  |  |  |  |  |
| Rolling quantile | 0 | ✓ | ✓ |  |  |  |  |  |  |
| EVT/GPD | 0 | ✓ | ✓ |  |  |  |  |  |  |
| Direct threshold MLP | 0 | ✓ | ✗ |  |  |  |  |  |  |
| Monotone regression | 0 | ✓ | ✓ |  |  |  |  |  |  |
| RC-IRSTD v5 | 0 | ✓ | ✓ |  |  |  |  |  |  |
| Target oracle | All | ✓ | — |  |  |  | — | — |  |

### 14.2 Detector 消融

| Detector | Background tail | Target tail | Domain aggregation | Pd@1e-5 | Pd@1e-6 | Oracle Pd | Tail gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 |  |  |  |  |  |  |  |
| D1 | ✓ |  | ✓ |  |  |  |  |
| D2 |  | ✓ | ✓ |  |  |  |  |
| D3 | ✓ | ✓ | ✓ |  |  |  |  |

### 14.3 Calibrator 消融

| Calibrator | Structural monotonicity | Risk-aligned loss | Oracle auxiliary | BSR | LogExcess | Pd |
|---|---:|---:|---:|---:|---:|---:|
| Direct MLP |  |  | ✓ |  |  |  |
| Monotone regression | ✓ |  | ✓ |  |  |  |
| Risk only | ✓ | ✓ |  |  |  |  |
| Full v5 | ✓ | ✓ | ✓ |  |  |  |

---

## 15. 工件与可复现性规范

每个 claim-bearing run 必须有：

```text
experiment_id
method_version
git_commit
git_tag
git_clean
source_zip_sha256
container_digest
python_version
torch_version
cuda_version
gpu_uuid
seed
source_domains
pseudo_target
outer_target
official_split_roles
dataset_manifest_sha256
checkpoint_selection
checkpoint_sha256
budget_grid
threshold_semantics
matching_rule
original_resolution
exact_replay_status
```

建议统一生成：

```text
run_contract.json
```

其中：

```json
{
  "method": "RC-IRSTD-v5-no-reject",
  "stage": 1,
  "claim_bearing": false,
  "git_clean": true,
  "checkpoint_selection": "fixed_last_no_test_selection",
  "primary_risk": "original_resolution_pixel_false_alarm_rate",
  "threshold_semantics": "prediction = probability > threshold",
  "connected_component_role": "compatibility_evaluation_only"
}
```

只有通过全部 verifier 的工件才能把：

```json
"claim_bearing": false
```

改为：

```json
"claim_bearing": true
```

---

## 16. 当前不应做的事情

不要：

- 再设计复杂 backbone；
- 在工作树未冻结时启动多 seed 长训练；
- 把 1-step smoke 写成性能结果；
- 把三域 Stage 2 诊断写成 nested-LODO 主结论；
- 使用 official_test 选 epoch；
- 根据 target 结果回头修改预算、窗口或特征；
- 把 connected-component FA 写成严格单调主风险；
- 只报告 threshold MAE；
- 只报告 `1e-6`；
- 跳过 rolling quantile 和 EVT/GPD；
- 用大量重叠窗口替代独立域；
- 把同一序列拆到 meta-train 和 meta-val；
- 把 deterministic plateau collapse 当作性能插件；
- 恢复 Reject 作为主方法捷径；
- 在没有 exact replay 时报告 surrogate risk；
- 在没有 near-duplicate audit 时声称数据域完全独立；
- 在没有第四域时启动 Stage 2 claim-bearing 长训练。

---

## 17. 最终执行顺序

### Step 1：冻结

- [ ] 提交并 tag v5 release candidate；
- [ ] 工作树 clean；
- [ ] 源码 ZIP + SHA；
- [ ] 环境与 GPU inventory；
- [x] split / data manifest；
- [ ] sealed preflight 复制进 release evidence。

### Step 2：数据与统计计划

- [x] near-duplicate audit；
- [x] domain independence decision；
- [x] analysis plan；
- [x] seeds；
- [x] budgets；
- [x] CI；
- [x] baselines 规格；
- [x] ablations 规格与 D0–D3 实现；
- [x] 8-run Stage 1 pilot 矩阵与严格运行产物合同；
- [x] `detector_diagnostic` 原分辨率评测角色与隔离验证；
- [x] stop criteria。

### Step 3：Stage 1 单 seed Gate

- [ ] D0 all-three；
- [ ] D3 all-three；
- [ ] D0 leave-NUAA；
- [ ] D3 leave-NUAA；
- [ ] D0 leave-NUDT；
- [ ] D3 leave-NUDT；
- [ ] D0 leave-IRSTD1K；
- [ ] D3 leave-IRSTD1K；
- [ ] original-resolution exact curves；
- [ ] source vs oracle diagnostic；
- [ ] Gate decision。

### Step 4：Stage 1 扩展

- [ ] D1/D2 单 seed ablation；
- [ ] D0/D3 三 seeds；
- [ ] bootstrap CI；
- [ ] failure cases；
- [ ] result freeze。

### Step 5：Stage 2 准备

- [ ] rolling quantile；
- [ ] EVT/GPD；
- [ ] direct MLP；
- [ ] monotone regression；
- [ ] feature ablations；
- [ ] 新增第 4 独立域；
- [ ] near-duplicate audit；
- [ ] legal outer-fold contract。

### Step 6：Stage 2 正式 pilot

- [ ] 1 outer；
- [ ] 1 seed；
- [ ] strict inner detectors；
- [ ] schema-v4 grouped episodes；
- [ ] train/val pseudo-target separation；
- [ ] v5 checkpoint selection；
- [ ] no-Reject online adaptation；
- [ ] CPU threshold recomputation；
- [ ] label-after-decision exact replay；
- [ ] Gate decision。

### Step 7：完整 AAAI 实验

- [ ] all outer targets；
- [ ] 3 seeds；
- [ ] all baselines；
- [ ] detector/calibrator ablations；
- [ ] CI；
- [ ] window sensitivity；
- [ ] target contamination；
- [ ] noise/resolution drift；
- [ ] second detector；
- [ ] external unseen targets；
- [ ] reproducibility package。

---

## 18. 最终训练决策

### 现在是否可以运行训练？

**尚不能立即启动性能训练；完成 clean commit/tag/archive 与新 sealed
preflight 后可以启动 Stage 1 pilot。**

#### 冻结门通过后可以启动

```text
冻结后的 Stage 1 单 seed、20–40 epoch 性能 pilot
```

包括：

- 三域 all-source detector；
- 三个两源 LODO detector；
- D0 baseline 与 D3 full 的公平配对；
- fixed-last；
- official_train `detector_fit` only for fitting；
- official_train-derived `detector_diagnostic` only for本轮 Gate；
- official_test 保持 sealed，不参与是否扩 seeds、消融或修改方法的决策；
- original-resolution exact replay。

#### 可以运行但只能作为工程诊断

```text
synthetic Stage 2 schema-v4 / v5 calibrator smoke
```

不得使用三域真实数据训练 Stage 2，不得进入论文主表。

#### 现在不能启动为主实验

```text
Stage 2 三域 claim-bearing nested-LODO
AAAI 多 seed 主表
完整结论性训练
```

硬阻塞：

- 只有三个独立域；
- 工作树尚未冻结；
- clean commit/tag/source archive 尚未生成；
- 当前修改后的 full pytest 与 sealed preflight 尚未完成；
- Stage 2 baseline runners 尚未闭环；
- 外部 unseen targets 未建立；
- 置信区间未运行。

### 最准确的项目状态

> **方法、D0–D3、数据隔离与分析协议：已实现，待最终 sealed preflight。**
> **Stage 1 工程烟测：完成。**
> **Stage 1 性能训练：冻结后 GO。**
> **Stage 2 synthetic/engineering 诊断：GO，但 non-claim-bearing；三域真实训练 NO-GO。**
> **Stage 2 正式主实验：新增独立域后 GO。**
> **AAAI 实验证据：尚未完成。**

---

## 19. 推荐的下一条实际动作

不要再修改方法结构。近重复处置、v2 split、数据契约和 analysis plan 已完成；
下一条实际动作应是：

```text
full pytest + 新 sealed preflight
→ 显式提交并 tag v5 frozen release candidate
→ 源码 ZIP/SHA 与环境证据
```

随后才运行：

```text
D0/D3 Stage 1 单 seed、固定 30 epoch、official-train diagnostic Gate
```

只有 Gate 通过，才扩展三 seeds 和 Stage 2。
