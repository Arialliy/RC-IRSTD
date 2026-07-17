# RC-IRSTD AAAI-27 下一步训练方案与 GO / NO-GO 决策
> **2026-07-17 最终覆盖**：当前唯一执行权威为 [RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md](RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md)。本文件仅保留历史决策过程，不得用于启动实验或判定模型成功。

> **历史决策快照（不再作为当前执行权威）**：本文件中的旧 D1/D2、
> C0–C8 编号和 Gate −1 状态已被
> `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md` 与
> `docs/AAAI27_STATISTICAL_ANALYSIS_PLAN.md` 取代。当前 D1/D2 分别为
> background-only / target-only 梯度分支；threshold baselines 使用 T0–T9，
> calibrator ablations 使用 C0–C6；Stage 1 Gate 只使用冻结的
> official-train-derived diagnostic IDs，official test 保持 sealed。

> 工作目录：`/home/md0/ly/RC-IRSTD`
> 决策日期：2026-07-14
> 投稿方法全称：**Two-Stage No-Reject RC-IRSTD**
> 当前决定：**只批准预检和非主张性工程烟测；不批准当前三域条件下的真实 Stage 2、outer 评估或任何论文结果主张。**

方案参考源为 `/home/md0/ly/RC-IRSTD_AAAI27_TwoStage_NoReject`。本文吸收其“两阶段、No-Reject”研究身份，但不把参考包较弱的训练/工件合同提升为正式主线；正式实现仍以本仓库 flat v5 为唯一权威。

本文实际文件名是：

```text
/home/md0/ly/RC-IRSTD/RC-IRSTD_AAAI27_下一步训练方案与_GO_NO_GO_决策.md
```

文件名不含星号；在 Shell 中直接引用时应整体加引号，不要把 `*GO_NO_GO*` 当作通配符。

---

## 0. 不可再混淆的三条结论

1. **完整模型不是 MSHNet。**完整投稿模型是“两阶段 No-Reject RC-IRSTD”；MSHNet 只是 Stage 1 的 detector backbone。
2. **唯一可承载论文主张的是 flat v5 主线。**`data_ext/`、`evaluation/`、`losses/`、`model/`、`rc/` 和 `scripts/train_multisource_tail.py` 共同构成严格主线。
3. **`rc_irstd/` 不是第二套可混用的正式实现。**它与 YAML、reference launcher 只用于参考兼容和合成烟测；它产生的 checkpoint、episode、metric 或部署输出不得与 flat v5 工件拼接，也不得进入主表。

如果后续命令、README 或实验记录把整套方法写成“MSHNet”，或把 `rc_irstd.*` 工件接到 flat v5，必须先停止运行并修正身份。

---

## 1. 最终架构与论文边界

### 1.1 完整模型

```text
Stage 1: detection and score formation
  MSHNet backbone
    + Domain-level Target-Lower-Tail / Background-Upper-Tail Separation
    + GT-neighbourhood exclusion
    + deterministic plateau collapse
    -> frozen native-resolution score maps

Stage 2: no-reject operating-point adaptation
  unlabeled context S
    -> compact unlabeled statistics z(S)
    -> monotone inverse pixel-risk curve
    -> one threshold tau(B) for each frozen false-alarm budget B
    -> threshold frozen before future query labels are opened
    -> native-resolution exact replay on disjoint future query Q
```

论文中的正确写法是：

- 方法：**Two-Stage No-Reject RC-IRSTD**；
- Stage 1 backbone：**MSHNet**；
- Stage 1 新机制：域级目标下尾—背景上尾分离；
- Stage 2 新机制：query-risk-aligned 的单调 inverse pixel-risk curve；
- 部署输出：预算对应阈值，不包含 Reject、abstention、`p_min` 或覆盖率拒绝决策。

`L_coverage` 是 partial exact suffix 的有效支持域约束，不是预测拒绝机制，不能据此把方法称为 reject model。

### 1.2 允许主张与禁止主张

在真实、预注册证据完成前，只能写“implemented”“contract-tested”或“engineering-verified”。当前禁止写：

- 已优于 SOTA 或强基线；
- 已证明跨域泛化；
- guaranteed、certified 或 distribution-free risk control；
- 已完成 AAAI 主表；
- MSHNet 是本文提出的新 backbone；
- synthetic smoke 的数值代表真实数据性能。

所有最终性能数字必须来自冻结阈值后的原分辨率 hard-threshold exact replay，不能由训练 surrogate、采样像素近似或 reference pipeline 输出代替。

---

## 2. 代码主线、入口和工件隔离

### 2.1 唯一 claim-bearing flat v5 主线

| 环节 | 权威代码 | 权威配置/入口 |
|---|---|---|
| 数据、身份、split、score/label manifest | `data_ext/` | 仅哈希冻结后的 manifest |
| Stage 1 backbone 与损失 | `model/MSHNet.py`、`losses/target_background_margin.py` | `configs/aaai27_detector_tail_sep.json` |
| Stage 1 trainer | `scripts/train_multisource_tail.py` | `$PYTHON_BIN -m scripts.train_multisource_tail` |
| Stage 1 单卡/三卡 launcher | `scripts/train_stage1_single_gpu.sh`、`scripts/train_rc_3gpu.sh` | 物理 GPU 仅限 0、1、2 |
| Stage 2 episode/source reference | `rc/build_meta_episodes.py`、`rc/build_source_reference.py`、`rc/meta_dataset.py` | schema、路径和哈希必须一致 |
| Stage 2 model/loss | `model/monotone_pixel_calibrator.py`、`losses/calibrator_risk.py` | `configs/aaai27_calibrator_risk_aligned.json` |
| Stage 2 trainer | `rc/train_calibrator_risk_aligned.py` | `$PYTHON_BIN -m rc.train_calibrator_risk_aligned` |
| Stage 2 严格 launcher | `scripts/train_calibrator_risk_aligned.sh`、`scripts/train_calibrator_v5_strict.sh` | 均解析到 flat v5；仅在 Stage 2 闸门通过后 |
| 原分辨率评测与回放 | `evaluation/` | exact replay 与独立 label attachment |

严格主线的最低工件合同包括：

- detector、split、score manifest、curve manifest、label manifest 和 source reference 的 SHA-256；
- JSON 配置及其 SHA-256；
- schema-v4 grouped episodes；
- v5 calibrator checkpoint；
- native-resolution global exact curve，或经审计的 event-exact high-tail suffix；
- context/query ID 不相交；
- query label 在阈值冻结后才附加；
- `probability > threshold` 的统一阈值语义；
- 8-connected、one-to-one 的冻结 matching contract；
- CPU 端阈值复算和 exact replay 一致。

### 2.2 reference-compatible 路径的边界

以下全部是 reference-compatible / synthetic-smoke 范围：

```text
rc_irstd/
configs/*.yaml
scripts/start_training.sh reference ...
scripts/start_training_reference.sh
scripts/train_calibrator_reference.sh
任何最终解析到 python -m rc_irstd.* 的 launcher
任何 rc-irstd-reference-* console entrypoint
reference README / runbook / legacy LODO scripts
```

这些内容可以用于：

- 对照上游方案；
- import、shape、loss、checkpoint 和合成两阶段链路烟测；
- 复现某个明确的软件合同 bug；
- 检查兼容接口。

这些内容不能用于：

- 生成 claim-bearing detector 或 calibrator；
- 与 flat v5 的 JSON、episode、source reference 或 checkpoint 互换；
- 替代 v5 的 coverage loss、full native replay、严格 provenance 或标签延迟读取；
- 提供 AAAI 表格数字。

判定入口是否合格时看**最终 Python 模块**，不看 Shell 文件名。任何解析到 `rc_irstd.*` 的入口一律标记：

```text
protocol_scope = reference_compatibility_or_synthetic_smoke
claim_bearing = false
```

若需要把 reference 中某项科学设置迁入正式主线，必须显式移植到 flat v5、补测试、生成新的版本化 JSON 并重新冻结 SHA；禁止直接用 YAML 覆盖 JSON，也禁止在同一 run 中同时引用两者。

### 2.3 当前入口使用规则

当前 strict 白名单为：

```text
严格预检:
  scripts/preflight_aaai27.sh

严格 flat-v5 launcher:
  scripts/start_training.sh detector|export-scores|export-labels
  scripts/start_training.sh build-source-reference|build-meta
  scripts/start_training.sh calibrator|online|audit
  scripts/train_rc_3gpu.sh
  scripts/train_stage1_single_gpu.sh
  scripts/train_calibrator_risk_aligned.sh
  scripts/train_calibrator_v5_strict.sh

严格 flat-v5 console entrypoint:
  rc-irstd-audit
  rc-irstd-train-detector
  rc-irstd-export-scores
  rc-irstd-export-labels
  rc-irstd-build-source-reference
  rc-irstd-build-meta
  rc-irstd-train-calibrator
  rc-irstd-apply-calibrator
  rc-irstd-evaluate-adapter
  rc-irstd-threshold-sweep
```

`scripts/start_training.sh` 的默认子命令和上列不含 `reference` 的 console command 已完成 flat-v5 分流；只有显式 `scripts/start_training.sh reference ...`、`rc-irstd-reference-*` 或最终解析到 `rc_irstd.*` 的命令进入兼容路径。入口已分流不等于 Stage 2 已获训练授权：当前 Stage 2 仍为 NO-GO。

---

## 3. 当前已验证事实与证据边界

### 3.1 软件状态

第一份候选 Gate 0 在 `outputs/preflight/aaai27_20260714_candidate/` 记录了：

```text
182 passed
10 subtests passed
```

该候选结果之后又补了依赖、入口方向、outer 身份、source fingerprint 与 bounded-smoke 门禁，因此已被后续源码修改自动作废，只保留为审计痕迹。

第二份候选 `outputs/preflight/aaai27_20260714_final_candidate/` 随后记录了：

```text
RUN_STATUS = PASS
189 passed
10 subtests passed
```

但该运行结束后，`scripts/audit_aaai_protocol.py` 又收敛了跨数据集 exact-hash 报告的公开字段；虽然对应聚焦回归已通过，该候选仍按“任何源码变更都会使旧预检失效”的规则自动过期。

本轮修改完成后，`outputs/preflight/aaai27_20260714_final/` 已重新执行并记录：

```text
RUN_STATUS = PASS
189 passed
10 subtests passed
validated_entrypoints = 31
validated_dispatcher_routes = 9
GPU = 0,1,2 / NVIDIA GeForce RTX 3090
all_split_contracts_passed = true
cross_dataset_exact_image_duplicate_group_count = 0
```

由于把该结果回填到本文档本身也会改变工作树，最终交付快照再以 `outputs/preflight/aaai27_20260714_final_sealed/` 封存；该目录必须在本文档最后一次修改后新建，且之后不再修改源码、配置或本文档。任何测试数字只是一份时间点记录，不是 Gate 0 的硬编码门限；后续每次修改后都必须用固定解释器重跑全套测试，并从新日志动态读取 collected/passed/failed/skipped/subtest 数。

已有工程证据包括：

- flat v5 Stage 1 尾部损失、确定性 global max 和关键数据协议测试；
- flat v5 Stage 2 单调曲线、risk-aligned loss、coverage、checkpoint 与 exact replay 合同测试；
- reference-compatible 包的 import/合成烟测；
- NUAA `Misc_111` 真实样本几何对齐回归；
- 当前三域 official train/test 清单的基本 ID 与重复审计。

这些证据仅证明代码合同可执行，不证明方法有效。

### 3.2 当前未满足项

当前至少存在以下科学准入缺口：

- `configs/aaai27_analysis_plan.json` 尚未以无占位、Git 跟踪和哈希冻结形式完成；
- 工作树尚未形成可供 claim run 使用的 clean commit；
- official train 内部诊断/元验证划分尚未全部预注册并冻结；
- baseline suite 尚未在同协议、同预算下完整实现和验收；
- 只有三个独立域；
- 没有合法第四 outer target 的 split、score、label、source-reference 和 episode 工件；
- 没有预注册的 primary budget、primary endpoint、minimum effect、Pd 非劣界、seed 和 paired CI 合同；
- 没有真实 Stage 2 claim-bearing 训练授权。

因此当前准确状态是：

```text
preflight: GO
strict flat-v5 Stage-1 bounded engineering smoke: GO after preflight
reference synthetic smoke: optional engineering-only
Stage-1 development/performance comparison: NO-GO until Gate -1
all real-data Stage-2 training: NO-GO
official-test outer evaluation: NO-GO
paper performance claim: NO-GO
```

---

## 4. 数据角色与泄漏防线

### 4.1 只有官方 train 和 test

当前三个域为：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

Stage 1 的完整 source 数据池固定为上述三个域的 `official_train`，缺少任何一个都不能称为“三域训练”。必须区分两种合法用途：

- 三域同时作为 source：仅用于当前工程连通性 smoke，或未来所有选择均冻结后的 final refit；
- 未知域 LODO：每个 fold 必须把一个域完整留出，只用另外两个域的 `official_train` 训练；三个 fold 合起来覆盖 NUAA、NUDT、IRSTD-1K，但不能把 held-out 域偷偷加入该 fold。

两种用途都禁止读取三个域的 `official_test` 来训练、验证、early-stop、选 checkpoint 或调参。

原始数据角色只有：

```text
official_train
official_test
```

不得把不存在的官方 validation 写成数据集事实。所有内部 `detector_train`、`detector_val`、`meta_train`、`meta_val` 都只能由 `official_train` 派生，并满足：

1. 在任何 official test 的内容型模型/性能读取前冻结划分规则、seed、ID 清单与 SHA-256；
2. 对序列数据先按序列/组划分，再构造 context/query 窗口；
3. 对非序列数据先按图像 ID 划分，再构造 episode；
4. 内部集合完全不相交；
5. official test 绝不用于 early stopping、checkpoint、超参、预算、窗口长度、继续/停止或基线选择。

工程烟测可以使用完整 `official_train`，但不得创建 validation、不得选择 checkpoint，只保存 fixed-last 工程工件。

### 4.2 official test 的唯一允许用途

official test 的**内容型模型/性能使用**只在代码、配置、超参、outer 角色、阈值策略和统计计划全部冻结后发生：

```text
unlabeled prefix S
  -> compute unlabeled statistics
  -> freeze tau(B)

disjoint future query Q
  -> attach labels only after tau(B) is frozen
  -> native-resolution exact replay
```

唯一例外是预先声明、不会生成性能数字的静态数据完整性审计，例如文件存在性、格式、尺寸、hash、配对关系以及 `Misc_111` 的固定几何回归。该审计不得计算可用于设计的方法指标、阈值或样本难度，也不得触发继续/停止。除这类静态审计外，任何 official test 标签提前被模型、指标或决策路径读取，整个对应 fold 必须标记 `invalid_protocol`，不能通过重新命名 run 修复。

### 4.3 NUAA Misc_111 的唯一处理

必须与 `/home/md0/ly/BasicIRSTD` 保持一致：

1. 读取图像与 mask 的原始 PIL 尺寸，顺序为 `(width, height)`；
2. 若尺寸相同，不处理；
3. 若尺寸不同，计算相对宽高比误差：

```text
relative_error =
  abs(image_width / image_height - mask_width / mask_height)
  / abs(image_width / image_height)
```

4. 仅当 `relative_error <= 0.01` 时，将 **mask** 用 `PIL.Image.Resampling.NEAREST` resize 到图像尺寸；
5. 图像不因 mask 而缩放；mask 对齐必须发生在任何共同 resize/crop/augmentation 之前；
6. 若误差大于 1%，fail closed，视为错误配对或数据损坏；
7. 对齐后重新二值化/检查，禁止 bilinear、bicubic 或 antialias 污染标签。

当前真实 `Misc_111`：

```text
image size = (325, 220)
mask size  = (592, 400)
relative aspect-ratio error = 0.1846%
decision = allowed
operation = mask -> (325, 220), PIL NEAREST
```

该规则必须同时由 flat v5 的 `data_ext/mask_alignment.py` 和 reference-compatible 的 `rc_irstd/data/mask_alignment.py` 回归测试约束。处理 `Misc_111` 不意味着可以无条件 resize 其他不匹配样本。

---

## 5. 当前可以立即执行的工作

### 5.1 固定解释器和 GPU

所有命令统一使用：

```bash
cd /home/md0/ly/RC-IRSTD
export PYTHON_BIN=/home/md0/ly/BasicIRSTD/infrarenet/bin/python
test -x "$PYTHON_BIN"
```

物理 GPU 只使用 `0,1,2`：

- `scripts/train_rc_3gpu.sh` 同时占用 0、1、2，映射为逻辑 cuda:0、1、2；
- `scripts/train_stage1_single_gpu.sh` 每个进程只占一个物理 GPU；
- 同一张卡不得同时被 DP run 和单卡 run 复用；
- 三卡 DP run 与三个单卡 run 的 batch、随机路径和吞吐不同，不能直接比较性能。

### 5.2 Gate 0：预检

```bash
PYTHON_BIN="$PYTHON_BIN" ./scripts/preflight_aaai27.sh --output-dir outputs/preflight/aaai27_20260714_final_sealed
```

Gate 0 必须保存并动态验证：

- Git HEAD、完整 binary diff、dirty status 和全部 untracked file SHA-256；
- 解释器、pip freeze、PyTorch/CUDA、GPU 型号和 UUID；
- compile、Shell syntax、受控 JSON；
- 全套 pytest 零失败、零错误；
- 数据/split/重复与几何审计；
- 预飞行目录不与旧证据混写。

工作树 dirty 时，Gate 0 仍可服务工程烟测，因为预飞行会完整取证；但它不构成 development/claim run 的 Gate -1 通过。

### 5.3 Stage 1 bounded engineering smoke

当前只批准严格 flat v5 的 bounded smoke。flat v5 的 proposed 默认设置已从参考方案显式移植到版本化 strict JSON：

```json
{
  "lambda_margin": 0.20,
  "background_tail_fraction": 0.05,
  "hard_object_fraction": 0.25,
  "object_top_pixel_fraction": 0.25,
  "peak_kernel_size": 5,
  "gt_exclusion_radius": 2,
  "smooth_worst_domain_gamma": 10.0
}
```

来源记录在 `configs/aaai27_detector_tail_sep.json` 的 `reference_package`、`reference_config` 和 `migration_policy` 字段中。reference YAML 不参与运行时解析；strict 主线仍保留 fixed-last、official-test 隔离、flat-v5 provenance 和 exact replay。三卡 DataParallel 的 `batch_per_domain=3` 是硬件切分适配，不伪称与参考包的单卡 batch 完全相同。

本轮已执行一次三域同时作为 source 的 DataParallel 连通性 smoke，物理 GPU 固定为 0、1、2：

```text
source = NUAA-SIRST official_train
       + NUDT-SIRST official_train
       + IRSTD-1K official_train
output = outputs/stage1_engineering_smoke/final-all-three-dp-s42/
scope  = engineering_smoke_not_paper_evidence
```

实际加载的 `official_train` 数量为 NUAA `213`、NUDT `663`、IRSTD-1K `800`；总 loss 为 `1.44091`，首次激活风险梯度范数为 `10.4649`。三个域均产生独立 tail/margin 诊断，所有 logits、梯度和更新后参数均为有限值。该运行只验证三域平衡采样、三卡切分、有限前反传和 fixed-last 保存，不能验证未知域泛化。随后也已执行三折 held-out 身份 smoke；它们验证 LODO 数据角色和 provenance，但同样不产生性能结论。

2026-07-14 已在三个物理 GPU 上并行执行过三折、每折两源域、1 epoch × 1 step 的候选 smoke。每条命令都包含：

```text
--engineering-smoke
--risk-objective margin
--warm-epoch 0
--risk-warmup-epochs 0
--risk-ramp-epochs 0
--lambda-margin 0.20
--tail-q 0.05
--miss-q 0.25
--object-pixel-q 0.25
--peak-kernel-size 5
--exclusion-radius 2
--deterministic
```

Gate 0 通过后已按同一映射重跑，最终工件为：

| 物理 GPU | held-out outer | official-train source | 审计工件目录 |
|---:|---|---|---|
| 0 | NUAA-SIRST | IRSTD-1K + NUDT-SIRST | `outputs/stage1_engineering_smoke/final-audited-outer-nuaa-s42/` |
| 1 | NUDT-SIRST | IRSTD-1K + NUAA-SIRST | `outputs/stage1_engineering_smoke/final-audited-outer-nudt-s42/` |
| 2 | IRSTD-1K | NUAA-SIRST + NUDT-SIRST | `outputs/stage1_engineering_smoke/final-audited-outer-irstd-s42/` |

最终 `final-audited-*` 工件均已保存并验证以下字段：

```text
protocol_scope = engineering_smoke_not_paper_evidence
checkpoint_selection = fixed_last_no_test_or_target_validation
risk_weight = 1.0
risk_gradient_checked = true
risk_gradients_finite = true
parameters_finite_after_epoch = true
```

最终三折的总 loss 分别为 `1.42499 / 1.43047 / 1.42230`，风险梯度范数分别为 `27.7972 / 14.0697 / 16.0842`。这些数字只证明有限前反传和风险分支被激活，不是性能指标，不进入任何主表、Gate G1 或继续/停止判断。此前的 `outer-*-s42` 与 `audited-outer-*-s42` 目录是在最终门禁加固前生成，只保留为调试审计痕迹；本轮只有新生成且合同复核通过的 `final-all-three-dp-s42` 与 `final-audited-*` 才是当前工程证据。

该 smoke 只能回答：

- 三域 official train 能否加载和平衡采样；
- MSHNet backbone、SLS 和域级尾部目标能否有限前反传；
- risk 梯度路径在 `warmup=0, ramp=0` 时是否真正激活；
- GPU 0/1/2、checkpoint 序列化和 provenance 是否工作。

原尺寸 score-map 导出/回放和 `Misc_111` 几何规则由 Gate 0 的独立测试与静态数据审计验证，不得声称这次只读 official train 的训练 smoke 覆盖了 official test 中的 `Misc_111`。

验收条件：

```text
all losses finite
all logits finite
all checked gradients finite
post-step parameters finite
risk effective weight > 0
per-domain tail diagnostics present
no official_test DataLoader created
fixed-last checkpoint saved and reloadable
config/provenance/hash evidence complete
```

失败就停止；修复后必须换新 run ID 重跑。成功只写 `engineering_smoke=PASS`，不得报告为一轮训练精度，也不得触发 Stage 2。

### 5.4 可选 reference synthetic smoke

reference-compatible synthetic smoke 只能在与 strict 工件完全隔离的目录执行，并写：

```text
RUN_STATUS = synthetic_reference_smoke
protocol_scope = reference_compatibility
claim_bearing = false
```

它只用于确认参考包未因集成而损坏，不是“第二套主线复现”。本次已经完成：

- `outputs/reference_detector_engineering_smoke/`：三域 official train、MSHNet + domain-tail separation、1 个优化步、无 validation，`run_scope=engineering_smoke_fixed_last_no_validation`；
- `outputs/reference_compatibility_smoke/`：合成 score/episode 上的 No-Reject calibrator 2 epoch 和 hard replay，`status=passed`、无 Reject head；
- 首次合成烟测因迁移脚本缺少执行位在进入 Python 前失败，修复执行位后用同一固定解释器和新产生的完整工件成功重跑。

上述 reference 工件一律 `claim_bearing=false`，不得和 audited flat-v5 三折 checkpoint 连接。

---

## 6. Gate -1：开发训练前必须冻结

Stage 1 工程 smoke 不要求 Gate -1；任何开发比较、调参或论文结果要求 Gate -1 全部通过：

```text
G_MINUS_1 =
  fixed_interpreter_and_sha
  AND clean_git_commit
  AND strict_flat_v5_authority_recorded
  AND analysis_plan_exists_and_is_git_tracked
  AND analysis_plan_has_no_TBD
  AND strict_JSON_hashes_frozen
  AND data_and_internal_split_hashes_frozen
  AND primary_budget_endpoint_baseline_frozen
  AND seeds_statistics_and_failure_policy_frozen
  AND confirmatory_outer_absent_from_development_decisions
```

`configs/aaai27_analysis_plan.json` 至少应冻结：

- development 与 confirmatory domain；
- official train 派生 split 的 ID、规则、seed 和 hash；
- Stage 1 exact epoch、risk schedule、D0–D3 设置；
- Stage 2 budget grid、primary budget 和 episode contract；
- primary baseline、primary endpoint、聚合公式和方向；
- minimum effect 与 Pd non-inferiority margin；
- seed 列表、paired comparison、bootstrap unit 和 confidence level；
- multiple-comparison、missing-run 和 low-FA estimability policy；
- matching、threshold、checkpoint、outer development/refit 规则；
- G1–G6 的完全可复算布尔表达式。

任何 `TBD_PRE_REGISTER`、dirty commit、未哈希 split 或未冻结统计规则存在时，development/claim 训练保持 NO-GO。

---

## 7. Stage 1 的后续科学实验

Gate -1 通过后才可运行以下同协议配对实验：

| ID | Stage 1 变体 | 机制问题 |
|---|---|---|
| D0 | MSHNet backbone + segmentation/SLS | backbone/segmentation baseline |
| D1 | D0 + legacy per-image hinge | 域级聚合是否必要 |
| D2 | D0 + domain-tail separation，但不做 GT-neighbour exclusion | 邻域排除是否必要 |
| D3 | D0 + 完整 domain-tail separation | proposed Stage 1 |

这里的 MSHNet 仍只是 backbone；`D3` 也不能简称为“MSHNet 新模型”。

所有可比较 run 必须固定：

- 相同 official-train 派生 split；
- 相同数据顺序、augmentation、batch-per-domain；
- 相同 optimizer、scheduler、epoch、seed；
- 相同输入尺寸/native-shape 恢复；
- fixed-last checkpoint 规则；
- 相同 GPU 启动模式；
- 全部 D0–D3，无选择性漏跑。

GPU 并行策略：

| 物理 GPU | 第一波 |
|---:|---|
| 0 | D0 |
| 1 | D1 |
| 2 | D3 |

D2 必须按预注册第二波运行，不能根据前三个结果决定取消。若用三卡 DP，则所有 D0–D3 都应以相同 DP 模式串行运行，不与单卡结果作主比较。

Stage 1 至少记录：

```text
segmentation loss
separation loss and effective weight
per-domain target lower tail
per-domain background upper tail
per-domain gap and margin violation
background peak q99 / q99.9 / q99.99
hard-object lower-tail quantiles
logit mean / std / max
internal diagnostic IoU / nIoU
internal native-resolution Pd and pixel Fa
config, split, code and checkpoint hashes
```

内部 validation 只能从 official train 冻结产生。当前 strict Stage 1 主协议使用 fixed-last；诊断指标不得暗中选 checkpoint。若未来改成 internal-val selection，必须新建协议版本、统一所有方法并重新预注册，绝不能使用 official test。

G1 在 analysis plan 中必须机械定义；在具体门限未冻结前，`G1=false`。不能用“趋势较好”“多数域改善”或看完结果后任选 gap/Pd/IoU 来判 GO。

---

## 8. Stage 2：当前固定 NO-GO

### 8.1 为什么三域不够

现有域只有 A、B、D。若 D 是 outer target：

```text
meta-source = A, B
pseudo-target A -> inner detector source only B
pseudo-target B -> inner detector source only A
```

每个 inner detector 都退化成单源。这不能支撑“两源及以上的多源 nested-LODO”主张。增加窗口、图片或 seed 不能增加独立域数。

同时，当前不存在一套合法、冻结且可审计的 outer-target 工件：

- 第四独立域的许可与来源审计；
- official train/test manifest 与 hash；
- 与现有域的 exact/near-duplicate 审计；
- outer detector checkpoint/source-domain provenance；
- native score manifest 和独立 label manifest；
- source reference、curve manifest、episode/fold contract；
- final refit、standardizer 和 outer policy；
- 完整公平 baseline suite。

因此以下全部为 NO-GO：

```text
three-domain real-data Stage-2 training
three-domain pseudo-target LODO pilot
three-domain single-source-inner outer smoke
official-test adapter evaluation
claim-bearing calibrator result
using reference rc_irstd training as a substitute
```

即使标注“non-claim”，当前也不运行三域真实 Stage 2，因为它会产生容易被误用的数字且不能解除协议缺陷。已有 synthetic/minimal 单元烟测已足以证明基本执行链路。

### 8.2 第四域后的最小非退化设计

加入独立第四域 D 后：

```text
meta-source = A, B, C

pseudo-target A -> detector sources B, C
pseudo-target B -> detector sources A, C
pseudo-target C -> detector sources A, B

outer detector -> sources A, B, C
outer D official test -> unlabeled prefix S then frozen-threshold query Q
```

这只是非退化 nested-LODO 的最小条件，不等于已经证明广义泛化。第四域必须在其标签被用于任何设计决定前冻结 development/confirmatory 身份。

未来 Stage 2 fold 可分别占用 GPU 0、1、2；但只有在第四域、Gate -1、G1、G2、真实 artifact preflight 和 baseline suite 全部通过后才允许启动。

### 8.3 Stage 2 必须保留的 v5 合同

```text
L =
  lambda_viol * L_viol
  + lambda_util * L_util
  + lambda_oracle * L_oracle
  + lambda_smooth * L_smooth
  + lambda_coverage * L_coverage
```

必须保留：

- 完整冻结预算网格；
- descending-budget structural monotonicity；
- log10-budget range 内插值；
- range 外 fail closed，禁止外推；
- global exact 或经审计的 event-exact suffix；
- coverage 支持域检查；
- native-resolution hard replay；
- 独立 label attachment；
- 无 reject 字段；
- checkpoint schema/fold/detector/standardizer hash 一致；
- checkpoint 按 `BSR max -> LogExcess min -> Pd max -> earlier epoch on exact tie` 的冻结字典序选择；
- 另设预注册 Pd anti-collapse Gate，防止全高阈值退化解。

G3 的 primary baseline、effect test、Pd 非劣界、surrogate/exact 差异界和成功 pseudo-target 数尚未冻结，因此当前 `G3=false`。

---

## 9. 证据计划：不填假结果

### 9.1 claim—evidence 矩阵

| 论文主张 | 必需证据 | 当前状态 |
|---|---|---|
| Stage 1 尾部分离改善低虚警排序 | D0–D3 配对、域级尾诊断、native Pd/Fa、paired seeds | 未运行，NO-GO until Gate -1 |
| Stage 2 优于简单阈值策略 | 同 score/context/query/budget 下的 C0–C7 公平比较 | baseline 不完整，NO-GO |
| No-Reject 单调曲线不违约且不塌陷 | BSR、LogExcess、Pd anti-collapse、exact replay | 仅软件合同有测试 |
| 对未知域有效 | 非退化 nested-LODO、全 outer、预注册 seeds/CI | 仅三域，NO-GO |
| 两阶段各有贡献 | D0/D3 × simple/C7 交叉消融 | 未运行 |
| 对 backbone 通用 | 第二个独立强 detector 的同协议证据 | 未具备；当前 claim 必须限定为 MSHNet backbone setting |

### 9.2 Stage 2 最小基线

| ID | 方法 | 用 outer 标签 | 状态 |
|---|---|---:|---|
| C0 | fixed 0.5 | 否 | 待同协议 runner 验收 |
| C1 | pooled-source threshold | 否 | 待实现/验收 |
| C2 | worst-source safe threshold | 否 | 待实现/验收 |
| C3 | rolling quantile | 否 | 待冻结参数与 runner |
| C4 | EVT/GPD | 否 | 待 tail-fit/fail-closed 合同 |
| C5 | direct threshold MLP | 否 | 待等特征/等预算公平版本 |
| C6 | monotone oracle-threshold regression | 否 | 待独立 runner/checkpoint |
| C7 | flat v5 No-Reject risk-aligned calibrator | 否 | 软件已实现，真实训练 NO-GO |
| C8 | target oracle | 是 | 仅冻结后上界，不参与选择 |

C0–C7 必须使用相同 detector score maps、context/query IDs、预算网格、matching、seed、调参预算和 exact replay。C8 的标签不能回流到任何 Gate。

### 9.3 两阶段交叉消融

```text
D0 + primary simple threshold baseline
D0 + C7
D3 + primary simple threshold baseline
D3 + C7
```

第二 backbone 未完成时，只能写“在 MSHNet backbone setting 下”，不能写 detector-agnostic。

### 9.4 指标与统计

每个冻结预算至少报告：

```text
Pd@B
BSR
LogExcess
exact pixel Fa
false-pixel count
background/total pixel count
mean-domain and worst-domain Pd
budget estimability flag
```

同时报告 IoU/nIoU 和 connected-component FA 作为兼容指标，但主风险仍是原图分辨率 pixel false-alarm rate。

统计要求：

- 唯一 primary budget 和 primary endpoint 在看结果前冻结；
- 不把像素当独立 Bernoulli 样本；
- 序列用 sequence-level block bootstrap，非序列用 image-level bootstrap；
- 主方法与 primary baseline 用相同 block 和 seed 的 paired difference CI；
- 每域原始结果与汇总结果同时报告；
- 预注册低虚警可估性、multiple comparison 和 missing-run policy；
- 不把训练 seed 当成独立域。

任何未知数值都保持 `TBD`，不得为文档完整性伪造结果、提升幅度、置信区间或排名。

---

## 10. GO / NO-GO 闸门

| Gate | GO 条件 | 当前状态 |
|---|---|---:|
| G0 软件/环境 | 当前 commit/worktree 快照下 compile、pytest、Shell、JSON、GPU、数据审计全通过 | **PASS（仅工程准入；以 `aaai27_20260714_final_sealed` 为准）** |
| G-1 身份/预注册 | clean commit、无占位 analysis plan、strict JSON/split/statistics 全冻结 | **NO-GO** |
| G1 Stage 1 | D0–D3 完整且预注册机械条件为真 | **NO-GO** |
| G2 校准问题成立 | 仅 official-train development 数据上，阈值漂移/oracle gap/Stage1 effect 达到预注册条件 | **NO-GO** |
| G3 Stage 2 | v5 软件合同、primary effect、Pd anti-collapse 和 baseline 合同全通过 | **NO-GO** |
| G4 协议非退化 | 独立域数至少 4，每个 inner detector 至少两个源域，工件/许可/去重/角色冻结 | **NO-GO** |
| G5 outer | G4、统计计划、outer development/confirmatory 模式和 final-refit 全冻结 | **NO-GO** |
| G6 AAAI 证据 | 所有预注册 outer/seeds/baselines/ablations/paired CI 完整报告 | **NO-GO** |

唯一当前 GO 是：

```text
G0 preflight
strict flat-v5 bounded Stage-1 engineering smoke after G0
optional isolated reference synthetic smoke
analysis-plan/split/baseline/fourth-domain preparation
```

工程烟测豁免 Gate -1，但它永远不能用于 G1–G6。

---

## 11. 硬停止条件

任一条件出现即停止对应 run 并标记 `invalid_protocol`：

```text
nonfinite loss, logits, gradients or parameters
score/label/native-shape contract failure
hard replay not exactly reproducible
CPU threshold recomputation mismatch
structural monotonicity violation
budget extrapolation not rejected
official-test label opened before frozen evaluation
context/query overlap or support-label leakage
checkpoint/split/artifact/schema hash mismatch
reference and flat-v5 artifacts mixed
YAML and strict JSON mixed in one run
confirmatory outer used for design or continue decision
new domain license/source/duplicate audit failure
three-domain Stage-2 result relabelled as allowed evidence
Misc_111 handled by unconditional or non-nearest interpolation
```

修 bug 后必须保存旧证据、换新 run ID，并从新的预检开始；不得覆盖旧目录。

---

## 12. 实验目录与状态

每个 run 至少保存：

```text
outputs/experiments/<experiment_id>/
  RUN_STATUS.txt
  protocol_scope.txt
  claim_bearing.txt
  command.sh
  config.json
  config_sha256.txt
  git_commit.txt
  git_status_porcelain.txt
  git_diff.patch
  untracked_sha256.tsv
  environment.freeze.txt
  gpu_environment.txt
  data_manifest.json
  split_hashes.json
  metrics.jsonl
  checkpoint_evidence.json
  score_manifest/
  curve_manifest/
  label_manifest/
  exact_replay.json
  notes.md
```

当前允许状态：

```text
preflight
engineering_smoke
synthetic_reference_smoke
invalid_protocol
```

只有 G-1 以后才可新增 `development_stage1`；只有 G4/G5 以后才可新增 `claim_candidate_outer`；只有 G6 与完整审计后才可标记 `final_claim_evidence`。

---

## 13. 立即执行顺序

### Step 1：已完成

```text
fix PYTHON_BIN
-> run scripts/preflight_aaai27.sh
-> require dynamic zero-failure tests
-> verify GPU 0/1/2
-> verify data identity, split and Misc_111 contract
```

### Step 2：已完成

```text
run one bounded strict flat-v5 Stage-1 engineering smoke
-> first run NUAA + NUDT + IRSTD-1K official_train together on GPU 0/1/2
-> then run the three predeclared two-source/one-held-out folds on isolated GPU 0/1/2
-> official_train only
-> fixed-last, no validation, no official test
-> save complete dirty-worktree evidence if still dirty
-> output only PASS/FAIL
```

### Step 3：烟测后只做准备，不继续 Stage 2

```text
freeze clean commit
-> create no-TBD analysis plan
-> freeze official-train internal splits
-> implement/equalize baselines
-> acquire and audit a fourth independent legal domain
```

### Step 4：未来满足条件后

```text
run D0-D3 under one strict protocol
-> evaluate G1/G2 using official-train-derived development data
-> require G4
-> build legal flat-v5 artifacts
-> run Stage 2 folds on GPU 0/1/2
-> freeze threshold before outer query labels
-> run all pre-registered outer/seeds and paired statistics
```

---

## 14. 最终批准意见

### 现在批准

- 固定解释器的预检；
- GPU 0/1/2 环境与数据审计；
- strict flat v5 Stage 1 的 bounded、非主张性工程烟测；
- 隔离的 reference synthetic smoke；
- analysis plan、官方 train 内部划分、基线和第四域准入准备。

### 现在不批准

- 任何真实 Stage 2 训练；
- 三域 pseudo-target LODO pilot；
- 三域 single-source-inner outer smoke；
- official test 上的调参、checkpoint 或继续/停止判断；
- `rc_irstd.*` 与 flat v5 工件混用；
- YAML 与 strict JSON 混用；
- reference console/script 输出进入论文表格；
- analysis plan 尚有占位时的 development 或 claim run；
- 把完整模型称为 MSHNet；
- 把工程 PASS 当作性能结果。

### 最终结论

> **按 Two-Stage No-Reject RC-IRSTD 的身份继续，但当前只运行 flat v5 预检和 Stage 1 工程烟测。MSHNet 仅是 Stage 1 backbone；flat v5 是唯一 claim-bearing 主线；`rc_irstd/` 仅作参考兼容/合成烟测。由于只有三个域且缺少合法 outer-target 工件，真实 Stage 2 与 outer 评估保持 NO-GO。**
