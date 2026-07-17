# RC-IRSTD AAAI-27：完整模型设计冻结与实验执行计划

> 状态：**08:00 前冻结中的 result-free 设计合同**  
> 仓库：`/home/md0/ly/RC-IRSTD`  
> 数据：仅 `NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K` 的冻结 official-train-derived 划分  
> 预注册实验矩阵与空白结果表：[RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md](RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md)  
> 设备：真实运行仅允许 GPU 0/1/2  
> 红线：在单独授权前，official test 的 ID、图像、标签、预测、指标和决策均保持封存

### 权威性与版本优先级

本文、配套预注册矩阵以及 `configs/aaai27_stage2_crossfit_v2.json` 共同构成
2026-07-17 Stage-2 权威合同。旧文件 `configs/aaai27_analysis_plan.json` 中的
`C32/Q64`、T6 hidden=192 和旧 C6 描述均已被本合同取代，**不得用于执行、审计、
恢复或解释本轮实验**。S2_I0 与后续 launch manifest 必须以外部 SHA-256 绑定
`aaai27_stage2_crossfit_v2.json`；只引用旧 JSON、文件名而没有 v2 字节哈希，均
fail closed。旧文件仅保留为历史记录，不是可执行配置。
> Result-free 范围：本冻结不产生或读取新的 Stage-2 观测性能结果；它仅以固定 SHA-256 绑定此前已经完成、含观测开发证据的 Stage-1 G1 结论，不能把“绑定 G1”写成“整个过程从未接触任何观测结果”。

## 1. 什么叫“模型设计成功”

本项目把三个容易混淆的结论严格分开：

1. **模型设计成功（S2_I0 PASS）**：完整模型、训练器、因果数据链、阈值族、exact replay、恢复机制和实验合同通过 result-free 合成验收。此时停止改模型。
2. **development 假设成立（S2_DGO GO）**：冻结模型在三域 × 三种子真实 development outer-LODO 上，T8 相对 T4 同时通过 BSR 效应和 Pd 非劣门。
3. **AAAI 证据完整（S2_EC PASS）**：主结果、T0–T9、C0–C6、稳健性、失败分析、效率、统计区间和完整性复审全部完成。

因此，S2_I0 PASS 只说明“完整模型已经设计并实现到可以展开正式实验”，不能提前写成性能成功或 AAAI 可接收。

### 1.1 科学问题、可证伪假设与贡献边界

科学问题是：当 detector 面对训练时未见的目标域，且部署端只有一个短的无标签
context、不能拒绝样本或读取未来标签时，能否从 detector score distribution 选择
同时适用于三个低 false-alarm budgets 的 operating points，并优于直接对 context
pixels 取尾部 order statistic？

本轮需要由实验分别支持、不得相互替代的设计点为：

1. **结构假设**：按从宽松到严格预算联合输出严格递增阈值，可减少预算次序违反；
2. **目标假设**：直接在 verified exact event curve 上优化 risk-aligned surrogate，
   比只回归 oracle thresholds 更有利于 BSR；
3. **信息假设**：score、local peaks、gray statistics 和 source distance 的组合在未知域
   context 中提供互补信息；
4. **两阶段假设**：Stage-1 tail separation 与 Stage-2 operating-point calibration 是
   不同机制，必须分别由 D0–D3 与 T6/T7/T8、C2/C4–C6 证据归因。

主门 T8−T4 只检验“完整学习式 calibrator 相对 training-free rolling quantile”的
端到端价值，**不能单独证明 monotonicity 或 risk-aligned loss 的必要性**。结构与目标
机制主张分别要求 T7−T6 和 T8−T7 的冻结消融支持；若对应差异不支持，不得用主门
结果代替机制证据。

## 2. 最终方法身份

RC-IRSTD 是一个 **Two-Stage / No-Reject** 系统，不是把 MSHNet 改名后的单阶段网络。MSHNet 仅是本轮冻结的 Stage-1 detector backbone。

### 2.1 Stage 1：跨域小目标检测器

- Backbone：MSHNet，三通道输入、原分辨率输出。
- D0：多尺度 SLS segmentation-only 对照。
- D3（正式 Stage-1）：D0 加 domain-level target-lower-tail / background-upper-tail separation。
- D3 风险项：
  - 背景：GT 膨胀排除后，每图确定性局部峰值的上尾；
  - 目标：每个 GT object 内 top-pixel logits 的对象级下尾；
  - 先在域内聚合，再做 margin hinge，再以 normalized smooth worst-domain 聚合；
  - target-free 图只贡献背景统计；target-free 域不进入 hinge；
  - D1/D2 保持同一 forward hinge，仅分别 stop-gradient 一侧，作为机制消融。
- 冻结超参：`lambda_margin=0.2`、logit margin `1.0`、background tail `0.05`、hard-object fraction `0.25`、object top-pixel fraction `0.25`、peak kernel `5`、GT exclusion radius `2`、smooth-worst gamma `10`。
- Detector checkpoint：fixed-last，不用 official test 或 outer target 选模。

### 2.2 Stage 2：未知域 No-Reject operating-point calibrator

每个完整窗口严格为按冻结顺序、不重叠的 `C14/Q28`：

```text
14 张无标签 context
  -> 93D 无标签统计
  -> No-Reject 三预算阈值曲线
  -> 在任何 Q28 label/mask 打开前封印 T0–T8
  -> 28 张未来 query exact replay
```

三个 benchmark 在本协议中仍按 iid-image 数据处理；“context first / query second”
只规定不可逆的信息访问顺序和因果审计边界，不声称原数据具有真实时间序列、在线
视频流或 temporal causality。论文中不得把冻结文件顺序包装成自然时间动态。

93D 特征按固定组组成：

- G_score：索引 0–38，共 39 维；
- G_peak：索引 39–78，共 40 维；
- G_gray：索引 79–86，共 8 维；
- G_source：索引 87–92，共 6 维。

标准化器只在 Stage-2 training contexts 上以 float64 拟合，scale floor 为 `1e-8`；feature mask 只能在标准化之后把排除维设为精确正零。Validation 和 outer target 对标准化器拟合的访问次数必须为 0。

### 2.3 主模型 T8

输入为 93D context statistics，预算网格固定为：

```text
[1e-4, 1e-5, 1e-6]  # 从宽松到严格
```

结构：

```text
93 -> Linear(32) -> GELU -> Dropout(0.1)
   -> Linear(4) interval masses
   -> bounded cumulative spacing
   -> 3 个严格递增 threshold logits
   -> sigmoid thresholds
```

- 类：`MonotoneNoRejectPixelRiskCalibrator`；
- 参数量：`(93×32+32)+(32×4+4)=3140`；
- logit bounds：`[-10, 18]`；
- minimum logit gap：`0.001`；
- reject/abstain head：无；
- missing-episode fallback：无；
- 域级 threshold override：无；
- 风险含义：empirical calibration，不声称 certified/distribution-free guarantee。

T8 loss 使用 verified global-exact query event curve 的当前预测阈值局部左右事件，
在 logit space 定义可微的分段线性 surrogate。这里“exact”修饰的是 uncapped event
curve 与左右 bracket；它**不表示**离散 empirical risk 在事件之间本身是线性的。
compact-bracket 与 full-curve 两种实现对这个冻结 surrogate 数值和梯度等价。完整
all-event curve 保留在 CPU/磁盘证据中；每步只将每 episode、每预算的 exact
bracket 搬到 GPU。这不是事件采样、截断或 cap，同时避免数百万事件 × batch 16 的
显存爆炸。

损失权重：

```text
lambda_violation = 4.0
lambda_utility   = 1.0
lambda_oracle    = 0.1
lambda_smoothness= 0.01
lambda_coverage  = 4.0  # exact-curve support coverage，不是 reject coverage
risk_epsilon     = 1e-12
Huber delta      = 1.0
```

### 2.4 机制对照 T6/T7

- T6：`93 -> 32 -> 3` direct No-Reject MLP，3107 参数；三个输出彼此独立、允许非单调，但每个 logit 都有界于 `[-10,18]`。只用 oracle-logit Huber。
- T7：与 T8 相同的 3140 参数结构和结构单调性，只用 oracle-logit Huber，不使用 risk-aligned exact-curve loss。
- T8−T7 隔离 risk-aligned objective；T7−T6 隔离 structural monotonicity。

## 3. 冻结阈值族

所有方法均使用 `prediction = probability > threshold`，均无 reject、无 target-label fallback：

| ID | 冻结定义 | Outer Q28 labels 用于决策 |
|---|---|---:|
| T0 | 三预算固定 0.5 | 否 |
| T1 | 两个 source-validation 域 pooled exact safe threshold | 否 |
| T2 | 两个 source-specific safe thresholds 的较大者 | 否 |
| T3 | 标准化特征 0–86 上最近 source，再用该 source safe threshold | 否 |
| T4 | 合并 14 张 context 的全部 float64 pixels；对每个预算 `b`、总数 `N`，取升序索引 `max(0, N-floor(bN)-1)` 的值；配合严格 `>`，context exceedance 不超过 `floor(bN)` | 否 |
| T5 | q0.95 以上 scipy GPD MLE，少于 50 个 exceedances 则 sealed missing、无 fallback | 否 |
| T6 | direct No-Reject checkpoint | 否 |
| T7 | monotone oracle-regression checkpoint | 否 |
| T8 | risk-aligned monotone No-Reject checkpoint | 否 |
| T9 | post-label target-future oracle，仅诊断 | 是，且永不进入选择/GO/主比较 |

T0–T8 必须以固定顺序形成完整 decision set。T5 可使用唯一的
`SEALED_MISSING_INSUFFICIENT_TAIL_NO_FALLBACK` 结果；其余八项必须是
`SEALED_COMPLETE`。T9 的 schema 与 API 独立，出现在 prelabel set 即为致命完整性错误。

## 4. 因果与数据隔离链

Outer target 的唯一合法顺序：

1. 验证 W05 C14/Q28 window、W03 float64 score-v4、W04 checkpoint-specific source reference；
2. 生成无标签 context package，禁止解析任何 context/query mask path；
3. 用相同 context/query/score/geometry/budget identities 封印 T0–T8；
4. 公共 verifier 重放 decision bundle、context package、commit、StatisticsConfig 和全部外部 SHA；
5. 只有前四步全部成功后，才允许 resolve/stat/open 28 个 query masks；
6. 生成 label attachment 与 uncapped exact curve；
7. 生成完整 episode-v5 并按已封印阈值 replay；
8. 如需 T9，只能走单独 post-label diagnostic API。

训练集使用 detector-OOF scores；source checkpoint validation 使用 detector-full-fit scores；outer target 不得进入训练、标准化器拟合或 checkpoint selection。以下四个边界均要求 train/validation/outer 零重叠：

- canonical ID；
- original image SHA-256；
- near-duplicate cluster ID（无簇使用 unique sentinel）；
- exclusion group ID。

NUAA `Misc_111` 只按 BasicIRSTD 规则以 nearest-neighbor 将 mask 对齐到 image geometry；禁止 crop、bilinear 或静默形变。

## 5. 训练合同

- Optimizer：AdamW，lr `1e-3`，weight decay `1e-4`；
- Scheduler：none；
- Batch size：16；
- Max epochs：100；
- Early stopping patience：20；
- Gradient clip norm：5；
- AMP：false；
- DataLoader workers：0；
- deterministic algorithms：true；
- seeds：42、123、3407，仅用冻结 SHA-256 domain-separated mapping；
- checkpoint：schema-v6，tensors/primitives-only，`weights_only=True` 重载；
- 每 epoch immutable generation；payload/sidecars 先发布，COMMIT 最后；
- resume 必须提供 generation COMMIT path + external SHA，并恢复 Python/NumPy/Torch/CUDA/DataLoader RNG。

Checkpoint selection 只看冻结 primary budget `1e-5`：

1. 两个 source 域各占 1/2；
2. 域内 BSR 和 `ln(max(FA/b,1))` 对全部 mandatory windows 等权平均；
3. 域内 Pd 使用整数 sufficient counts pooled TP / pooled GT；
4. 排序：macro-source BSR 最大、LogExcess 最小、Pd 最大、完全平局时更早 epoch。

Outer target 的任何值都不得参与 early stopping 或 checkpoint rank。

## 6. S2_I0：模型设计冻结门

只有下列全部 PASS，才写“RC-IRSTD 完整模型设计成功”并停止改模型：

- T6/T7/T8 参数量分别为 3107/3140/3140；
- T7/T8 对全部有限输入零结构单调性违反；T6 不被错误排序；
- T8 loss 对模型参数有有限、非零梯度，T6/T7 只路由 Huber；
- C14/Q28 context、episode-v5 和三类 collection completeness 全部重放；
- train/validation 四身份边界零重叠，outer target 零访问；
- StatisticsConfig 外部绑定、93D train-only standardizer、1e-8 floor 全部重放；
- T0–T8 decision set 完整，T9 无法进入；
- decision/context 任一缺失或被篡改时，mask resolver 调用次数严格为 0；成功后才为 28；
- exact DSU curve 与 brute-force 随机小图逐行、canonical digest 完全一致；
- 28×64×64 全唯一事件测试输出全部 114,690 个工作点、legacy matcher/event 调用为 0、CPU 小于 30 秒；
- million-event ragged CPU + compact exact-bracket 路径不生成 padded full-curve GPU tensor；
- checkpoint 原子故障注入、weights-only load 和 CPU interruption/resume 等价通过；
- W01 的 53-file materialization index 与独立审计、27 个 run contracts、B1 integration PASS、9×7 seed manifest 均按外部 SHA-256 重放；
- W10 deployment、W11 paired bootstrap、legacy consumer 回归与 W13 端到端 orchestrator fail-closed 合同通过；
- W12 的 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三域 metadata、pre-open plan、index 及 sidecar 全部按外部 SHA-256 绑定，且不物化 official-test IDs；
- W06–W13 每个工作包的固定 required-node collection 均完整，official-access sentinel tests 已收集；固定 synthetic/fault-injection allowlist 执行不少于 100 个测试且 failure/error/skip 均为 0；
- B3 implementation integration 为 PASS；B4 scoped release 完整绑定 manifest/archive/environment/COMMIT 的外部 SHA-256，并明确不声称共享 Git worktree clean 或已 tag；
- compile 与 `git diff --check` 通过；后者只检查 diff whitespace，不等价于 clean worktree。

S2_I0 审计器不接受 dataset、checkpoint 或 official-test 参数，并在子进程中隐藏 CUDA；固定 sentinel suite 检查关键 official-access fail-closed 边界。该证据不能反向证明未被 instrument 的系统级文件访问事实，因此报告只写“未授权、未接收相关参数、CUDA 对审计子进程不可见”，不写未经观测支持的全系统访问次数。

S2_I0 报告不含模型性能数字，且 `execution_authorized=false`。通过后仍须单独签发真实 experiment launch authorization。

## 7. 冻结后的实验顺序

### 7.1 S2_DGO：只运行主假设所需最小矩阵

先运行 T8 vs T4，不先看完整基线/消融：

- 3 outer domains × 3 base seeds；
- 每 cell 使用全部 mandatory outer windows：NUAA 1、NUDT 3、IRSTD-1K 3；
- 每个 window 严格 C14/Q28，原始点估计中每个单位恰好一次；
- 主预算：`1e-5`；
- 主 endpoint：T8−T4 macro-domain BSR；
- Pd 为联合非劣约束；
- paired bootstrap：10,000 次，固定 domain strata，域内 seed → window → query image 分层有放回抽样；T8/T4 使用 byte-identical indices；阈值和 checkpoint 在 bootstrap 中固定。

S2_DGO 仅在以下全部成立时 GO：

```text
delta macro BSR point >= 0.05
delta macro BSR 95% CI lower > 0
delta macro Pd point >= -0.02
delta macro Pd 95% CI lower >= -0.02
```

还必须满足 9 个 primary cells 全部 complete/finite/estimable、无补值、无身份错误、无 T8 单调性违反、official test 完全缺席。任何一项不满足即 HOLD，次要诊断不得救门。

其中 `0.05` 和 `0.02` 都是绝对比例，即 5 和 2 个百分点，不是相对百分比。点估计
固定为：window 内先由 28 张 query 汇总 sufficient counts；BSR/LogExcess 依次对
window、seed、domain 等权平均；Pd 在每个 domain×seed 内先对全部 mandatory
windows 池化整数 TP/GT，再对 seed、domain 等权平均。bootstrap 固定三个 domain
strata，不重采样 domain；百分位区间使用双侧 2.5%/97.5%、Hyndman–Fan type 7。
NUAA 只有一个 window，因此其 window 层重采样退化，不能声称估计了 NUAA 的
between-window variation。三域与三种子只支持对这三个冻结 benchmark/settings 的
经验推断，不支持把 domain 当作随机总体后声称任意未知域泛化。

### 7.2 S2_EC：主门 GO 后补齐 AAAI 证据

- 阈值基线：T0–T9；
- 机制消融：C0–C6，其中 T6/T7/T8 复用 C0/C1/C3；
- 特征消融：score only、score+peak、score+peak+gray、full 93D；
- Stage-1 消融：D0/D1/D2/D3；
- detector portability：使用本机 `BasicIRSTD/model/DNANet` 作为唯一预注册第二骨干，重新按相同三域 official-train-derived outer-LODO、相同三种子与 fixed-last 规则产生连续 score maps；DNANet 内部 T8/T4 的 context/query/geometry/bootstrap identities 必须 byte-identical（score maps 必然是 DNANet-specific），不复用历史日志、不因结果更换第二骨干；
- 三预算完整曲线与每域结果；
- robustness：context 顺序/大小的预注册敏感性、score perturbation、domain-distance 特征去除；
- failure analysis：T5 insufficient-tail、zero-GT、低预算不可估、域级失败；
- efficiency：Stage1 参数/FLOPs/吞吐，Stage2 参数、context/decision latency、curve build 时间、CPU RAM/磁盘与 GPU peak memory；
- 统计：主比较不做多重校正；次要比较 Holm family-wise alpha 0.05；
- 完整性复审：claim–evidence、数字、identity、图表和 artifact hashes 一致。

DNANet 的冻结方向性判据为：9 个 domain×seed cells 全部 complete/finite，macro
`delta BSR > 0` 且 macro `delta Pd >= -0.02`。满足它只允许写“方向一致”；只有同时
通过与主门相同的 BSR effect/CI 与 Pd non-inferiority/CI 四条件，才允许写“在第二
backbone 上得到统计支持的复现”。DNANet 无论成功或失败均不改变、不补救 S2_DGO。

次要比较的 Holm families、context/score 扰动值和精确重采样/检验规则以配套
预注册矩阵第 6 节为准；不得在看到结果后移动 family、删除失败条件或更换扰动。

### 7.3 Confirmatory one-look

只有 S2_DGO GO、三个数据集的 confirmatory identity 冻结且另有显式 opening
authorization 后，才允许原子式 one-look 打开 official test。每个数据集固定 first-14
context、all-remaining suffix query；context labels 不打开；阈值先封印；不在线
更新、不替换 context、不截断 query、不重新选阈值、不按结果重跑。

每个 outer dataset 使用 development 阶段已经冻结的三个 seed-specific Stage-1
fixed-last checkpoints 和三个 seed-specific T8 source-validation-selected checkpoints；
T4 与同 seed 的 detector scores 配对。禁止按 official-test 结果挑 seed、做 score
ensemble 或重选 checkpoint。点估计在 dataset×seed 内汇总一个 suffix-query
window，随后 seed 等权、dataset 等权；T8/T4 的 query-image bootstrap indices 必须
配对相同。one-look 的 `S2_CONFIRM_SUPPORTIVE` 判据复用 S2_DGO 四条件与 10,000
次 paired 95% CI，并要求 3 datasets×3 seeds 全部 complete；它只决定 confirmatory
结果报告为 supportive 或 not supportive，**不得回写、推翻或挽救 development 门，
也不得触发模型修改**。无论结果如何都按一次性结果完整报告。

该 first-14/suffix setting 是 transductive context-calibration 协议，不等同于使用完整
official-test 的标准 segmentation evaluation。confirmatory 表只能比较遵守同一
context/query identity 的方法；不得把 suffix-only T8/T4 数字与使用完整 test set 的
published numbers 放在同一列宣称 SOTA。若引用外部方法，必须在完全相同的 suffix
协议下重跑或明确标成不可直接比较。

## 8. GPU 0/1/2 调度原则

- Detector training/score export 才是主要 GPU 工作；按 outer-fold/seed 独立任务分配到 GPU 0/1/2。
- Stage2 只有约 3k 参数，单个训练进程不可能长期占满 3090；通过三张卡并行独立 cells 提高总吞吐，而不伪造“高利用率”。
- exact curve 构建、SHA 校验和 CSV/array I/O 是 CPU/磁盘阶段，GPU 利用率自然会下降；这不是模型空转。
- full curves 保持 ragged CPU；GPU 只接收 93D batch、三预算输出和 compact exact brackets。
- 每个真实任务记录 GPU、PID、seed、outer fold、输入/代码/release SHA、开始结束时间、峰值显存、失败类型；只允许预注册 infrastructure/implementation failure 重跑。

## 9. AAAI 主张边界

若所有门均通过，可主张：在固定三 benchmark outer-LODO、MSHNet-backbone 范围内，context-conditioned、risk-aligned、monotone No-Reject calibrator 相对 rolling context quantile 改善低 false-alarm budget satisfaction，同时保持 object Pd 非劣。只有预注册 DNANet portability 矩阵也完整通过相同方向性检查，才可把表述扩展为“在两个 detector backbone 上复现”；它不参与 S2_DGO，也不能挽救 MSHNet 主门失败。

不得主张：任意未知域保证、distribution-free/certified risk control、外部真实开放世界泛化，或只凭 S2_I0 工程 PASS 推断性能成功。
