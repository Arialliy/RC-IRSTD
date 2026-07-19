# RC-IRSTD AAAI-27：RC5 完整模型设计与冻结门

> 状态：**result-free RC5 candidate；尚未获得 S2_I0 执行授权**  
> 日期：2026-07-17  
> 仓库：`/home/md0/ly/RC-IRSTD`  
> 红线：S2_I0 PASS 前不得启动真实 Stage-2 性能实验；任何真实结果不得反向修改本候选稿中的主门、预算、方法定义或统计规则。

## 0. 权威性与当前结论

本文与 `configs/aaai27_stage2_crossfit_rc5_v3.json` 描述 RC5 候选设计。旧文件
`RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md` 及
`configs/aaai27_stage2_crossfit_v2.json` 只保留 RC4 历史，不得用于 RC5 训练、恢复、
部署或结果解释。

RC5 当前已完成模型数学核心、端点表示、解析锚点、部署 checkpoint、坐标损失、
因果推理 transcript 和 crossed bootstrap 的合成实现；但 schema-v6 数据链、完整可恢复
训练发布合同、T0--T8 原子 decision set、detector RUN_COMPLETE-v2 与 S2_I0 总审计尚未
全部闭合。因此当前准确结论是：**模型核心已设计，完整模型尚未冻结，不得开始正式实验。**

三个结论必须分开：

1. `S2_I0 PASS`：设计、代码和因果合同合成验收通过，可以启动预注册实验；
2. `S2_DGO GO`：真实三域三种子结果通过预注册性能门；
3. `S2_EC PASS`：机制、基线、第二骨干、第四域、稳健性、失败与效率证据完整。

`S2_I0 PASS` 不等于性能成功，`S2_DGO GO` 也不等于 AAAI 接收保证。

## 1. 科学问题与安全的新颖性边界

科学问题是：在 detector 和 feature extractor 均冻结、目标域仅提供一个短无标签
context、不能拒绝样本且不能读取未来 query 标签时，能否一次性产生满足三个原生像素
false-alarm 预算的阈值曲线，并在未知域 query 上优于仅用 context 尾部统计的解析阈值？

RC5 的可辩护方法身份为：

> short unlabeled target context -> frozen IRSTD detector -> explicit native-pixel
> false-alarm-budget curve, structurally monotone across budgets, no reject, with an
> analytic target-tail anchor plus a source-trained correction.

截至 2026-07-17 的标准检索筛选了 25 项工作，保留 15 项高相关证据；未发现与上述完整
部署组合直接同构的工作，但无标签 threshold calibration、test-time adaptation、
Neyman--Pearson/risk control 和 monotone threshold prediction 均已有先例。最接近的机制
风险是 CVPR 2024 OpenGCN。因此论文不得使用以下表述：

- first unlabeled/test-time threshold calibration；
- certified、distribution-free 或 guaranteed false-alarm control；
- Neyman--Pearson optimal；
- source-free、zero-shot 或 domain generalization；
- calibrated probabilities；
- 对任意未知域的总体泛化结论。

允许的表述是：IRSTD 特定的经验 operating-point calibration；在冻结三域/种子/骨干及
新增独立域上得到的经验结果；精确、可重放的预算与因果执行合同。

## 2. 系统身份

RC-IRSTD 是 Two-Stage / No-Reject 系统。

### 2.1 Stage 1

主 detector 为 MSHNet；D0--D3 的 tail-separation 设计保持既有冻结定义。Stage 1 的
贡献必须由 D0--D3 与 Stage1×Stage2 交互实验独立支持，不得用 Stage-2 主门代替。

### 2.2 Stage 2 validation/outer evaluation 的 mandatory 因果单位

对每个冻结 source-validation 或 outer-development role 的 `N` 个有序 records：

```text
W = floor(N / (14 + 28))
total_query = N - 14W
Q_i = floor(total_query / W) 或 ceil(total_query / W)
```

前 `total_query mod W` 个窗口取得较大 `Q_i`。每个窗口均为 14 个 context 后接动态
`Q_i >= 28` 个 query；全部 `N` 个记录按原顺序、不重叠、恰好消费一次，不再丢弃
suffix。该顺序只定义信息访问协议，不声称 benchmark 具有自然时间因果性。

这套 dynamic-Q/all-once geometry 只用于 mandatory validation 与 outer evaluation，
不得拿它的少量窗口数当作 Stage-2 training episode 数。

### 2.3 Source OOF training-only cyclic episodes

每个经过验证的 source OOF training role 使用独立的
`rc-irstd.stage2-source-cyclic-training-geometry.v1`。若有 `N>=42` 个有序 records，
对每个 cyclic start `s in [0,N)` 产生一个 C14/Q28 episode，局部 offset 的索引固定为
`(s+offset) % N`。因此每个 role 恰有 `N` 个 raw training episodes；任一 episode
内 context/query 不相交，完整 cyclic 集合中每条 record 恰作为 context 14 次、query
28 次。该几何只允许 `oof_holdout_stage2_fit`，source validation 与 outer target
明确禁止使用，后两者仍执行上一节 mandatory variable-Q/all-once geometry。

纯 manifest 核算得到的 pooled cyclic episode count 按 source domain 分别为：
NUAA-SIRST `85+85=170`、NUDT-SIRST `255+254=509`、IRSTD-1K
`319+319=638`。对应 outer fold 两个 source 的 raw total 分别为：
leave-NUAA `509+638=1147`、leave-NUDT `170+638=808`、leave-IRSTD-1K
`170+509=679`。这些都是 training-only cyclic raw episodes，不是 mandatory
evaluation windows。

每个 outer fold 的两个 source domain 使用 equal-source-domain、without-replacement、
rotating-subset sampler；每 epoch 每域抽取两域 pooled count 的较小值，所以
leave-NUAA epoch size 为 `2*min(509,638)=1018`，leave-NUDT 为
`2*min(170,638)=340`，leave-IRSTD-1K 为 `2*min(170,509)=340`。
sampler live schema 为
`rc-irstd.stage2-domain-balanced-cyclic-epoch-sampler.v1`，算法固定为
`sha256_fixed_permutation_rotating_slice_domain_pairs_v1`。输出按两域 episode pair
排列，所以冻结 batch size 16 的每个 batch 严格为 8/8，epoch 尾也保持两域平衡；
域内每 epoch 无放回，并用固定 SHA-256 permutation 的 rotating slice 跨 epoch 覆盖
较大域。seed 只能来自 verified training manifest，禁止 Python builtin `hash` 与手工
覆盖。trainer 必须逐项消费 sampler 的 `ordered_selection`；DataLoader
`shuffle=False` 且禁止任何二次随机 shuffle，`drop_last=False`。否则 8/8 batch
平衡与 interruption/resume 的 byte replay 均失效。sampler 已有 result-free 实现与
合成测试，但仍处于
`LIVE_IMPLEMENTED_PENDING_TRAINER_INTEGRATION_S2_I0`；trainer 接入、manifest SHA
绑定与总审计未通过前不能启动正式训练。

### 2.4 Context features and standardization

每个 context 产生固定 93D 无标签特征：score 39D、local peak 40D、gray 8D、source
distance 6D。标准化器仅由 Stage-2 training contexts 以 float64 拟合，scale floor 为
`1e-8`；模型输入在标准化完成后转为 float32。Validation 与 outer target 对标准化器
拟合的访问次数必须为 0。

## 3. 极低虚警阈值表示 EATC-v2

普通 clipped logit 不能区分极接近 1 的内部概率与精确 endpoint `p=1`。RC5 使用
endpoint-aware tail coordinate：

```text
p <= 0.5:  s = p
0.5 < p < 1: s = 0.5 - log(2(1-p))
p = 1:      s = s_endpoint
```

其中最大 binary64 内部坐标为 `0.5 + 52 ln 2`，精确上端点坐标再增加 1。模型先在有界
raw coordinate 上输出，再通过 hard-forward/identity-backward canonicalization 映射到
内部区间或离散 endpoint，最后解码为概率阈值。

只主张以下性质：精确 `p=1` 与所有 binary64 内部概率可区分；canonicalization
幂等；训练和部署使用同一 live contract。不得声称对任意实数存在全局双射，也不得把
有限样本的 ULP 测试写成数学证明。

所有阈值使用 `prediction = probability > threshold`。预算按从宽松到严格排列：

```text
[(1, 10_000), (1, 100_000), (1, 1_000_000)]
```

任何可行 false-positive count 均只允许整数公式
`(numerator * pixel_count) // denominator`；禁止由 `float_budget * N` 或
`Fraction.from_float` 决定离散计数。

## 4. T4 解析 target-tail anchor

合并 14 张 context 的全部原生分辨率 float64 probability pixels。对预算 `a/b`、总像素
数 `N_c`：

```text
k = floor(a * N_c / b)
r = N_c - k - 1
t_anchor = ascending_order_statistic(r)
```

在严格 `>` 语义下，context 中超过该阈值的像素数不超过 `k`；ties 可以使实际数量更
少。锚点 artifact 必须绑定 14 张 score map 的 shape、内容 SHA、context identity、精确
有理预算、rank、阈值概率和 EATC coordinate，且 verifier 从 score maps 完整重放。

这只是解析经验锚点，不是对未来 query false alarms 的有限样本保证。

## 5. T6/T7/T8 模型

三者共享 93D 输入、hidden width 32、GELU、dropout 0.1、三个预算和同一个 T4 anchor。

令 `a in R^3` 为非递减 anchor coordinates，`l_theta(x) in R^3` 为 learned raw
coordinates，`alpha = sigmoid(g)` 为一个全局可训练标量，初值 0.1：

```text
r = (1 - alpha) * a + alpha * l_theta(x)
s = hard_canonicalize(r)
t = EATC_decode(s)
```

混合发生在 canonicalization 之前。`alpha` 对所有样本和预算共享，部署时必须从
checkpoint state 重放，禁止调用方覆盖。

| 方法 | learned branch | 目标 | 结构性质 | 参数量 |
|---|---|---|---|---:|
| T6 | `93->32->3` direct | coordinate Huber | 不要求单调 | 3108 |
| T7 | `93->32->4` positive interval masses | coordinate Huber | raw 严格递增；canonical/threshold 非递减 | 3141 |
| T8 | 与 T7 相同 | verified global exact-event curve 上的 piecewise-linear differentiable risk surrogate + oracle-coordinate Huber | 同 T7 | 3141 |

由于 anchor 非递减、`0<alpha<1` 且 T7/T8 learned raw 严格递增，T7/T8 最终 raw
coordinates 严格递增；canonicalization 后允许多个预算共同落到精确 endpoint，因此只
要求 canonical coordinates 与 decoded thresholds 非递减、endpoint 为 suffix。

### 5.1 训练损失

loss 是 method-routed，而不是全局统一的 coordinate Huber。T6/T7 只使用有效 oracle
EATC coordinates 上的 Huber loss，delta=1。T8 则在 verified、uncapped global
exact-event curve 上，对每个 live prediction 的相邻左右 event 使用 piecewise-linear
differentiable risk surrogate，并同时加入 oracle-coordinate Huber 辅助项。完整 ragged
event curve 保留在 CPU，只将三个 live prediction 各自左右 event 的并集（每 episode
最多 6 行）搬到训练设备。这里 exact 只修饰完整 event set 与经验证的相邻 bracket；
它不修饰 risk，也不声称离散 empirical risk 在事件之间本身线性。因此不得写
“exact risk”或“精确风险优化”。

cyclic training 的 query 由可能跨 wrap-around 的 28 个 records 组成，禁止为每个
cyclic episode 预先物化 full aggregate curve。冻结方向是保留 per-image exact-curve
bank，训练时按 episode identity live composition，再提取相邻 verified brackets；
`per_image_exact_curve_bank=true`、`aggregate_curve_materialization=false`。live
provider 尚待 S2_I0 接口绑定，未完成前不得启动 T8 正式训练。

冻结权重：`lambda_violation=4`、`lambda_utility=1`、`lambda_oracle=0.1`、
`lambda_smoothness=0.01`、`lambda_coverage=0`、`risk_epsilon=1e-12`。coverage 项因使用
verified global exact-event curve 而移除；不得恢复 RC4 的 coverage=4。

## 6. 机制归因与预注册消融

以下比较各自只支持一个问题：

- `T8 - T4`：完整 learned correction 相对解析 anchor-only 的端到端价值；
- `T7 - T6`：结构单调性，在相同 anchor 与 oracle-coordinate 目标下的作用；
- `T8 - T7`：risk-aligned objective，在相同结构和 anchor 下的作用；
- `T8 - T8-no-anchor`：解析 anchor 对 full risk-aligned model 的作用；
- feature ablations：score-only、score+peak、去 source-distance；
- D0--D3 与 Stage1×Stage2：Stage-1 tail separation 及两阶段交互。

`T8-no-anchor` 使用同一 93->32->4 monotone learned branch 和同一 T8 loss，直接
canonicalize learned raw coordinates，不读取 anchor，参数量 3140；这一 1 参数差异必须
透明报告。它只能作为预注册消融，不能通过运行时 flag 改写 claim-bearing T8 checkpoint。

主门通过不能替代任一机制消融；对应差异不支持时，必须删除或收缩该机制主张。

## 7. Checkpoint、推理与封存链

部署 checkpoint-v7 必须包含并验证：方法/类/参数量、sorted tensor name/dtype/shape/
CPU-byte state digest、93D feature schema、float64 mean/scale 与 floor、training-contract
SHA、live EATC contract、state-derived alpha，以及 CPU inference smoke test。

完整训练 generation/run-v2 还必须另外保存 optimizer、epoch/rank/history、Python/
NumPy/Torch/CUDA/DataLoader RNG；部署 checkpoint 本身不得被误写成可恢复训练状态。

Outer target 的唯一合法因果链为：

```text
verified detector RUN_COMPLETE
 -> score-manifest
 -> variable-Q window
 -> label-blind context package + commit
 -> replayed T4 tail anchor
 -> verified calibrator checkpoint-v7
 -> 93D standardize -> model inference
 -> canonical threshold transcript
 -> atomic T0--T8 decision set
 -> only then query mask resolver
 -> label attachment + exact curve + episode-v6
```

推理 core 只能接受 verifier-issued checkpoint、schema-v6
`VerifiedStage2ContextV2` 经 `VerifiedContextInferenceMaterialV2` 投影得到的 query-free
material，以及与同一 context identity 绑定的 anchor capability；不得接受自由 feature
vector、自由 threshold、reject、fallback 或 query 输入。Verifier 必须重新执行
checkpoint loading、context-v2 canonical replay、query-free material assertion、
standardization、CPU forward、anchor mix、canonicalization 与 decode，并与 transcript
byte-for-byte 比较。动态 Q 不进入 material；相同 context/features/anchor 在 Q29 与 Q39
合成重放中必须产生 byte-identical decision。

hard-Q28 context-v1 迁移 adapter 已从 RC5 inference core 移除。但裸
`VerifiedStage2ContextV2` 只证明 canonical context payload，不单独证明 score maps 到
93D features 的语义 provenance。最终公开 API 必须再收紧为 label-blind RC5 producer
bundle capability，由它绑定 variable-Q window、score-v4、source reference、
statistics config、context-v2、T4 anchor 与 commit；该 producer 接口及原子 T0--T8
decision set 未闭合前，本节仍未 PASS。

## 8. S2_I0 设计冻结门

以下全部 PASS 后，才可写“完整模型设计成功”并停止改 claim-bearing 模型：

1. RC5 config、设计稿、模型类、参数量、预算、EATC 与 loss 字节身份一致；
2. T6/T7/T8 forward/backward 有限且非零；T6/T7 只路由 Huber，T8 路由 verified
   global exact-event curve 上的 piecewise-linear differentiable risk surrogate 与
   oracle-coordinate Huber，不得称为 exact risk；
3. T7/T8 在随机、边界和 endpoint 输入上满足结构性质，T6 不被错误排序；
4. source OOF cyclic C14/Q28 training geometry、equal-domain sampler 与 mandatory
   variable-Q/all-once validation/outer geometry 各自按 live schema 重放，角色不可互换；
5. 四身份边界无重叠，train/validation 含两个 source 域且 outer target 完全缺席；
6. 93D train-only float64 standardizer、scale floor 和 feature masks 可重放；
7. T4 anchor 从 14 张 maps 重放，精确有理预算不经过 float count；
8. checkpoint-v7 deployment state 与 generation/run-v2 resume state 各自完整，原子故障注入通过；
9. CPU interruption/resume 与 uninterrupted training 在允许的确定性范围内等价；
10. checkpoint->context->anchor->threshold transcript 被完整重算，调用方无法注入 thresholds；
11. T0--T8 原子且先于 label resolver；任一绑定缺失/篡改时 mask resolver 调用数为 0；
12. exact curve 与 brute-force 小图一致，million-event 路径不产生 padded GPU full curve；
13. detector launch authorization、immutable generations、RUN_COMPLETE-v2 和 score export 因果闭合；
14. crossed bootstrap 仅验证一次输入，seed 与 window/query 真正交叉，T8/T4 共用 byte-identical draws；
15. learned-only anchor ablation、近期强基线、第二骨干、第四域与失败/效率表均已预注册为空表；
16. 固定 synthetic/fault-injection suite 全通过，compileall 与 `git diff --check` 通过；
17. 审计报告不含真实性能数字，`execution_authorized=false`，另行发布 scoped launch authorization。

任一项失败即保持候选状态；不得用“主要代码已完成”替代 S2_I0。

## 9. S2_DGO 主性能门

S2_I0 PASS 后只先运行 3 outer domains × 3 base seeds 的 T8 与 T4。每个 cell 使用全部
mandatory variable-Q windows。主预算为 `1e-5`，主比较为 T8−T4 macro-domain BSR；
Pd 为联合非劣约束。

Crossed paired bootstrap 固定三个 domain strata；重采样三个 seed slots，并在每个 domain
replicate 中独立抽一次 window/query hierarchy；该 window/query draw 对三个 seed slots 与
T8/T4 共享，draw preimage 不包含 method 或 selected seed。10,000 次，95% 双侧 percentile
CI，Hyndman--Fan type 7。

只有下列全部成立才 GO：

```text
delta macro BSR point >= 0.05
delta macro BSR 95% CI lower > 0
delta macro Pd point >= -0.02
delta macro Pd 95% CI lower >= -0.02
```

每个 primary cell 还必须 finite、complete、identity-valid 且满足
`1e-5 * total_background_pixels >= 20` 的可估性门。背景像素只用于可估性判断；主 false
alarm denominator 始终是全部原生 query pixels。缺失或不可估不得插补，次要指标不得救门。

## 10. S2_EC 完整证据

主门 GO 后才补齐：T0--T9、T6/T7/T8 机制、anchor/feature 消融、D0--D3 与交互、
MSHNet 之外第二骨干、至少第四独立域、近期强 IRSTD baselines、合理的 TTA comparator、
稳健性、失败案例、计算/存储/延迟，以及论文主张--表格--代码--checkpoint 一致性审计。

如果真实证据不支持主张，只能执行透明诊断、result-free 重新设计并重新冻结，或 NO-GO；
不得移动主门、选择有利 seed/domain、用次要指标替代主 endpoint，或伪造结果。
