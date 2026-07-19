# RC-IRSTD AAAI-27：RC5+ 预算条件化单调曲线候选设计

> 状态：`RESULT_FREE / ISOLATED_RC5PLUS_CANDIDATE / NOT_EXECUTION_AUTHORIZED`  
> 日期：2026-07-18  
> 当前实现：`model/budget_conditioned_residual_transport_calibrator.py`  
> 红线：本文不是正式实验授权，也不覆盖 RC5 的冻结配置；RC5+ 只有通过本文的准入门、
> 近期文献复核和新的 S2_I0 后，才能替换现有三点 RC5。

## 1. 当前判定

RC5 的三个固定输出已经实现了 endpoint-aware、anchor-aware 和预算间结构单调，但其
learned branch 本质上仍是三个坐标头，任意预算只在输出后做 log-budget 插值。这不足以
把“预算条件化 operating-point function”作为强方法贡献。RC5+ 候选把整个预算轴变成
同一个可训练、可查询的函数，并保留 RC5 的全部核心机制。

当前已经由 result-free 实现与测试证明：

- 九个精确有理预算结点、同预算 anchor-v2、等容量直接控制和结构单调候选已经实现；
- label→all-N exact-curve provider、九预算 loss、source OOF cyclic trainer、强制 feature
  mask、九预算 source validation 与 variable-Q sanity 已接入；
- T6+、T7+、T8+ 的参数量均为 `3339`，T7+ 与 T8+ 使用同一模型类；
- checkpoint-v8、producer→same-map anchor-v2→masked context→threshold 的 sealed inference
  以及 T6+/T7+/T8+ pre-label atomic learned decision set 已接入；
- 唯一 result-free 配置为 `configs/aaai27_stage2_crossfit_rc5plus_v1.json`，旧三预算
  checkpoint-v7 配置不能被解释为 RC5+；
- T8+-no-anchor 的等容量模型、无 anchor 训练/验证、checkpoint-v8 与独立
  pre-label seal 已实现；
- baseline-inclusive T0--T8 已组成 commit-last 原子 pre-label decision set，并经真实
  producer capability 重放、上游篡改与中断故障测试；依赖 target labels 的 T9 Oracle
  只能使用独立 post-label diagnostic schema，不得进入 pre-label 原子集；
- checkpoint-v8 专用 generation-v3 已隔离 deployment/resume state，绑定 optimizer、
  全部 RNG、training-view 与 source-only primary epoch rank；中断恢复的下一 dropout
  更新已与不中断执行位级一致；
- 固定 17 项 RC5+ S2_I0 审计已覆盖上述数学、训练、resume、validation、
  checkpoint、sealed inference、原子决策与故障路径；
- 未产生、读取或推断任何真实性能数字；创新性尚未达到“已证实 4/5”。

因此当前结论是：**RC5+ 的 result-free 实现设计可冻结；但“完整模型设计
成功、足以支撑 AAAI”仍需新颖性门与真实性能门的独立证据，在新的 hash-bound
launch 签发前正式实验继续 HOLD。**

## 2. 问题、缺口与核心洞察

### 2.1 问题

在 detector 与 feature extractor 冻结、目标域只有短无标签 context、不能读取未来
query、不能拒绝域或样本的条件下，输出一个面向原生像素虚警预算的阈值函数：

```text
(unlabeled target context x, exact rational budget b) -> threshold t(x,b)
```

该函数需要在极低虚警端正确表示精确上端点，预算收紧时阈值不能下降，并在部署时由
checkpoint、context 和解析 anchor 唯一决定。

### 2.2 现有 RC5 的局限

RC5 只学习 `1e-4 / 1e-5 / 1e-6` 三个 ordinates。即使三点结构单调，它仍可能被严格
评审解释为“三头 MLP + 后处理插值”，预算没有成为训练中的显式函数变量，三点之间的
形状也缺少密集监督。

### 2.3 RC5+ 洞察

把原生像素预算定义为精确有理数，在归一化 log-budget 轴上学习一条 context-conditioned
连续曲线；使用正区间质量参数化保证整个有效预算区间结构单调；在每一个请求预算上重新
计算同预算的解析 target-tail anchor，再进行 source-only learned correction。这样：

1. 训练与部署查询同一个函数，而不是训练三头后临时插值；
2. 单调性覆盖所有有效、数值可区分的有序预算查询，而不只覆盖三个主点；
3. 解析 target evidence 与 learned source correction 在同一个精确预算语义下组合；
4. EATC-v2 继续区分最大内部概率与精确 `p=1`；
5. 预算计数、训练曲线、checkpoint 和 sealed decision 可以绑定为同一可重放对象。

## 3. 冻结候选的数学定义

### 3.1 精确预算格点

从宽松到严格固定九个最低项有理数：

```text
1/10000, 1/17783, 1/31623, 1/56234, 1/100000,
1/177828, 1/316228, 1/562341, 1/1000000
```

三个 claim-bearing 预算是索引 `0, 4, 8`，因此仍精确等于 `1e-4, 1e-5, 1e-6`。
整数 false-positive budget 只能计算为：

```text
k(b,N) = (numerator(b) * N) // denominator(b)
```

不得用 binary64 预算乘像素数决定离散计数。浮点投影只用于连续函数的 log-budget
位置，不参与 false-positive count。

令结点预算为 `b_i`，曲线位置为：

```text
u_i = [log(b_i) - log(b_0)] / [log(b_8) - log(b_0)]
```

故 `u_0=0`、`u_8=1`，严格预算对应更大的 `u`。

### 3.2 Context encoder

沿用 RC5 的 93D label-blind context features、train-only float64 standardizer、hidden
width 32、GELU 与 dropout 0.1：

```text
h_theta(x) = Dropout(GELU(Wx+c))
```

不得读取 query features、query labels、official test 或调用方自由 feature vectors。

### 3.3 T6+：等容量非单调 residual-transport 控制

T6+ 使用 `93->32->10` transport head：一个 residual 原点、一个 anchor-slope 调制量和
八个有符号区间增量。九个 residual ordinates 为：

```text
G_0(x) = q_0(x)
G_i(x) = q_0(x) + sum_{j=1}^{i} Delta_j(x),  i=1,...,8
```

其中 `Delta_j` 不受符号约束，因此 T6+ 不声明结构单调。它与 T7+/T8+ 共享同一九结点
格点、同预算 anchor、transport 方程和 coordinate-Huber 监督，并具有完全相同的
`3339` 个可训练参数。

T6+ 的目的不是成为候选主模型，而是让 `T7+−T6+` 只回答结构单调参数化的作用。

### 3.4 T7+/T8+：正增量单调 residual function

T7+/T8+ 使用与 T6+ 完全相同的 `93->32->10` 容量，只把八个区间增量参数化为：

```text
Delta_j(x) = softplus(q_j(x)) + epsilon,  epsilon=1e-6
G_0(x) = q_0(x)
G_i(x) = G_0(x) + sum_{j=1}^{i} Delta_j(x)
```

实数算术下 `G_{i+1}(x)>G_i(x)`；有限精度累加允许相邻值因舍入相等，但禁止下降。
T7+ 和 T8+ 必须使用完全相同的模型类、参数量和前向
计算，只允许训练目标不同；T6+ 也保持 `3339` 参数，避免把容量差异伪装成单调机制收益。

### 3.5 连续预算查询与精确结点回放

对范围内的最低项有理预算 `b`，先按精确分数判断合法范围与顺序，再计算 `u(b)`；在
相邻结点之间对 residual ordinates 做分段线性插值：

```text
G_theta(x,b) = lerp(G_i(x), G_{i+1}(x),
                    [u(b)-u_i]/[u_{i+1}-u_i])
```

如果请求恰好等于九个冻结有理结点之一，必须直接 gather 对应 ordinate，保证位级回放，
不得通过权重为 0/1 的浮点算术近似回放。

接口使用对齐的 int64 numerator/denominator，不接受 float budget。两个不同有理预算若在
float64 log 坐标中坍缩为同一位置，则它们不是该连续曲线接口的有效查询；这是输入合同
错误，不是对 context、domain 或样本的 decision reject，也不能触发 fallback。

### 3.6 同预算解析 anchor

对每个 context `x` 和每个请求预算 `b=a/d`，必须从同一组 14 张原生分辨率 float64
context score maps 重新计算：

```text
k = floor(a*N_c/d)
r = N_c-k-1
t_anchor(x,b) = ascending_order_statistic(r)
a_x(b) = EATC_encode(t_anchor(x,b))
```

RC5+ 禁止把三个旧 anchor 做插值后冒充请求预算的解析 anchor。九结点训练、任意预算
推理都必须携带同预算、同 context identity 的 anchor capability。现有三预算 anchor-v1
不能直接满足本合同，必须新增独立 schema 的 generalized anchor-v2。

### 3.7 Anchor-aware residual transport 与 EATC-v2

令 `alpha=sigmoid(g)` 为 checkpoint 内的全局可训练标量，初始值 0.1；把同预算
anchor coordinate `a_x(b)` 先映射到冻结 raw EATC 有界区间的 logit latent：

```text
q_x(b) = [a_x(b)-r_min] / [r_max-r_min]
z_anchor(x,b) = logit(q_x(b))
beta(x) = exp(alpha * tanh(q_beta(x)))
z_theta(x,b) = beta(x)*z_anchor(x,b) + alpha*G_theta(x,b)
r_theta(x,b) = r_min + [r_max-r_min]*sigmoid(z_theta(x,b))
s_theta(x,b) = hard_canonicalize_EATC_v2(r_theta(x,b))
t_theta(x,b) = decode_EATC_v2(s_theta(x,b))
```

`alpha` 对所有样本和预算共享，调用方不能覆盖；当 `alpha→0` 时 transport 连续回到
解析 anchor identity。由于同预算 anchor 非递减、`beta>0`、T7+/T8+ residual function
在实数算术下严格递增且 `alpha>0`，transport latent 与 raw curve 数学上严格递增；
在 float64 饱和与舍入下，所有中间量、canonical coordinates 和 decoded thresholds 的
可执行保证是非递减，精确 endpoint 只能形成后缀。审计不得把机器精度相等夸大为严格
不等式，也不得容许任何下降。

该性质是结构性质，不是未来 query false-alarm 的 certified guarantee。

## 4. 训练目标与机制可归因性

九个结点同时进入训练；三个主预算仍只用于冻结主评估，不因增加训练结点而改变。

| 方法 | 模型 | 训练目标 | 参数量 | 唯一机制问题 |
| --- | --- | --- | ---: | --- |
| T6+ | signed-increment residual transport | 九结点有效 oracle EATC coordinate Huber | 3339 | 等容量非单调控制 |
| T7+ | positive-increment residual transport | 与 T6+ 相同 Huber | 3339 | `T7+−T6+`：结构单调 |
| T8+ | 与 T7+ byte-identical | 九结点 risk-aligned surrogate + 同一 Huber | 3339 | `T8+−T7+`：risk alignment |
| T8+-no-anchor | positive-increment learned-only curve | 与 T8+ 相同 loss；API 与训练批次均不接受 target anchor | 3339 | `T8+−no-anchor`：解析 anchor |

T8+-no-anchor 保留相同 encoder、十输出 head、正 residual 增量和全局 sigmoid 标量，去掉
target anchor 后使用：

```text
z_no_anchor(x,b) = alpha * beta(x) * G_theta(x,b)
```

它不接收 `anchor_coordinates`，checkpoint 明确锁定 `anchor_overlay_required=false`；该对比
是等参数量的 target analytic-anchor ablation，不得在实现中偷偷换成三点 anchor、自由
threshold 或 target-labelled source prior。

source OOF cyclic episode 的 label→all-N exact-curve bank 保持不变，但 provider 必须为
九个预测分别提取相邻 event brackets。device 上每 episode 最多是 18 行 bracket union，
不得回退到 padded full curve，也不得继续沿用旧的“最多 6 行”声明。

T8+ 中 `exact` 仍只修饰 verified uncapped event set、预算整数语义和相邻 bracket；
piecewise-linear differentiable risk surrogate 不是 certified risk，也不是离散 empirical
risk 的精确连续化。

## 5. 对既有创新的保留映射

| 既有核心创新 | RC5+ 中的保留方式 |
| --- | --- |
| Two-Stage / No-Reject | Stage 1 与 Stage 2 身份不变；模型无 reject/fallback head |
| Stage-1 tail-separation | D0--D3 与 Stage1×Stage2 独立归因，不由曲线模块替代 |
| 短目标域无标签 context adaptation | 93D context-only 输入与 C14 协议不变 |
| EATC-v2 | raw→canonical→decode 全链保留 |
| 原生像素 FA budget | 所有计数继续使用精确有理整数公式 |
| 解析 target-tail anchor | 扩展为每个请求预算的 anchor-v2，而不是删除或插值 |
| source-only learned correction | learned function 只由 source OOF labels 训练 |
| risk-aligned exact-event supervision | 从 3 个预算扩展到 9 个预算，exactness 边界不夸大 |
| 预算—阈值结构单调 | 从三点单调增强为有效区间内的函数单调 |
| variable-Q/all-once | validation/outer evaluation geometry 不变 |
| source OOF cyclic | C14/Q28 cyclic training geometry与 equal-domain sampler 不变 |
| label→all-N exact-curve bank | 继续按 image bank live composition，不物化每 episode full curve |
| sealed causal chain | 升级 checkpoint/context/anchor/curve/threshold 绑定，不开放自由阈值 |

任何接入方案若通过减少预算、删除 anchor、允许 query feature、增加 fallback 或改为自由
threshold 来简化实现，均自动判定为 RC5+ 设计失败。

## 6. 新颖性边界

候选的安全 novelty delta 不是“首次阈值校准”“首次 TTA”或“首次单调函数”，而是以下
IRSTD 特定组合：

> frozen detector 下，从短无标签目标 context 构造同预算解析 tail anchor，再用
> source OOF exact-event evidence 学习一个 exact-rational、EATC endpoint-aware、
> native-pixel-budget-conditioned 的连续单调 operating-point correction function，
> 并以 no-reject sealed decision chain 部署。

截至现有 2026-07-17 检索，尚未发现完整同构方法，但 threshold calibration、quantile/
tail estimation、monotonic prediction、risk control 和 test-time adaptation 均已有先例。
因此当前不得写“novelty 已达到 4/5”。RC5+ 只有经过更新检索、直接先例审查、与最接近
方法逐项差异表和严格 idea review 后 novelty `>=4/5` 且无 fatal prior-art，才可进入
最终冻结。

## 7. RC5+ 准入与冻结门

以下门按顺序执行，任何一门失败均保持候选状态：

1. **C0 数学核心**：九结点、精确请求、单调结构、EATC、endpoint suffix、梯度、错误
   输入和能力合同全部有合成测试并通过 focused replay。
2. **C1 anchor-v2**：任意有效预算 anchor、整数 rank、score-map digest、context identity、
   exact replay 和 fault injection 完成；禁止三点 anchor 插值。
3. **C2 source curve/loss**：九结点 oracle coordinates、每预测相邻 event、最多 18 行
   device bracket、T6+/T7+/T8+ method routing 和 finite backward 完成。
4. **C3 checkpoint/training**：使用新 schema，不把 RC5 checkpoint-v7 静默解释为 RC5+；
   generation-v3 分离 checkpoint-v8 deployment state 与 resume state，绑定 optimizer、全部
   RNG、training-view、feature masks、参数量与 state digest，并证明中断恢复的下一步
   与不中断执行位级一致。
5. **C4 sealed inference**：唯一链为 checkpoint→verified context→same-budget anchor→
   budget-conditioned function→EATC threshold→atomic decision；调用方不能注入曲线、anchor、
   alpha 或 threshold。
6. **C5 production E2E**：T0--T8 在任何 label resolver 之前完成原子 decision，T9 保持
   post-label 隔离；variable-Q identity invariance、故障时 label resolver 零调用、
   interruption/resume、compileall 和 diff check 通过。
7. **C6 novelty**：更新近期检索，严格 novelty >=4/5，无未处理直接先例。
8. **C7 S2_I0-RC5+**：原固定 17 类审计按 RC5+ 新 schema 全部 17/17 PASS，报告仍为
   result-free 且 `execution_authorized=false`；之后另行签发 hash-bound launch。

原 RC5 的测试通过不能替代 RC5+ 审计；任一局部模块测试也不能替代 C1--C7。

## 8. 性能成功门不变

RC5+ 只有通过工程冻结后才能运行预注册真实性能实验，且只有全部性能证据成立才叫
“完整模型设计成功、足以支撑 AAAI”：

- MSHNet 三 outer domains×三 seeds 的 `T8+−T4` macro-domain BSR@1e-5 点增益
  `>=0.05`，配对 95% CI lower `>0`；
- macro-domain Pd 差值 `>=-0.02`，配对 95% CI lower `>=-0.02`；
- `T8+−T7+`、`T7+−T6+`、`T8+−T8+-no-anchor`、Stage-1 D3 关键对比和所主张
  feature contributions 均为正向 macro BSR，通过预注册 family 内 Holm 校正，且 Pd
  非劣于 `-0.02`；
- 冻结近期强 IRSTD、operating-point calibration 和合理 TTA baselines，同协议比较；
- DNANet 与至少第四独立域方向一致，稳健性、失败和效率证据完整；
- 最终冻结后，未参与设计选择的 independent confirmatory one-look 得到支持。

单元测试、S2_I0、单域改善、单 seed 改善、次要预算或更好看的次要指标均不能单独满足
性能成功门。

## 9. 失败处理

- C0--C7 失败：继续 result-free 修复或增强并重新审计，不启动真实性能实验；
- 主性能或机制消融失败：透明诊断，在仍有合法未开封 development evidence 时重新设计、
  重新预注册和重新冻结；
- 无独立证据可继续验证、直接先例消除安全 novelty delta，或 confirmatory one-look 失败：
  诚实 NO-GO 或收缩主张；
- 禁止删除失败创新后仍保留原贡献叙述，禁止移动主门、挑 seed/domain、泄漏 target labels
  或伪造结果。
