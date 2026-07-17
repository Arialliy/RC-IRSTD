# RC-IRSTD AAAI-27：预注册实验矩阵与结果表模板

> 状态：`RESULT_FREE / FROZEN BEFORE REAL STAGE-2 EXECUTION`  
> 方法身份：Two-Stage No-Reject RC-IRSTD；Stage-1 backbone 为 MSHNet，Stage-2 主方法为 T8。  
> 数据范围：仅 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 的冻结 official-train-derived development 划分。  
> 结果规则：下表所有 `TBD` 只能由 S2_I0 PASS 后的哈希绑定运行填写；不得估算、补值或手工挑选。

> **执行配置优先级**：本矩阵只与
> `configs/aaai27_stage2_crossfit_v2.json` 共同生效，S2_I0/launch 必须外部
> SHA-256 绑定该 v2 文件。旧 `configs/aaai27_analysis_plan.json` 中的 C32/Q64、
> hidden=192 和旧 C6 定义已经 superseded，仅作历史记录，禁止执行或恢复。
> Result-free 范围：本模板不含新的 Stage-2 观测结果；此前含观测开发证据的 Stage-1 G1 只作为固定 SHA-256 前置条件被绑定。

## 1. 唯一主假设与最小开发门

主假设：在冻结三 benchmark outer-LODO 协议下，T8 相对 T4 提高主预算
`1e-5` 的 macro-domain budget satisfaction rate（BSR），同时保持 object Pd
非劣。

最小矩阵 S2_DGO：

| Outer domain | Base seeds | Mandatory windows/seed | T4 | T8 |
| --- | --- | ---: | ---: | ---: |
| NUAA-SIRST | 42, 123, 3407 | 1 | TBD | TBD |
| NUDT-SIRST | 42, 123, 3407 | 3 | TBD | TBD |
| IRSTD-1K | 42, 123, 3407 | 3 | TBD | TBD |

每个 cell 必须使用全部冻结窗口，每个窗口严格 C14/Q28；T4/T8 的 context、query、detector score、geometry、budget 和 bootstrap draws 必须 byte-identical。

T4 是不使用 source-domain 训练的 rolling order-statistic 基线，而 T8 使用 source
episodes 学习，二者不是 capacity-matched ablation。因此 T8−T4 只支持完整方法的
端到端价值；risk-aligned objective 和结构单调性的归因必须分别来自 T8−T7 和
T7−T6，不能由主比较替代。

| Primary endpoint | T4 | T8 | T8−T4 | Paired 95% CI | Frozen criterion | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Macro-domain BSR @ 1e-5 | TBD | TBD | TBD | TBD | point ≥ 0.05 且 lower > 0 | TBD |
| Macro-domain Pd @ 1e-5 | TBD | TBD | TBD | TBD | point ≥ −0.02 且 lower ≥ −0.02 | TBD |

S2_DGO 只有在 9 个 domain×seed cells 全部 finite、complete、estimable，且无身份错误、补值、T8 单调性违反或 official-test 访问时才可判定。任何缺失均为 HOLD；次要结果不能挽救主门。

## 2. 指标与统计量

对方法 `m`、域 `d`、种子 `s`、窗口 `w`、预算 `b`：

- pixel risk：`FP_pixels / total_native_query_pixels`；
- BSR：窗口级指示量 `1[pixel_risk <= b]`；
- LogExcess：`ln(max(pixel_risk / b, 1))`；
- Pd：先池化整数 `TP_objects / GT_objects`，禁止对窗口 Pd 直接平均；
- 三预算：`1e-4, 1e-5, 1e-6`，主预算固定 `1e-5`；
- threshold semantics：`prediction = probability > threshold`；
- object matching：native resolution、8-connected、maximum-cardinality one-to-one overlap。

除 primary BSR/LogExcess 外，每个方法、domain、seed、budget 都必须报告原始 pooled
pixel risk，并同时换算为 `FA pixels per 10^6 native pixels = pixel_risk * 10^6`；
这属于描述性透明度要求，不能用 LogExcess 或 BSR 隐去绝对 false-alarm 水平。
benchmark 仍按 iid images 处理，C14→Q28 仅是信息隔离顺序，不作 temporal claim。

主点估计的聚合顺序固定为：

- `BSR_macro = mean_domain(mean_seed(mean_window(BSR_window)))`；
- LogExcess 使用相同 window → seed → domain 等权顺序；
- Pd 在每个 domain×seed 内先对全部 mandatory windows 和 query images 池化整数
  `sum(TP_objects)/sum(GT_objects)`，再对 seed、domain 等权平均；
- 三个 domain 是固定等权 strata，不作为随机总体重采样。

主比较使用 10,000 次 paired hierarchical bootstrap：固定 domain strata，域内按
seed → window → query image 分层有放回抽样；T8/T4 共用 byte-identical indices；阈值、context 和 checkpoint 在 bootstrap 中固定。主比较不做多重校正；次要比较按预先声明 family 使用 Holm FWER 0.05。
95% 区间固定为双侧 percentile 2.5%/97.5%、Hyndman–Fan type 7。NUAA 只有
一个 mandatory window，其 window 层重采样退化；因此不得声称估计 NUAA 的
between-window variation。三域三种子的区间只描述冻结 benchmark/settings，不代表
任意未知域总体。

## 3. 主门通过后的完整阈值比较

| Method | Definition | No-Reject | Uses outer Q labels to decide | BSR @1e-5 | Pd @1e-5 | LogExcess @1e-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T0 | fixed 0.5 | yes | no | TBD | TBD | TBD |
| T1 | pooled two-source safe | yes | no | TBD | TBD | TBD |
| T2 | safer source threshold | yes | no | TBD | TBD | TBD |
| T3 | nearest-source safe | yes | no | TBD | TBD | TBD |
| T4 | 合并 C14 全部 float64 pixels；预算 `b`、总数 `N` 时取升序索引 `max(0,N-floor(bN)-1)`，并使用严格 `>` | yes | no | TBD | TBD | TBD |
| T5 | context EVT/GPD | yes | no | TBD | TBD | TBD |
| T6 | direct oracle-Huber MLP | yes | no | TBD | TBD | TBD |
| T7 | monotone oracle-Huber MLP | yes | no | TBD | TBD | TBD |
| **T8** | **risk-aligned monotone calibrator** | **yes** | **no** | **TBD** | **TBD** | **TBD** |
| T9 | post-label future oracle diagnostic | yes | **yes** | TBD | TBD | TBD |

T5 尾部不足必须报告 `SEALED_MISSING_INSUFFICIENT_TAIL_NO_FALLBACK`；T9 不进入 checkpoint selection、主门、bootstrap 主比较或 confirmatory 成功判据。

论文结果表还必须为 T0–T5 补齐来源、实现版本和公平性说明。本文没有检索或虚构
外部引用；在这些来源核验完成前，不得把该组内部基线称为“覆盖全部当前 SOTA”。

## 4. 机制消融

| ID | Structural monotonicity | Risk-aligned loss | Oracle auxiliary | Features | Purpose | BSR | Pd |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| C0 / T6 | no | no | yes | 0–92 | direct capacity control | TBD | TBD |
| C1 / T7 | yes | no | yes | 0–92 | isolate monotonic structure | TBD | TBD |
| C2 | yes | yes | no | 0–92 | remove oracle auxiliary | TBD | TBD |
| **C3 / T8** | **yes** | **yes** | **yes** | **0–92** | **full method** | **TBD** | **TBD** |
| C4 | yes | yes | yes | 0–38 (score only) | score sufficiency | TBD | TBD |
| C5 | yes | yes | yes | 0–78 (score + peak) | add local-peak evidence | TBD | TBD |
| C6 | yes | yes | yes | 0–86 (score + peak + gray) | remove source distance | TBD | TBD |

Primary mechanism contrasts are `T8−T7`（risk-aligned objective）and `T7−T6`（structural monotonicity）。参数量必须同时报告：T6=3,107；T7/T8=3,140。

## 5. Stage-1 消融与两阶段归因

| Stage-1 variant | Background-side gradient | Target-side gradient | Smooth worst-domain | Stage-1 IoU/nIoU | Stage-2 BSR | Stage-2 Pd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | none | none | no | TBD | TBD | TBD |
| D1 | trainable | stop-gradient | yes | TBD | TBD | TBD |
| D2 | stop-gradient | trainable | yes | TBD | TBD | TBD |
| **D3** | **trainable** | **trainable** | **yes** | **TBD** | **TBD** | **TBD** |

Stage-1 segmentation 和 Stage-2 low-FA operating point 必须分表报告；不得只用 Stage-2 阈值修正掩盖 detector 退化。




### 5.1 第二骨干可迁移性（主门通过后）

唯一预注册第二骨干为本机 BasicIRSTD/model/DNANet。必须重新使用相同三域 official-train-derived outer-LODO、seeds 42/123/3407、fixed-last detector checkpoint、C14/Q28、三预算、identity/causal gate 和 exact replay；禁止复用历史日志、按结果更换骨干或修改 T8/T4 公平性合同。

| Backbone | Method pair | NUAA-SIRST | NUDT-SIRST | IRSTD-1K | Macro BSR Δ @1e-5 | Macro Pd Δ @1e-5 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| DNANet | T8−T4 | TBD | TBD | TBD | TBD | TBD | TBD |

该矩阵是 detector-portability 次要证据，不参与 S2_DGO，不能挽救 MSHNet 主比较失败。未完成时，论文主张必须继续限定为 MSHNet-backbone setting；完整且同方向时才允许写“在两个 detector backbone 上复现”。

“同方向”固定定义为 9 个 DNANet domain×seed cells 全部
complete/finite、macro `delta BSR > 0` 且 macro `delta Pd >= -0.02`。满足该条件
只能写“方向一致”；只有 DNANet 也通过 S2_DGO 相同的 BSR point/CI 与 Pd
point/CI 四条件，才可写“在第二 backbone 上得到统计支持的复现”。无论哪种结果，
DNANet 均不得改变、推翻或补救 MSHNet 的 S2_DGO。

## 6. 稳健性、失败与效率

稳健性矩阵（全部为次要、预注册诊断；均使用冻结 checkpoints，不重训、不选优）：

| Test | Frozen perturbation | Endpoint | Result |
| --- | --- | --- | ---: |
| Context size | 从原 C14 分别取 ordered first-7 与 last-7，形成两个 C7 诊断；同一 Q28 不变；T4 用实际 `N` 的同一 order-statistic 公式，T8 checkpoint 不变 | 两个子集分别报告 threshold、BSR/Pd 与相对 C14 的差值 | TBD |
| Context order | C14 membership 不变，使用 original、reverse、按 `SHA256(canonical_id)` 升序三种顺序 | threshold bytes/hash、BSR/Pd；非完全不变即实现/契约失败 | TBD |
| Monotone score transform | 对 context/query 同时应用 `p'=sigmoid(logit(clip(p,1e-6,1-1e-6))/tau)`，`tau in {0.8,1.25}`；T4/T8 输入 byte-identical | BSR/Pd | TBD |
| Score noise | 对 context/query 同时应用 `p'=sigmoid(logit(clip(p,1e-6,1-1e-6))+e)`，`e~N(0,sigma^2)`，`sigma in {0.05,0.10}`；每 pixel 由冻结 SHA-256 domain/seed/image/pixel substream 生成，T4/T8 共用 bytes | BSR/Pd | TBD |
| Source distance removal | C3 vs C6 | BSR/Pd | TBD |
| Domain failure | per-domain T8−T4 | BSR/Pd | TBD |

Context-size 两个 C7 结果和所有扰动强度必须逐项报告，禁止只取较好者。所有 score
perturbations 在 threshold decision 前施加，并对该 cell 的 context 与 query 一致
施加；labels、checkpoints 与 bootstrap draws 不变。C7 的可变 `N` T4 只存在于
独立 robustness evaluator，不改变或覆盖 C14 主方法实现。

Score-noise 正态值不用库默认 RNG：对
`(domain, base_seed, image_sha256, row, column, sigma)` 形成 UTF-8 canonical tuple，
分别加 domain tags `rc-irstd-noise-u1-v1`/`rc-irstd-noise-u2-v1` 后做 SHA-256；
取各 digest 前 8 bytes 为 unsigned big-endian `h1,h2`，令
`u1=(h1+0.5)/2^64`、`u2=(h2+0.5)/2^64`、
`z=sqrt(-2*ln(u1))*cos(2*pi*u2)`、`e=sigma*z`。该定义与平台 RNG 状态无关。

次要推断的 Holm families 固定如下，主比较 T8−T4@1e-5 不在任何 family 内：

| Family | Directional BSR hypotheses @ frozen cells | Count |
| --- | --- | ---: |
| H1 threshold baselines | T8−T0, T8−T1, T8−T2, T8−T3, T8−T5 @1e-5；主比较 T8−T4 明确排除 | 5 |
| H2 mechanisms | T8−T7, T7−T6, T8−C2 @1e-5 | 3 |
| H3 features | T8−C4, T8−C5, T8−C6 @1e-5 | 3 |
| H4 Stage-1 contribution | D3−D0, D3−D1, D3−D2 的 Stage-2 BSR @1e-5 | 3 |
| H5 off-primary budgets | T8−T4 @1e-4、T8−T4 @1e-6 | 2 |

每个 hypothesis 先形成 9 个 domain×seed cell-level paired BSR deltas（cell 内使用
全部 mandatory windows），以 domain、seed 等权的 macro delta 为统计量；穷举
`2^9=512` 个 paired sign flips，单侧 exact `p = count(T_perm >= T_obs)/512`，ties
计入，再在各 family 内做 Holm step-down、FWER 0.05。任一 mandatory cell 缺失时，
该 hypothesis 固定 `p=1` 且不得主张支持。次要 Pd、Stage-1 IoU/nIoU、稳健性和
效率均完整报告 effect/interval，但不作额外显著性或非劣主张；若论文以后要作此类
主张，必须在相应运行前另行冻结 family，不能事后添加。

失败分析至少包含：T5 insufficient-tail、zero-GT window、不可估 cell、低预算无目标检出、identity/causal gate fail-closed，以及 NUAA `Misc_111` nearest-neighbor mask 对齐的审计计数。

| Efficiency item | Unit | D0 | D3 | T4 | T8 |
| --- | --- | ---: | ---: | ---: | ---: |
| Parameters | count | TBD | TBD | 0 | 3,140 |
| Stage-1 compute | FLOPs / image | TBD | TBD | n/a | n/a |
| Detector throughput | images/s | TBD | TBD | n/a | n/a |
| Context + decision latency | ms/window | n/a | n/a | TBD | TBD |
| Exact curve build | s/window CPU | n/a | n/a | shared | shared |
| Peak CPU memory / disk | MiB | n/a | n/a | TBD | TBD |
| Peak GPU memory | MiB | TBD | TBD | 0 | TBD |

## 7. 执行顺序与 GPU 0/1/2

1. S2_I0 必须先绑定 W06–W13 required-node 零跳过测试证据、三域 W12 pre-open bundle、B3 integration PASS 与 B4 scoped-release manifest/archive/environment/COMMIT 哈希；PASS 后再单独签发 launch authorization；
2. 只启动 S2_DGO 的 T4/T8 九格主矩阵；
3. GPU 0/1/2 各运行独立 outer-fold/seed cell，禁止一个约 3k 参数的 Stage-2 作业强行独占多卡；
4. exact curves、SHA 和 bootstrap 是 CPU/I/O 阶段，GPU 利用率下降属于正常流水线；
5. S2_DGO GO 后才展开第 3–6 节；HOLD 时只按预注册诊断定位，不查看 official test；
6. confirmatory 只在另一个一次性 opening authorization 后进行。

### 7.1 Official-test confirmatory one-look

三个 official-test datasets 必须在同一 pre-open contract 中冻结 identity；每个数据集
只使用 first-14 unlabeled context 和 all-remaining suffix query。每个 outer dataset
保留 development 已冻结的三个 seed-specific Stage-1 fixed-last checkpoints 与三个
seed-specific T8 source-validation-selected checkpoints；T4 使用同 seed detector
scores。禁止挑 seed、score ensemble、重选 checkpoint 或使用 context labels。

每个 dataset×seed 产生一个 suffix-window sufficient-count result；点估计先在 seed
内保持配对、再 seed 等权、dataset 等权。10,000 次 paired bootstrap 固定 dataset
strata，在 dataset 内重采样 seed slots 与 suffix query images，T8/T4 使用
byte-identical indices，区间仍为 type-7 percentile 95%。`S2_CONFIRM_SUPPORTIVE`
要求 3 datasets×3 seeds 全部 complete，并通过与 S2_DGO 相同的四个 BSR/Pd
point/CI 条件；否则标记 `NOT_SUPPORTIVE`。该状态只用于一次性 confirmatory 报告，
**不得改变、推翻或挽救 S2_DGO，不得触发模型、阈值、seed 或分析修改**。

该 first-14/suffix setting 是 transductive context-calibration protocol，不是完整
official-test segmentation protocol。confirmatory 只允许同 identity 的 paired
T8/T4（及事先实现的同协议基线）横向比较；suffix-only 数字不得与 published
full-test 数字同列排名或声称 SOTA，除非外部方法在相同 suffix identity 下重跑。

每个运行必须记录：outer fold、base/derived seed、GPU、PID、输入/代码/config/release SHA、开始/结束时间、peak memory、completion commit 和失败分类。只允许预注册 infrastructure/implementation failure 重跑，禁止按指标重跑。

B4 是共享脏工作树条件下的可复现 scoped-release 替代，不声称 Git clean/tag；任何真实 launch 必须同时绑定 S2_I0 report/COMMIT 与 B4 manifest/archive/COMMIT SHA-256，缺一即 HOLD。

## 8. 主张边界

只有完整证据通过后，论文才可限定主张：在三 benchmark、冻结 outer-LODO、MSHNet-backbone 范围内，context-conditioned risk-aligned monotone No-Reject calibrator 相对 rolling context quantile 改善低 false-alarm budget satisfaction，并满足 object-Pd 非劣。

不得声称 distribution-free/certified risk guarantee、任意未知域保证、外部开放世界泛化，或仅凭 S2_I0 工程 PASS 推断性能成功。
