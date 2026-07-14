# RC-IRSTD：AAAI-27 两阶段、无 Reject 定稿方案

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

**版本：0.4.0-aaai27-two-stage-no-reject**
**方法边界：经验性 budget-aware operating-point adaptation；不提供 distribution-free、certified 或 guaranteed risk control。**

---

## 1. 一句话问题定义

给定多个有标签源域，在冻结的红外小目标检测器部署到未知目标流时，仅利用目标流前缀的无标签图像与连续预测分数，预测满足用户像素虚警预算的阈值，并将该阈值固定用于未来、不重叠的 query 数据。

这不是“再设计一个分割 backbone”，而是一个在域偏移下学习部署工作点的问题。

---

## 2. 最终方法身份

```text
Stage 1
MSHNet
  └─ cross-domain target–background domain-tail separation
          ↓ freeze detector

Stage 2
unseen-domain unlabeled prefix S
  └─ compact, label-free statistics z(S)
          ↓
monotone inverse pixel-risk curve calibrator
          ↓
threshold τ(B) for a requested budget B
          ↓ freeze threshold
future, disjoint query Q
```

主文中只有两个提出的模块：

1. **Domain Tail Separation**：提高极低虚警区的目标—背景排序间隔；
2. **Monotone Inverse Pixel-Risk Calibration**：从无标签前缀预测完整预算—阈值曲线。

MSHNet 是基础检测器，不是新贡献；Reject、CRC、双风险曲线和复杂 TTA 不属于主方法。

---

## 3. 主风险与评测边界

### 3.1 主约束

主约束统一为原始图像画布上的 pixel false-alarm rate：

\[
F_a^{\mathrm{pix}}(\tau)
=
\frac{\sum_{i,p}(1-y_{i,p})\mathbf 1[\ell_{i,p}>\eta]}
{\sum_i |\Omega_i|},
\qquad \tau=\sigma(\eta).
\]

原因：在固定 score map 上，阈值升高时像素预测集合单调缩小，适合构造单调逆风险曲线。

### 3.2 固定局部候选

固定局部峰值候选只用于：

- detector 的背景上尾；
- 无标签 support 统计；
- 可选的固定候选风险诊断。

候选集合由连续 logit map 上的统一规则一次确定；阈值只过滤候选，不重新连通分量。

### 3.3 Connected-component FA

传统 connected-component FA/MP 仅用于兼容已有 IRSTD 文献。它不进入主校准损失和单调性论证，因为阈值升高可能导致连通域分裂，组件数不保证单调。

### 3.4 删除 Reject

主方法总是返回预算范围内的阈值，不包含 Reject head。若预算位于训练网格之外，代码 fail closed 并禁止外推，而不是返回 Reject。

---

## 4. Stage 1：Domain Tail Separation

### 4.1 总损失

\[
\mathcal L_{\mathrm{det}}
=
\mathcal L_{\mathrm{seg}}
+
\lambda_{\mathrm{sep}}
\operatorname{SmoothMax}_d
\left[m+R_d^- -R_d^+\right]_+.
\]

所有尾部运算在 **logit 空间**完成。对所有 logits 加同一常数不会改变尾部间隔，从而避免通过整体压低分数获得虚假的低背景风险。

### 4.2 背景尾 \(R_d^-\)

对每张图像：

1. 将 GT mask 膨胀，排除目标邻域；
2. 在剩余背景上用固定核提取局部 logit 峰；
3. 对相等平台只保留一个确定性代表点；
4. 计算该图的背景峰值上尾均值；
5. 域内对图像等权平均。

因此无目标图像仍是有效的背景风险样本，大图像不会仅因像素更多而主导域统计。

### 4.3 目标尾 \(R_d^+\)

对每个 GT 连通目标：

1. 在目标内部取 top-fraction logit mean，得到一个目标分数；
2. 域内对所有目标分数取困难目标下尾；
3. 每个目标等权，不让大目标按面积主导。

### 4.4 域聚合

先形成每个域的 \(R_d^-\) 和 \(R_d^+\)，再计算域级 hinge。域间使用归一化 log-mean-exp：

\[
\operatorname{SmoothMax}_d(a_d)
=
\frac{1}{\gamma}
\left(\log\sum_d e^{\gamma a_d}-\log K\right).
\]

减去 \(\log K\) 后，损失不会因源域数量变化产生无关常数偏移。

### 4.5 训练稳定性

- 前若干 epoch 只训练 segmentation loss；
- 风险项在 `risk_start_epoch` 后线性 ramp；
- 记录 `loss_seg`、`loss_sep`、每域 background tail、target tail 和 gap；
- 同时监控 source-validation IoU、原图 pixel risk、logit 均值/方差和极值；
- 主比较为相同 backbone、相同数据、相同训练预算下的 segmentation-only 与 tail-separation 配对实验。

---

## 5. Stage 2：Monotone Inverse Pixel-Risk Curve

### 5.1 Grouped multi-budget episode

一个元样本是：

\[
E=(S,Q,\{B_1,\ldots,B_J\}),
\qquad B_1>B_2>\cdots>B_J.
\]

同一个 support/query episode 的多个预算必须作为一个 `[J]` 样本训练，不能复制成 J 个相互独立的标量预算样本。

### 5.2 Support 编码

编码器只能读取无标签 support \(S\)：

- probability/logit histogram；
- 分位数与高分比例；
- 固定局部峰值数量、分布和密度；
- 少量灰度、梯度与噪声统计；
- 到源域参考集合的 permutation-invariant 距离编码。

支持路径明确以 `load_mask=False` 读取 score records。统计提取函数不接受 target mask。

### 5.3 结构单调的阈值曲线

模型一次输出预算网格上的 threshold logits：

\[
\eta_j
=
\eta_{\min}
+j\delta
+C\sum_{k=1}^{j}\operatorname{softmax}(a)_k,
\qquad
\tau_j=\sigma(\eta_j).
\]

实现中的等价参数化为起始 logit、正总跨度和通过 softmax 分配的非负增量。因此：

\[
B_{j+1}<B_j
\Rightarrow
\eta_{j+1}\ge \eta_j.
\]

单调性由结构保证，不增加额外 monotonicity penalty。

### 5.4 未见预算

- 仅在训练预算的 `log10(B)` 区间内插；
- 不允许超出最松或最严预算的范围；
- 越界直接抛错并记录 provenance；
- 论文中必须区分网格预算结果与区间内插结果。

---

## 6. Query-risk-aligned objective

给定 query 像素 logit \(\ell_p\) 和阈值 logit \(\eta\)：

\[
u_p(\eta)=\sigma\left(\frac{\ell_p-\eta}{T_p}\right).
\]

可微像素虚警代理：

\[
\widetilde F_a^Q(\eta)
=
\frac{1}{|\Omega_Q|}
\sum_{p\in Q}(1-y_p)u_p(\eta).
\]

对每个 GT 目标先聚合内部 top-fraction logit 得到 \(r_o^+\)，再定义：

\[
\widetilde P_d^Q(\eta)
=
\frac{1}{N_{\mathrm{obj}}}
\sum_o
\sigma\left(\frac{r_o^+-\eta}{T_o}\right).
\]

最终损失：

\[
\mathcal L_{\mathrm{cal}}
=
\lambda_v
\left[
\log\frac{\widetilde F_a^Q+\epsilon}{B+\epsilon}
\right]_+^2
+
\lambda_u(1-\widetilde P_d^Q)
+
\lambda_o\operatorname{Huber}(\widehat\eta-\eta^*)
+
\lambda_s\mathcal L_{\mathrm{smooth}}.
\]

其中：

- `violation` 是主安全目标；
- `utility` 防止通过过高阈值获得空预测；
- oracle logit 只作低权重辅助；
- smoothness 约束相邻预算的增量变化，不承担单调性；
- 最终指标全部由 hard-threshold exact replay 计算。

---

## 7. 严格训练协议

### 7.1 外层与内层

对最终 unseen target \(D_t\)：

1. 其 official test 标签不进入 detector、calibrator、窗口、特征、超参和 checkpoint 选择；
2. detector 只在其余 meta-source 域上训练；
3. calibrator 的 inner pseudo-target detector 必须排除当前 pseudo-target 域；
4. 每个 pseudo-target 的元训练数据来自该域 official train 的预先独立子集；
5. meta train 与 meta validation 先按图像/序列拆分，再构造窗口。

### 7.2 Support/query 因果边界

- \(S\cap Q=\varnothing\)；
- temporal 数据必须 support 在前、query 在后；
- 无可验证采集时序的静态数据称为 `prefix_holdout`，不声称真实在线流；
- support 标签不加载；
- query 标签只用于 meta-training loss 或最终离线评价。

### 7.3 Checkpoint 预注册顺序

calibrator checkpoint 依据独立 pseudo-target validation 的 exact hard replay 选择：

1. **BSR 最大**；
2. **LogExcess 最小**；
3. **mean \(P_d\) 最大**。

阈值 MSE、surrogate loss 和 outer target 结果均不得覆盖该排序。

---

## 8. 必须比较的基线

### 8.1 Detector

- MSHNet segmentation-only；
- MSHNet + domain-tail separation；
- pixel top-k hard negative；
- local-peak upper-tail only；
- 可行时加入第二 backbone，保持同一校准协议。

### 8.2 Operating point

- fixed 0.5；
- pooled-source threshold；
- worst-source safe threshold；
- rolling quantile；
- EVT/GPD；
- direct threshold MLP；
- monotone oracle-regression baseline；
- proposed monotone + query-risk-aligned loss；
- target-label oracle 上界。

旧 direct+reject 可作为普通补充基线，但不进入主方法图和贡献。

---

## 9. 主指标

每个预算至少报告：

- \(P_d@B\)；
- Budget Satisfaction Rate；
- LogExcess；
- worst-domain \(P_d\)；
- exact pixel false-alarm count/rate；
- threshold 与 oracle threshold（诊断，不作为主指标）；
- IoU、nIoU、hIoU；
- component FA/MP（兼容附表）。

建议预算主次顺序为 `1e-4`、`1e-5`、`1e-6`。`1e-6` 是最严格压力设置，不应成为唯一结论。

---

## 10. AAAI 主张边界

可以主张：

- 未知域无标签前缀上的经验性 budget-aware operating-point adaptation；
- disjoint support-query meta-training；
- 结构单调的 inverse pixel-risk curve；
- domain-tail separation 提高低虚警区可校准性；
- external unseen-domain 和 prefix-to-future 评测协议。

不得主张：

- 任意目标域满足预算；
- distribution-free guarantee；
- certified false-alarm control；
- support 分数唯一识别真实背景风险；
- component FA 被严格单调控制。

---

## 11. 推荐论文题目与三条贡献

推荐题目：

> **Learning Budget-Aware Operating Points from Unlabeled Target Streams for Cross-Domain Infrared Small Target Detection**

贡献压缩为：

1. 形式化未知域、零目标标签、显式像素虚警预算下的 causal/prefix operating-point adaptation，并说明无附加假设时背景风险不可由无标签混合分数唯一识别；
2. 提出 domain-tail separation 与 disjoint support-query 的结构单调 inverse-risk calibrator；
3. 建立 external unseen-domain、原图 pixel risk、BSR、LogExcess、污染/漂移压力测试的评测协议。

---

## 12. 当前证据状态

代码层面已经完成：

- 真正域级两尾 hinge；
- 无目标图像背景尾；
- GT 膨胀与平台去重；
- `[J]` grouped-budget episode；
- 无 Reject 结构单调校准器；
- query-risk-aligned loss；
- exact hard replay；
- BSR → LogExcess → Pd checkpoint 选择；
- label-free prefix deployment；
- 单元测试与合成闭环。

尚未由本包证明：

- 真实多域上的性能提升；
- 至少四个独立 meta-source 的 strict nested LODO；
- 2–3 个 external unseen target；
- 三随机种子显著性；
- 新颖性相对于所有 2025–2026 prior art 的最终检索结论。

这些是下一阶段实验任务，不能从 synthetic smoke 结果外推。
