# RC-IRSTD：未知域红外小目标检测的风险曲线适配与少样本风险控制

## 方案设计、模型定义、完整代码工程、训练启动与投稿实现

> **建议英文题目**
> **Risk Curves under Domain Shift: Zero-Label Operating-Point Adaptation and Few-Shot Conformal Control for Infrared Small Target Detection**
>
> **短题目**
> **Risk-Controlled Infrared Small Target Detection under Unseen Domains**
>
> **实现代码包**
> `RC_IRSTD_AAAI_Implementation/`
>
> **文档定位**
> 本文不是一份“如何写投稿”的泛化建议，而是一套从科学问题、理论定义、模型结构、数据协议、代码实现、训练启动、实验矩阵到最终论文产物的完整落地方案。

---

# 1. 最终研究目标

本研究解决的不是“再设计一个红外小目标分割网络”，而是一个更接近真实部署的问题：

> 检测器进入此前未见的传感器域、背景域或噪声域后，如何利用无标签 warm-up 数据判断自身的虚警风险，并在严格虚警预算下自动选择工作点；当少量目标域标注可用时，如何进一步对预算违反风险进行有限样本控制。

最终系统提供两种部署模式。

## 1.1 模式 A：Zero-Label Operating-Point Adaptation

目标域只有无标签 warm-up 窗口：

\[
z_A=\Phi\big(\{x_i,f_\theta(x_i)\}_{i\in A}\big),
\]

风险曲线预测器输出：

\[
U_\phi(z_A,\tau)
=
\left[
U_\phi^{\mathrm{pix}}(z_A,\tau),
U_\phi^{\mathrm{peak}}(z_A,\tau)
\right].
\]

对给定像素虚警预算 \(B_p\) 和固定峰值候选虚警预算 \(B_c\)，选择：

\[
\tau_{\mathrm{zero}}(z_A)
=
\min\left\{
\tau:
U_\phi^{\mathrm{pix}}(z_A,\tau)\leq \log_{10}B_p,
\quad
U_\phi^{\mathrm{peak}}(z_A,\tau)\leq \log_{10}B_c
\right\}.
\]

该模式是**经验风险预测与工作点适配**，不声称 distribution-free guarantee。

## 1.2 模式 B：Few-Shot Conformal Risk Control

当目标域可提供少量有标注校准 block 时，在样本自适应工作点上校准一个共享、非负的阈值索引偏移：

\[
k_i(\delta)
=
\min(k_i^{\mathrm{zero}}+\delta,T-1).
\]

通过二元联合预算违反损失：

\[
L_i(\delta)
=
\mathbf 1\left\{
F^{\mathrm{pix}}_i(k_i(\delta))>B_p
\ \lor\
F^{\mathrm{peak}}_i(k_i(\delta))>B_c
\right\},
\]

使用标准 Conformal Risk Control 修正选择 \(\widehat\delta\)。该模式在交换性和嵌套损失族条件下，控制未来随机 block 违反任一预算的边际概率。

---

# 2. 相比原方案必须完成的理论修正

本实现没有机械照搬原始设计，而是对三个关键问题做了结构性修正。

## 2.1 不能直接认证连通域虚警数

阈值升高时，前景像素集合缩小，但一个连通区域可能因连接桥消失而分裂成两个区域。因此：

\[
\tau_1<\tau_2
\not\Rightarrow
N_{\mathrm{components}}(\tau_2)
\leq
N_{\mathrm{components}}(\tau_1).
\]

而标准 CRC 需要损失关于控制参数形成嵌套、单调族。

### 最终处理

本方案在连续 score map 上只提取一次**阈值无关的固定局部峰值候选集**：

\[
\mathcal C(x)
=
\{(s_j,y_j,x_j,g_j)\}_{j=1}^{M}.
\]

其中：

- \(s_j\)：候选分数；
- \((y_j,x_j)\)：候选位置；
- \(g_j=0\)：背景候选；
- \(g_j>0\)：匹配到的 GT 目标 ID。

阈值只决定候选是否保留：

\[
\mathcal C_\tau(x)=\{j:s_j\geq\tau\}.
\]

随着阈值增大，候选只能被删除，因此：

\[
N_{\mathrm{false\ peak}}(\tau)
\]

严格单调不增。若多个峰值落入同一 GT 的容差区域，只把分数最高的一个峰值固定标记为 true candidate，其余峰值固定标记为 false candidate；该一对一分配在阈值扫描前完成，因此不会破坏嵌套性。

论文仍可以报告 connected-component FA/MP，但**形式化保证只对应 fixed false peak risk**。

## 2.2 CRC 声明必须与评价指标一致

连续截断损失：

\[
\min(N_{\mathrm{FP}}/K,1)
\]

控制的是期望归一化损失，不等于 BSR，也不等于每张图都满足预算。

因此最终实现采用：

\[
L_i(\delta)
=
\mathbf 1\{\text{pixel budget violation 或 peak budget violation}\}.
\]

这样：

\[
\mathbb E[L_{m+1}(\widehat\delta)]\leq\alpha
\]

可以准确解释为边际意义下：

\[
P(\text{未来 block 违反至少一个预算})\leq\alpha,
\]

即：

\[
\mathrm{JointBSR}\geq1-\alpha.
\]

## 2.3 训练必须是 warm-up 预测未来，而不是同窗自预测

错误协议是：

\[
z_W\rightarrow r_W(\tau).
\]

它让模型根据同一批样本的无标签统计预测同一批样本的风险，属于 transductive 自估计，无法严格对应“warm-up 后部署到未来数据”。

最终协议为：

\[
z_A\rightarrow r_E(\tau),
\qquad A\cap E=\varnothing,
\]

其中：

- \(A\)：无标签 context/warm-up block；
- \(E\)：其后的 future/evaluation block；
- 输入只使用 \(A\) 的图像和 score；
- 标签风险只在 \(E\) 上计算；
- 正式实验建议 `stride >= context_size + horizon`，避免 block 重叠。

---

# 3. 完整系统结构

```text
多个有标注源域
       │
       ▼
MSHNet + SLS
       │
       ├── 背景固定局部峰值 Tail-CVaR
       └── 困难目标 Miss-CVaR
       │
       ▼
风险敏感检测器 fθ
       │
       ▼
连续概率图 p = sigmoid(fθ(x))
       │
       ├── 像素分数 survival statistics
       ├── 固定峰值 survival statistics
       └── 成像/噪声统计
       │
       ▼
无标签 warm-up 表征 zA
       │
       ▼
结构性单调双风险曲线 Uφ(zA, τ)
       │
       ├── Zero-label：选择 τzero
       │
       └── Few-shot：CRC 校准共享 offset δ
       │
       ▼
严格虚警预算下的最终工作点
```

代码采用分阶段设计：

1. detector 训练；
2. score map 离线导出；
3. causal episode 构造；
4. risk-curve predictor 训练；
5. zero-label 评估；
6. few-shot CRC；
7. baseline 与论文表格汇总。

该设计比端到端联合训练更适合研究阶段，因为能明确区分：

- 表征失效；
- 风险估计失效；
- 工作点选择失效；
- CRC 过度保守；
- 数据泄漏或协议错误。

---

# 4. 模块 A：风险敏感检测器

## 4.1 基础检测器

使用 MSHNet：

\[
P=f_\theta(x),
\qquad P\in\mathbb R^{H\times W}.
\]

代码适配当前 MSHNet 风格接口：

```python
masks, pred = model(image, training_tag)
```

基础分割项严格保留 MSHNet 的多尺度监督：对最终 `pred` 与所有辅助 mask 分别计算 SLS，再等权平均；Tail-CVaR 与 Miss-CVaR 只施加在最终连续 score map 上。

## 4.2 基础损失

首选：

\[
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{SLS}}.
\]

当不在 MSHNet 工程中运行时，代码提供：

\[
\mathcal L_{\mathrm{BCE+Dice}}
=
\mathcal L_{\mathrm{BCE}}
+
\mathcal L_{\mathrm{softDice}}
\]

作为 TinyUNet smoke test 和通用 detector fallback。

## 4.3 背景局部峰值 Tail-CVaR

对概率图：

\[
p=\sigma(P),
\]

使用局部最大池化得到背景候选代理：

\[
\mathcal A_d^{-}
=
\operatorname{LocalPeak}(p_d)
\cap
\operatorname{Background}(Y_d).
\]

为避免目标边缘被误当作背景 hard negative，对 GT 区域做半径 \(r\) 的膨胀排除。

每个源域计算高分尾部均值：

\[
R_d^{-}
=
\operatorname{CVaR}_{q_-}
(\mathcal A_d^{-}).
\]

对各源域使用平滑最坏域聚合：

\[
\mathcal L_{\mathrm{tail}}
=
\frac{1}{\gamma}
\left[
\log\sum_{d=1}^{K}\exp(\gamma R_d^{-})
-
\log K
\right].
\]

它直接惩罚最危险源域中的高置信背景峰值，而不是让大量容易背景像素淹没训练信号。

## 4.4 困难目标 Miss-CVaR

对每个真实目标连通区域 \(G_j\)，用归一化 LogSumExp 聚合响应：

\[
a_j^+
=
\frac{1}{\beta}
\log\left(
\frac{1}{|G_j|}
\sum_{p\in G_j}
\exp(\beta p_p)
\right).
\]

困难目标损失：

\[
\mathcal L_{\mathrm{miss}}
=
\operatorname{CVaR}_{q_+}
\left(\{1-a_j^+\}_j\right).
\]

它重点处理响应最低的一组目标，防止只压低背景后导致弱目标进一步消失。

## 4.5 总检测损失

\[
\boxed{
\mathcal L_{\mathrm{det}}
=
\mathcal L_{\mathrm{base}}
+
\lambda_{\mathrm{tail}}\mathcal L_{\mathrm{tail}}
+
\lambda_{\mathrm{miss}}\mathcal L_{\mathrm{miss}}
}
\]

推荐初值：

| 参数 | 初值 | 搜索范围 |
|---|---:|---:|
| \(q_-\) | 0.95 | 0.90 / 0.95 / 0.99 |
| \(q_+\) | 0.80 | 0.70 / 0.80 / 0.90 |
| \(\lambda_{\mathrm{tail}}\) | 0.10 | 0.03 / 0.10 / 0.30 |
| \(\lambda_{\mathrm{miss}}\) | 0.10 | 0.03 / 0.10 / 0.30 |
| local peak kernel | 5 | 3 / 5 / 7 |
| GT exclusion radius | 2 | 1 / 2 / 3 |
| worst-domain \(\gamma\) | 10 | 5 / 10 / 20 |

Tail/Miss 部分应作为可消融组件。若实际结果不稳定，论文主体可固定原始 MSHNet，将创新集中在风险曲线和 CRC。

---

# 5. 模块 B：固定候选与风险定义

## 5.1 阈值无关候选集

对每张连续概率图执行一次确定性局部峰值提取：

1. `maximum_filter` 找局部极大值；
2. 对平坦 plateau 只保留一个点；
3. 使用字典序打破并列，保证跨平台确定性；
4. 按分数降序排序；
5. 可限制最大候选数。

固定候选集提取以后，所有阈值共用同一候选集合。

## 5.2 像素风险

对 future block \(E\)：

\[
F_E^{\mathrm{pix}}(\tau)
=
\frac{
\sum_{x\in E}
\sum_p
\mathbf 1[p_x(p)\geq\tau]\mathbf 1[Y_x(p)=0]
}{
\sum_{x\in E}|x|
}.
\]

使用 log-risk 作为回归目标：

\[
r_E^{\mathrm{pix}}(\tau)
=
\log_{10}(F_E^{\mathrm{pix}}(\tau)+\epsilon_p).
\]

## 5.3 固定峰值虚警风险

候选与 GT 通过以下规则匹配：

- 候选位于 GT 内；或
- 候选到最近 GT 的距离不超过容差 \(d\)。

future block 的 false peak/MP：

\[
F_E^{\mathrm{peak}}(\tau)
=
\frac{
N_{\mathrm{false\ peak}}(E,\tau)
}{
\sum_{x\in E}|x|/10^6
}.
\]

对应：

\[
r_E^{\mathrm{peak}}(\tau)
=
\log_{10}(F_E^{\mathrm{peak}}(\tau)+\epsilon_c).
\]

## 5.4 目标检测概率

\[
P_d(E,\tau)
=
\frac{
N_{\mathrm{matched\ GT}}(E,\tau)
}{
N_{\mathrm{GT}}(E)
}.
\]

由于固定候选集合只随阈值删减，false peak count 和 matched GT count 都关于阈值单调不增。

---

# 6. 模块 C：无标签 warm-up 表征

风险预测器不直接读取标签，只读取 warm-up context 中的无标签统计。

## 6.1 像素分数统计

对一组固定阈值 \(\mathcal S\)：

\[
q_t^{\mathrm{pix}}
=
\frac{1}{|A|}
\sum_{x\in A}
\frac{1}{|x|}
\sum_p
\mathbf 1[p_x(p)\geq t].
\]

代码使用多个高分辨率尾部阈值，并对图像间均值和标准差同时编码。

同时加入：

- score quantiles；
- 图像间 quantile 方差；
- 高阈值 survival tail。

## 6.2 固定峰值统计

对每张 warm-up 图像提取固定峰值：

- peak count/MP；
- peak score survival curve；
- peak score quantiles；
- 图像间均值和标准差。

这部分比单纯像素直方图更接近部署关注的虚警对象。

## 6.3 成像与噪声统计

代码导出：

- 灰度均值、标准差、MAD；
- 1%、10%、50%、90%、99% 分位数；
- Sobel 梯度均值、标准差和 q95；
- Laplacian 标准差；
- 局部对比度均值和 q95；
- 256-bin entropy；
- 高频能量比例。

## 6.4 最终特征

\[
z_A
=
[
\text{pixel survival},
\text{pixel quantiles},
\text{peak survival},
\text{peak quantiles},
\text{image/noise statistics},
\text{window size statistics}
].
\]

每组逐图统计均使用：

\[
[\operatorname{mean},\operatorname{std}]
\]

聚合，使模型既感知中心位置，也感知窗口内部不稳定程度。

---

# 7. 模块 D：结构性单调双风险曲线网络

## 7.1 特征编码器

\[
h
=
\operatorname{MLP}(\operatorname{LayerNorm}(z_A)).
\]

默认结构：

```text
LayerNorm
→ Linear(D, 256)
→ GELU
→ Dropout(0.1)
→ Linear(256, 256)
→ GELU
→ Dropout(0.1)
```

## 7.2 单调 head

对每个风险 head，网络输出：

- 初始风险值 \(s\)；
- 总下降量原始值 \(d\)；
- 各阈值间隔分配 logits \(a_1,\ldots,a_{T-1}\)。

令：

\[
D=\operatorname{softplus}(d)>0,
\]

\[
\pi_j
=
\frac{\exp(a_j)}{\sum_k\exp(a_k)}.
\]

定义：

\[
U_0=s,
\]

\[
U_t
=
s-D\sum_{j=1}^{t}\pi_j.
\]

于是：

\[
U_{t+1}\leq U_t
\]

由网络结构直接保证，不需要额外 monotonicity penalty。

相比朴素 cumulative-softplus，该结构把一个正的总下降量分配到所有间隔，避免数百个阈值初始累计产生数值爆炸。

## 7.3 双风险输出

\[
U_\phi(z_A)
=
\left[
U_\phi^{\mathrm{pix}}(z_A),
U_\phi^{\mathrm{peak}}(z_A)
\right],
\]

每个输出长度为 \(T\)，对应同一固定阈值网格。

阈值网格最后包含一个略大于 1 的显式空预测动作。由于 sigmoid 分数位于 \([0,1]\)，该动作保证候选集为空；一旦工作点落在该端点，评价代码将其记为 reject/abstention，而不是普通的预算满足样本。

## 7.4 高分位数训练

对真实 future log-risk \(r\)，使用 pinball loss：

\[
\ell_\rho(\widehat r,r)
=
\max\big(
\rho(r-\widehat r),
(\rho-1)(r-\widehat r)
\big).
\]

总损失：

\[
\boxed{
\mathcal L_{\mathrm{curve}}
=
\mathcal L_{\mathrm{pinball}}^{\mathrm{pix}}
+
\lambda_c
\mathcal L_{\mathrm{pinball}}^{\mathrm{peak}}
}
\]

主实验建议：

- mean-risk：\(\rho=0.50\)；
- upper-risk：\(\rho=0.90\)；
- 更保守消融：\(\rho=0.95\)。

必须准确表述为“条件上分位风险预测”，而不是“无条件风险上界”。

---

# 8. Zero-label 工作点选择

对每个 episode 或部署窗口，寻找第一个同时满足预测预算的阈值索引：

\[
k_i^{\mathrm{zero}}
=
\min\left\{
k:
U_{i,k}^{\mathrm{pix}}\leq\log_{10}B_p,
\quad
U_{i,k}^{\mathrm{peak}}\leq\log_{10}B_c
\right\}.
\]

若没有任何阈值满足：

- 使用最保守阈值索引 \(T-1\)；
- 输出 `rejected=True`；
- 主表单独报告 rejection rate；
- `effective_pd_with_rejects` 把 reject 的检测概率记为 0。

这能防止方法通过大量拒判制造虚假的高 BSR。

---

# 9. Few-shot CRC 设计

## 9.1 样本自适应阈值族

对第 \(i\) 个校准 episode：

\[
k_i(\delta)
=
\min(k_i^{\mathrm{zero}}+\delta,T-1),
\qquad \delta\geq0.
\]

这不是把一个目标域常数阈值平移，而是对每个样本自己的 zero-label 工作点添加共享残差，因此其函数族不同于单一 global threshold。

## 9.2 联合预算违反损失

\[
L_i(\delta)
=
\mathbf 1\left\{
F_{i,k_i(\delta)}^{\mathrm{pix}}>B_p
\lor
F_{i,k_i(\delta)}^{\mathrm{peak}}>B_c
\right\}.
\]

因为两个风险都关于阈值索引单调不增，所以：

\[
L_i(\delta+1)\leq L_i(\delta).
\]

## 9.3 CRC 修正

经验风险：

\[
\widehat R_m(\delta)
=
\frac1m\sum_{i=1}^{m}L_i(\delta).
\]

选择最小安全偏移：

\[
\widehat\delta
=
\inf\left\{
\delta:
\frac{m}{m+1}\widehat R_m(\delta)
+
\frac{1}{m+1}
\leq\alpha
\right\}.
\]

## 9.4 小样本可行性

即使经验风险为 0，修正后最小风险仍为：

\[
\frac{1}{m+1}.
\]

在 \(\alpha=0.1\) 时：

| 校准量 \(m\) | 最小修正值 | 结论 |
|---:|---:|---|
| 5 | 0.1667 | 不可能形式化可行 |
| 9 | 0.1000 | 仅允许零经验违反 |
| 10 | 0.0909 | 可行但极严 |
| 20 | 0.0476 | 可用于正式结果 |
| 50 | 0.0196 | 更实用 |

因此：

- `m=5` 只能作为 empirical calibration；
- `m=10` 是 severe diagnostic；
- 主认证结果应使用 `m=20` 和 `m=50`；
- 每个设置使用多次随机 sequence-block split。

## 9.5 校准与测试隔离

代码的 CRC split 满足：

- calibration 和 test 使用完全不同的 domain/sequence group；
- 选中 calibration sequence 后，该序列中未被选为 calibration 的 episode 直接丢弃，不进入 test；
- test 只来自未触碰序列；
- 形式化论文应把 episode/block 作为抽样单位，并说明 exchangeability 假设。

---

# 10. 数据协议：严格 Nested Leave-One-Domain-Out

设共有 \(K\) 个域。

## 10.1 外层未见域

选目标域 \(\mathcal D_t\)：

- detector 训练不使用 \(\mathcal D_t\)；
- risk-curve predictor 训练不使用 \(\mathcal D_t\)；
- 目标域无标签 warm-up 只用于 zero-label 特征；
- few-shot 模式只额外使用固定数量 calibration block；
- test 标签仅用于最终评价。

## 10.2 内层伪目标域

对每个非目标源域 \(\mathcal D_p\)，训练 episode detector 时排除：

\[
\mathcal D_t\quad\text{和}\quad\mathcal D_p.
\]

即：

\[
\theta_{t,p}
\leftarrow
\operatorname{Train}
\left(
\{\mathcal D_k:k\notin\{t,p\}\}
\right).
\]

然后在 \(\mathcal D_p\) 上导出 score、构造 warm-up-to-future episode，作为风险曲线训练数据。

这一设计避免 risk predictor 只看到“检测器曾经训练过的域”。

## 10.3 最终 detector

对外层目标 \(\mathcal D_t\)，最终 detector 使用全部非目标源域：

\[
\theta_t^{\mathrm{final}}
\leftarrow
\operatorname{Train}
\left(
\{\mathcal D_k:k\neq t\}
\right).
\]

## 10.4 数据泄漏检查

必须执行：

- 按序列划分视频帧；
- 对近重复图像做哈希与感知哈希；
- 避免同时使用包含关系数据集；
- 每个结果记录 source domains、target domain、split、seed、checkpoint、commit；
- 风险曲线超参数不能根据最终外层目标测试结果调节；
- threshold grid 在训练、验证、测试中完全一致。

---

# 11. 实现代码结构

```text
RC_IRSTD_AAAI_Implementation/
├── README.md
├── DELIVERY.md
├── ANONYMIZATION.md
├── NOTICE.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── lodo_example.yaml
│   └── lodo_fold.example.yaml
├── scripts/
│   ├── setup.sh
│   ├── smoke_test.sh
│   ├── train_detector_mshnet.sh
│   ├── launch_lodo_fold.sh
│   ├── run_lodo.sh
│   ├── aggregate_paper_results.sh
│   └── build_anonymous_supplement.sh
├── rc_irstd/
│   ├── calibration/
│   │   └── crc.py
│   ├── candidates/
│   │   └── peaks.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── sampler.py
│   │   ├── score_records.py
│   │   └── windows.py
│   ├── episodes/
│   │   ├── builder.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── evaluation/
│   │   ├── budget.py
│   │   ├── curves.py
│   │   ├── operating_point.py
│   │   ├── risk_curve_metrics.py
│   │   └── segmentation.py
│   ├── features/
│   │   ├── image_stats.py
│   │   └── window_stats.py
│   ├── losses/
│   │   ├── cvar.py
│   │   ├── quantile.py
│   │   └── risk_aware.py
│   ├── models/
│   │   ├── detector_adapter.py
│   │   ├── risk_curve.py
│   │   ├── risk_io.py
│   │   └── tiny_detector.py
│   └── pipelines/
│       ├── train_detector.py
│       ├── export_scores.py
│       ├── evaluate_scores.py
│       ├── build_episodes.py
│       ├── train_curve.py
│       ├── evaluate_zero_label.py
│       ├── predict_unlabeled.py
│       ├── calibrate_and_evaluate.py
│       ├── evaluate_baselines.py
│       ├── run_lodo.py
│       ├── aggregate_results.py
│       ├── build_supplement.py
│       ├── make_synthetic_data.py
│       └── smoke.py
└── tests/
    ├── test_crc.py
    ├── test_dataset_paths.py
    ├── test_episode_metrics.py
    ├── test_feature_config.py
    ├── test_lodo_protocol.py
    ├── test_operating_point.py
    ├── test_peaks.py
    ├── test_risk_aware_loss.py
    ├── test_risk_curve.py
    ├── test_sampler.py
    ├── test_splits.py
    └── test_windows.py
```

---

# 12. 环境安装与软件验证

## 12.1 安装代码包

```bash
cd /path/to/RC_IRSTD_AAAI_Implementation
python -m pip install -e ".[dev]"
```

若使用 MSHNet：

1. 先创建并安装 MSHNet 原工程环境；
2. 在同一 Python 环境安装本代码包；
3. 运行时把 MSHNet 根目录加入 `PYTHONPATH`。

## 12.2 一键 smoke test

```bash
cd /path/to/RC_IRSTD_AAAI_Implementation
./scripts/smoke_test.sh
```

该命令自动完成：

1. 生成三个合成域；
2. TinyUNet 多源训练；
3. 连续 score map 导出；
4. causal episode 构造；
5. 单调 risk curve 训练；
6. zero-label 评估；
7. CRC 校准；
8. 单元测试。

当前代码已通过：

```text
12 passed
```

并完成了完整 synthetic end-to-end smoke pipeline，以及完整 Nested LODO 命令图的 dry-run 校验。dry-run 会检查外层目标留出、内层伪目标检测器的源域排除关系、训练/评估 episode 步长、逐数据集 split 和所有预期产物路径。

这只证明软件链路闭合，不代表真实 MSHNet 或真实数据集结果。

---

# 13. 训练启动：单独训练 MSHNet 风险敏感 detector

## 13.1 直接命令

```bash
export PYTHONPATH=/path/to/MSHNet:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
cd /path/to/MSHNet

python -m rc_irstd.pipelines.train_detector \
  --source-dataset /data/NUAA-SIRST \
  --source-dataset /data/NUDT-SIRST \
  --source-dataset /data/IRSTD-1K \
  --train-split train \
  --val-split test \
  --detector mshnet \
  --base-loss auto \
  --resize 256 256 \
  --batch-size 6 \
  --epochs 400 \
  --warm-epoch 5 \
  --optimizer adagrad \
  --lr 0.05 \
  --weight-decay 0.0 \
  --lambda-tail 0.10 \
  --lambda-miss 0.10 \
  --tail-quantile 0.95 \
  --miss-quantile 0.80 \
  --peak-kernel 5 \
  --exclusion-radius 2 \
  --worst-gamma 10.0 \
  --auxiliary-weight 1.0 \
  --num-workers 4 \
  --device cuda \
  --amp \
  --deterministic \
  --seed 42 \
  --output-dir outputs/rc_irstd_detector
```

每个源数据集也可分别指定 split：按 `--source-dataset` 的顺序重复传入 `--source-train-split` 与 `--source-val-split`。未提供时回退到全局 `--train-split` 和 `--val-split`。Nested LODO 启动器会自动读取 YAML 中每个数据集的 `train_split` 与 `eval_split`。

## 13.2 Shell 启动器

修改数据路径后：

```bash
MSHNET_ROOT=/path/to/MSHNet \
RUN_ROOT=/path/to/output/detector \
CUDA_VISIBLE_DEVICES=0 \
/path/to/RC_IRSTD_AAAI_Implementation/scripts/train_detector_mshnet.sh \
  /data/NUAA-SIRST /data/NUDT-SIRST /data/IRSTD-1K
```

## 13.3 断点续训

```bash
python -m rc_irstd.pipelines.train_detector \
  ... \
  --resume outputs/rc_irstd_detector/last.pt \
  --output-dir outputs/rc_irstd_detector
```

输出：

```text
outputs/rc_irstd_detector/
├── arguments.json
├── metrics.jsonl
├── last.pt
├── best.pt
└── epoch_XXXX.pt
```

---

# 14. 分阶段训练与评估命令

## 14.1 导出连续 score map

```bash
python -m rc_irstd.pipelines.export_scores \
  --dataset-dir /data/RealScene-ISTD \
  --split test \
  --detector mshnet \
  --checkpoint outputs/rc_irstd_detector/best.pt \
  --resize 256 256 \
  --restore-original \
  --include-mask \
  --num-workers 4 \
  --device cuda \
  --output-dir outputs/scores/RealScene-ISTD
```

每张图保存一个 `.npz`：

- continuous probability；
- mask（离线训练/评价时）；
- image ID；
- domain；
- sequence ID；
- frame index；
- original resolution；
- image/noise statistics；
- source checkpoint。

## 14.2 画完整低虚警曲线

```bash
python -m rc_irstd.pipelines.evaluate_scores \
  --score-dir outputs/scores/RealScene-ISTD \
  --pixel-budget 1e-6 \
  --pixel-budget 1e-5 \
  --peak-budget 1.0 \
  --peak-budget 5.0 \
  --output-dir outputs/score_diagnostics/RealScene-ISTD
```

在继续训练 risk predictor 之前，必须先确认：

- oracle threshold 跨域明显变化；
- oracle 在相同预算下能恢复明显的 \(P_d\)；
- 固定/source threshold 存在预算违反；
- 问题不是完全由 detector 表征崩溃主导。

## 14.3 构造 causal episode

```bash
python -m rc_irstd.pipelines.build_episodes \
  --score-dir outputs/scores/PseudoTarget \
  --output outputs/episodes/PseudoTarget.npz \
  --context-size 32 \
  --horizon 16 \
  --stride 48 \
  --peak-min-distance 2 \
  --peak-min-score 0.0 \
  --peak-tolerance 2.0 \
  --max-candidates 0
```

正式风险控制实验使用 `--max-candidates 0`，即不截断固定候选集。有限候选上限只用于 smoke test 或明确报告的效率消融，因为候选截断会改变被控制的风险定义。

## 14.4 训练单调双风险曲线

```bash
python -m rc_irstd.pipelines.train_curve \
  --train-episode outputs/episodes/Pseudo-A.npz \
  --train-episode outputs/episodes/Pseudo-B.npz \
  --train-episode outputs/episodes/Pseudo-C.npz \
  --quantile 0.90 \
  --hidden-dim 256 \
  --dropout 0.10 \
  --lambda-peak 1.0 \
  --batch-size 64 \
  --epochs 300 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 40 \
  --pixel-budget 1e-6 \
  --peak-budget 1.0 \
  --device cuda \
  --output-dir outputs/risk_curve
```

## 14.5 Zero-label 评价

```bash
python -m rc_irstd.pipelines.evaluate_zero_label \
  --episode outputs/episodes/FinalTarget.npz \
  --curve-checkpoint outputs/risk_curve/best.pt \
  --pixel-budget 1e-6 \
  --peak-budget 1.0 \
  --device cuda \
  --output-dir outputs/zero_label
```

## 14.6 Few-shot CRC

```bash
python -m rc_irstd.pipelines.calibrate_and_evaluate \
  --episode outputs/episodes/FinalTarget.npz \
  --curve-checkpoint outputs/risk_curve/best.pt \
  --pixel-budget 1e-6 \
  --peak-budget 1.0 \
  --alpha 0.10 \
  --calibration-sizes 10 20 50 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --offset-step 1 \
  --device cuda \
  --output-dir outputs/few_shot_crc
```

输出同时比较：

- adaptive risk-curve CRC offset；
- adaptive empirical offset；
- raw global-threshold CRC；
- zero-label no calibration。

## 14.7 真正无标签部署

先对无标签 warm-up 数据导出 score，不保存 mask：

```bash
python -m rc_irstd.pipelines.export_scores \
  --dataset-dir /data/UNLABELED-TARGET \
  --split warmup \
  --detector mshnet \
  --checkpoint outputs/rc_irstd_detector/best.pt \
  --resize 256 256 \
  --restore-original \
  --no-include-mask \
  --device cuda \
  --output-dir outputs/deployment/warmup_scores
```

再选择工作点：

```bash
python -m rc_irstd.pipelines.predict_unlabeled \
  --score-dir outputs/deployment/warmup_scores \
  --curve-checkpoint outputs/risk_curve/best.pt \
  --last-n 32 \
  --pixel-budget 1e-6 \
  --peak-budget 1.0 \
  --device cuda \
  --output outputs/deployment/operating_point.json
```

---

# 15. 一键 Nested LODO 训练启动

## 15.1 配置文件

复制：

```bash
cp configs/lodo_example.yaml configs/paper.yaml
```

配置关键项（相对路径均以 YAML 文件所在目录为基准）：

```yaml
python: python
working_directory: /path/to/MSHNet
output_root: /path/to/outputs/rc_irstd_nested_lodo

datasets:
  NUAA-SIRST:
    path: /data/NUAA-SIRST
    train_split: train
    eval_split: test
  # ...

outer_targets:
  - RealScene-ISTD

detector:
  name: mshnet
  base_loss: auto
  resize: [256, 256]
  per_domain_batch: 2
  epochs: 400
  lambda_tail: 0.10
  lambda_miss: 0.10

episodes:
  context_size: 32
  horizon: 16
  # 伪目标域训练 episode 可重叠，以提高风险曲线训练样本量。
  train_stride: 16
  # 目标域校准/测试 episode 必须不共享图像。
  eval_stride: 48
  # 正式形式化结果不截断固定候选族；0 表示关闭截断。
  max_candidates: 0

curve:
  quantile: 0.90
  hidden_dim: 256
  epochs: 300

budgets:
  pixel: 1.0e-6
  peak_per_mp: 1.0

calibration:
  alpha: 0.10
  sizes: [10, 20, 50]
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## 15.2 先 dry-run

```bash
python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml \
  --outer-target RealScene-ISTD \
  --dry-run
```

所有命令会写入：

```text
outer_<target>/commands.log
```

## 15.3 启动一个完整 fold

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml \
  --outer-target RealScene-ISTD
```

## 15.4 分阶段启动

```bash
# 只训练 detector
python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml \
  --outer-target RealScene-ISTD \
  --stages detector

# 导出 score 并构造 episode
python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml \
  --outer-target RealScene-ISTD \
  --stages export episodes

# 风险曲线、zero-label、CRC、基线
python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml \
  --outer-target RealScene-ISTD \
  --stages curve zero calibrate baselines
```

默认 `resume-existing=True`，已有目标产物会自动跳过。

## 15.5 多 GPU 外层并行

代码采用“一个 outer fold 一个进程”，不在单个 detector 内强制 DDP。示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml --outer-target NUAA-SIRST

CUDA_VISIBLE_DEVICES=1 python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml --outer-target NUDT-SIRST

CUDA_VISIBLE_DEVICES=2 python -m rc_irstd.pipelines.run_lodo \
  --config configs/paper.yaml --outer-target RealScene-ISTD
```

不同 outer target 写入独立目录，可以安全共享同一个 `output_root`。

---

# 16. 已实现基线

## 16.1 工作点基线

```bash
python -m rc_irstd.pipelines.evaluate_baselines \
  --target-episode outputs/target.npz \
  --source-episode outputs/source_A.npz \
  --source-episode outputs/source_B.npz \
  --fixed-threshold 0.5 \
  --pixel-budget 1e-6 \
  --peak-budget 1.0 \
  --output-dir outputs/baselines
```

已实现：

1. `fixed_0.5`；
2. `source_pooled`；
3. `source_worst_domain`；
4. `nearest_source_curve`；
5. `context_all_detection_upper`；
6. `target_future_oracle`；
7. raw global-threshold CRC；
8. empirical adaptive offset；
9. zero-label no calibration。

## 16.2 All-detection upper bound

在 warm-up context 中，把所有预测前景像素和所有峰值候选都暂时视为 false：

\[
\widetilde F_A^{\mathrm{pix}}(\tau)
=
\frac{N_{\mathrm{predicted\ pixels}}(A,\tau)}{|A|},
\]

\[
\widetilde F_A^{\mathrm{peak}}(\tau)
=
\frac{N_{\mathrm{predicted\ peaks}}(A,\tau)}{|A|/10^6}.
\]

它是 context 自身实际虚警的确定性上界，因为其中还包含 true positives；但把该上界迁移到 future block 仍需要时间平稳假设，不具有 distribution-free 保证。

这是风险曲线模型必须显著优于的强无学习基线。

## 16.3 外部 detector baseline

RealScene-ISTD、S²CPNet、Ivan-ISTD、SCTransNet、DNANet 等应使用官方实现训练或加载 checkpoint，再把连续 score map 接入本工程统一评价。

不要为对齐本方法而修改这些模型的测试标签或目标域训练协议。

---

# 17. 评价指标

## 17.1 Joint Budget Satisfaction Rate

\[
\mathrm{JointBSR}
=
\frac1N\sum_{i=1}^{N}
\mathbf 1\left\{
F_i^{\mathrm{pix}}\leq B_p
\land
F_i^{\mathrm{peak}}\leq B_c
\right\}.
\]

同时报告：

- Pixel BSR；
- Peak BSR；
- Worst-domain BSR。

## 17.2 Excess

\[
\mathrm{PixelExcess}
=
\frac1N\sum_i
[F_i^{\mathrm{pix}}-B_p]_+,
\]

\[
\mathrm{PeakExcess}
=
\frac1N\sum_i
[F_i^{\mathrm{peak}}-B_c]_+.
\]

## 17.3 检测性能

报告：

- mean selected \(P_d\)；
- effective \(P_d\) with rejects；
- conditional \(P_d\) on non-rejected windows；
- worst-domain selected \(P_d\)；
- \(P_d@F_a^{\mathrm{pix}}\leq10^{-6}\)；
- \(P_d@F_a^{\mathrm{pix}}\leq10^{-5}\)；
- \(P_d@F_a^{\mathrm{peak/MP}}\leq B_c\)。

## 17.4 风险曲线质量

- pixel log-risk MAE；
- peak log-risk MAE；
- pointwise upper-quantile coverage；
- joint pointwise coverage；
- underestimation MAE；
- monotonicity violations；
- selected-point budget violation；
- rejection rate。

点态 coverage 不等于 selected-point guarantee，因此两类指标必须分开。

---

# 18. 实验矩阵

## 18.1 主结果

| 组别 | 目标标签 | 方法 |
|---|---:|---|
| 固定工作点 | 0 | Fixed 0.5 |
| 源域阈值 | 0 | Source pooled / worst source |
| 最近源域 | 0 | Nearest-source curve |
| 强保守基线 | 0 | Context all-detection upper |
| 直接风险适配 | 0 | RC q50 / q90 / q95 |
| 经验少样本 | 10/20/50 | Empirical adaptive offset |
| 全局 CRC | 10/20/50 | Raw global threshold CRC |
| 提出方法 | 10/20/50 | Adaptive risk-curve CRC offset |
| 上限 | 全部 | Target future oracle |

## 18.2 检测器消融

1. MSHNet + SLS；
2. + background Tail-CVaR；
3. + Miss-CVaR；
4. + Tail-CVaR + Miss-CVaR；
5. 第二 backbone。

## 18.3 风险曲线消融

1. pixel only；
2. peak only；
3. dual risk；
4. q50 / q90 / q95；
5. 仅 pixel score features；
6. + fixed peak features；
7. + image/noise features；
8. 无结构单调 head；
9. 结构单调 head；
10. same-window transductive；
11. causal warm-up-to-future。

## 18.4 CRC 消融

1. global raw threshold CRC；
2. sample-adaptive base threshold + empirical offset；
3. sample-adaptive base threshold + CRC offset；
4. calibration size 10 / 20 / 50；
5. alpha 0.05 / 0.10 / 0.20；
6. offset grid step 1 / 2 / 4；
7. overlapping episode vs non-overlapping sequence block，仅作为协议敏感性分析。

## 18.5 鲁棒性

- 噪声；
- 模糊；
- 对比度下降；
- 条纹噪声；
- 分辨率变化；
- warm-up 长度；
- 目标密度污染；
- domain shift 强度；
- sequence drift。

---

# 19. 阶段性科学闸门

## Gate 1：问题是否真实存在

至少三个未见域中：

- oracle threshold 明显不同；
- source threshold 经常违反预算或过度保守；
- oracle 在相同预算下比 source threshold 提升至少约 3 个百分点 \(P_d\)。

若 oracle 也无法恢复性能，先解决表示学习，不继续堆校准器。

## Gate 2：无标签统计是否具有预测力

- 简单线性/MLP 能预测风险曲线大体位置；
- q90 优于 nearest-source 和 all-detection baseline 的效用-安全折中；
- selected-point BSR 和 Excess 均改善；
- 不只是预测常数曲线。

## Gate 3：完整风险曲线是否优于直接阈值

必须证明：

- 支持训练中未显式使用的预算；
- risk curve 比固定预算 direct threshold 更易迁移；
- 曲线诊断和 coverage 指标提供额外价值。

若完整曲线没有收益，应退回直接阈值模型，避免不必要复杂度。

## Gate 4：CRC 是否提供实际价值

- formal feasible fraction 足够高；
- JointBSR 接近或超过 \(1-\alpha\)；
- adaptive CRC 比 raw global CRC 保留更高 \(P_d\)；
- calibration size 增加时偏移和保守性下降；
- 不把 infeasible fallback 误标为 certified。

## Gate 5：跨 backbone 与跨域稳定

- 至少两个 detector backbone；
- 至少五个 outer target；
- 至少三个 detector seed；
- CRC 至少十个 calibration split；
- 改进不是由单个数据集主导。

---

# 20. 计算资源控制

完整 nested LODO 的 detector 训练量很大。推荐以下策略。

## 20.1 先缓存 score map

所有风险定义、阈值网格和 CRC 都基于离线 score map，因此 detector 一旦固定，不需要反复推理。

## 20.2 三阶段实验顺序

### MVP

- 3 个域；
- 1 个 backbone；
- 1 个 detector seed；
- q50/q90；
- zero-label；
- m=20 CRC。

### 核心论文

- 5–6 个域；
- 2 个 backbone；
- 3 个 detector seed；
- 10 个 calibration split；
- 主基线与关键消融。

### 补充材料

- 扩展扰动；
- 更多预算；
- q95；
- 额外窗口长度；
- 额外 detector。

## 20.3 不要过早联合训练

第一篇版本不要把 detector 和 risk predictor 端到端联合，否则：

- target 风险标签容易泄漏；
- 无法复用 score map；
- 调试代价大；
- 理论对象不清晰；
- 消融难以解释。

---

# 21. 输出目录与论文表格

完整 LODO 输出示例：

```text
outputs/rc_irstd_nested_lodo/
├── protocol.json
├── outer_RealScene-ISTD/
│   ├── commands.log
│   ├── final_detector/
│   ├── target_scores/
│   ├── target_episodes.npz
│   ├── pseudo/
│   │   ├── NUAA-SIRST/
│   │   ├── NUDT-SIRST/
│   │   └── ...
│   ├── final_source/
│   ├── risk_curve/
│   ├── zero_label/
│   ├── few_shot_crc/
│   └── baselines/
└── outer_<other-target>/
```

汇总：

```bash
python -m rc_irstd.pipelines.aggregate_results \
  --lodo-root outputs/rc_irstd_nested_lodo \
  --output-dir outputs/rc_irstd_nested_lodo/paper_tables
```

生成：

- `zero_label_by_domain.csv`；
- `zero_label_summary.csv/.md`；
- `baselines_by_domain.csv`；
- `baselines_summary.csv/.md`；
- `few_shot_crc_all_runs.csv`；
- `few_shot_crc_summary.csv/.md`；
- `manifest.json`。

---

# 22. 最终论文方法贡献

贡献建议压缩为三条。

## Contribution 1：问题和无泄漏协议

提出严格虚警预算下的未知域 IRSTD 部署问题，明确区分：

- zero-label empirical adaptation；
- few-shot conformal risk control；

并设计 nested LODO、causal warm-up-to-future、sequence-disjoint calibration/test 协议。

## Contribution 2：单调双风险曲线

提出由无标签目标统计预测的阈值条件化双风险曲线：

- pixel false risk；
- fixed false peak risk；

通过结构性单调 head 统一支持多个预算、风险诊断和样本自适应工作点选择。

## Contribution 3：Adaptive Conformal Residual

在样本自适应 zero-label 工作点上校准共享 threshold-index residual，使用与 JointBSR 精确对应的二元联合预算违反损失，在少量目标标注下进行有限样本边际风险控制。

Tail-CVaR 只有在跨 backbone、跨域结果稳定时才写入主贡献，否则作为 detector enhancement 或补充消融。

---

# 23. 论文正文结构

七页正文建议：

| 内容 | 页数 |
|---|---:|
| Introduction | 0.8 |
| Related Work + Problem Formulation | 0.8 |
| Fixed-Candidate Risk Definition | 0.6 |
| Monotone Risk-Curve Adaptation | 1.2 |
| Conformal Residual Control | 0.7 |
| Experimental Protocol | 0.6 |
| Main Results + Ablations | 1.9 |
| Limitations + Conclusion | 0.4 |

正文图表：

1. 总体流程图；
2. 固定候选和风险曲线示意；
3. 真实/预测 risk curve；
4. calibration size 与 \(P_d\)/BSR 曲线；
5. zero-label 主表；
6. few-shot CRC 主表。

代码文件树、完整特征列表、全部阈值网格、扩展消融放补充材料。

---

# 24. 摘要骨架

> Infrared small target detectors are commonly evaluated at fixed thresholds under dataset-specific train-test splits, whereas practical deployment requires maintaining high detection probability under strict false-alarm budgets in previously unseen sensor domains. We formulate this setting as deployment-time risk adaptation. Our method first constructs a threshold-independent local-peak candidate family, yielding monotone pixel-level and candidate-level false-alarm risks. From an unlabeled target-domain warm-up block, a structurally monotone predictor estimates threshold-conditioned upper-quantile risk curves and selects sample-adaptive operating points for arbitrary deployment budgets. When a small labeled target calibration set is available, a shared conformal threshold residual is calibrated on top of these adaptive operating points using a joint budget-violation loss. Under exchangeability and nested-loss assumptions, the resulting procedure controls the marginal probability of violating either false-alarm budget. A nested leave-one-domain-out protocol with disjoint warm-up, calibration, and future-test blocks evaluates detection probability, joint budget satisfaction, excess risk, rejection, and worst-domain performance.

---

# 25. 失败分析必须包含的内容

1. **表征失败**：目标和背景峰值排序已经崩溃，oracle 也不能恢复；
2. **风险低估**：预测曲线低于真实曲线，导致 budget violation；
3. **过度保守**：q95 或 CRC offset 过大，导致 \(P_d\) 明显下降；
4. **warm-up 非平稳**：context 与 future 背景分布快速变化；
5. **目标污染**：warm-up 目标密度异常高，使无标签 score 统计偏移；
6. **候选定义失配**：fixed peak 与传统 connected-component 指标排序不同；
7. **校准样本不足**：CRC correction floor 导致无形式可行解；
8. **sequence 相关性**：名义样本数大，但有效独立 block 很少；
9. **拒判集中**：某一目标域 rejection rate 过高；
10. **域外偏移过远**：目标统计超出源 episode 支持范围。

建议在风险模型中增加 OOD/reject 诊断，但第一版不需要把它包装为额外理论贡献。

---

# 26. 当前实现状态与诚实边界

## 已完成并验证

- 完整 Python package；
- MSHNet adapter；
- 多源域平衡采样；
- SLS/BCE-Dice 基础损失；
- Tail-CVaR 和 Miss-CVaR；
- 连续 score map 导出；
- 原始分辨率恢复；
- 固定峰值候选；
- 单调风险曲线计算；
- causal episode；
- 无标签特征；
- 结构单调双风险网络；
- q90/q95 pinball 训练；
- zero-label 阈值选择与 reject；
- sequence-disjoint CRC；
- raw CRC 与 empirical offset 对照；
- fixed/source/nearest/count-all/oracle 基线；
- nested LODO 一键启动与命令图 dry-run；
- 每个源数据集独立 train/validation split；
- 结果汇总；
- 16 个单元测试；
- synthetic end-to-end smoke test。

## 尚不能声称已完成的部分

由于当前没有提供真实数据、已配置的 MSHNet 运行环境和 GPU，本交付不能声称：

- 已获得真实 IRSTD benchmark 数值；
- 已完成六域完整 LODO；
- 已复现所有第三方 detector baseline；
- 已证明方法达到 AAAI 接收水平；
- 已验证真实 MSHNet 长训练的显存和速度；
- 已产生最终论文主表。

下一步不是继续扩展代码模块，而是按 Gate 1–4 在真实 score map 上验证核心假设。

---

# 27. 最终执行顺序

```text
Step 0  安装代码并通过 smoke_test.sh
Step 1  在一个 source→三个 target 上导出连续 score map
Step 2  画完整低虚警曲线，确认 oracle gap
Step 3  固定 detector，构造 causal episodes
Step 4  训练 q50/q90 risk curve
Step 5  比较 fixed/source/count-all/nearest-source/oracle
Step 6  验证 zero-label JointBSR、Excess、Pd、reject
Step 7  在 m=20/50 上运行 adaptive CRC 与 raw CRC
Step 8  决定 Tail-CVaR 是否进入主方法
Step 9  扩展到完整 nested LODO、第二 backbone、多个 seed
Step 10 汇总表格、绘图、错误分析、撰写七页正文
```

---

# 28. 双盲投稿代码打包

正式提交 supplementary code 前，不要直接上传开发仓库、Git 历史、真实数据路径、原始 checkpoint、score map 或本地实验输出。代码包提供匿名源码打包器：

```bash
./scripts/build_anonymous_supplement.sh \
  dist/RC_IRSTD_Anonymous_Supplement.zip
```

打包器会：

- 排除 `.git`、缓存、数据、checkpoint、score map 和结果目录；
- 扫描常见个人 home path 与邮箱，并支持通过 `--forbid` 增加开发仓库、账号、机构等项目特定标识；
- 在 ZIP 内生成 `ANONYMOUS_MANIFEST.json`；
- 只保留复现实验所需源码、配置、测试、文档和启动脚本。

仍需人工检查本地生成的 `arguments.json`、`protocol.json` 和 manifest，因为这些运行产物可能记录绝对数据路径或 checkpoint 路径，不应直接进入双盲附件。

---

# 29. 一句话最终定位

> **RC-IRSTD 研究的是：在此前未见的红外传感器域中，先从无标签 warm-up 数据预测结构单调的虚警风险曲线并选择样本自适应工作点，再在少量目标标注可用时，用 conformal residual 将经验适配升级为与联合预算满足率严格匹配的有限样本风险控制。**
