# RC-IRSTD：未知域零标注风险校准红外小目标检测方案

> **英文暂定题目**  
> **RC-IRSTD: Risk-Calibrated Infrared Small Target Detection under Unseen Domains**
>
> **更严谨的备用题目**  
> **Budget-Aware Infrared Small Target Detection under Unseen Domains**
>
> **方案定位**  
> 在多个有标注源域上训练风险敏感检测器；测试时仅观察未知目标域的无标签图像统计，由一个轻量校准器直接预测部署阈值和拒判状态。
>
> **推荐程度**：新颖性高、部署设定强，但理论风险最大。  
> **目标域标注需求**：0。  
> **是否具备形式化保证**：不具备无条件保证，应表述为经验性风险校准。  
> **当前代码基础**：继续使用现有 MSHNet 工程，新增多源域训练、尾部损失和阈值校准器。

---

## 0. 2026-07-14 实施审计与协议更正

### 0.1 研究定位

本方案当前以 AAAI-27 Main Technical Track 为时间敏感的 working target，同时保留通用 CCF-A 方法/视觉评审标准。当届页限和详细评审规则必须以官方后续发布页面为准。最诚实的贡献层级是：

1. **主贡献**：未知域、零目标标注、给定虚警预算的 IRSTD 部署问题与可审计协议；
2. **次贡献**：“多源尾部风险训练 + 无标签窗口统计到阈值的经验映射”；
3. **必要边界**：该映射不是无条件风险认证，新颖性在完成当前 prior-art 检索前一律标记为 `needs-literature-search`。

即使最终不超越 SOTA，也应当通过“什么时候无标签尾部可以/不可以预测安全阈值”的失败分析产生可复用结论。

### 0.2 唯一有效的协议定义

| 项目 | 统一定义 |
|---|---|
| 外层 unseen target | 最终目标域的图像可以作无标签 context，其标签不得进入 detector/calibrator 训练、调参、early stopping 或 checkpoint 选择。 |
| 内层元 episode | 只在当前外层的源域内做 leave-one-source-domain-out；不得把外层目标域当伪目标。 |
| Causal 边界 | `context_image_ids` 只生成无标签统计，`query_image_ids` 必须与其不相交，且只用 query 隐藏标签生成 oracle/评价。无可验证时间顺序的数据集只能称为 **prefix holdout**，不得声称真实在线流。 |
| Oracle | 在同时满足所有启用预算的工作点中最大化 (P_d)；若 (P_d) 并列，选最低阈值。候选必须显式包含 0/1 并加入 query 高尾唯一 score events；若 cap 后 oracle 不在完整事件后缀内，该 episode 不得标记为 verified。 |
| Reject | 由 oracle (P_d<P_{\min})、明确 OOD/不确定性规则产生，不由“扫描网格无可行点”产生；阈值 1 的空预测应总能满足非负预算。 |
| 预算 schema | 像素预算与连通域/MP 预算分别保存 `value + active mask`；可只启用其一，也可同时施加双约束，不用特殊数值伪装“缺省”。 |
| 评价空间 | 论文主结果在原始尺寸空间恢复 score map 后匹配与计数；256×256 强制 resize 仅可作诊断实验。 |
| 匹配规则 | 主规则为 8-连通、GT/预测一对一 overlap 最大基数匹配（候选按 overlap 大小确定性排序）；`fp_pixels` 是所有 GT 外的预测像素，`fp_components` 是未匹配预测连通域数。质心距离仅作旧文献兼容结果。 |

### 0.3 当前本地证据与阻塞

- 本地实际有 IRSTD-1K、NUDT-SIRST 和 NUAA-SIRST 三个域；NUAA 掩码同时存在 `<id>.png` 与 `<id>_pixels0.png` 变体，不能用 `name.*` 否则会误选 XML。本地镜像中 `Misc_111` 的掩码画布与图像尺寸不同，评估 loader 使用 nearest-neighbor 对齐到图像画布，并在 artifact/manifest 中保留原掩码尺寸与对齐标志。
- 三个域可用于代码 smoke test，但固定一个 outer target 后再做 inner LODO 时，detector 只剩一个训练域。这类运行必须标记为 `single_source_inner_smoke_not_main_result`；主实验至少需第四个去重独立域。
- 主机 Python 当前未安装 PyTorch；仓库已补齐依赖声明，并已在现有 PyTorch GPU 容器中完成三域/多 GPU 一步训练、201 张 score-map 导出和 query 高尾 sweep smoke test。这些数字只记录在 `baseline_results.md`；所有 claim-bearing 结果单元格仍保持 `TBD`，禁止从旧日志推导新结论。

---

## 1. 核心研究问题

给定多个有标注源域：

\[
\mathcal D_s
=
\{\mathcal D_1,\ldots,\mathcal D_K\},
\]

测试时输入来自未知、无标注的目标域：

\[
\mathcal D_t=\{x_i^t\}_{i=1}^{N_t}.
\]

目标是在不使用目标域标签的情况下，根据指定虚警预算 \(B\) 自动选择阈值：

\[
\max_\tau P_d^t(\tau),
\qquad
\text{s.t.}\quad
F_a^t(\tau)\leq B.
\]

核心评价：

\[
P_d@F_a=10^{-6},
\qquad
P_d@F_a=10^{-5},
\]

以及：

- 最差目标域 \(P_d\)；
- 预算满足率；
- 相对预算超限量；
- 跨域 \(P_d-F_a\) 曲线面积；
- hIoU、IoU、nIoU。

---

## 2. 核心洞察与必要限定

红外小目标在整幅图像中占比很小，因此无标签目标图像中的大多数像素属于背景。目标域预测分数的主体分布可以反映：

- 传感器噪声水平；
- 背景复杂度；
- 置信度整体漂移；
- 高响应杂波的数量和强度；
- 固定阈值在当前域中的潜在虚警风险。

但是必须明确：

\[
P_t(S>\tau)
=
P_t(S>\tau,Y=0)
+
P_t(S>\tau,Y=1).
\]

无标签条件下可观察的是总预测尾部，而真实虚警是其中的背景部分。因此：

> 无标签分数尾部不能在无附加假设时唯一确定真实虚警率。

RC-IRSTD 应被定位为：

> 利用源域元训练学习“无标签域统计到安全部署阈值”的经验映射，而不是声称在任意未知域提供 distribution-free guarantee。

论文中建议使用：

- risk-aware；
- budget-aware；
- empirical risk calibration；
- unseen-domain threshold adaptation。

应谨慎使用：

- guaranteed false-alarm control；
- distribution-free control；
- certified calibration。

---

## 3. 总体方法

```text
多个有标注源域
     │
     ▼
MSHNet + SLS Loss
     │
     ├── 背景局部峰值 Tail-CVaR
     └── 困难目标 Miss-CVaR
     │
     ▼
风险敏感检测器
     │
     ▼
Leave-One-Source-Domain-Out 元训练
     │
     ▼
无标签域统计 → 阈值校准器
     │
     ▼
目标域阈值 τ_B + 拒判状态
```

方法只保留两个主要模块：

1. **跨域尾部风险训练**；
2. **无标注域直接阈值校准器**。

---

## 4. 模块 A：跨域尾部风险训练

### 4.1 基础检测器

使用 MSHNet 作为主干，保留 SLS Loss：

\[
\mathcal L_{\mathrm{SLS}}.
\]

第一版不修改 MSHNet 编码器、解码器和多尺度头，以保证创新点集中在风险学习和部署校准。

### 4.2 背景局部峰值

对输出 logits 做 sigmoid：

\[
P=\sigma(Z).
\]

利用 GT 背景掩膜提取背景响应：

\[
P^- = P\odot(1-Y).
\]

使用局部最大池化提取高响应背景峰值：

\[
\mathcal A_d^-
=
\operatorname{LocalPeak}(P_d^-).
\]

推荐实现：

```python
peak_map = (
    background_prob
    == F.max_pool2d(
        background_prob,
        kernel_size=3,
        stride=1,
        padding=1,
    )
)
peak_scores = background_prob[peak_map]
```

训练阶段使用局部峰值代理连通候选，原因是：

- 可端到端反向传播；
- 比像素级 top-k 更接近目标级虚警；
- 避免在训练图中使用不可微连通域操作。

### 4.3 背景 Tail-CVaR

对于域 \(d\) 的背景峰值分数集合 \(\mathcal A_d^-\)，定义：

\[
R_d^-
=
\operatorname{CVaR}_{q_-}
(\mathcal A_d^-).
\]

简单实现：

\[
\operatorname{CVaR}_{q}(a)
=
\frac{1}{k}
\sum_{i\in\operatorname{TopK}(a,k)}
a_i,
\qquad
k=\lceil q|a|\rceil.
\]

建议：

\[
q_-\in\{0.001,0.005,0.01,0.05\}.
\]

### 4.4 最坏源域聚合

不要直接使用不稳定的硬最大值：

\[
\max_d R_d^-.
\]

使用平滑最大值：

\[
\mathcal L_{\mathrm{tail}}
=
\frac{1}{\gamma}
\left[
\log
\sum_{d=1}^{K}
\exp(\gamma R_d^-)
-\log K
\right].
\]

\(\gamma\) 越大，越接近最坏域风险。减去 \(\log K/\gamma\) 可避免不同源域数量带来与风险无关的常数偏移。

### 4.5 困难目标漏检损失

对每个真实目标区域 \(G_j\) 聚合目标响应：

\[
a_j^+
=
\operatorname{Pool}_{p\in G_j}P_p.
\]

可以使用 max、top-k mean 或 LogSumExp。

定义困难目标损失：

\[
\mathcal L_{\mathrm{miss}}
=
\operatorname{CVaR}_{q_+}
(
\{1-a_j^+\}_j
).
\]

这会重点抬高最低置信度的真实目标，而不是只优化平均目标。

### 4.6 总损失

\[
\boxed{
\mathcal L_{\mathrm{det}}
=
\mathcal L_{\mathrm{SLS}}
+
\lambda_{\mathrm{tail}}
\mathcal L_{\mathrm{tail}}
+
\lambda_{\mathrm{miss}}
\mathcal L_{\mathrm{miss}}
}
\]

建议初始搜索：

```text
lambda_tail ∈ {0.05, 0.1, 0.2, 0.5}
lambda_miss ∈ {0.05, 0.1, 0.2}
q_minus ∈ {0.005, 0.01, 0.05}
q_plus ∈ {0.1, 0.2, 0.5}
gamma ∈ {5, 10, 20}
```

---

## 5. 模块 B：无标签目标域阈值校准器

### 5.1 为什么直接预测阈值

原始设想同时输出温度 \(T\) 和概率阈值 \(t\)，但：

\[
\sigma(z/T)>t
\Longleftrightarrow
z>T\log\frac{t}{1-t}.
\]

二者最终等价为一个 logit 阈值，存在参数冗余。

RC-IRSTD 第一版建议直接预测：

\[
\widehat\tau_B
=
g_\phi(z_W,B),
\]

其中：

- \(z_W\)：目标域无标签窗口统计；
- \(B\)：用户指定的虚警预算；
- \(\widehat\tau_B\)：预测的部署阈值。

可额外输出拒判概率：

\[
r_W=g_\phi^{\mathrm{reject}}(z_W,B).
\]

### 5.2 无标签窗口统计

使用一个无标签目标窗口：

\[
W=\{x_1,\ldots,x_M\}.
\]

推荐窗口长度：

\[
M\in\{8,16,32,64,128\}.
\]

输入统计包括：

#### 分数分布

- 固定区间概率直方图；
- logit 直方图；
- 0.9、0.95、0.99、0.995、0.999 分位数；
- 均值、方差、偏度、峰度；
- 高于若干固定阈值的像素比例。

#### 局部峰值统计

- 局部峰值数量；
- 峰值分数直方图；
- 峰值 top-k 均值；
- 峰值空间密度；
- 峰值区域面积分布；
- 峰值聚集程度。

#### 图像统计

- 灰度均值和标准差；
- MAD；
- 局部熵；
- 梯度密度；
- Laplacian 噪声估计；
- 高频能量比例；
- 条纹方向性；
- 局部对比度分布。

#### 域偏移特征

为每个源域建立统计中心 \(\mu_d\)，计算：

\[
\delta_d
=
\|z_W-\mu_d\|_2.
\]

也可使用 Mahalanobis 距离。

### 5.3 校准器结构

建议使用小型 MLP：

```text
Input statistics
  → LayerNorm
  → MLP 256
  → GELU
  → MLP 128
  → GELU
  ├── threshold head
  └── reject head
```

参数量应远小于主检测器。

---

## 6. Leave-One-Source-Domain-Out 元训练

假设源域共有 \(K\) 个。

每个元训练 episode：

1. 选择一个源域 \(\mathcal D_k\) 作为伪目标域；
2. 检测器只能在其余源域训练；
3. 从 \(\mathcal D_k\) 抽取无标签 context 窗口；
4. 再抽取与 context 不相交的 query 集；
5. 校准器只能观察 context 的无标签统计；
6. 仅使用 query 的隐藏标签离线计算真实 \(P_d-F_a\) 曲线；
7. 生成预算 \(B\) 对应的 oracle 安全阈值；
8. 训练校准器预测该阈值。

### 6.1 Oracle 安全阈值标签

阈值网格 \(\Lambda\) 必须包含 0 和 1。定义可行集：

\[
\mathcal F_B
=
\{\tau\in\Lambda:
F_a^{\mathrm{pix}}(\tau)\leq B_p
\ \land\ 
F_a^{\mathrm{comp}}(\tau)\leq B_c\},
\]

其中只对启用的预算施加约束。Oracle 工作点定义为：

\[
\tau_B^*
=
\arg\max_{\tau\in\mathcal F_B}
\big(P_d(\tau),-\tau\big),
\]

即先最大化 \(P_d\)，并列时选更低阈值。由于 \(\tau=1\) 对应空预测，对非负预算应始终存在可行点；若实现返回空可行集，应视为曲线/网格错误并立即失败，而不是生成 reject 标签。

### 6.2 非对称阈值回归损失

阈值预测偏低更危险，因为可能导致虚警超限。定义：

\[
\mathcal L_\tau
=
w_{\mathrm{under}}
\mathbf 1[\widehat\tau<\tau^*]
(\widehat\tau-\tau^*)^2
+
w_{\mathrm{over}}
\mathbf 1[\widehat\tau\geq\tau^*]
(\widehat\tau-\tau^*)^2,
\]

其中：

\[
w_{\mathrm{under}}>w_{\mathrm{over}}.
\]

建议：

```text
w_under = 4
w_over = 1
```

也可以将阈值离散为 100–500 个有序 bin，使用 ordinal classification。

### 6.3 拒判损失

如果即使使用很高阈值也无法同时获得可用检测率和预算满足，可以定义拒判标签：

\[
y_{\mathrm{reject}}
=
\mathbf 1[
P_d(\tau_B^*)<P_{\min}
].
\]

使用 BCE 训练拒判头：

\[
\mathcal L_{\mathrm{reject}}
=
\operatorname{BCE}
(r_W,y_{\mathrm{reject}}).
\]

### 6.4 校准器总损失

\[
\mathcal L_{\mathrm{cal}}
=
\mathcal L_\tau
+
\eta
\mathcal L_{\mathrm{reject}}.
\]

---

## 7. 测试时流程

测试时不使用目标标签，也不更新 MSHNet 参数。

```text
1. 收集前 M 张无标签目标图像；
2. MSHNet 输出连续概率图；
3. 计算窗口统计 z_W；
4. 输入用户预算 B；
5. 校准器输出阈值 τ_B 和拒判概率；
6. 固定该阈值处理后续图像；
7. 可选：仅使用过去图像滚动更新窗口。
```

需要明确两种协议：

### Transductive

使用整个无标签目标测试集估计统计，再在同一集合上评价。

优点：稳定。  
缺点：不符合严格在线部署。

### Causal / Online

使用目标流的前 \(M\) 张图像估计阈值，后续图像测试；或使用只包含过去帧的滑动窗口。

最终论文应以 causal 设置为主，transductive 设置作为上界或补充。

---

## 8. 极端尾部的样本量问题

一张 \(512\times512\) 图像约有：

\[
2.62\times10^5
\]

个像素。

在 \(10^{-6}\) 尾部，一张图像的理论期望样本数仅约：

\[
0.262.
\]

而像素之间存在明显空间相关性，实际有效样本量更小。因此：

- 不应从单张图直接估计 \(10^{-6}\) 风险；
- 必须使用图像窗口；
- 应报告窗口长度敏感性；
- 应加入简单 EVT/GPD 或滚动分位数基线；
- 主要校准特征应包含候选级统计，而不是只看单像素尾部。

---

## 9. 在当前 MSHNet 仓库中的代码改动

当前工作树的实际上游为：

```text
https://github.com/ying-fu/MSHNet
```

建议新增：

```text
<repo-root>/
├── data/
│   ├── multi_source_dataset.py
│   ├── domain_balanced_sampler.py
│   └── target_window_dataset.py
│
├── losses/
│   ├── tail_cvar_loss.py
│   ├── hard_target_loss.py
│   └── smooth_worst_domain.py
│
├── calibration/
│   ├── extract_domain_statistics.py
│   ├── build_meta_episodes.py
│   ├── oracle_threshold.py
│   ├── train_threshold_calibrator.py
│   └── online_threshold_adapter.py
│
├── model/
│   ├── MSHNet.py
│   └── threshold_calibrator.py
│
├── evaluation/
│   ├── export_score_maps.py
│   ├── threshold_sweep.py
│   ├── component_matching.py
│   ├── budget_metrics.py
│   └── error_analysis.py
│
└── scripts/
    ├── train_multisource_detector.py
    ├── export_meta_data.py
    ├── train_rc_calibrator.py
    └── evaluate_rc.py
```

---

## 10. 多源域数据加载

每个样本必须返回：

```python
{
    "image": image,
    "mask": mask,
    "domain_id": domain_id,
    "dataset_name": dataset_name,
    "image_id": image_id,
    "original_size": (height, width),
}
```

不建议简单：

```python
ConcatDataset(datasets) + shuffle=True
```

因为大数据集会主导训练。

建议每个 batch 均衡采样，例如：

```text
batch size = 12
domain A = 4
domain B = 4
domain C = 4
```

最简单实现是每个域一个 DataLoader，每一步分别取一个小 batch 后拼接。

---

## 11. 数据集与协议

### 11.1 推荐数据集

- NUAA-SIRST；
- NUDT-SIRST；
- IRSTD-1K；
- SIRST-UAVB；
- RealScene-ISTD；
- NUDT-SIRST-Sea。

### 11.2 不建议直接混用

- SIRST3 与其组成数据集；
- WideIRSTD 与其公开组成数据集；
- SIRST 与 SIRST-v2，除非完成去重。

### 11.3 嵌套 leave-one-domain-out

最终目标域为 \(\mathcal D_k\) 时：

1. 检测器只使用其余 \(K-1\) 个域；
2. 校准器元训练也只使用其余 \(K-1\) 个域构造 episode；
3. \(\mathcal D_k\) 的标签不能用于 early stopping；
4. oracle threshold 只作为最终性能上界；
5. 目标域无标签图像可用于统计，但必须明确 transductive 或 causal 协议。

---

## 12. 必须比较的基线

### 检测器

- MSHNet + SLS；
- MSHNet + Focal；
- MSHNet + top-k hard negative；
- MSHNet + pixel CVaR；
- MSHNet + local-peak CVaR；
- SCTransNet 或 DNANet；
- 可获得代码时加入 RealScene 方法、S²CPNet 或 Ivan-ISTD。

### 阈值方法

1. 固定 0.5；
2. pooled-source 最优阈值；
3. worst-source 安全阈值；
4. 最近源域阈值；
5. 滚动分位数阈值；
6. EVT/GPD 尾部阈值；
7. 直接阈值校准器；
8. 目标域标签 oracle threshold。

### 测试时适配

- Source-only；
- BatchNorm 统计更新；
- Tent；
- EATA；
- 普通温度缩放。

Tent/EATA 是补充基线，不应替代风险预算相关基线。

---

## 13. 评价指标

### 检测指标

- \(P_d@F_a^{\mathrm{pix}}=10^{-6}\)；
- \(P_d@F_a^{\mathrm{pix}}=10^{-5}\)；
- \(P_d@FA^{\mathrm{comp/MP}}\leq B_c\)；
- 最差域 \(P_d\)；
- \(P_d-F_a\) 曲线面积；
- hIoU、IoU、nIoU。

### 校准指标

预算满足率：

\[
\mathrm{BSR}(B)
=
\frac{1}{K}
\sum_k
\mathbf 1[
F_{a,k}(\widehat\tau_B)\leq B
].
\]

相对超限量：

\[
\mathrm{Excess}(B)
=
\frac{1}{K}
\sum_k
\frac{
[F_{a,k}(\widehat\tau_B)-B]_+
}{
B
}.
\]

阈值误差：

\[
|\widehat\tau_B-\tau_B^*|.
\]

拒判评价：

- AUROC；
- AUPRC；
- Coverage；
- 非拒判样本上的预算满足率。

---

## 14. 关键实验

| 实验 | 回答的问题 |
|---|---|
| 跨数据集阈值矩阵 | 固定阈值是否在未知域失效 |
| Source vs Oracle | 性能下降有多少来自阈值错误 |
| Tail-CVaR 消融 | 是否压低最高背景响应 |
| Pixel top-k vs local-peak CVaR | 候选级代理是否更有效 |
| Hard miss loss | 是否改善低虚警区目标召回 |
| 校准器 vs 分位数 | 学习映射是否超过简单规则 |
| 不同预算输入 | 一个校准器能否适配多预算 |
| 不同窗口长度 | 极端尾部估计是否稳定 |
| Causal vs transductive | 在线部署代价有多大 |
| 目标污染实验 | 目标数量增加时背景假设何时失效 |
| 噪声与分辨率扰动 | 对传感器变化是否稳定 |
| 错误类型分析 | 云边、海浪、建筑热源等虚警来源 |
| 三个随机种子 | 结果是否稳定 |

### 14.1 Claim–evidence 对齐与停止条件

| 主要 claim | 评审问题 | 必要证据 | 主指标 | 停止/转向条件 | 状态 |
|---|---|---|---|---|---|
| 未知域会使固定/源域阈值违反虚警预算 | 这是真实新问题还是人为协议？ | 多源→多目标阈值迁移矩阵，fixed/source/worst-source/oracle | (P_d@F_a)、BSR、Excess、原始 FP 计数 | 若 oracle 也不能恢复低虚警 (P_d)，转表征学习 | TBD |
| local-peak Tail-CVaR 能减少未见域高分背景峰且不破坏目标 | 收益是否只来自额外损失强度？ | SLS、pixel top-k、local-peak CVaR、Tail+Miss 等预算消融；尾部分布和失败案例 | 背景峰分位数、(P_d@F_a)、IoU/nIoU | 若只降低整体分数且 oracle (P_d) 不升，删除模块 | TBD |
| 无标签统计可预测预算工作点 | 小 MLP 是否只记住数据集 ID？ | 严格 held-out pseudo-target；线性/随机森林诊断；nearest-source、rolling quantile、EVT/GPD 基线 | BSR、Excess、threshold MAE、coverage 与 non-reject (P_d) | 若不稳定超过 rolling quantile 或靠大量 reject 取得 BSR，转 ConFA | TBD |
| prefix context 可支撑零标注部署 | 是否依赖 transductive 看完全测试集？ | context/query 分离；M=8/16/32/64/128；目标污染、噪声、分辨率应力 | BSR、Excess、coverage、置信区间 | 若只有 transductive 有效，收缩为离线批处理设定 | TBD |

> 本表不包含任何预设结果。`TBD` 只能由实际实验、协议一致的可验证公开数字或明确失败记录替换。

---

## 15. 两天可行性诊断

先不训练校准器，只做：

1. 在一个源域训练 MSHNet；
2. 直接测试 3–4 个数据集；
3. 导出连续概率图；
4. 计算不同域的 \(P_d-F_a\) 曲线；
5. 比较固定阈值、源域阈值和目标域 oracle 阈值；
6. 画真实目标候选与背景候选分数分布；
7. 测试滚动分位数阈值。

继续 RC-IRSTD 的条件：

- 多数目标域的 oracle 阈值明显不同；
- oracle threshold 能显著恢复低虚警区 \(P_d\)；
- 滚动分位数有帮助但预算超限仍明显；
- 无标签统计和 oracle 阈值之间存在可预测关系；
- 至少三个目标域显示一致趋势。

转向表示学习的条件：

- oracle threshold 也无法恢复 \(P_d\)；
- 低虚警区没有真实目标候选；
- 目标和背景候选分数严重重叠。

转向 ConFA 的条件：

- 无标签阈值校准器与简单分位数相当；
- 未见目标域预算超限率仍很高；
- 留一源域元训练不能泛化。

---

## 16. 消融设计

| 版本 | SLS | Tail-CVaR | Miss-CVaR | Direct calibrator | Reject |
|---|---:|---:|---:|---:|---:|
| Baseline | ✓ |  |  |  |  |
| Risk training A | ✓ | ✓ |  |  |  |
| Risk training B | ✓ | ✓ | ✓ |  |  |
| Calibration only | ✓ |  |  | ✓ |  |
| Full RC | ✓ | ✓ | ✓ | ✓ |  |
| Full RC + reject | ✓ | ✓ | ✓ | ✓ | ✓ |

校准器特征消融：

- 仅分数直方图；
- 直方图 + 分位数；
- 加局部峰值；
- 加图像噪声；
- 加源域距离；
- 全部特征。

---

## 17. 论文贡献写法

建议贡献：

1. 提出未知域、固定虚警预算、零目标标签的 IRSTD 部署问题；
2. 提出最坏源域局部峰值 Tail-CVaR 训练，专门压低高置信背景候选；
3. 提出 leave-one-source-domain-out 的无标签阈值元校准器；
4. 建立预算满足率、预算超限、窗口长度和目标污染等风险评测协议。

不建议声称：

- 无标签条件下精确识别背景尾部；
- 任意未知域都满足 \(F_a\leq B\)；
- 方法具有 conformal 或 distribution-free guarantee；
- 预测阈值等价于真实风险认证。

---

## 18. 预期主表

| Method | Target labels | \(P_d@10^{-6}\) | \(P_d@10^{-5}\) | BSR ↑ | Excess ↓ | Worst-domain \(P_d\) ↑ | AUC\(_{Pd-Fa}\) ↑ | hIoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0 |  |  |  |  |  |  |  |
| Source threshold | 0 |  |  |  |  |  |  |  |
| Rolling quantile | 0 |  |  |  |  |  |  |  |
| EVT/GPD | 0 |  |  |  |  |  |  |  |
| Tail-risk detector | 0 |  |  |  |  |  |  |  |
| RC-IRSTD | 0 |  |  |  |  |  |  |  |
| Oracle | All |  |  | — | — |  |  |  |

---

## 19. 实现顺序

### 阶段 0：诊断基础设施

- 连续分数导出；
- 精确阈值扫描；
- 候选匹配；
- \(P_d-F_a\) 曲线；
- oracle 阈值分析。

### 阶段 1：风险敏感检测器

- 多源域 loader；
- domain-balanced batch；
- local-peak Tail-CVaR；
- hard-target Miss-CVaR；
- 最坏域平滑聚合。

### 阶段 2：元数据构建

- 留一源域 episode；
- 无标签统计抽取；
- 多预算 oracle threshold 标签；
- 不同窗口长度。

### 阶段 3：阈值校准器

- MLP threshold head；
- 非对称回归；
- reject head；
- causal 在线测试。

### 阶段 4：第二 backbone 和完整消融

- SCTransNet 或 DNANet；
- 传统分位数和 EVT；
- 噪声、模糊、分辨率扰动；
- 错误类型分析。

---

## 20. 主要风险与应对

### 风险 1：无标签风险不可识别

应对：

- 明确方案是经验校准；
- 使用多源域元训练建立结构假设；
- 输出拒判状态；
- 报告预算超限而不是只报平均性能；
- 将 RC-v2 的少样本认证作为增强路线。

### 风险 2：校准器过拟合数据集 ID

应对：

- 在统计特征上训练，不输入数据集名称；
- 使用嵌套 leave-one-domain-out；
- 做统计扰动增强；
- 限制模型容量；
- 增加合成噪声与背景扰动 episode。

### 风险 3：窗口中真实目标污染尾部

应对：

- 使用局部峰值形态统计；
- 增加目标数量污染实验；
- 对极高分峰值做稳健截断；
- 使用中高分段而不是只看最极端 top-1；
- 必要时触发 reject。

### 风险 4：直接阈值回归无法适配多个预算

应对：

- 将预算 \(B\) 作为输入；
- 使用对数预算 \(\log_{10}B\)；
- 训练时随机采样多个预算；
- RC-v2 中升级为完整风险曲线预测。

---

## 21. 推荐论文结构

1. Introduction  
2. Related Work  
   - IRSTD and Low-False-Alarm Detection  
   - Cross-Domain IRSTD  
   - Test-Time Adaptation and Risk Calibration  
3. Problem Formulation  
4. RC-IRSTD  
   - Worst-Domain Tail-Risk Learning  
   - Unlabeled Domain Statistics  
   - Leave-One-Domain-Out Threshold Meta-Calibration  
   - Rejection Mechanism  
5. Experiments  
6. Limitations and Identifiability Discussion  
7. Conclusion  

---

## 22. 参考文献与代码

1. Liu et al., **Infrared Small Target Detection with Scale and Location Sensitivity**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html

2. Pang et al., **Rethinking Evaluation of Infrared Small Target Detection**, NeurIPS 2025 Datasets and Benchmarks.  
   https://arxiv.org/abs/2509.16888

3. Lu et al., **Rethinking Generalizable Infrared Small Target Detection: A Real-scene Benchmark and Cross-view Representation Learning**.  
   https://arxiv.org/abs/2504.16487

4. **Rethinking Representations for Cross-Domain Infrared Small Target Detection: A Generalizable Perspective from the Frequency Domain**.  
   https://arxiv.org/abs/2604.01934

5. Li et al., **Ivan-ISTD: Rethinking Cross-domain Heteroscedastic Noise Perturbations in Infrared Small Target Detection**.  
   https://arxiv.org/abs/2510.12241

6. Ciocarlan et al., **A Contrario Paradigm for YOLO-based Infrared Small Target Detection**.  
   https://arxiv.org/abs/2402.02288

7. Wang et al., **Tent: Fully Test-Time Adaptation by Entropy Minimization**.  
   https://openreview.net/forum?id=uXl3bZLkr3c

8. Niu et al., **Efficient Test-Time Model Adaptation without Forgetting**.  
   https://proceedings.mlr.press/v162/niu22a.html

9. 当前开发仓库。  
   https://github.com/ying-fu/MSHNet

10. PyIRSTDMetrics。  
    https://github.com/lartpang/PyIRSTDMetrics

---

## 23. 一句话定位

> **RC-IRSTD 研究的是：在完全没有目标域标签的情况下，能否根据未知域的无标签分数和成像统计，经验性地预测满足指定虚警预算的部署阈值。**


---

# 第二部分：基于当前 MSHNet 仓库的代码修改与执行步骤

> **适用仓库**：`https://github.com/ying-fu/MSHNet`  
> **原则**：不把 MSHNet 迁回 BasicIRSTD；继续以当前仓库为主工程。先建立三种方案共用的评估基础设施，再在独立 Git 分支开发各自模块。  
> **重要说明**：下面的文件名和命令是一套推荐实现；现有 `main.py`、`train.py`、`test.py` 保留用于复现原始 MSHNet，不应一开始就全部重写。
> **工作树安全**：当前存在用户未提交修改和本地数据/权重。任何分支或提交命令前必须先检查 `git status --short`；禁止直接执行 `git add .`。

## A. 三个方案共用的分支结构

同时做三个方案时，不要复制三份仓库。建议采用以下分支：

```bash
# 0. 保存当前可运行基线
git checkout main
git checkout -b backup/mshnet-baseline

# 1. 共用评估基础设施
git checkout main
git checkout -b feat/risk-eval-core

# 完成 score map 导出、阈值扫描和预算指标后，只 stage 本次源码
git add evaluation data_ext tests .gitignore
git commit -m "feat: add cross-domain score export and budget evaluation"

# 2. ConFA 独立分支
git checkout -b exp/confa-irstd

# 3. 风险敏感检测器公共分支
git checkout feat/risk-eval-core
git checkout -b feat/tail-risk-detector

# 完成多源域训练和 Tail-CVaR 后，只 stage 本次源码
git add losses data_ext/multi_source_dataset.py data_ext/balanced_domain_loader.py scripts/train_multisource_tail.py tests
git commit -m "feat: add multi-source tail-risk detector"

# 4. RC 分支
git checkout -b exp/rc-irstd

# 5. RC-v2 分支
git checkout feat/tail-risk-detector
git checkout -b exp/rc-irstd-v2
```

推荐依赖关系：

```text
main
 └── feat/risk-eval-core
      ├── exp/confa-irstd
      └── feat/tail-risk-detector
           ├── exp/rc-irstd
           └── exp/rc-irstd-v2
```

这样三个方案共享：

- 数据集读取；
- 连续概率图导出；
- 阈值扫描；
- 目标匹配；
- \(P_d\)、像素虚警和连通域虚警；
- 实验日志格式。

---

## B. 当前仓库必须先修正的五个问题

当前仓库已经可以读取 BasicIRSTD 风格的：

```text
datasets/<name>/
├── images/
├── masks/
└── img_idx/
```

但用于风险研究前仍需修正以下问题。

### 问题 1：测试 loader 只返回 `image, mask`

风险评估还需要：

- `image_id`；
- `dataset_name`；
- 原始高宽；
- resize 后高宽；
- 可选的序列 ID。

### 问题 2：现有 `PD_FA` 不能用于精确低虚警曲线

当前 `utils/metric.py` 中的 `PD_FA` 对原始 logits 使用 0–255 阈值，并且主程序只打印第一个工作点。该实现可保留用于对照原仓库结果，但不能作为 RC/ConFA 的最终评价器。

### 问题 3：验证图像统一 resize 到 `base_size × base_size`

第一阶段诊断可以继续使用 256×256；最终论文应增加原比例 pad 推理，否则“每百万原始像素虚警”的物理含义会改变。

### 问题 4：`main.py` 同时承担训练、测试和保存

不要继续把候选提取、风险校准和曲线预测全部塞入 `main.py`。新增独立脚本，保持原始复现入口可运行。

### 问题 5：原训练入口用官方 test split 选最佳 checkpoint

当前 `main.py` 每个 epoch 读取 `test.txt`/`img_idx/test*.txt` 并按其 mIoU 保存 best weight。这个入口只能用于兼容原仓库复现，不得用于 RC 论文实验。RC 训练脚本必须使用源域内部 validation，或预先固定训练 epoch 并保存 last checkpoint；任何外层 unseen target 都不能参与选择。

---

## C. 共用目录结构

在仓库根目录新增（下面的 `<repo-root>` 就是当前 `RC-IRSTD/`，不再创建一层 `M/`）：

```text
<repo-root>/
├── evaluation/
│   ├── __init__.py
│   ├── export_score_maps.py
│   ├── component_matching.py
│   ├── threshold_sweep.py
│   ├── budget_metrics.py
│   └── operating_point.py
│
├── data_ext/
│   ├── __init__.py
│   ├── eval_dataset.py
│   ├── dataset_meta.py
│   └── split_utils.py
│
├── scripts/
│   ├── smoke_test.sh
│   ├── export_cross_domain_scores.sh
│   └── evaluate_cross_domain.sh
│
└── outputs/
    ├── score_maps/
    ├── curves/
    └── tables/
```

把 `outputs/` 加入 `.gitignore`：

```gitignore
outputs/
*.npy
*.npz
*.csv
*.pth
*.pkl
repro_runs/
__pycache__/
*.pyc
.DS_Store
```

---

## D. 共用 Step 0：冻结基线结果

### 目标

确认修改前的 MSHNet 能正常训练、加载和测试。

### 命令

```bash
python train.py \
  --dataset-dir datasets/NUDT-SIRST \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --save-dir repro_runs/smoke
```

随后：

```bash
python test.py \
  --dataset-dir datasets/NUDT-SIRST \
  --weight-path repro_runs/smoke/<run-name>/weight.pkl \
  --num-workers 0
```

### 产物

```text
repro_runs/smoke/<run-name>/
├── weight.pkl
├── checkpoint.pkl
└── metric.log
```

### 验收条件

- loss 为有限数值；
- checkpoint 可重新加载；
- 测试不报 shape 错误；
- 当前基线 IoU/Pd/Fa 被记录到 `baseline_results.md`；
- 此后所有新代码都不能破坏以上命令。

---

## E. 共用 Step 1：新增带元数据的评估 Dataset

不要直接破坏现有 `IRSTD_Dataset` 返回格式。新增：

```text
data_ext/eval_dataset.py
```

推荐接口：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class SampleMeta:
    image_id: str
    dataset_name: str
    original_hw: Tuple[int, int]
    input_hw: Tuple[int, int]

    def to_collatable(self):
        return {
            "image_id": self.image_id,
            "dataset_name": self.dataset_name,
            "original_hw": torch.tensor(self.original_hw),
            "input_hw": torch.tensor(self.input_hw),
        }


class IRSTDEvalDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        base_size: int = 256,
    ) -> None:
        self.root = Path(dataset_dir)
        self.dataset_name = self.root.name
        self.base_size = base_size

        split_files = sorted(
            (self.root / "img_idx").glob("test*.txt")
        )
        if split_files:
            split_file = split_files[0]
        else:
            split_file = self.root / "test.txt"
            if not split_file.exists():
                raise FileNotFoundError(
                    f"No test split found under {self.root}"
                )

        self.names = [
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [.485, .456, .406],
                [.229, .224, .225],
            ),
        ])

    def _resolve(
        self,
        folder: str,
        image_id: str,
    ) -> Path:
        matches = sorted(
            (self.root / folder).glob(f"{image_id}.*")
        )
        if not matches:
            raise FileNotFoundError(
                f"Missing {folder} file for {image_id}"
            )
        return matches[0]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:
        image_id = Path(self.names[index]).stem
        image = Image.open(
            self._resolve("images", image_id)
        ).convert("RGB")
        mask = Image.open(
            self._resolve("masks", image_id)
        ).convert("L")

        original_hw = (image.height, image.width)

        image = image.resize(
            (self.base_size, self.base_size),
            Image.BILINEAR,
        )
        mask = mask.resize(
            (self.base_size, self.base_size),
            Image.NEAREST,
        )

        image_tensor = self.to_tensor(image)
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        meta = SampleMeta(
            image_id=image_id,
            dataset_name=self.dataset_name,
            original_hw=original_hw,
            input_hw=(
                self.base_size,
                self.base_size,
            ),
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            # 实际实现必须转为普通 dict/tensor，
            # 否则 default DataLoader collate 不支持 dataclass。
            "meta": meta.to_collatable(),
        }
```

### 验收条件

运行一个 DataLoader 后能打印：

```text
image: [1, 3, 256, 256]
mask:  [1, 1, 256, 256]
image_id: ...
dataset_name: ...
original_hw: (...)
```

---

## F. 共用 Step 2：导出连续概率图

新增：

```text
evaluation/export_score_maps.py
```

核心原则：

```python
prob = torch.sigmoid(logits)
```

而不是：

```python
binary = logits > 0
```

推荐每张图保存一个压缩 `.npz`：

```python
np.savez_compressed(
    output_path,
    prob=prob.astype(np.float32),
    mask=mask.astype(np.uint8),
    image_id=image_id,
    dataset_name=dataset_name,
    original_hw=np.asarray(
        original_hw,
        dtype=np.int32,
    ),
    input_hw=np.asarray(
        input_hw,
        dtype=np.int32,
    ),
)
```

命令接口建议：

```bash
python -m evaluation.export_score_maps \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path repro_runs/NUDT/weight.pkl \
  --output-dir outputs/score_maps/NUDT_to_IRSTD1K \
  --base-size 256 \
  --device cuda
```

### 输出目录

```text
outputs/score_maps/NUDT_to_IRSTD1K/
├── manifest.json
├── image_0001.npz
├── image_0002.npz
└── ...
```

`manifest.json` 至少记录：

```json
{
  "source_dataset": "NUDT-SIRST",
  "target_dataset": "IRSTD-1K",
  "weight_path": "...",
  "base_size": 256,
  "score_type": "sigmoid_probability",
  "num_images": 1000
}
```

### 验收条件

随机读取 10 张：

```python
assert np.isfinite(prob).all()
assert prob.min() >= 0.0
assert prob.max() <= 1.0
assert set(np.unique(mask)).issubset({0, 1})
assert prob.shape == mask.shape
```

---

## G. 共用 Step 3：实现目标匹配和预算指标

新增：

```text
evaluation/component_matching.py
```

至少支持两种匹配规则：

1. **Overlap matching**：预测连通域与 GT 有交集；
2. **Centroid-distance matching**：用于兼容旧 IRSTD 文献。

推荐主结果使用固定且公开的 matching rule，旧版质心规则作为兼容结果。

返回结构建议：

```python
@dataclass
class MatchResult:
    num_gt: int
    num_tp_objects: int
    num_fp_components: int
    num_fp_pixels: int
    matched_pairs: list[tuple[int, int]]
```

定义指标：

```python
pd = (
    total_tp_objects
    / max(total_gt_objects, 1)
)

fa_pixel = (
    total_fp_pixels
    / max(total_pixels, 1)
)

fa_component_mp = (
    total_fp_components
    / max(
        total_pixels / 1_000_000.0,
        1e-12,
    )
)
```

新增：

```text
evaluation/threshold_sweep.py
```

阈值建议：

```python
thresholds = np.unique(np.concatenate([
    np.asarray([0.0]),
    np.linspace(0.00, 0.90, 91),
    np.linspace(0.90, 0.99, 181),
    np.linspace(0.99, 0.999, 181),
    np.linspace(0.999, 0.99999, 201),
    np.asarray([1.0]),
]))
```

网格版本必须写入 curve/episode manifest。极低虚警主结果应进一步用唯一 score 排序或自适应尾部网格复核，避免固定网格漏掉工作点。

输出：

```text
outputs/curves/NUDT_to_IRSTD1K.csv
```

列：

```text
threshold,pd,fa_pixel,fa_component_mp,
tp_objects,gt_objects,fp_components,
fp_pixels,total_pixels
```

### 验收条件

- 随阈值升高，`fa_pixel` 单调不增；
- 阈值 1.0 附近的预测接近空集；
- 阈值 0.0 附近的虚警很高；
- 同一 score map 重复运行得到完全相同结果。

---

## H. 共用 Step 4：实现预算工作点选择

新增：

```text
evaluation/operating_point.py
```

```python
def select_operating_point(
    rows,
    pixel_budget=None,
    component_budget=None,
):
    feasible = rows.copy()

    if pixel_budget is not None:
        feasible = feasible[
            feasible["fa_pixel"]
            <= pixel_budget
        ]

    if component_budget is not None:
        feasible = feasible[
            feasible["fa_component_mp"]
            <= component_budget
        ]

    if len(feasible) == 0:
        return None

    return feasible.sort_values(
        ["pd", "threshold"],
        ascending=[False, True],
    ).iloc[0]
```

主预算建议同时支持：

```text
pixel Fa: 1e-6, 1e-5
component Fa/MP: 0.5, 1, 5, 10
```

### 验收条件

生成首个诊断表：

| Source | Target | Fixed-0.5 Pd | Source-threshold Pd | Oracle Pd | Oracle threshold |
|---|---|---:|---:|---:|---:|
| NUDT | IRSTD-1K |  |  |  |  |
| NUDT | NUAA |  |  |  |  |
| IRSTD-1K | NUAA |  |  |  |  |

只有完成这一步后，才进入三个方案的专用开发。

---

## I. 共用 Step 5：三方案可行性闸门

必须先回答：

1. 不同目标域的 oracle threshold 是否明显不同？
2. oracle threshold 是否能在相同预算下显著恢复 \(P_d\)？
3. fixed/source threshold 是否频繁违反预算？
4. 目标候选和背景候选在高分区是否仍有一定排序空间？

建议至少满足：

```text
- 三个目标域中至少两个存在明显阈值漂移；
- Oracle Pd 比 source-threshold Pd 提升 ≥ 3 个百分点；
- 固定阈值的预算超限具有跨域差异；
- 曲线不是完全由表征失效主导。
```

若 oracle threshold 也无法恢复性能，应先加强表示学习，而不是继续堆校准器。

---

## J. 共用实验日志规范

每次实验写入：

```text
outputs/experiments/<experiment_id>/
├── config.json
├── git_commit.txt
├── command.txt
├── metrics.json
├── curve.csv
└── notes.md
```

`config.json` 至少包含：

```json
{
  "seed": 42,
  "source_domains": [],
  "target_domain": "",
  "base_size": 256,
  "matching_rule": "overlap",
  "pixel_budgets": [1e-6, 1e-5],
  "component_budgets": [1.0, 5.0],
  "checkpoint": ""
}
```

任何结果没有记录 checkpoint、split、seed 和阈值选择来源，都不进入论文主表。

---

# 第三部分：RC-IRSTD 专用代码修改与完整步骤

## 1. RC-IRSTD 的最小落地版本

建议把 RC 分成两段完成：

```text
阶段 A：多源域 Tail-CVaR 风险敏感检测器
阶段 B：无标签统计 → 单一部署阈值
```

第一版不要同时预测温度和阈值；直接预测阈值，并将预算 \(\log_{10}B\) 作为输入。

---

## 2. RC 文件级改动清单

在 `feat/tail-risk-detector` 和 `exp/rc-irstd` 分支新增：

```text
<repo-root>/
├── data_ext/
│   ├── multi_source_dataset.py
│   ├── balanced_domain_loader.py
│   └── target_window_dataset.py
│
├── losses/
│   ├── __init__.py
│   ├── local_peak_cvar.py
│   ├── hard_target_loss.py
│   └── smooth_worst_domain.py
│
├── model/
│   └── threshold_calibrator.py
│
├── rc/
│   ├── __init__.py
│   ├── domain_statistics.py
│   ├── oracle_threshold.py
│   ├── build_meta_episodes.py
│   ├── meta_dataset.py
│   ├── train_calibrator.py
│   └── online_adapter.py
│
└── scripts/
    ├── train_multisource_tail.py
    ├── build_rc_episodes.sh
    ├── train_rc_calibrator.sh
    └── test_rc.sh
```

---

## 3. Step R1：多源域 Dataset 与均衡 batch

新增：

```text
data_ext/multi_source_dataset.py
```

```python
from typing import Dict
import torch
from torch.utils.data import Dataset


class DomainDataset(Dataset):
    def __init__(
        self,
        dataset,
        domain_id: int,
        domain_name: str,
    ) -> None:
        self.dataset = dataset
        self.domain_id = domain_id
        self.domain_name = domain_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:
        image, mask = self.dataset[index]
        return {
            "image": image,
            "mask": mask,
            "domain_id": torch.tensor(
                self.domain_id,
                dtype=torch.long,
            ),
            "domain_name": self.domain_name,
        }
```

第一版使用每域一个 DataLoader，每一步拼接：

```python
images = torch.cat(
    [batch["image"] for batch in batches],
    dim=0,
)
masks = torch.cat(
    [batch["mask"] for batch in batches],
    dim=0,
)
domain_ids = torch.cat(
    [batch["domain_id"] for batch in batches],
    dim=0,
)
```

建议：

```text
3 个源域，每域 batch=2 或 4
```

### 验收条件

每个 step 的域样本数一致。

---

## 4. Step R2：局部峰值 Tail-CVaR

新增：

```text
losses/local_peak_cvar.py
```

```python
import math
import torch
import torch.nn.functional as F


def top_fraction_mean(
    values: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    values = values.flatten()
    if values.numel() == 0:
        return values.sum() * 0.0

    k = max(
        1,
        int(math.ceil(
            fraction * values.numel()
        )),
    )
    return torch.topk(
        values,
        k=k,
    ).values.mean()


def local_background_peak_scores(
    logits: torch.Tensor,
    masks: torch.Tensor,
    kernel_size: int = 3,
    min_score: float = 0.05,
) -> list[torch.Tensor]:
    prob = torch.sigmoid(logits)
    background = prob * (1.0 - masks)

    pooled = F.max_pool2d(
        background,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    peak_mask = (
        (background >= pooled - 1e-7)
        & (background >= min_score)
        & (masks < 0.5)
    )

    return [
        background[b][peak_mask[b]]
        for b in range(
            background.shape[0]
        )
    ]
```

按域聚合后使用平滑最大：

```python
loss_tail = (
    torch.logsumexp(
        gamma * domain_risks,
        dim=0,
    )
    - math.log(domain_risks.numel())
) / gamma
```

不要遗漏 `min_score`，否则零值平台会被当作大量峰值。同一平台也只能保留一个确定性代表峰；需用合成 plateau 单测验证，不得把整片相等值全部计为候选。

---

## 5. Step R3：困难目标 Miss-CVaR

新增：

```text
losses/hard_target_loss.py
```

GT 连通域提取可以离线或使用 `skimage.measure.label`。目标区域的预测分数聚合必须保留梯度。

推荐每个 GT 目标使用 top-25% mean，再对最难目标做 CVaR：

```python
miss = 1.0 - object_scores
loss_miss = top_fraction_mean(
    miss,
    fraction=0.2,
)
```

### 验收条件

日志包含：

```text
loss_sls
loss_tail
loss_miss
tail_risk_domain_0
tail_risk_domain_1
...
```

---

## 6. Step R4：多源域风险训练脚本

新增：

```text
scripts/train_multisource_tail.py
```

核心：

```python
aux_masks, logits = model(
    images,
    warm_flag,
)

loss_sls = sls_loss(
    logits,
    masks,
    warm_epoch,
    epoch,
)

domain_risks = domain_tail_risks(
    logits,
    masks,
    domain_ids,
    q=args.tail_q,
)

loss_tail = smooth_max(
    domain_risks,
    gamma=args.tail_gamma,
)

loss_miss = hard_target_miss_loss(
    logits,
    masks,
    q=args.miss_q,
)

loss = (
    loss_sls
    + args.lambda_tail * loss_tail
    + args.lambda_miss * loss_miss
)
```

辅助尺度仍使用原 SLS；风险损失先只加最终输出。

命令：

```bash
python -m scripts.train_multisource_tail \
  --source-dirs \
    datasets/NUAA-SIRST \
    datasets/NUDT-SIRST \
    datasets/IRSTD-1K \
  --batch-per-domain 2 \
  --epochs 400 \
  --lr 0.05 \
  --lambda-tail 0.1 \
  --lambda-miss 0.1 \
  --tail-q 0.01 \
  --miss-q 0.2 \
  --tail-gamma 10 \
  --save-dir repro_runs/rc_tail
```

必须训练：

```text
A. SLS only
B. SLS + pixel top-k
C. SLS + local-peak Tail-CVaR
D. SLS + Tail-CVaR + Miss-CVaR
```

---

## 7. Step R5：严格嵌套 LODO 元 episode

必须先固定外层 unseen target，再于剩余源域中构造元 episode。例如外层目标为 D，可用源域只有 A、B、C：

```text
Detector BC → pseudo-target A
Detector AC → pseudo-target B
Detector AB → pseudo-target C
```

严禁构造 `Detector ABC → pseudo-target D`，因为这会把最终 D 的隐藏标签泄漏给 calibrator。只有当 A/B/C/D 全部是源域、另有独立外部目标 E 时，才能让四个源域互相作 pseudo-target。

新增：

```text
rc/build_meta_episodes.py
```

每个 episode：

```json
{
  "pseudo_target": "IRSTD-1K",
  "context_image_ids": ["...", "..."],
  "query_image_ids": ["...", "..."],
  "window_size": 32,
  "statistics": [],
  "pixel_budget": 1e-6,
  "pixel_budget_active": true,
  "component_budget": null,
  "component_budget_active": false,
  "oracle_threshold": 0.973,
  "oracle_pd": 0.86,
  "oracle_fa_pixel": 0.00000094,
  "oracle_fa_component_mp": 0.0,
  "p_min": 0.2,
  "reject_label": 0
}
```

Episode builder 必须断言 `context_image_ids` 与 `query_image_ids` 不相交。Budget 的 `log10` 变换只在数值特征层执行，JSON 保留原始单位以便审计。

窗口：

```text
size ∈ {8, 16, 32, 64}
budget ∈ {1e-4, 1e-5, 1e-6}
```

---

## 8. Step R6：无标签域统计

新增：

```text
rc/domain_statistics.py
```

第一版统计：

```text
- 所有像素分数直方图：32 bins
- 分数分位数：7
- 局部峰值直方图：32 bins
- 局部峰值分位数：7
- 每 MP 峰值数：1
- 灰度 mean/std/MAD：3
- 梯度 mean/q95：2
- Laplacian 噪声：1
- 高频能量比例：1
- 与源域统计中心的固定维聚合：available、L2 min/mean/max/std、cosine min，共 6
```

使用固定维聚合距离而不直接拼接 K 个距离，以保证不同 LODO fold 的输入维度一致。所有输入只能来自无标签图像与预测，不使用目标 mask；代码接口不应接受 mask 参数。

---

## 9. Step R7：生成 oracle threshold 标签

新增：

```text
rc/oracle_threshold.py
```

```python
def oracle_safe_threshold(
    curve_df,
    pixel_budget=None,
    component_budget=None,
    p_min=0.2,
):
    feasible = curve_df.copy()

    if pixel_budget is not None:
        feasible = feasible[
            feasible.fa_pixel
            <= pixel_budget
        ]

    if component_budget is not None:
        feasible = feasible[
            feasible.fa_component_mp
            <= component_budget
        ]

    if len(feasible) == 0:
        raise ValueError(
            "No feasible point: the curve must include threshold=1.0"
        )

    best = feasible.sort_values(
        ["pd", "threshold"],
        ascending=[False, True],
    ).iloc[0]

    reject = float(best.pd) < p_min
    return {
        "threshold": float(best.threshold),
        "pd": float(best.pd),
        "fa_pixel": float(best.fa_pixel),
        "fa_component_mp": float(best.fa_component_mp),
        "reject": reject,
    }
```

---

## 10. Step R8：训练直接阈值校准器

新增：

```text
model/threshold_calibrator.py
```

```python
import torch
import torch.nn as nn


class ThresholdCalibrator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )
        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.reject_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, x):
        hidden = self.encoder(x)
        threshold = self.threshold_head(
            hidden
        ).squeeze(-1)
        reject_logit = self.reject_head(
            hidden
        ).squeeze(-1)
        return threshold, reject_logit
```

输入：

```text
[domain statistics,
 log10(pixel_budget_or_1), pixel_budget_active,
 log10(component_budget_or_1), component_budget_active]
```

其中预算未启用时数值占位为 1（因此 `log10=0`），真正含义由对应 active mask 决定。训练集特征标准化参数只能在当前内层训练 pseudo-target 上拟合。

非对称损失：

```python
def asymmetric_threshold_loss(
    pred,
    target,
    under_weight=4.0,
):
    error = pred - target
    weight = torch.where(
        error < 0,
        torch.full_like(
            error,
            under_weight,
        ),
        torch.ones_like(error),
    )
    return (
        weight * error.square()
    ).mean()
```

### 验收条件

在完全留出的伪目标域：

- BSR 优于 rolling quantile；
- Excess 下降；
- threshold MAE 优于 nearest-source；
- 不靠大规模 reject 获得虚假安全。

---

## 11. Step R9：未知目标域零标注测试

以“前 \(M\) 张无标签图像估计阈值，后续图像测试”为主协议：

```bash
python -m rc.online_adapter \
  --manifest outputs/scores/realscene/manifest.json \
  --calibrator-checkpoint outputs/rc/threshold_calibrator.pt \
  --target-domain RealScene-ISTD \
  --context-size 32 \
  --pixel-budget 1e-6 \
  --component-budget 1.0 \
  --output outputs/rc/realscene_zero_label.json

python -m evaluation.evaluate_adapter_output \
  --adapter-output outputs/rc/realscene_zero_label.json \
  --score-manifest outputs/scores/realscene/manifest.json \
  --output outputs/rc/realscene_query_metrics.json
```

第一条命令只读取无标签 context score maps，输出阈值/reject 及与 query IDs 的 SHA-256 绑定；第二条命令是独立的 label-using offline replay，仅在核验 target domain、detector hash、manifest hash 和 query 顺序后计算 Pd/FA。若 score manifest 的顺序不是经证实的时间流，结果必须报告为 `prefix_holdout`；只有显式提供 `--temporal-order-verified` 时才能报告 `causal_online`。

禁止使用目标标签选择：

- 阈值；
- 窗口长度；
- 特征组合；
- checkpoint。

---

## 12. RC 完整步骤清单

代码完成不等于实验证据完成。以下 `[x]` 只表示输入契约、执行路径和回归测试已存在：

- [x] 共用 Step 0–5 的数据解析、原分辨率 score-map 和评估基础；
- [x] Step R1：多源域均衡且 DataParallel-safe 的 round-robin batch；
- [x] Step R2：local-peak Tail-CVaR；
- [x] Step R3：object Miss-CVaR；
- [x] Step R4：风险敏感 detector 训练入口（fixed-last，不构造 target/test loader）；
- [ ] Step R5：严格 LODO detectors 的全量 checkpoint（契约已实现，实验产物未生成）；
- [x] Step R6：无标签窗口统计与 fold-specific source reference；
- [x] Step R7：自适应高尾 curve 与 oracle threshold 元标签；
- [x] Step R8：直接阈值校准器基线及 schema-v2 溯源；
- [x] Step R9：prefix-holdout/causal online 适配与独立 query replay；
- [ ] rolling quantile 和 EVT；
- [ ] 不同窗口长度；
- [ ] 目标污染实验；
- [ ] reject coverage 分析；
- [ ] 至少 3 个 unseen domains；
- [ ] 三个随机种子。

另外，AAAI 评审稿建议的“单调逆风险曲线 + 尾部间隔学习”属于 RC-v2/候选主方法，当前仓库中的 direct calibrator 不得被改名为该方法。

---

## 13. RC 阶段性闸门

### Gate 1：表征

Tail-CVaR 是否减少 unseen-domain 高分背景峰值？

### Gate 2：可校准性

先用线性回归或随机森林检查无标签统计能否预测 oracle threshold。若简单模型完全无效，MLP 很可能也不稳定。

### Gate 3：真正 unseen domain

必须优于 rolling quantile，而不仅是在伪目标源域上有效。

### Gate 4：安全性

必须同时报告 Pd、BSR、Excess 和 coverage。

---

## 14. RC 转向条件

转向 ConFA：

```text
- zero-label 校准器频繁超预算；
- 与 rolling quantile 无显著差异；
- 少量目标域标签能明显解决问题。
```

升级到 RC-v2：

```text
- 直接阈值回归部分有效；
- 但跨预算不一致；
- 需要风险解释和少样本认证。
```
