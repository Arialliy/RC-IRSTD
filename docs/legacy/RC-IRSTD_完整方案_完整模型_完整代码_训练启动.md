# RC-IRSTD 完整方案、完整模型、完整代码与训练启动

> 本文档面向当前工程实现阶段。重点是把 RC-IRSTD 做成一个能训练、能验证、能输出风险工作点、能部署的完整系统，而不是组织最终论文材料。

---

## 0. 本次续建的事实基础

用户指定的远端地址为：

```text
https://github.com/Arialliy/RC-IRSTD
```

本次运行环境无法解析 `github.com`，浏览器也没有该仓库的可用缓存，因此不能诚实地声称已经取得远端当前 HEAD。工程续建使用的是用户文件库中最新的源码归档：

```text
RC_IRSTD_AAAI_Implementation.zip
```

随后重新核对了公开的 MSHNet 官方实现，并将 MSHNet 与 SLS 以自包含形式整合进 RC-IRSTD。详细来源记录见：

```text
SOURCE_PROVENANCE.md
```

因此，最终 ZIP 是：

> 基于当前可获得 RC-IRSTD 源码快照完成的完整续建版本，而不是未经验证的远端 HEAD 镜像。

推送到 GitHub 前，应再与远端分支做一次 diff，保留远端可能存在的更新。

---

# 第一部分：对原工程的代码级分析

## 1. 原工程已经具备的研究骨架

原快照并不是空工程，它已经包含：

1. 多源域检测器训练入口；
2. Tail-CVaR 和 Miss-CVaR；
3. 连续 score record；
4. support/future episode；
5. 单调风险曲线预测器；
6. 零标签阈值选择；
7. CRC offset 校准；
8. Nested LODO 调度；
9. synthetic smoke test；
10. 16 个基础单元测试。

这意味着正确做法不是再创建一个脱离仓库的“新框架”，而是保留该主干，修复模型、协议、评价和部署层的缺口。

## 2. 原快照中影响完整训练的主要问题

### 2.1 MSHNet 不在仓库内

原实现通过动态导入：

```python
importlib.import_module("model.MSHNet")
```

实际训练依赖另一个仓库，并且 smoke 只测试 TinyUNet。因此：

- 仓库无法独立启动真实主模型；
- 外部 fork 的 forward API 可能不同；
- SLS 路径没有稳定的本地集成测试；
- checkpoint 是否兼容无法在仓库内验证。

### 2.2 detector checkpoint 只按 IoU 选择

原训练以固定阈值 0.5 的 IoU 选择最好模型，而研究目标是：

\[
P_d@F_a\le B.
\]

IoU 更高不等于极低虚警区排序更好。因此风险损失即使有效，也可能被错误的 checkpoint 选择覆盖。

### 2.3 “shot”与真实标注图数量不一致

原 calibration size 统计 episode 数，而每个 episode 的 future 内包含多张图。比如：

```text
20 episodes × 16 future images = 320 labeled images
```

这不能称为 20-shot image calibration。

### 2.4 静态图像被当作时间序列

纯数字文件名会落入一个默认 sequence，然后按编号构造“未来”。这不会直接产生标签泄漏，但时间因果语义是错误的。

### 2.5 component false alarms 不严格单调

阈值升高可能把一个连通域分裂成两个连通域。因此 connected-component 数量不一定随阈值单调，不能作为形式风险曲线和 CRC 的唯一风险族。

### 2.6 固定 256 resize 后上采样不等于原分辨率推理

先压缩到 256，再将 score map 插值回原图会：

- 丢失原始单像素目标；
- 改变 false pixel 面积；
- 改变候选间距和容差的物理意义。

### 2.7 风险曲线损失被大量极端阈值支配

若所有阈值等权，模型容易主要拟合高阈值区域的风险 floor，而非预算 crossing 附近。pointwise quantile coverage 可以正常，但实际选点仍然不准。

### 2.8 旧结果仅按文件存在与否复用

修改配置、数据、阈值网格、checkpoint 或代码后，只要旧文件仍存在，原调度器就可能直接跳过，导致实验静默污染。

### 2.9 部署只输出阈值

缺少：

- 将阈值作用到 future score maps；
- mask 保存；
- candidate 保存；
- rolling warm-up；
- threshold history；
- OOD/reject；
- 完整 deployment state。

---

# 第二部分：最终完整系统设计

## 3. 问题定义

给定多个有标注源域：

\[
\mathcal D_s=\{\mathcal D_1,\ldots,\mathcal D_K\},
\]

未知目标域仅提供无标签图像流：

\[
\mathcal D_t=\{x_i^t\}_{i=1}^{N_t}.
\]

目标是根据用户预算选择部署阈值：

\[
\max_\tau P_d^t(\tau)
\]

满足：

\[
F_{a,\mathrm{pix}}^t(\tau)\le B_p,
\qquad
F_{a,\mathrm{peak/MP}}^t(\tau)\le B_c.
\]

其中：

- `pixel false rate`：错误预测背景像素数 / 总像素数；
- `fixed false peaks/MP`：在阈值扫描前固定提取的局部峰值候选中，错误候选数 / 百万像素；
- `connected-component FA/MP`：保留为文献兼容评价，不承担风险曲线的单调性假设。

## 4. 总体流水线

```text
Stage A  多源风险感知检测器
         MSHNet + SLS + Tail-CVaR + Miss-CVaR
                         │
                         ▼
Stage B  连续 score maps 与双轨风险评价
         fixed peaks + pixel / connected components
                         │
                         ▼
Stage C  support 窗口无标签特征
         score survival + peak survival + acquisition stats
                         │
                         ▼
Stage D  单调双风险曲线网络
         U_pix(z, τ), U_peak(z, τ)
                         │
                         ▼
Stage E  零标签工作点
         最小满足双预算的阈值 / empty-action reject
                         │
                         ▼
Stage F  可选 image-shot 或 block CRC offset
                         │
                         ▼
Stage G  future masks / candidates / deployment history
```

该系统不强制把所有模块端到端联合训练。检测器和风险曲线分阶段训练，便于：

- 固定 detector 后重复构造多个元 episode；
- 明确 support 无标签、future 隐藏标签的边界；
- 单独诊断表示失败与校准失败；
- 缓存 score maps，减少重复 GPU 推理；
- 保持部署阶段不更新 detector。

---

# 第三部分：Stage A——完整风险感知 MSHNet

## 5. 内置 MSHNet

新增：

```text
rc_irstd/models/mshnet.py
```

结构：

```text
Input 3×H×W
   │
1×1 projection → 16 channels
   │
Encoder 0: 16
   │ pool
Encoder 1: 32
   │ pool
Encoder 2: 64
   │ pool
Encoder 3: 128
   │ pool
Middle: 256
   │
Decoder 3: [128 + 256] → 128
Decoder 2: [64  + 128] → 64
Decoder 1: [32  + 64 ] → 32
Decoder 0: [16  + 32 ] → 16
   │
4 multi-scale heads
   │ resize + concatenate
3×3 fusion head
   │
final logit map
```

每个残差块使用：

```text
Conv-BN-ReLU
→ Conv-BN
→ Channel Attention
→ Spatial Attention
→ residual addition
→ ReLU
```

保留以下公开 checkpoint 常用模块名：

```text
conv_init
encoder_0 ... encoder_3
middle_layer
decoder_3 ... decoder_0
output_0 ... output_3
final
```

默认通道：

```python
(16, 32, 64, 128, 256)
```

测试可以实例化更小通道，但正式训练使用默认结构。

## 6. 内置稳定 SLS-IoU

新增：

```text
rc_irstd/losses/sls.py
```

warm-up 阶段：

\[
\mathcal L_{IoU}=1-\frac{|P\cap Y|+\epsilon}{|P\cup Y|+\epsilon}.
\]

warm-up 后加入尺度权重：

\[
d=\left(\frac{|P|-|Y|}{2}\right)^2,
\]

\[
\alpha=
\frac{\min(|P|,|Y|)+d+\epsilon}
{\max(|P|,|Y|)+d+\epsilon}.
\]

以及位置损失：

\[
\mathcal L_{loc}
=
1-\frac{\min(l_P,l_Y)}{\max(l_P,l_Y)+\epsilon}
+
\frac{4}{\pi^2}(\theta_P-\theta_Y)^2.
\]

最终：

\[
\mathcal L_{SLS}=1-\operatorname{mean}(\alpha IoU)+\mathcal L_{loc}.
\]

对空目标图像不计算没有意义的位置项，从而避免 NaN。

## 7. 背景峰值 Tail-CVaR

最终概率：

\[
P=\sigma(Z).
\]

先膨胀 GT 排除区域，避免目标 PSF 边缘被当作背景：

\[
E=\operatorname{Dilate}(Y,r).
\]

在背景上计算局部最大值：

\[
\mathcal A_i^-
=
\{P_p:p\notin E,\,P_p=\operatorname{MaxPool}(P)_p\}.
\]

每张图先计算：

\[
R_i^-=\operatorname{CVaR}_{q_-}(\mathcal A_i^-).
\]

再按域平均：

\[
R_d^-=\frac1{|I_d|}\sum_{i\in I_d}R_i^-.
\]

最后对域风险做 smooth upper max：

\[
\mathcal L_{tail}
=
\frac1\gamma\log\sum_d\exp(\gamma R_d^-).
\]

“每图先聚合，再按域聚合”防止候选多的单张图支配整个域。

## 8. 困难目标 Miss-CVaR

数据加载时预先计算 GT component label map，不在每个训练 forward 中调用 SciPy。

对目标组件 \(G_j\)：

\[
s_j^+
=
\frac1T\log\left(
\frac1{|G_j|}
\sum_{p\in G_j}\exp(TP_p)
\right).
\]

困难目标风险：

\[
\mathcal L_{miss}
=
\operatorname{CVaR}_{q_+}\{1-s_j^+\}.
\]

## 9. 检测器总损失

\[
\boxed{
\mathcal L_{det}
=
\mathcal L_{SLS,multiscale}
+
\lambda_{tail}\mathcal L_{tail}
+
\lambda_{miss}\mathcal L_{miss}
}
\]

默认：

```text
lambda_tail = 0.10
lambda_miss = 0.10
tail_quantile = 0.95
miss_quantile = 0.80
worst_gamma = 10
```

Tail/Miss 仅作用于最终 score map；多尺度辅助头继续使用 SLS。

## 10. 多源均衡 batch

新增/保留：

```text
rc_irstd/data/sampler.py
```

假设 3 个域、每域 2 张：

```text
batch_size = 6
A: 2
B: 2
C: 2
```

禁止简单 `ConcatDataset + global shuffle` 让大数据集主导每个 step。

## 11. 风险导向 checkpoint

新增：

```text
rc_irstd/evaluation/detector_selection.py
```

每个 validation 周期：

1. 对每个源域导出验证概率；
2. 扫描简化阈值网格；
3. 计算 pixel 与 fixed-peak 风险；
4. 每域选择满足双预算的最小阈值；
5. 计算 mean-domain Pd 和 worst-domain Pd；
6. 字典序选择 checkpoint。

排序键：

```text
1. worst-domain Pd
2. mean-domain Pd
3. IoU
4. -rejection rate
```

保存：

```text
best_budget.pt
best_iou.pt
last.pt
best.pt -> 兼容性复制 best_budget
```

---

# 第四部分：数据、分数与双轨评价

## 12. 8/16 位红外数据

新增：

```text
rc_irstd/data/transforms.py
```

支持：

```text
uint8
uint16
其他整数位深
float
```

不再强制：

```python
Image.open(...).convert("RGB")
```

归一化：

```text
imagenet   按原 dtype 位深缩放到 [0,1]，复制为 3 通道，再 ImageNet 标准化
minmax     单图 min-max
percentile 单图 0.5–99.5 percentile robust scaling
none       保持浮点值
```

正式实验应固定并记录归一化方式，不能根据未知目标标签调节。

## 13. 目标保留 mask resize

普通 nearest resize 可能跳过一像素目标。当前下采样使用 adaptive max pooling：

```python
resized = F.adaptive_max_pool2d(mask, target_hw)
```

只要原 mask 中存在正像素，缩放结果至少保留一个正输出 bin。混合缩放（一个轴下采样、另一个轴上采样）也使用该策略。

## 14. score export 三协议

入口：

```text
rc_irstd/pipelines/export_scores.py
```

### 14.1 resize

```text
resize → inference → optional restore original
```

用于兼容历史 MSHNet 256×256 协议。

### 14.2 native_pad

```text
保持原高宽
→ pad 到 stride multiple
→ inference
→ 去 padding
```

推荐作为主原分辨率协议。

### 14.3 tiled

```text
原图滑窗
→ 每 tile native pad
→ 重叠区域 logit 平均
→ 合并完整 score map
```

用于大分辨率图像和显存受限场景。

每张 score record 保存：

```text
probability
optional mask
image statistics
image_id
dataset_name
sequence_id
frame_index
original_hw
source_checkpoint
dataset_type
inference_mode
```

## 15. Formal fixed-peak 风险

固定候选在阈值扫描前提取：

```text
score map
→ deterministic local maxima
→ candidate scores + coordinates
```

阈值只决定保留哪些候选，因此：

\[
N_{peak}(\tau_2)\le N_{peak}(\tau_1),
\quad \tau_2>\tau_1.
\]

用于：

- 风险曲线训练；
- 零标签预算；
- CRC nested loss；
- detector budget checkpoint。

## 16. Connected-component 文献兼容评价

新增：

```text
rc_irstd/evaluation/component_curves.py
rc_irstd/evaluation/irstd_metrics.py
```

输出：

- IoU；
- nIoU；
- foreground/background IoU；
- hIoU；
- precision；
- recall；
- F1；
- object Pd；
- connected-component FA/MP；
- false pixel rate。

Formal risk 与 component metrics 不能混称。

---

# 第五部分：Stage C——无标签目标域特征

## 17. WindowFeatureExtractor

文件：

```text
rc_irstd/features/window_stats.py
```

输入只包含：

```text
unlabeled images
continuous detector scores
```

不读取 mask。

### 17.1 Pixel survival curve

固定阈值：

```text
0.01, 0.03, 0.05, 0.10, ..., 0.9999
```

每张图：

\[
s^{pix}_i(t)=\frac1{HW}\sum_p\mathbf 1[P_{ip}\ge t].
\]

窗口内保存 mean 与 std，并在 survival 比例上使用 log10。

### 17.2 Pixel quantiles

```text
0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995
```

### 17.3 Fixed-peak survival curve

\[
s^{peak}_i(t)=
\frac{\#\{a\in A_i:a\ge t\}}
{HW/10^6}.
\]

同样聚合 mean/std。

### 17.4 Peak quantiles

固定候选分数的多分位数。

### 17.5 Acquisition statistics

包括紧凑的图像统计，例如：

- intensity mean/std/MAD；
- gradient；
- Laplacian；
- high-frequency energy；
- score distribution higher moments。

### 17.6 Window metadata

- `log1p(window size)`；
- 总像素；
- 单图像素规模 mean/std；
- 候选密度 mean/std。

特征名随 checkpoint 一起保存。部署时若 schema 不一致会直接报错，避免静默错位。

---

# 第六部分：Stage D——结构单调双风险曲线模型

## 18. 为什么不是直接回归阈值

直接学习：

\[
\hat\tau=g(z,B)
\]

存在：

- 不支持未见预算；
- 不能检查风险曲线形状；
- threshold MAE 与预算超限不等价；
- 很难接入 CRC；
- 多预算单调性需要额外约束。

当前模型学习：

\[
U_\phi^{pix}(z,\tau),
\qquad
U_\phi^{peak}(z,\tau),
\]

输出 log10 risk curve。

## 19. 风险曲线网络

文件：

```text
rc_irstd/models/risk_curve.py
```

共享 encoder：

```text
LayerNorm
→ Linear(input_dim, hidden_dim)
→ GELU
→ Dropout
→ Linear(hidden_dim, hidden_dim)
→ GELU
→ Dropout
```

两个独立 head：

```text
pixel log-risk head
peak log-risk head
```

## 20. 精确结构单调参数化

对升序阈值网格 \(\tau_0<\cdots<\tau_{T-1}\)，head 输出：

- 起点 \(u_0\)；
- 正总下降量 \(D=\operatorname{softplus}(d)\)；
- interval softmax 权重 \(a_k\)。

\[
\Delta_k=D a_k,
\qquad \Delta_k\ge0,
\qquad \sum_k\Delta_k=D.
\]

\[
U(\tau_k)=u_0-\sum_{j<k}\Delta_j.
\]

因此结构上保证：

\[
U(\tau_{k+1})\le U(\tau_k).
\]

与直接累计 softplus 相比，总下降量分配不会在几百个阈值上初始化爆炸。

## 21. 阈值网格和 empty action

正式网格最后包含大于 1 的哨兵阈值：

```text
threshold > 1.0
```

它表示：

```text
empty prediction / abstention
```

而不是普通有效检测点。

## 22. Budget-focused quantile loss

目标 log risk：

\[
r_t=\log_{10}(F_a(\tau_t)+\epsilon).
\]

使用上分位 pinball loss，鼓励保守预测。预算附近权重：

\[
w_t=w_0+\lambda_B
\exp\left(-\frac{|r_t-\log_{10}B|}{s}\right).
\]

empty action 权重单独降低，避免模型主要拟合风险 floor。

## 23. Crossing loss

对每个阈值构造真实预算可行标签：

\[
y_t=\mathbf1[r_t\le\log_{10}B].
\]

预测可行概率：

\[
p_t=\sigma\left(
\frac{\log_{10}B-U_\phi(z,\tau_t)}{T}
\right).
\]

使用 BCE：

\[
\mathcal L_{cross}=BCE(p_t,y_t).
\]

总损失：

\[
\mathcal L_{curve}
=
\mathcal L_{pin,pix}
+
\lambda_p\mathcal L_{pin,peak}
+
\lambda_c\mathcal L_{cross}.
\]

## 24. 风险曲线 checkpoint 选择

不是只看 validation pinball。排序：

```text
1. selected-point joint excess 越小越好
2. effective Pd 越高越好
3. rejection 越低越好
4. pinball 越小越好
```

保存：

```text
best_selected.pt
best_pinball.pt
best.pt -> best_selected
last.pt
```

---

# 第七部分：Stage E/F——零标签工作点与 CRC

## 25. 零标签工作点

给定预测曲线：

\[
\hat r_t^{pix},\hat r_t^{peak},
\]

选择第一个同时满足预算的阈值：

\[
k^0=
\min\{k:
\hat r_k^{pix}\le\log_{10}B_p,
\hat r_k^{peak}\le\log_{10}B_c\}.
\]

若只能选择最后的 empty action，则标记 reject。

输出：

```text
threshold index
threshold
predicted pixel risk
predicted peak risk/MP
reject
feature OOD score
```

## 26. CRC offset

对样本自适应基础索引 \(k_i^0\)，构造非负 offset：

\[
k_i(\Delta)=\min(k_i^0+\Delta,T-1).
\]

联合违反预算损失：

\[
L_i(\Delta)=
\mathbf1[
F_{i,pix}(k_i(\Delta))>B_p
\lor
F_{i,peak}(k_i(\Delta))>B_c
].
\]

标准有限样本修正：

\[
\widehat R^+_m(\Delta)
=
\frac{m}{m+1}\widehat R_m(\Delta)
+
\frac1{m+1}.
\]

选择最小可行 offset：

\[
\hat\Delta=
\min\{\Delta:\widehat R^+_m(\Delta)\le\alpha\}.
\]

## 27. 校准单位

### 27.1 `--calibration-unit image`

- 每个样本是一张 future 标注图；
- `calibration_size=20` 就是 20 张图；
- episode 文件必须保存 per-future-image risk curves；
- 同一 image ID 不允许重复；
- 空目标图像提供虚警证据，但 Pd 记为 NaN，不压低 target-bearing Pd。

### 27.2 `--calibration-unit episode`

- 每个样本是一个 future block；
- 输出同时记录 block 数和标签图总数；
- 不能把 20 blocks 写成 20 labeled images。

## 28. IID 与 temporal 划分

### IID images

```text
固定 seed 随机排列
→ support/query block
→ image IDs 全部不重叠
→ calibration/test unique image split
```

不声明真实时间因果，只声明：

> 无标签 support 预测独立 query block 的工作点。

### Temporal

```text
sequence 内按 frame_index 排序
→ past context
→ future horizon
→ calibration/test 按 sequence 或 blocked unit 隔离
```

---

# 第八部分：部署系统

## 29. DeploymentState

新增：

```text
rc_irstd/deployment/session.py
```

保存：

```text
detector checkpoint
curve checkpoint
score directory
budgets
warmup size
offset
created time
all threshold updates
```

每次 update 保存：

```text
sequence_id
update_index
warmup image IDs
base/final threshold index
threshold
predicted risks
rejected
feature OOD score
```

## 30. OOD 与滚动更新

无标签特征经训练 normalizer 转换后，计算 RMS z-score：

\[
d_{ood}(z)=
\sqrt{\frac1D\sum_j\left(\frac{z_j-\mu_j}{\sigma_j}\right)^2}.
\]

超过阈值时选择 empty action/reject。

`update_every > 0` 时：

```text
只使用当前时刻之前最近 warmup_size 张
→ 重新预测风险曲线
→ 更新 future threshold
```

不会使用未来图像或目标标签。

## 31. 部署输出

```text
masks/<image_id>.png
candidates.csv
deployment_state.json
summary.json
```

---

# 第九部分：Nested LODO

## 32. 严格排除规则

outer target 为 \(t\)，pseudo-target 为 \(p\)：

```text
final detector sources = all domains excluding t
pseudo detector sources = all domains excluding t and p
risk curve train episodes = p produced by corresponding pseudo detector
final target t = never enters detector/curve fitting
```

目标域标签只用于最终离线评价或明确声明的 few-shot calibration。

## 33. source-set 去重

六域时，原始嵌套循环会重复训练相同 source set。当前 detector 输出目录按：

```text
sorted source names + detector config
```

哈希缓存。

## 34. artifact fingerprint

每个声明产物旁保存 manifest，指纹包括：

```text
command
working directory
Python source tree hash
checkpoint/config file hash
dataset split/manifest hash
directory metadata
```

只有：

```text
artifact exists
AND manifest exists
AND fingerprint exact match
```

才会跳过。

---

# 第十部分：文件级实现清单

## 35. 核心新增或重写文件

### 模型与损失

```text
rc_irstd/models/mshnet.py
rc_irstd/models/detector_adapter.py
rc_irstd/models/risk_curve.py
rc_irstd/losses/sls.py
rc_irstd/losses/cvar.py
rc_irstd/losses/risk_aware.py
rc_irstd/losses/quantile.py
```

### 数据与协议

```text
rc_irstd/data/transforms.py
rc_irstd/data/dataset.py
rc_irstd/data/windows.py
rc_irstd/data/score_records.py
rc_irstd/episodes/builder.py
rc_irstd/episodes/dataset.py
rc_irstd/episodes/splits.py
```

### 评价与校准

```text
rc_irstd/evaluation/detector_selection.py
rc_irstd/evaluation/component_curves.py
rc_irstd/evaluation/irstd_metrics.py
rc_irstd/calibration/samples.py
```

### 训练、导出、部署

```text
rc_irstd/pipelines/train_detector.py
rc_irstd/pipelines/export_scores.py
rc_irstd/pipelines/build_episodes.py
rc_irstd/pipelines/train_curve.py
rc_irstd/pipelines/evaluate_scores.py
rc_irstd/pipelines/calibrate_and_evaluate.py
rc_irstd/pipelines/apply_operating_point.py
rc_irstd/pipelines/run_deployment.py
rc_irstd/pipelines/run_lodo.py
```

### 可复现性

```text
rc_irstd/engine/worker_seed.py
rc_irstd/provenance/fingerprint.py
rc_irstd/provenance/manifest.py
```

### 启动器

```text
scripts/start_training.sh
scripts/train_detector_mshnet.sh
scripts/full_pipeline_start.sh
scripts/deploy_target.sh
scripts/mshnet_integration_test.sh
scripts/validate_release.sh
```

### 配置

```text
configs/lodo_example.yaml
configs/lodo_temporal_example.yaml
```

完整文件内容展开在：

```text
docs/RC-IRSTD_完整代码清单.md
```

---

# 第十一部分：训练启动

## 36. 环境安装

```bash
unzip RC-IRSTD_Rebuilt_Complete.zip
cd RC-IRSTD_Rebuilt
bash scripts/setup.sh
```

建议先安装适配本机 CUDA 的 PyTorch，再运行 editable install。

## 37. 完整验证

```bash
bash scripts/validate_release.sh /tmp/rc_irstd_validation
```

## 38. detector 训练

```bash
bash scripts/start_training.sh detector \
  /data/NUAA-SIRST \
  /data/NUDT-SIRST \
  /data/IRSTD-1K
```

指定输出和参数：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT=outputs/detector_main \
PER_DOMAIN_BATCH=2 \
EPOCHS=400 \
LR=0.05 \
LAMBDA_TAIL=0.10 \
LAMBDA_MISS=0.10 \
SEED=42 \
bash scripts/train_detector_mshnet.sh \
  /data/NUAA-SIRST \
  /data/NUDT-SIRST \
  /data/IRSTD-1K
```

## 39. 一键 LODO

```bash
cp configs/lodo_example.yaml configs/my_lodo.yaml
# 修改全部 path

bash scripts/full_pipeline_start.sh configs/my_lodo.yaml \
  --outer-target RealScene-ISTD \
  --dry-run

bash scripts/full_pipeline_start.sh configs/my_lodo.yaml \
  --outer-target RealScene-ISTD
```

## 40. 分阶段启动

```bash
python -m rc_irstd.pipelines.run_lodo \
  --config configs/my_lodo.yaml \
  --outer-target RealScene-ISTD \
  --stages detector

python -m rc_irstd.pipelines.run_lodo \
  --config configs/my_lodo.yaml \
  --outer-target RealScene-ISTD \
  --stages export episodes

python -m rc_irstd.pipelines.run_lodo \
  --config configs/my_lodo.yaml \
  --outer-target RealScene-ISTD \
  --stages curve zero calibrate baselines
```

## 41. 未知域部署

```bash
DEVICE=cuda \
WARMUP_SIZE=32 \
UPDATE_EVERY=0 \
PIXEL_BUDGET=1e-6 \
PEAK_BUDGET=1.0 \
bash scripts/deploy_target.sh \
  /data/UNLABELED_TARGET \
  test \
  outputs/detector_main/best_budget.pt \
  outputs/risk_curve/best.pt \
  outputs/deployment/UNLABELED_TARGET
```

---

# 第十二部分：验证状态

## 42. 已执行验证

当前发布树已实际执行：

```text
python -m compileall -q rc_irstd tests
pytest -q
synthetic end-to-end smoke
MSHNet forward/backward/checkpoint roundtrip
shell syntax validation
```

测试覆盖：

1. CRC 选择；
2. dataset path；
3. episode metrics；
4. feature schema；
5. LODO protocol；
6. operating point；
7. fixed peaks；
8. risk-aware loss；
9. monotone risk curve；
10. balanced sampler；
11. grouped split；
12. windows；
13. bundled MSHNet integration；
14. 16-bit + target-preserving resize；
15. IRSTD metrics + provenance；
16. deployment + calibration units。

当前数量：

```text
25 passed
```

Synthetic smoke 实际执行：

```text
3 synthetic domains
→ detector train
→ budget checkpoint selection
→ score export
→ IID support/query episodes
→ monotone risk curve train
→ zero-label evaluation
→ explicit-unit CRC
```

## 43. 尚未声称完成的内容

本包没有用户的真实 IRSTD 数据，因此没有生成：

- 真实 benchmark 最终权重；
- 真实跨域性能数字；
- 真实低至 \(10^{-6}\) 的统计稳定性结论；
- 真实多 GPU 训练吞吐报告。

这些必须在用户提供数据后运行。

---

# 第十三部分：推荐执行顺序

## 44. Gate 0：问题诊断

先训练一个 baseline detector，并比较：

```text
fixed 0.5
source budget threshold
target oracle threshold
```

继续风险校准路线的条件：

- 至少两个目标域存在明显工作点漂移；
- target oracle 可在同预算恢复 Pd；
- 固定/source 阈值存在预算超限；
- 高分目标与背景候选仍有排序空间。

## 45. Gate 1：风险感知 detector

对比：

```text
SLS
SLS + Tail
SLS + Tail + Miss
```

检查：

- unseen 背景高分峰值；
- worst-domain Pd；
- best_budget 与 best_iou checkpoint 差异。

## 46. Gate 2：风险曲线

先检查：

- monotonicity violations = 0；
- selected crossing error；
- BSR/Excess；
- zero-label 是否优于简单 source threshold。

## 47. Gate 3：严格 unseen deployment

最终目标域不能参与：

- detector early stopping；
- risk feature selection；
- curve checkpoint；
- window size选择；
- threshold grid 调参。

---

# 第十四部分：当前版本的边界和下一步

## 48. 理论边界

无标签分数尾部满足：

\[
P(S>\tau)
=
P(S>\tau,Y=0)
+
P(S>\tau,Y=1).
\]

仅观察无标签分数，不能在任意目标分布下唯一识别真实背景虚警。因此 zero-label 模块应理解为：

> 在跨源域元分布假设下学习的经验风险曲线，而不是任意未知域的 distribution-free guarantee。

CRC 只在声明的校准统计单位及其交换性条件下解释。

## 49. 工程边界

当前正式训练为单进程单设备；可以通过不同 outer fold 分配到不同 GPU 并行运行。未来需要时可增加 DDP，但不影响当前完整算法闭环。

## 50. 下一轮真实实验优先级

```text
P0: 数据 manifest 和重复样本审计
P0: baseline / oracle 阈值漂移矩阵
P0: 1 个 MSHNet seed 的三域小规模闭环
P1: 完整 3 seed LODO
P1: native_pad 与 resize 对比
P1: window 8/16/32/64
P1: Tail/Miss 消融
P2: 第二 backbone adapter
P2: direct threshold、rolling quantile、EVT baseline
P2: target contamination 和传感器扰动
```

---

# 最终工程定义

本次完成后的 RC-IRSTD 不再是“手工统计 + 一个阈值 MLP”的概念代码，而是：

```text
自包含 MSHNet
+ 风险感知多源训练
+ 双轨候选/评价
+ IID/temporal 明确协议
+ support → disjoint query 元 episode
+ 结构单调双风险曲线
+ 预算 crossing 优化
+ 零标签工作点
+ 显式 image/block CRC
+ 原分辨率/切片导出
+ 完整 future mask 与 candidate 部署
+ 指纹化 Nested LODO
+ 自动测试和训练启动器
```

这构成了当前阶段可以直接接入真实数据继续训练和迭代的完整模型与完整代码基础。
