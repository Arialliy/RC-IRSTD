---
title: "RC-IRSTD：AAAI-27 投稿可行性评审与修改建议"
date: "2026-07-14"
language: "zh-CN"
source_document: "02_RC-IRSTD_方案_代码_步骤.md"
target_venue: "AAAI-27 Main Technical Track"
review_basis: "2026-07-14 local code audit + official AAAI-27 timetable"
---

# RC-IRSTD：AAAI-27 投稿可行性评审与修改建议

> 分析对象：`02_RC-IRSTD_方案_代码_步骤.md` 与当前 `RC-IRSTD/` 工作树  
> 投稿目标：AAAI-27 Main Technical Track  
> 分析日期：2026 年 7 月 14 日  
> 核心判断：**方向可以冲 AAAI，但不建议按当前方案原样投稿。**

---

## 目录

1. [总体结论](#1-总体结论)
2. [AAAI-27 投稿约束](#2-aaai-27-投稿约束)
3. [为什么该方向具备 AAAI 潜力](#3-为什么该方向具备-aaai-潜力)
4. [当前方案中应保留的部分](#4-当前方案中应保留的部分)
5. [当前版本最严重的技术问题](#5-当前版本最严重的技术问题)
6. [推荐的 AAAI 版核心方法](#6-推荐的-aaai-版核心方法)
7. [Reject 机制的处理建议](#7-reject-机制的处理建议)
8. [AAAI 所需实验包](#8-aaai-所需实验包)
9. [评价指标的重新组织](#9-评价指标的重新组织)
10. [代码方案中必须立即修正的问题](#10-代码方案中必须立即修正的问题)
11. [论文定位、题目与投稿关键词](#11-论文定位题目与投稿关键词)
12. [推荐的三条论文贡献](#12-推荐的三条论文贡献)
13. [建议加入的理论结果](#13-建议加入的理论结果)
14. [7 页主文的结构与取舍](#14-7-页主文的结构与取舍)
15. [AAAI-27 最小可行投稿版本](#15-aaai-27-最小可行投稿版本)
16. [优先执行路线与 Go/No-Go 闸门](#16-优先执行路线与-gono-go-闸门)
17. [最终判断](#17-最终判断)
18. [参考资料](#18-参考资料)

---

# 1. 总体结论

**RC-IRSTD 具备 AAAI 论文的研究内核，但当前文档更接近一份高质量的应用研究与工程实施方案，还不是一篇已经闭环的 AAAI 方法论文。**

当前方案最有价值的研究问题是：

> 在多个有标签源域上训练密集检测器后，部署到完全未知且无标签的目标流时，如何利用有限的无标签前缀数据，根据用户给定的虚警预算，自动选择后续数据的安全工作点？

这个问题同时涉及：

- domain generalization；
- unsupervised calibration；
- test-time deployment；
- selective prediction；
- operational risk control；
- dense prediction；
- extreme class imbalance。

要达到 AAAI 主会所需的方法完整度，建议完成三项决定性升级：

1. **从 IRSTD 专用阈值技巧提升为广义 AI/ML 问题**：  
   将论文核心抽象为“无标签目标流下的预算约束工作点适配”。

2. **把元训练协议改成严格因果形式**：  
   使用无标签 support 窗口预测独立 future query 窗口上的工作点，而不是用同一窗口同时生成统计和 oracle 标签。

3. **把普通阈值回归升级为风险对齐、预算单调的逆风险曲线预测**：  
   方法应直接针对预算违反和检测效用进行学习，并显式保证不同预算下阈值预测的单调一致性。

建议将论文核心凝练为：

> **Disjoint causal meta-episodes + monotone inverse risk curve + formal identifiability analysis**

---

## 1.1 当前方案的投稿状态评估

| 维度 | 当前状态 | AAAI 所需状态 |
|---|---|---|
| 问题重要性 | 高 | 基本满足 |
| IRSTD 领域内新颖性 | 较高 | 基本满足 |
| 对广义 AI 社区的意义 | 中等 | 需要重新抽象问题 |
| 方法严谨性 | 中等偏低 | 必须修正 episode、oracle 与损失 |
| 理论完整性 | 偏弱 | 至少加入不可识别性命题 |
| 实验说服力 | 尚未形成闭环 | 需要严格 external unseen-domain 与 causal protocol |
| 工程可复现性 | 较好 | 保持，具体实现细节放补充材料 |
| 当前直接投稿风险 | 高 | 完成关键升级后可显著下降 |

## 1.2 代码审计后的边界（2026-07-14）

本文后续的“建议方法”不等于“已实现方法”。当前工作树应按以下三层理解：

| 层级 | 当前状态 | 可用范围 |
|---|---|---|
| 评估与协议基础 | 已有原分辨率 score-map、双虚警指标、support/query 拆分、固定最后 checkpoint 和无标签统计的代码 | 可做可重复性 smoke test 与基线实验 |
| RC 直接阈值校准器 | 作为最小基线保留；必须通过跨域溯源、query replay 和独立外层目标审计后才能进入主表 | 不得写成单调逆风险曲线 |
| AAAI 升级版 | 单调逆风险曲线、风险对齐损失、尾部间隔学习和可识别性分析尚待独立实现与实证 | 只能作为 RC-v2/候选主方法，不得标记为已完成 |

截至本次审计，本地可直接验证的数据域只有 IRSTD-1K、NUDT-SIRST 和 NUAA-SIRST。固定一个 outer target 后只剩两个源域，再做 inner LODO 时 detector 只有一个训练域。这可用于显式标记的诊断/smoke test，但不足以支撑 AAAI 主结论；主实验需增加至少第四个合法独立域。

---

# 2. AAAI-27 投稿约束

截至 2026-07-14，[AAAI-27 官方主页](https://aaai.org/conference/aaai/aaai-27/) 已确认下列主会时间表，所有截止时间均为 AoE，即 UTC-12：

| 事项 | 官方截止时间 | 中国标准时间 UTC+8 |
|---|---:|---:|
| 摘要截止 | 2026-07-21 23:59 AoE | 2026-07-22 19:59 |
| 全文截止 | 2026-07-28 23:59 AoE | 2026-07-29 19:59 |
| 补充材料与代码截止 | 2026-07-31 23:59 AoE | 2026-08-01 19:59 |
| Phase 1 拒稿通知 | 2026-09-24 | — |
| Author Feedback | 2026-10-19 至 2026-10-25 | — |
| 最终录用通知 | 2026-11-30 | — |

AAAI-27 主页当前可确认 Phase 1 通知和 Author Feedback 时间，但本次核验未在官方 AAAI-27 专页中找到已发布的 Main Technical Track Call/Review Process 正文。因此，下列内容是基于往届规则的**排版与风险管理假设**，不是本文已核实的 AAAI-27 官方事实：

- 按 **7 页主内容**预留版面；
- 按往届“额外页只放参考文献”规则准备；
- 双盲评审；
- 两阶段评审；
- Phase 1 负面评价足够明显时会直接拒稿，没有作者回复机会；
- 补充材料不保证被评审阅读；
- 预先准备 reproducibility checklist。

一旦 AAAI-27 Author Kit、Main Track Call 或 Review Process 页面更新，必须在摘要提交前重新核对页限、匿名、补充材料和 checklist；不得用 AAAI-26 规则替代 AAAI-27 规则。

这意味着以下内容必须在主文中一次讲清楚：

- causal protocol；
- support/query 是否独立；
- 目标域是否发生标签泄漏；
- “risk calibration” 是否具有形式化保证；
- 预算指标的定义；
- 方法为何不是手工特征加 MLP 的工程启发式。

## 2.1 时间上的现实判断

### 情况 A：当前只有方案，尚未形成跨域连续分数图与核心实验

这种情况下，强行冲 AAAI-27 的风险很高。更合理的选择是：

- 立即完成可行性诊断；
- 形成完整方法与实验闭环；
- 以 AAAI-28 或同等级会议为稳健目标。

### 情况 B：MSHNet 基线已经跑通，多个数据集已整理，能够立即导出 score maps

仍可压缩范围冲 AAAI-27，但必须避免同时铺开所有扩展。优先级应为：

1. causal support-query 元协议；
2. 单调预算阈值模型；
3. 三个以上完全未知目标域；
4. rolling quantile、EVT 和 oracle；
5. 风险指标与不可识别性讨论。

以下模块可以暂缓：

- 独立 reject head；
- 多套 Tail-CVaR 变体；
- 同时控制多种非单调 component budget；
- 第三个 backbone；
- 复杂 EVT 与 conformal 扩展；
- 大量手工图像统计。

---

# 3. 为什么该方向具备 AAAI 潜力

## 3.1 不应把论文写成 IRSTD 阈值工程

不建议将核心贡献表述为：

> 给 MSHNet 增加 Tail-CVaR 损失，再用 MLP 预测红外分割阈值。

这种叙述容易被认为：

- 过度依赖单一任务；
- 方法由多个启发式模块拼接；
- 对广义 AI 社区的意义不足；
- 新颖性主要来自特定评测设置。

更适合 AAAI 的问题表述是：

> 给定在多个有标签环境中训练的密集预测器，部署到未知、无标签目标流时，如何利用有限的无标签前缀，在用户给定的错误预算下自动选择后续流的操作工作点？

这样，IRSTD 是一个具有强现实意义的验证场景，而不是方法只能成立的唯一场景。

## 3.2 IRSTD 是有说服力的验证任务

IRSTD 的特点使其非常适合验证预算工作点适配：

- 目标像素极少；
- 背景占比极高；
- 高分杂波会显著影响虚警；
- 固定阈值跨传感器和场景容易失效；
- 实际系统通常关心低虚警区域，而不是单一 IoU；
- 部署时往往没有目标域标签。

因此，该任务可以支撑一个更一般的研究主张：

> 平均精度优化与部署工作点控制是两个不同问题；模型在未知域上的概率排序、概率尺度和安全阈值可能同时发生漂移。

## 3.3 必须与已有校准与风险控制研究区分

论文相关工作至少要覆盖：

- Unsupervised Temperature Scaling；
- TransCal；
- PseudoCal；
- domain-generalized calibration；
- Risk-Controlling Prediction Sets；
- Conformal Risk Control；
- test-time adaptation；
- cross-domain IRSTD。

需要明确说明当前问题与这些工作的差异：

1. 研究对象是极端类别不平衡的密集检测，而非普通分类概率校准；
2. 目标是用户指定预算下的操作阈值，而不仅是 ECE/NLL；
3. 主协议是前缀到未来的 causal deployment，而不是同一测试集上的 transductive calibration；
4. 在零目标标签条件下，不声称 distribution-free 风险保证；
5. 需要同时评估预算满足、超限量和检测效用。

---

# 4. 当前方案中应保留的部分

## 4.1 问题设定值得保留

当前方案定义：

\[
\max_\tau P_d^t(\tau)
\qquad
\text{s.t.}\quad
F_a^t(\tau)\le B.
\]

这是一个比“最大化平均 IoU”更接近实际部署的目标，建议保留并作为论文主问题。

## 4.2 对不可识别性的认识是正确的

当前方案已经指出：

\[
P_t(S>\tau)
=
P_t(S>\tau,Y=0)
+
P_t(S>\tau,Y=1).
\]

无标签目标域中只能观察总预测尾部，不能直接知道其中有多少属于真实背景。因此：

> 无标签分数尾部不能在无附加假设时唯一确定真实虚警率。

这一点不应只放在 limitation 中，而应提升为形式化命题，用于界定方法能保证什么、不能保证什么。

## 4.3 “Budget-Aware” 比 “Guaranteed Risk Control” 更严谨

建议继续使用以下措辞：

- budget-aware；
- risk-aware；
- empirical risk calibration；
- unseen-domain operating-point adaptation；
- meta-calibrated deployment。

应避免：

- guaranteed false-alarm control；
- certified calibration；
- distribution-free control；
- arbitrary-domain guarantee。

## 4.4 Causal 测试协议方向正确

当前方案主张：

1. 收集前 \(M\) 张无标签目标图像；
2. 提取统计；
3. 预测阈值；
4. 固定阈值处理后续图像。

这是合理的部署协议。最终论文应以 causal protocol 为主，transductive same-set 结果只作为上界或附加实验。

## 4.5 对极端尾部样本量问题的判断正确

在 \(10^{-6}\) 预算附近：

- 单张图像中的有效尾部样本极少；
- 像素存在空间相关性；
- 单图经验分位数非常不稳定；
- 必须使用窗口、候选级统计和窗口长度敏感性分析。

这一部分应保留，并用于解释为何需要窗口级元学习，而不是简单地在单图上取分位数。

## 4.6 嵌套未知域原则正确

最终未知目标域不能用于：

- detector 训练；
- calibrator 训练；
- early stopping；
- 特征组合选择；
- 窗口长度选择；
- checkpoint 选择；
- 超参数搜索。

目标域标签只能用于最终评估和 oracle 上界。

---

# 5. 当前版本最严重的技术问题

# 5.1 元训练 episode 与 causal 测试不一致

当前元训练逻辑大致是：

1. 从伪目标域抽取一个窗口；
2. 用该窗口的无标签图像计算统计；
3. 用该窗口的隐藏标签计算 \(P_d-F_a\) 曲线；
4. 生成该窗口对应的 oracle threshold；
5. 学习从统计到该阈值的映射。

这相当于学习：

\[
z(W)\longrightarrow \tau^*(W).
\]

但真实部署要求：

\[
z(W_{\mathrm{support}})
\longrightarrow
\tau^*(W_{\mathrm{future}}).
\]

两者不是同一个任务。

如果 support 与产生 oracle 标签的 query 是同一批图像，校准器可能利用当前集合的偶然峰值结构，预测当前集合最优阈值。这会导致：

- transductive 乐观偏差；
- causal claim 不成立；
- support/query 标签泄漏；
- 在未来流上性能显著下降。

## 必须修改为 disjoint support-query episode

每个元 episode 定义为：

\[
E=(S,Q),
\qquad
S\cap Q=\varnothing.
\]

其中：

- \(S\)：support，仅允许使用图像和模型预测，不使用标签；
- \(Q\)：future query，仅在元训练时使用标签计算风险曲线和元损失；
- 校准器学习：

\[
g_\phi(z(S),B)
\rightarrow
\tau_B^*(Q).
\]

最终目标域测试严格复现：

\[
\text{prefix support}
\rightarrow
\widehat\tau_B
\rightarrow
\text{future query}.
\]

这是整个方案最优先、最不能妥协的修正。

---

# 5.2 Oracle threshold 与 reject 定义存在逻辑矛盾

当前方案定义：

\[
\tau_B^*
=
\min\{\tau:F_a(\tau)\le B\},
\]

并在“没有阈值满足预算”时设置 \(\tau=1\) 并 reject。

问题是，只要允许空预测：

- 当阈值高于最大预测分数时，预测为空；
- 空预测的虚警为 0；
- 因而虚警预算通常总能被满足。

所以“没有安全阈值”通常不成立。

真正可能不可行的是：

\[
F_a(\tau)\le B
\quad\text{且}\quad
P_d(\tau)\ge P_{\min}
\]

无法同时成立。

更合理的 reject 标签是：

\[
y_{\mathrm{reject}}
=
\mathbf 1
\left[
\max_{\tau:F_a(\tau)\le B}
P_d(\tau)
<
P_{\min}
\right].
\]

其含义是：

> 预算可以通过空预测满足，但当前域不存在同时具有安全性和有效检测能力的非退化工作点。

## Oracle 的建议定义

对严格单调风险，可定义：

\[
\tau_B^*
=
\arg\max_{\tau}
P_d^Q(\tau)
\quad
\text{s.t.}\quad
F_a^Q(\tau)\le B.
\]

若存在多个阈值达到相同 \(P_d\)，再选择其中最低阈值或最稳定阈值，并明确 tie-breaking rule。

---

# 5.3 Connected-component 虚警不一定随阈值单调

像素级虚警通常随阈值升高而单调不增。

但连通域数量未必单调：

- 低阈值下，一个杂波区域可能是一个大连通域；
- 阈值提高后，该区域可能断裂为两个或多个小连通域；
- 虚警连通域数量可能先上升再下降。

这会破坏以下隐含假设：

- 更高阈值一定更安全；
- 最小可行阈值一定是最优工作点；
- 预算到阈值的逆映射天然单调；
- 可以直接使用普通逆分位数控制 component budget。

## 推荐处理

主校准预算应优先选择严格或近似单调的风险：

1. **像素背景虚警率**；或
2. **固定候选集合上的假峰值数量/MP**。

第二种方案与当前方法尤其契合：

1. 在连续 score map 上先固定提取局部峰值候选；
2. 每个候选拥有固定分数；
3. 阈值只决定保留哪些候选；
4. 候选虚警数随阈值严格单调不增。

传统 connected-component 指标仍可作为兼容评估，但不建议作为主理论推导中的唯一预算风险。

---

# 5.4 直接回归阈值与非对称 MSE 不足以支撑风险校准

当前损失主要惩罚：

\[
(\widehat\tau-\tau^*)^2,
\]

并给阈值低估更大权重。

问题是，阈值误差与风险误差并不等价：

- 在风险曲线平坦区，阈值误差 0.02 可能几乎无影响；
- 在极陡尾部，阈值误差 0.0005 可能造成数量级的虚警增加；
- threshold MAE 很小，不代表预算满足率高；
- 在概率接近 1 时直接用 Sigmoid head 回归，容易发生梯度饱和。

## 应改为风险对齐训练

在 query 上直接优化预算违反：

\[
\mathcal L_{\mathrm{viol}}
=
\left[
\log
\frac{F_a^Q(\widehat\tau)+\epsilon}
{B}
\right]_+.
\]

同时保留检测效用：

\[
\mathcal L_{\mathrm{util}}
=
1-P_d^Q(\widehat\tau).
\]

总损失建议为：

\[
\mathcal L_{\mathrm{cal}}
=
\lambda_v\mathcal L_{\mathrm{viol}}
+
\lambda_u\mathcal L_{\mathrm{util}}
+
\lambda_o\mathcal L_{\mathrm{oracle}}
+
\lambda_m\mathcal L_{\mathrm{mono}}.
\]

其中：

- \(\mathcal L_{\mathrm{viol}}\)：预算违反损失；
- \(\mathcal L_{\mathrm{util}}\)：检测效用损失；
- \(\mathcal L_{\mathrm{oracle}}\)：辅助阈值回归；
- \(\mathcal L_{\mathrm{mono}}\)：预算单调性损失或结构约束。

## 阈值应在 logit 空间预测

定义：

\[
\eta
=
\log\frac{\tau}{1-\tau},
\qquad
\tau=\sigma(\eta).
\]

模型输出 \(\eta\)，而不是直接输出接近 1 的概率阈值。

---

# 5.5 多预算输入缺少单调性保证

当前方案把 \(\log_{10}B\) 作为普通 MLP 输入，但没有保证：

\[
B_1<B_2
\quad\Longrightarrow\quad
\widehat\tau(B_1)\ge
\widehat\tau(B_2).
\]

即，更严格的预算可能被模型预测出更低阈值。

这会造成：

- 风险曲线自相矛盾；
- 未见预算插值不稳定；
- 难以解释模型行为；
- 评审容易质疑方法是否真正学习了预算关系。

## 推荐改为单调逆风险曲线

设预算网格：

\[
B_1>B_2>\cdots>B_J.
\]

模型输出基础 logit 与非负增量：

\[
\eta_1=a_1,
\]

\[
\eta_j
=
\eta_{j-1}
+
\operatorname{softplus}(\Delta_j),
\qquad j\ge2.
\]

从而天然保证：

\[
\tau(B_1)
\le
\tau(B_2)
\le
\cdots
\le
\tau(B_J).
\]

对训练网格之外的预算，在 \(\log B\) 空间做单调插值。

这比“预算作为普通特征输入 MLP”更像完整的方法贡献。

---

# 5.6 Tail-CVaR 可能只学到全局降分

背景 Tail-CVaR 的目标是压低最高背景响应，Miss-CVaR 的目标是提高困难目标响应。方向合理，但存在退化解：

> 将整张预测图的 logits 整体向下平移。

由于部署时还会重新选择阈值，仅改变概率尺度不一定改善低虚警区的目标—背景排序。

## 推荐改为尾部间隔学习

定义背景危险候选：

\[
A_d^-.
\]

定义真实目标候选分数：

\[
A_d^+.
\]

使用尾部间隔损失：

\[
\mathcal L_{\mathrm{sep}}
=
\operatorname{SmoothMax}_d
\left[
m
+
\operatorname{CVaR}_{q_-}(A_d^-)
-
\operatorname{LowerTail}_{q_+}(A_d^+)
\right]_+.
\]

这样直接要求：

> 最困难真实目标的分数仍高于最危险背景候选至少一个 margin。

相较于两个独立损失，这更能解释为何低虚警区域会改善。

## 实现上还应修正

1. **GT mask 适度膨胀**  
   防止真实目标 PSF 边缘被当作背景高响应惩罚。

2. **避免小 batch 下 CVaR 退化为 top-1**  
   每域 batch 仅 2–4 张时，\(q=0.001\) 通常极不稳定。

3. **使用候选 memory bank 或 EMA 风险**  
   跨多个 batch 累积候选，降低尾部估计方差。

4. **每图等权或每域等权聚合**  
   防止大图、复杂图或候选多的图像支配损失。

5. **固定候选提取规则**  
   尽量让训练风险、校准预算和最终评估使用一致的候选定义。

---

# 5.7 极端预算下 oracle 元标签本身可能很噪声

以 \(M=32\)、输入分辨率 \(256\times256\) 为例，总像素约为：

\[
32\times256\times256
\approx
2.1\times10^6.
\]

在 \(10^{-6}\) 预算下，独立像素假设下的期望尾部事件约为 2 个。考虑空间相关性后，有效样本更少。

因此，用一个小 query window 的经验曲线直接生成“精确 oracle threshold”会造成：

- oracle threshold 高方差；
- 相邻窗口标签剧烈跳变；
- 校准器主要拟合抽样噪声；
- \(10^{-6}\) 结果对随机种子极度敏感。

## 推荐修正

- support 可以使用 \(8,16,32,64\) 张；
- query 应显著大于 support，或使用多个不重叠 query block；
- 使用图像级或序列级 block bootstrap；
- 不把像素视为独立伯努利样本；
- 同时报告 \(10^{-5}\)、\(10^{-4}\) 等较稳定预算；
- 将 \(10^{-6}\) 作为最严格设置，而不是唯一结论；
- 可使用风险上置信界生成保守元标签：

\[
\tau_{B,\alpha}^*
=
\arg\max_\tau P_d^Q(\tau)
\quad
\text{s.t.}\quad
\operatorname{UCB}_\alpha(F_a^Q(\tau))
\le B.
\]

---

# 5.8 “很多窗口”不能替代“很多独立域”

即使从每个数据集生成大量窗口，真正独立的域可能只有 4–6 个。

重叠窗口会制造很大的样本数，但不会自动提供跨域泛化证据。

必须避免：

- 同一图像出现在 train 与 validation 窗口；
- 同一序列被拆到元训练与元验证；
- 先构造重叠窗口，再随机划分；
- 使用目标域结果选择窗口长度；
- 使用目标域结果选择统计特征；
- 使用目标域结果挑选 checkpoint。

## 推荐的域划分方式

更现实且严谨的设计是：

1. 选择 4 个数据集作为 meta-source；
2. 只在 meta-source 内做 LODO 超参数选择；
3. 额外保留至少 2–3 个完全外部数据集作为 final unseen targets；
4. final target 标签只在最终一次评估时打开；
5. 不根据 final target 的结果回头修改方法。

这种设计通常比大量高度相关的窗口更有说服力。

---

# 6. 推荐的 AAAI 版核心方法

建议将问题命名为：

> **Unlabeled Budgeted Operating-Point Adaptation under Domain Shift**

针对 IRSTD 的完整题目可以是：

> **Budget-Aware Operating-Point Adaptation for Unseen-Domain Infrared Small Target Detection**

整体方法只保留两个主模块，减少堆叠感。

---

## 6.1 模块一：跨域尾部间隔学习

基础检测器保持通用，不大幅修改 backbone。

\[
\mathcal L_{\mathrm{det}}
=
\mathcal L_{\mathrm{base}}
+
\lambda_{\mathrm{sep}}
\mathcal L_{\mathrm{tail\text{-}sep}}.
\]

其中：

\[
\mathcal L_{\mathrm{tail\text{-}sep}}
=
\operatorname{SmoothMax}_d
\left[
m+
R_d^- - R_d^+
\right]_+,
\]

\[
R_d^-=
\operatorname{CVaR}_{q_-}(A_d^-),
\]

\[
R_d^+=
\operatorname{LowerTail}_{q_+}(A_d^+).
\]

该模块同时处理：

- 高置信背景候选；
- 低置信真实目标；
- 最坏源域；
- 极低虚警区的排序间隔。

这样，Miss-CVaR 不必再作为第三个独立大模块，而可并入统一的 tail separation。

---

## 6.2 模块二：因果、单调的逆风险曲线元校准器

每个元 episode：

\[
E_{d,e}
=
(S_{d,e},Q_{d,e},B).
\]

训练过程：

1. detector 对 \(S\) 输出连续 score maps；
2. 仅从 \(S\) 提取无标签统计 \(z_S\)；
3. 标签只用于 \(Q\) 上构造风险曲线；
4. 校准器输出预算对应的 logit threshold；
5. 在 \(Q\) 上计算预算违反、检测效用和辅助 oracle 损失。

形式化为：

\[
\widehat\tau_B
=
g_\phi(z_S,\log B).
\]

实现上不使用普通标量 MLP，而使用单调逆风险曲线。

---

## 6.3 输入统计应收缩并去除域 ID 记忆风险

当前方案的统计特征较多，容易被评审认为是 dataset fingerprint engineering。

建议主文只保留四组：

### 1. Score distribution

- 固定 bins 的 score histogram；
- logit histogram；
- 少量关键分位数；
- 高分比例。

### 2. Fixed candidate statistics

- 局部峰值分数直方图；
- 每 MP 候选数；
- 峰值分位数；
- top-k candidate mean。

### 3. Compact image-noise statistics

- 灰度均值、标准差、MAD；
- 梯度统计；
- Laplacian 噪声；
- 高频能量比例。

### 4. Permutation-invariant source-distance statistics

不建议将长度为 \(K\) 的：

\[
(\delta_1,\ldots,\delta_K)
\]

直接输入模型，因为其维度和顺序依赖源域集合，也容易记住数据集身份。

推荐使用：

\[
\min_d\delta_d,
\quad
\operatorname{mean}_d\delta_d,
\quad
\operatorname{std}_d\delta_d,
\quad
\operatorname{softmin}_d\delta_d,
\]

或使用 DeepSets 聚合源域中心。

---

## 6.4 推荐的训练目标

一个可行的总目标是：

\[
\mathcal L
=
\mathcal L_{\mathrm{det}}
+
\lambda_{\mathrm{cal}}
\mathcal L_{\mathrm{cal}}.
\]

其中：

\[
\mathcal L_{\mathrm{cal}}
=
\lambda_v
\left[
\log
\frac{F_a^Q(\widehat\tau_B)+\epsilon}{B}
\right]_+
+
\lambda_u
\left(1-P_d^Q(\widehat\tau_B)\right)
+
\lambda_o
\left|
\widehat\eta_B-\eta_B^*
\right|
+
\lambda_m
\mathcal L_{\mathrm{mono}}.
\]

若阈值扫描不可微，可采用：

- soft threshold；
- differentiable histogram；
- soft candidate survival；
- ordinal bin prediction；
- 离线 curve supervision 加风险加权回归。

---

# 7. Reject 机制的处理建议

当前简单 BCE reject head 会成为第三个松散模块，而且 reject 标签的原定义不充分。

有两个可行选择。

---

## 7.1 选择 A：主文暂时删除 reject

主论文只做：

- threshold prediction；
- budget satisfaction；
- excess risk；
- detection utility；
- causal evaluation。

在 limitations 中说明：

> 当目标域完全超出 meta-domain support 时，零标签条件下缺少可靠的风险认证与拒判机制。

这是当前时间下最稳妥的选择。

---

## 7.2 选择 B：让 reject 成为风险模型的自然结果

拒判条件可以定义为：

\[
\max_{\tau:\operatorname{UCB}(F_a)\le B}
\widehat P_d(\tau)
<
P_{\min},
\]

或者：

\[
d(z_S,\mathcal Z_{\mathrm{source}})
>
\delta_{\max}.
\]

报告：

- coverage；
- non-rejected BSR；
- risk-coverage curve；
- non-rejected \(P_d\)；
- rejection rate；
- 每个域的拒判比例。

必须防止模型通过拒判绝大多数 episode 获得表面安全。

在没有可靠 uncertainty construction 前，建议使用选择 A。

---

# 8. AAAI 所需实验包

实验必须围绕论文主命题组织，而不是把所有可做实验平铺。

---

## 8.1 第一组：问题是否真实存在

先回答：

1. source-optimal threshold 到不同目标域后是否漂移？
2. fixed/source threshold 是否频繁违反同一预算？
3. target oracle 是否能在相同预算下恢复 \(P_d\)？
4. oracle 也失败时，是否说明问题主要来自表征失效？

建议绘制：

- source-target threshold matrix；
- 每个目标域的 \(P_d-F_a\) curve；
- source threshold、rolling threshold、predicted threshold、oracle threshold 的位置；
- 真实目标候选与背景候选的高分尾部分布。

若 oracle threshold 无法恢复低虚警区 \(P_d\)，继续做校准器的意义有限，应优先转向表示学习。

---

## 8.2 第二组：校准方法比较

至少包括：

1. Fixed 0.5；
2. pooled-source threshold；
3. worst-source safe threshold；
4. nearest-source threshold；
5. rolling quantile；
6. EVT/GPD；
7. 普通直接阈值 MLP；
8. 单调逆风险曲线；
9. target-label oracle。

还建议加入合理适配版本：

- UTS；
- TransCal；
- PseudoCal；
- source calibration / conformal-style threshold；
- domain-generalized calibration。

这些方法不一定能原样控制 IRSTD 的极低虚警预算，但需要说明如何适配，以及为何仍不足以解决当前问题。

---

## 8.3 第三组：协议严谨性

主结果必须使用：

- disjoint support/query；
- causal prefix-to-future；
- 至少 3 个完全 unseen target domains；
- 至少 3 个随机种子；
- source/meta-source 内完成所有模型选择；
- 图像级或序列级 bootstrap 置信区间。

Transductive same-set 结果只能作为上界。

---

## 8.4 第四组：方法是否具有一般性

至少采用两个结构明显不同的 detector：

- MSHNet；
- SCTransNet 或 DNANet。

可以有两种设置：

### Detector-specific calibrator

每个 detector 单独训练 calibrator，验证方法结论是否一致。

### Cross-detector calibrator

在一个 detector 上训练 calibrator，再测试另一个 detector，用于验证统计到风险映射的可迁移性。

若跨 detector 过于困难，主文可采用第一种，但至少要证明方法不只对 MSHNet 有效。

---

## 8.5 第五组：压力测试

主文优先保留：

1. support window length；
2. 目标数量污染；
3. 传感器噪声或分辨率变化；
4. support-query temporal drift。

特别建议加入：

> support 与未来 query 存在轻微时间漂移时，阈值是否仍稳定？

这比仅报告 transductive 结果更符合目标流部署叙述。

---

## 8.6 推荐主表

| Method | Target labels | Causal | Monotone | \(P_d@10^{-6}\) ↑ | \(P_d@10^{-5}\) ↑ | BSR ↑ | LogExcess ↓ | Worst-domain \(P_d\) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0 | ✓ | — |  |  |  |  |  |
| Source threshold | 0 | ✓ | — |  |  |  |  |  |
| Rolling quantile | 0 | ✓ | ✓ |  |  |  |  |  |
| EVT/GPD | 0 | ✓ | ✓ |  |  |  |  |  |
| Direct threshold MLP | 0 | ✓ | ✗ |  |  |  |  |  |
| Proposed monotone calibrator | 0 | ✓ | ✓ |  |  |  |  |  |
| Target oracle | All | ✓ | — |  |  | — | — |  |

---

# 9. 评价指标的重新组织

建议主指标围绕四个量：

\[
P_d@B,
\qquad
\mathrm{BSR},
\qquad
\mathrm{Excess},
\qquad
\mathrm{Coverage}.
\]

---

## 9.1 Budget Satisfaction Rate

\[
\mathrm{BSR}(B)
=
\frac{1}{K}
\sum_k
\mathbf 1[
F_{a,k}(\widehat\tau_B)\le B
].
\]

但仅在 3 个目标域上计算 domain-level BSR，会得到：

\[
0,\frac13,\frac23,1
\]

这种过于粗糙的结果。

因此应同时报告：

- domain-level BSR；
- episode-level BSR；
- 每个域单独的 BSR；
- bootstrap confidence interval。

---

## 9.2 Relative Excess

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

当 \(B=10^{-6}\) 时，该指标可能极端放大。

建议增加：

\[
\mathrm{LogExcess}
=
\left[
\log_{10}
\frac{F_a+\epsilon}{B}
\right]_+.
\]

这能更稳定地表示超预算的数量级。

---

## 9.3 Detection utility

至少报告：

- \(P_d@B\)；
- worst-domain \(P_d@B\)；
- average \(P_d@B\)；
- oracle gap；
- hIoU / IoU / nIoU 作为辅助指标。

---

## 9.4 Threshold error 只能作为诊断指标

\[
|\widehat\tau-\tau^*|
\]

不能作为主要成功标准，因为：

- 风险曲线不同区域的阈值敏感度不同；
- 阈值 MAE 小不代表 BSR 高；
- 阈值 MAE 大也可能不影响风险。

建议只在消融或 calibration diagnosis 中报告。

---

# 10. 代码方案中必须立即修正的问题

当前工作树对本节问题的处理状态如下。后文保留原审查理由，便于追溯为什么需要这些契约：

| 问题 | 代码状态 | 剩余证据缺口 |
|---|---|---|
| 10.1 collate | 已修正；metadata 为默认 DataLoader 可拼接字典 | 已有 batch smoke test |
| 10.2 空预测/高尾网格 | 已加 `0/1`、adaptive/exact query events 及 cap 审计 | 主实验需报告覆盖下界，capped 不得写 global exact |
| 10.3 原图预算 | 已在阈值与 matching 前恢复原始 canvas | 需在方法/补充材料报告 interpolation 和 mask 对齐记录 |
| 10.4 plateau | detector 与 RC 统计统一为 `kernel_local_row_major_rank_nms` | 需消融证明该候选定义合理 |
| 10.5 support/query 泄漏 | schema v2 强制 disjoint IDs、curve/manifest SHA、outer/pseudo/source 隔离；手写 IDs 不能训练主结果 | 需生成真实 nested-LODO artifacts |
| 10.6 sigmoid/direct head | 仍保留为 RC 最小基线 | 单调逆风险曲线尚未实现 |
| 10.7 source distance | 已用 permutation-invariant aggregate，并将 fold-specific domains/centers/scale/hash 嵌入 checkpoint | 需证明不只是 dataset fingerprint |
| 10.8 component/candidate | pixel 与 8-connected component metrics 已分开 | 严格单调 candidate-risk 路径尚未实现 |
| 在线后 query 评估 | 已新增 hash-bound replay，reject 时不读取/输出标签指标 | 需完整外层结果与置信区间 |

## 10.1 `SampleMeta` dataclass 与 DataLoader collate

默认 `collate_fn` 不一定稳定处理任意 dataclass。

建议直接返回普通字典：

```python
"meta": {
    "image_id": image_id,
    "dataset_name": self.dataset_name,
    "original_hw": original_hw,
    "input_hw": (self.base_size, self.base_size),
}
```

或者显式实现 `collate_fn`。

---

## 10.2 阈值网格应包含严格空预测工作点

当前最高阈值若仅到 `0.99999`，仍可能低于最大预测分数，从而人为制造“无可行阈值”。在当前 `prediction = probability > threshold` 且 probability 被约束到 `[0,1]` 的契约下，`threshold=1.0` 本身已是严格空预测哨兵；不应再把超出概率域的阈值混入同一 schema。

建议加入：

```python
thresholds = np.unique(np.concatenate([
    thresholds,
    [0.0, 1.0],
]))
```

但“存在空预测点”不代表“oracle 精确”。对 `1e-6` 级虚警预算，固定网格可能跳过 `0.99999` 与 `1.0` 之间的安全非空工作点。主协议必须把 query 中的高尾唯一 score/event thresholds 并入 sweep，并在 curve manifest 记录 exact/adaptive 模式、事件数、覆盖下界与是否因 cap 而失去精确性。

---

## 10.3 原图像素预算不能在 resize 空间直接解释

保存 `original_hw` 并不会自动恢复物理意义。

若在 \(256\times256\) 上阈值与计数，则 `Fa/MP` 仍是 resize 空间的指标。

最终论文应采用：

- 原比例 pad 推理；或
- 将连续预测图上采样回原始分辨率后再阈值与匹配；
- 明确 interpolation；
- 明确 padding；
- 明确有效区域 mask。

---

## 10.4 局部峰值平台会产生重复候选

如下规则：

```python
background >= pooled - 1e-7
```

可能将平坦高分平台上的多个像素全部标记为峰值。

需要：

- plateau-aware NMS；
- 连通平台只保留一个代表点；
- 局部对比度过滤；
- 固定最小间距；
- 或使用 soft-peak weighting。

---

## 10.5 重叠窗口会造成数据泄漏

正确顺序是：

1. 先按域、序列或图像划分 train/validation/test；
2. 再在各自划分内部构造窗口。

错误顺序是：

1. 先构造大量重叠窗口；
2. 再随机拆分 train/validation。

---

## 10.6 Sigmoid threshold head 不适合极端尾部

不建议：

```python
self.threshold_head = nn.Sequential(
    nn.Linear(hidden_dim, 1),
    nn.Sigmoid(),
)
```

建议：

```python
self.threshold_logit_head = nn.Linear(hidden_dim, 1)
```

训练时预测标准化 logit threshold，部署时再做：

```python
threshold = torch.sigmoid(threshold_logit)
```

---

## 10.7 Source-distance 特征需要 permutation invariance

不要直接输入：

```text
distance_to_source_1
distance_to_source_2
...
distance_to_source_K
```

建议输入聚合量，或使用 DeepSets：

```python
source_embedding = deepsets(source_distances)
```

避免模型记住源域顺序和数据集身份。

---

## 10.8 连通域预算与候选预算需要分开实现

建议明确两个评估路径：

### 路径 A：单调候选风险

用于：

- 训练；
- 元校准；
- 单调阈值预测；
- 理论分析。

### 路径 B：传统 connected-component metric

用于：

- 与既有 IRSTD 工作比较；
- 结果兼容；
- 补充评估。

不要在方法推导中默认两者完全等价。

---

# 11. 论文定位、题目与投稿关键词

## 11.1 不推荐的题目

> RC-IRSTD: Risk-Calibrated Infrared Small Target Detection under Unseen Domains

问题：

- “Risk-Calibrated” 容易被理解为具有风险保证；
- 方法目前只提供经验性跨域元校准；
- 审稿人可能据此要求 conformal 或 finite-sample guarantee。

---

## 11.2 推荐题目

### 首选

**Learning Budget-Aware Operating Points from Unlabeled Target Streams for Cross-Domain Infrared Small Target Detection**

### 更方法化

**Unlabeled Budgeted Operating-Point Adaptation under Domain Shift**

可在副标题或摘要中说明 IRSTD 是主要验证任务。

### 更简洁

**Meta-Calibrated Low-False-Alarm Detection under Unseen Domains**

其中必须说明 meta-calibrated 是跨源域经验映射，不是任意目标域认证。

---

## 11.3 推荐投稿关键词

### Primary

**ML: Transfer, Domain Adaptation & Continual Learning**

### Secondary

- CV: Object Detection, Segmentation & Scene Understanding
- ML: Bayesian Learning & Uncertainty Quantification
- ML: Evaluation, Benchmarking, Datasets & Analysis
- RU: Uncertainty Representations
- ML: Time-Series & Data Streams

Primary 应根据方法贡献选择，而不是根据红外应用场景选择。

---

# 12. 推荐的三条论文贡献

建议最终贡献压缩为三条。

## Contribution 1：问题贡献

> 提出未知域、零目标标签、显式虚警预算下的 causal operating-point adaptation 问题，并形式化说明仅依赖无标签目标分布时，真实背景虚警风险一般不可识别。

## Contribution 2：方法贡献

> 提出基于 disjoint support-query 元 episode 的单调逆风险曲线校准器，并通过跨域目标—背景尾部间隔学习提高低虚警区域的可校准性。

## Contribution 3：评测贡献

> 建立 external unseen-domain、causal prefix-to-future、budget satisfaction、excess risk、window-length sensitivity 和 contamination stress test 的严格评测协议。

不建议把以下内容写成主要贡献：

- 新增若干 Python 文件；
- 使用 MSHNet；
- 增加一个普通 reject head；
- 使用一个小型 MLP；
- 扫描多个阈值。

---

# 13. 建议加入的理论结果

即使不提供无条件风险保证，也建议加入一个简短、清晰的不可识别性命题。

## 13.1 命题：无标签目标风险不可识别

考虑两个目标联合分布 \(P_t\) 和 \(Q_t\)，满足：

\[
P_t(X)=Q_t(X),
\]

并且模型分数的无标签边缘分布相同：

\[
P_t(S)=Q_t(S).
\]

但在高分尾部：

- \(P_t\) 中高分样本主要是真实目标；
- \(Q_t\) 中相同高分样本主要是背景。

任何只观察无标签图像或分数统计的算法，在两个分布上都会输出相同阈值，但其真实虚警风险可能完全不同。

因此：

> 不存在一个只依赖无标签目标分布、并对任意目标域都提供真实虚警控制的通用算法。

## 13.2 该命题的作用

它可以解释：

- 为什么不能使用 guaranteed、certified、distribution-free；
- 为什么需要跨域元分布假设；
- 为什么需要 external-domain evaluation；
- 为什么 BSR 与 Excess 是经验指标；
- 为什么 reject/OOD detection 是潜在扩展；
- 为什么 conformal 方法需要标签校准集或额外交换性条件。

## 13.3 建议明确的条件化假设

方法可以在以下经验假设下工作：

1. 目标域来自与 meta-source 相关的环境族；
2. 无标签统计与风险曲线之间存在稳定关系；
3. support 与 future query 的分布漂移有限；
4. detector 在目标域上仍保留一定目标—背景排序能力；
5. 候选提取过程在不同域间具有可比性。

论文应明确这些是经验结构假设，不是无条件保证。

---

# 14. 7 页主文的结构与取舍

以“7 页主内容”作为当前保守排版假设，不能把完整工程方案全部搬入主文。AAAI-27 页限仍须以后续发布的当届 Author Kit/Main Track Call 为准。

## 14.1 推荐页数分配

| 部分 | 建议页数 |
|---|---:|
| Introduction | 0.8 |
| Related Work | 0.6 |
| Problem Formulation + Identifiability | 0.8 |
| Method | 2.0 |
| Experimental Setup | 0.8 |
| Main Results + Ablations | 1.7 |
| Limitations + Conclusion | 0.3 |

## 14.2 主文必须保留

- 问题定义；
- 不可识别性命题；
- causal support-query protocol；
- 单调逆风险曲线；
- 统一尾部间隔损失；
- 最关键主表；
- source vs oracle 可行性分析；
- 核心消融；
- limitations。

## 14.3 放入补充材料

- 完整代码目录；
- 所有统计特征定义；
- 多套 matching rule；
- 完整阈值矩阵；
- 更多预算；
- 全部窗口长度；
- 全部噪声、模糊与分辨率扰动；
- 完整 qualitative cases；
- 所有训练命令；
- 超参数网格；
- 第二套 component metric；
- 更多 backbone 结果。

---

# 15. AAAI-27 最小可行投稿版本

若必须冲当前 AAAI-27，建议将范围压缩如下。

## 15.1 必须保留

- 已训练或可快速训练的 MSHNet；
- 连续 score map 导出；
- disjoint support-query episode；
- causal prefix-to-future 测试；
- rolling quantile；
- EVT/GPD；
- 普通 MLP 基线；
- 单调预算阈值校准器；
- 至少 3 个 external unseen domains（且需额外独立域支撑 inner LODO）；
- \(P_d@B\)、BSR、LogExcess；
- target oracle；
- 不可识别性命题。

## 15.2 可以暂缓

- 独立 Miss-CVaR 模块；
- 普通 BCE reject head；
- 同时控制 pixel budget 与非单调 component budget；
- 过长的手工统计列表；
- 完整嵌套所有数据集的 detector 重训练；
- 大量 TTA 基线；
- 第三个 backbone；
- 完整 conformal 扩展。

## 15.3 最小方法结构

```text
Multi-source detector
        │
        ▼
Tail-separation learning
        │
        ▼
Unlabeled support window
        │
        ▼
Compact score/candidate statistics
        │
        ▼
Monotone inverse-risk calibrator
        │
        ▼
Threshold for future query stream
```

---

# 16. 优先执行路线与 Go/No-Go 闸门

不建议继续优先扩展代码目录。应先验证研究假设。

---

## 16.1 Gate 0：评估基础设施

必须先完成：

- 连续 score map 导出；
- 精确阈值扫描；
- 像素风险；
- 固定候选风险；
- connected-component 兼容指标；
- 原图分辨率评估；
- oracle threshold；
- source-target threshold matrix。

---

## 16.2 Gate 1：问题是否成立

至少满足：

1. 三个目标域中至少两个存在明显 threshold drift；
2. source/fixed threshold 经常违反预算；
3. target oracle 在相同预算下能显著恢复 \(P_d\)；
4. oracle gap 不是完全由表征失效造成；
5. 目标与背景候选在高分区仍存在一定排序空间。

若 oracle 也无法恢复性能，应优先改进表示学习。

---

## 16.3 Gate 2：无标签统计是否可预测 future risk

先使用简单模型：

- 线性回归；
- Ridge；
- Random Forest；
- Gradient Boosting。

任务必须是：

\[
z(S)
\rightarrow
\tau^*(Q),
\qquad
S\cap Q=\varnothing.
\]

而不是：

\[
z(W)
\rightarrow
\tau^*(W).
\]

若简单模型完全无效，直接堆 MLP 通常不会解决本质问题。

---

## 16.4 Gate 3：必须超过简单无监督基线

至少需要在真正 unseen target 上优于：

- rolling quantile；
- EVT/GPD；
- pooled-source threshold；
- worst-source safe threshold。

且不是只在伪目标源域上有效。

---

## 16.5 Gate 4：预算单调性

检查：

\[
B_1<B_2
\Rightarrow
\widehat\tau(B_1)\ge\widehat\tau(B_2).
\]

必须报告：

- monotonicity violation rate；
- 未见预算插值；
- 多预算共享模型与单预算模型比较。

---

## 16.6 Gate 5：安全性与效用必须同时报告

每个实验同时报告：

- \(P_d\)；
- BSR；
- LogExcess；
- worst-domain performance；
- coverage，若使用 reject。

不允许仅通过过高阈值或大量拒判获得表面安全。

---

## 16.7 推荐执行顺序

### 阶段 1：两天诊断

- 导出 3–4 个跨域 score maps；
- 绘制 \(P_d-F_a\) curves；
- 比较 fixed/source/oracle；
- 检查 threshold drift；
- 测试 rolling quantile。

### 阶段 2：因果元数据

- 按域与序列划分；
- 构建 disjoint support/query；
- 生成 future-query oracle；
- 统计 oracle 方差；
- 评估窗口长度。

### 阶段 3：简单可预测性验证

- Ridge；
- Random Forest；
- rolling quantile；
- EVT；
- threshold MAE 与 BSR。

### 阶段 4：单调逆风险曲线

- logit threshold；
- monotone increments；
- risk-aligned loss；
- 多预算训练；
- unseen-budget interpolation。

### 阶段 5：尾部间隔学习

- SLS baseline；
- pixel top-k；
- local-peak Tail-CVaR；
- unified tail separation；
- worst-domain aggregation。

### 阶段 6：完整外部评测

- 至少 3 个 unseen domains；
- 3 个随机种子；
- causal 主结果；
- transductive 上界；
- bootstrap confidence intervals。

---

# 17. 最终判断

**当前方案可以发展成 AAAI 论文，但不能按现有形式直接投稿。**

经过本轮代码修正，support/query、oracle/reject 和 provenance 已有可审计契约，但这些还只是“代码条件具备”，不是“论文证据已完成”。当前最可能导致 Phase 1 拒稿的原因依次是：

1. 手工统计加 MLP 回归阈值显得启发式；
2. 尚无完整 nested-LODO artifacts 和 external target 结果证明协议确实被遵守；
3. 当前直接阈值基线尚未通过 query replay 证明安全性与效用的联合收益；
4. component 风险的非单调性未处理；
5. 阈值回归损失与真实风险不对齐；
6. 多预算预测缺少单调性约束；
7. 独立域数量少，窗口数量造成伪样本规模；
8. Tail-CVaR 可能只造成整体 logit 平移；
9. “Risk-Calibrated” 容易被误解为形式化保证；
10. 方法只在 IRSTD 与 MSHNet 上验证，广义 AI 意义不足。

完成以下三项后，论文的 AAAI 匹配度会发生实质提升：

> **Disjoint causal meta-episodes**  
> **Monotone inverse risk curve**  
> **Formal identifiability analysis**

推荐最终论文定位：

> 研究在未知、无标签目标流中，如何根据有限无标签前缀与用户指定风险预算，经验性地选择后续流的部署工作点。

推荐最终 Primary Area：

> **ML: Transfer, Domain Adaptation & Continual Learning**

---

# 18. 参考资料

## 18.1 AAAI-27 官方信息

1. [AAAI-27 Conference Homepage（截止日期与会议时间已核实）](https://aaai.org/conference/aaai/aaai-27/)
2. AAAI-27 Main Technical Track Call、Review Process、Areas and Topics：截至 2026-07-14 本次核验未找到可确认的当届正文；待官方发布后补链并重新审核第 2、14 节。

## 18.2 校准与风险控制相关工作

> 以下是评审草稿的 working bibliography，本轮未对每条文献的出版元数据和正文支撑关系做逐条引用审计。写入论文前必须核验 DOI/arXiv/会议信息以及对应 claim；本列表本身不是已审核的 BibTeX 来源。

1. Mozafari et al.  
   *A Novel Unsupervised Post-Processing Calibration Method for DNNs with Robustness to Domain Shift.*  
   Unsupervised Temperature Scaling, 2019.  
   <https://arxiv.org/abs/1911.11195>

2. Wang et al.  
   *Transferable Calibration with Lower Bias and Variance in Domain Adaptation.*  
   TransCal, 2020.  
   <https://arxiv.org/abs/2007.08259>

3. Hu et al.  
   *PseudoCal: A Source-Free Approach to Unsupervised Uncertainty Calibration in Domain Adaptation.*  
   2023.  
   <https://arxiv.org/abs/2307.07489>

4. Gong et al.  
   *Confidence Calibration for Domain Generalization under Covariate Shift.*  
   2021.  
   <https://arxiv.org/abs/2104.00742>

5. Bates et al.  
   *Distribution-Free, Risk-Controlling Prediction Sets.*  
   2021.  
   <https://arxiv.org/abs/2101.02703>

6. Cohen et al.  
   *Cross-Validation Conformal Risk Control.*  
   2024.  
   <https://arxiv.org/abs/2401.11974>

## 18.3 IRSTD 与跨域相关工作

1. Liu et al.  
   *Infrared Small Target Detection with Scale and Location Sensitivity.*  
   CVPR 2024.  
   <https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html>

2. Pang et al.  
   *Rethinking Evaluation of Infrared Small Target Detection.*  
   <https://arxiv.org/abs/2509.16888>

3. Lu et al.  
   *Rethinking Generalizable Infrared Small Target Detection: A Real-scene Benchmark and Cross-view Representation Learning.*  
   <https://arxiv.org/abs/2504.16487>

4. *Rethinking Representations for Cross-Domain Infrared Small Target Detection: A Generalizable Perspective from the Frequency Domain.*  
   <https://arxiv.org/abs/2604.01934>

5. Li et al.  
   *Ivan-ISTD: Rethinking Cross-domain Heteroscedastic Noise Perturbations in Infrared Small Target Detection.*  
   <https://arxiv.org/abs/2510.12241>

---

# 附录 A：一句话投稿定位

> **本文研究在完全没有目标域标签的情况下，能否根据未知目标流前缀的无标签分数与成像统计，经验性地预测满足指定虚警预算的后续部署工作点。**

# 附录 B：论文摘要逻辑骨架

```text
Motivation:
Fixed operating points fail under unseen-domain shifts, especially in
extreme low-false-alarm dense detection.

Problem:
Given labeled source domains and an unlabeled target-stream prefix,
predict a deployment threshold for future target samples under a
user-specified false-alarm budget.

Challenge:
True target-domain background risk is not identifiable from unlabeled
scores without additional assumptions.

Method:
1. Cross-domain target-background tail-separation learning.
2. Disjoint support-query meta-training.
3. Monotone inverse-risk curve prediction conditioned on unlabeled
   target statistics and budget.

Evaluation:
External unseen domains, causal prefix-to-future protocol, budget
satisfaction, excess risk, worst-domain detection, and stress tests.

Claim:
Empirical budget-aware adaptation under a learned meta-domain
assumption, not distribution-free risk certification.
```

# 附录 C：投稿前检查清单

- [ ] support 与 query 完全独立；
- [ ] final target 不参与任何模型选择；
- [ ] 明确 transductive 与 causal 的区别；
- [ ] 阈值预测满足预算单调性；
- [ ] oracle 定义允许空预测但不把它误写为不可行；
- [ ] reject 定义基于安全性与效用同时不可满足；
- [ ] 主风险指标具有单调性；
- [ ] component metric 与 candidate metric 分开；
- [ ] 使用原始分辨率或明确等价变换；
- [ ] 报告 \(P_d\)、BSR、LogExcess 与 worst-domain result；
- [ ] 使用至少 3 个真正 unseen domains；
- [ ] 使用至少 3 个随机种子；
- [ ] 使用图像级或序列级置信区间；
- [ ] 至少比较 rolling quantile 与 EVT；
- [ ] 至少使用两个 detector；
- [ ] 不使用 guaranteed、certified 或 distribution-free 等过强措辞；
- [ ] 关键方法和协议全部放入当届页限内（当前按 7 页假设排版）；
- [ ] 代码、数据划分、checkpoint 与命令可复现；
- [ ] 双盲版本不泄露作者和仓库身份；
- [ ] 所有参考文献真实存在且与正文主张一致。
