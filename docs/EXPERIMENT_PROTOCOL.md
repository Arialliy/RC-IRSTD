# RC-IRSTD 严格实验协议

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

本文档冻结 claim-bearing 实验的输入、划分、阈值、模型选择和指标契约。

## 1. 数据角色

每个数据域只能在一个 outer fold 中扮演以下角色之一：

- `meta_source`：可在 official train 内用于 detector 与 calibrator 的内层训练/验证；
- `outer_target`：official test 标签仅在最终一次离线评价时打开；
- `external_target`：从未参与任何开发决策，最终一次评测。

数据集间存在重叠、派生或重新打包关系时，不能把它们计作独立域。必须保存去重审计记录。

## 2. 数据划分顺序

正确顺序：

1. 按数据集、序列或图像 ID 固定 `detector_train`、`detector_val`、`meta_train`、`meta_val`；
2. 保存有序 ID 和 SHA-256；
3. 在每个子集内部构造 support/query 窗口；
4. 检查所有角色的 ID 交集为空。

禁止先生成大量重叠窗口再随机拆分。

## 3. Detector 协议

对 outer target `D`：

- detector 不读取 `D` 标签；
- detector checkpoint 只由 source-internal validation 选择；
- main selection 使用原图 pixel budget，不使用 outer target test；
- segmentation-only 与 tail-separation 使用相同初始化、数据增强、epoch 和随机种子；
- 训练日志必须记录 loss ramp 和 logit scale。

对 inner pseudo-target `A`：

- inner detector 训练源中必须排除 `A`；
- `A` 的 official train 子集只用于构建 pseudo-target episodes；
- 若排除后只剩一个训练域，该 fold 只能标记为 engineering smoke，不能作为 strict multi-source main result。

## 4. Score artifact 协议

每张图的 score record 至少包含：

- 连续 probability/logit map；
- image ID 与 sequence ID；
- original shape 与 inference shape；
- detector checkpoint hash；
- split role；
- inference mode 和 normalization。

正式结果优先使用 `native_pad` 或 `tiled` 恢复到原始画布。强制 resize 只作诊断。

最终目标的 label-free score artifact 不嵌入 mask。标签工件与 score 工件分目录保存。

## 5. Meta episode 协议

一个 episode 为：

```text
support IDs S
query IDs Q
budget grid B[0:J]
label-free support features
query background logits/weights
query object logits
oracle threshold logits (auxiliary only)
provenance
```

硬约束：

- `S ∩ Q = ∅`；
- `S` 读取时 `load_mask=False`；
- budget grid 严格从松到严递减；
- 同一 episode 的预算合并为 `[J]`；
- meta train 与 meta val 的 image/sequence IDs 不相交；
- static IID 称为 `prefix_holdout`；只有可验证时序数据才称为 temporal/causal。

## 6. Calibrator 训练协议

- feature normalizer 仅在 meta train 拟合；
- source-distance 通过 permutation-invariant encoder；
- 主损失为 query risk violation + utility；
- oracle regression 只作低权重辅助；
- structural monotonicity 在每次验证中断言；
- 不包含 Reject head；
- 训练预算之外禁止外推。

## 7. Exact hard replay

所有主结果使用：

```python
prediction = logit > predicted_threshold_logit
```

对每个 query image 重放后累加：

- background false pixels；
- total original pixels；
- GT object count；
- detected object count；
- pixel false-alarm rate；
- object Pd。

surrogate 风险和 surrogate Pd 只用于训练，不进入论文结果表。

## 8. Model selection

预注册 rank key：

```text
(BSR, -LogExcess, mean_Pd)
```

按字典序最大化。不得使用：

- threshold MAE 单独选模；
- outer target 结果；
- target oracle 的事后观察；
- test split IoU；
- 人工查看 final target qualitative cases 后改超参。

## 9. 风险指标

对 episode-budget 对 `(e,j)`：

```text
satisfied[e,j] = Fa_pixel[e,j] <= B[j]
excess[e,j] = max(Fa_pixel[e,j] / B[j], 1)
log_excess[e,j] = log(excess[e,j])
```

报告：

- episode-level BSR；
- domain-level BSR；
- mean LogExcess；
- `Pd@B`；
- worst-domain Pd；
- bootstrap confidence interval（按图像/序列或 episode block 重采样）。

## 10. Component 兼容评测

connected-component 评价使用固定的 connectivity、匹配规则和原始画布。建议主兼容规则：

- 8-connectivity；
- GT/预测一对一 overlap matching；
- 未匹配预测分量计作 FP component；
- centroid-distance 仅作旧文献兼容。

该指标不进入 monotone calibrator 的主理论和 checkpoint 选择。

## 11. 最低实验矩阵

### Problem diagnosis

- source→target threshold matrix；
- fixed/source/worst-source/oracle；
- target/background high-tail separation；
- oracle recoverability。

### Detector ablation

- segmentation-only；
- background tail only；
- old per-image hinge；
- true domain-tail separation；
- dilation/plateau/ramp ablation。

### Calibrator ablation

- rolling quantile；
- EVT/GPD；
- direct MLP；
- monotone oracle regression；
- proposed risk-aligned objective；
- feature groups；
- budget interpolation。

### Stress tests

- support sizes 8/16/32/64/128；
- query size；
- target-density contamination；
- noise/blur/resolution shift；
- support-to-query temporal drift；
- source-order permutation；
- second backbone。

## 12. Stop/redirect conditions

停止 calibrator 主线并转向 representation learning：

- target oracle 在低虚警区也不能恢复 Pd；
- 真实目标在高分尾部不存在；
- target/background 候选排序完全重叠。

删除或降级 tail-separation：

- 只导致所有 logits 整体移动，oracle Pd 不改善；
- source-val IoU 明显恶化；
- 收益不能在至少两个 unseen targets 重现。

降级 calibrator 为 batch/offline setting：

- 只有 transductive same-set 统计有效；
- prefix-to-future BSR/LogExcess 不优于 rolling quantile；
- 改善主要来自过高阈值导致 Pd 塌陷。
