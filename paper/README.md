# 论文结果产出目录说明

本目录不包含虚构的 benchmark 数值。真实实验完成后，论文材料必须围绕最终两阶段、无 Reject 方法生成。

## 主结果

- original-resolution pixel-risk 主表；
- `Pd@1e-4/1e-5/1e-6`；
- BSR、LogExcess、worst-domain Pd；
- fixed/source/worst-source、rolling quantile、EVT/GPD、direct MLP、monotone regression、proposed、oracle；
- external unseen-domain prefix-to-future 结果；
- connected-component FA/MP 兼容附表。

## 方法消融

- segmentation-only vs domain-tail separation；
- per-image hinge vs true domain-level two-tail hinge；
- GT dilation、plateau collapse、risk ramp；
- scalar-budget vs grouped `[J]`；
- oracle regression vs query-risk-aligned loss；
- support statistics groups；
- budget interpolation；
- support size、target contamination、noise/resolution/temporal drift。

## 投稿前强制检查

- support/query 完全不重叠；
- support 标签未加载；
- pseudo-target detector 排除该域；
- outer target 不参与模型选择；
- checkpoint 按 BSR → LogExcess → Pd；
- 主方法无 Reject；
- 预算范围外不外推；
- 所有最终数字来自 hard-threshold exact replay；
- component metric 不写成单调受控主风险；
- 不使用 certified、guaranteed、distribution-free 等过强措辞；
- 所有数字可追溯到 commit/hash、config、split、seed、checkpoint 和命令。
