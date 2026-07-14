# RC-IRSTD 工程验证报告

验证日期：2026-07-14

## 1. 源码与环境边界

本次执行环境无法解析或下载用户指定的实时 GitHub 地址：

```text
https://github.com/Arialliy/RC-IRSTD
```

因此工程以用户文件库中最新保存的源码快照：

```text
RC_IRSTD_AAAI_Implementation.zip
```

为基础续建。公开 MSHNet 的网络结构和 SLS 接口另行核对，并以内置实现集成到本仓库。该目录不是对无法验证的远端 commit 的逐字节镜像；推回 GitHub 前应再与远端分支做一次 diff。

## 2. 静态验证

已执行：

```bash
python -m compileall -q rc_irstd tests
for script in scripts/*.sh; do bash -n "$script"; done
pytest -q
```

结果：

```text
25 passed
```

测试覆盖包括：

- CRC 小样本可行性与联合预算损失；
- 路径、数据集元数据和均衡采样；
- 8/16-bit 图像读取；
- 单像素目标保留缩放；
- 静态 IID 与 temporal 窗口；
- fixed local-peak 单调性；
- SLS + Tail-CVaR + Miss-CVaR；
- 内置 MSHNet 前向/反向/checkpoint roundtrip；
- 单调双风险曲线；
- formal 风险与标准 IRSTD 指标；
- image-shot 校准单位精确计数；
- deployment state、漂移/OOD 监控；
- 单候选坐标序列化回归；
- LODO 排除协议与 artifact fingerprint。

## 3. 完整 synthetic 闭环

执行：

```bash
bash scripts/validate_release.sh /mnt/data/rc_irstd_release_validation_final
```

闭环实际包含：

```text
3 synthetic domains
→ domain-balanced detector training
→ best_budget / best_iou checkpoint
→ continuous score export
→ IID support/query episode construction
→ budget-focused monotone risk-curve training
→ zero-label operating-point evaluation
→ explicit-unit CRC evaluation
```

最终：

```text
status = passed
```

关键产物：

```text
run/detector/best.pt
run/detector/best_budget.pt
run/detector/best_iou.pt
run/curve/best.pt
run/zero/summary.json
run/crc/summary.json
smoke_summary.json
```

## 4. 双轨评测验证

在 synthetic DomainC score records 上实际执行：

```bash
python -m rc_irstd.pipelines.evaluate_scores ...
```

成功生成：

```text
formal_curve.csv
component_curve.csv
summary.json
```

Formal 曲线使用：

- pixel false rate；
- threshold-independent fixed false local peaks/MP。

文献兼容曲线使用：

- connected-component FA/MP；
- object Pd；
- IoU、nIoU、hIoU；
- Precision、Recall、F1。

## 5. 部署闭环验证

实际执行 causal zero-label deployment：

```text
12 input score records
2 images/sequence warm-up
8 future images processed
2 threshold updates
8 masks written
65 candidates written
0 rejected future images
```

生成：

```text
deployment_state.json
summary.json
candidates.csv
masks/*.png
```

该验证期间发现并修复了单候选坐标被错误当作二维 coordinate 的问题，并新增回归测试。

## 6. Nested LODO 调度验证

使用 `configs/lodo_smoke.yaml` 执行 dry-run，解析出 **17 条命令**，覆盖：

- final source-set detector；
- inner pseudo-target detector；
- source-set 去重缓存；
- target/pseudo/source score export；
- train/eval stride 分离；
- risk episode；
- risk curve；
- zero-label；
- image-unit CRC；
- baseline evaluation。

所有 artifact 跳过逻辑均要求 fingerprint 匹配，不再仅凭文件存在就复用。

## 7. 命令入口验证

安装后共加载 **16 个** `rc-irstd-*` console entry points，包括：

```text
train-detector
export-scores
evaluate-scores
build-episodes
train-curve
eval-zero
calibrate
run-lodo
apply-threshold
deploy
smoke
```

## 8. 验证边界

已验证的是软件与协议闭环，不是最终研究结论。本环境未执行：

- 真实公开 IRSTD 数据的 400-epoch MSHNet 长训练；
- 六域、多 seed、第二 backbone 的完整实验；
- 真实目标域在 `1e-6` 预算下的统计功效验证；
- 最终 benchmark 权重和性能表；
- 多机多卡吞吐与显存报告。

因此发布包不附带真实 benchmark 权重，也不把 synthetic 数值解释为真实模型性能。
