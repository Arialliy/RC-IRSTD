# RC-IRSTD 基线与 smoke-test 记录

> 更新：2026-07-14  
> 性质：工程验收记录，**不是论文结果表**。除非明确写明，下列数字均不得用于支撑方法有效性或 AAAI 投稿主张。

## 1. 历史 detector smoke 环境

- 容器：`rrunet-course:latest`；
- PyTorch：2.1.2；
- GPU：3 × NVIDIA GeForce RTX 3090（24 GiB）；
- 工作树：包含本轮未提交修改，因此不将当前 Git revision 声称为可复现结果标识。

本节环境对应第 3--4 节的历史 smoke 记录；第 5 节当前回归使用另一项目虚拟
环境，二者不可混作同一次可复现实验。

## 2. 本地数据入口

| 数据集 | train | test | 状态 |
|---|---:|---:|---|
| IRSTD-1K | 800 | 201 | split、image 和 mask 解析通过 |
| NUDT-SIRST | 663 | 664 | split、image 和 mask 解析通过 |
| NUAA-SIRST | 213 | 214 | `_pixels0` mask 解析通过；原 image/mask 尺寸不一时记录并最近邻对齐 |

这三个域足以检查代码路径，但不足以支撑“严格 nested LODO + 至少 3 个 final unseen targets”的主结论。固定一个 outer target 后仅剩两源域，inner LODO detector 只有一个训练域；此情形只允许标记为 `single_source_inner_smoke_not_main_result`。

## 3. 风险训练 smoke test

> 本节数字来自旧 `separate` Tail/Miss 路径的本地一步记录，不是最终
> `margin` 方法。当前工作树未保留可逐项核验这些数值的原始日志、命令和
> artifact hash，因此只作为 legacy locally-unverified engineering note
> 留档；重新运行并归档前不得进入论文或补充材料结果表。

### 三域、单 GPU、一步

| 项目 | 实测值 |
|---|---:|
| 输入/裁剪 | 64 × 64 |
| 每域 batch | 1 |
| 优化步 | 1 |
| total loss | 2.2709577 |
| SLS loss | 2.1527669 |
| Tail loss | 0.6793334 |
| Miss loss | 0.5025737 |

三个域均产生了 tail-risk 记录，checkpoint 策略为 `fixed_last_no_test_or_target_validation`。该次运行只证明 forward/backward/checkpoint 路径可执行，不证明收敛或泛化。

### 两域、两 GPU DataParallel、一步

| 项目 | 实测值 |
|---|---:|
| 输入/裁剪 | 64 × 64 |
| 每域 batch | 2 |
| 总 batch | 4 |
| 优化步 | 1 |
| total loss | 2.2452281 |

域样本以 round-robin 顺序交错，两个 replica 均收到两域混合。入口已要求 `batch_per_domain % visible_gpu_count == 0`，否则提前报错，避免 BatchNorm running statistics 与首域绑定。

## 4. 历史 score-map 与高尾阈值 smoke test

使用现有 IRSTD-1K paper weight，以 64 × 64 网络输入完成 201 张 test image 推理，随后把连续概率图恢复到每张原始分辨率：

- 导出 201/201 个 NPZ；
- manifest 标记 `restored_to_original_hw=true`；
- checkpoint SHA-256：`82d29a4cffbd507fcf3c9dcb830af07c920474acf0ffdb07e8368ef31a6dfe66`；
- 该历史 weight 无源域/fold metadata，因此 provenance 正确标记为 `legacy_unverified`，不得进入 RC 主协议。

对前 5 张 query 进行 adaptive 高尾 sweep（event cap=128）：

- 754 个操作点；
- 102 个 `score >= 0.99` 的唯一 query event thresholds 全部覆盖；
- `event_thresholds_capped=false`；
- 曲线包含严格 `threshold=1.0` 空预测点；
- 末两个非空/空点为：`threshold=0.9999994636, FP_pixels=1`，`threshold=0.9999995828, FP_pixels=0`。

该 5-image 曲线只用于验证高尾 event 与 manifest 契约，不是数据集性能结果。

这组导出产生于严格 score/label 拆分之前，只是历史工程记录。当前 schema-v3 score 主协议不允许 score NPZ 嵌入 mask，因此不得把该旧产物直接改名为 verified curve/episode；必须用当前 `export_score_maps` 重新生成无标签 score，再用 `export_label_maps` 生成独立标签附件。

## 5. 协议回归验收

2026-07-14 使用项目虚拟环境（Python 3.12.3，PyTorch 2.9.1+cu130）对当前
工作树执行全量测试：

```text
139 passed, 10 subtests passed
```

已覆盖的关键 fail-closed 路径包括：

- 目标域无 `masks/` 目录时仍可完成 label-free score export，score NPZ 不含 mask；
- 训练 source split 的逐图像内容 SHA、mask SHA 与 target/source leaf collision audit；
- 错数据根、错 label attachment、score/label 同目录、文件篡改均被拒绝；
- curve threshold plan、event audit、matching contract 和所有风险/计数列均从 query score+label 独立重算；
- Stage 1 域级双尾分离、GT 邻域排除与 plateau collapse；
- Stage 2 完整预算单调曲线、verified query-risk loss、原分辨率 exact replay 和 v5 no-Reject checkpoint 合同；
- adapter 在 CPU 确定性复算 context 决策，v5 使用复算阈值评估；旧 reject 基线的 reject 分支不打开 label manifest。

这些是协议与实现验收，仍不是检测/校准方法有效性证据。

## 6. 不得当作当前基线的旧记录

`repro_runs/` 中的历史 `metric.log` 来自 legacy 训练/评估路径：它在每轮使用官方 test split 选最佳 checkpoint，且旧 PD/FA 实现与本轮原分辨率、一对一 8-connected matching 契约不同。因此本文不复制这些数字，也不用它们声称已复现论文结果。

## 7. 尚缺的 claim-bearing evidence

- 完整 outer-fold 和 inner-LODO detector checkpoints；
- 至少第四个独立合法数据域才能运行两源 inner-LODO；推荐主证据仍需至少 4 个 meta-source 与 3 个额外 external unseen targets；
- 至少 3 个随机种子；
- rolling quantile、EVT/GPD、nearest-source 和 oracle 对比；
- 直接 RC 与单调逆风险曲线的消融；
- query replay 后的 Pd、BSR、Excess、coverage 和 worst-domain 置信区间。

在这些证据完成之前，结果表应保留 `TBD`，不填入推测值或合成数字。
