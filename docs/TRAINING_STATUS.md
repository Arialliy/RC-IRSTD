# 训练启动与验证状态

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

## 已实际运行

### 1. 静态和单元验证

```text
compileall: PASS
all shell launchers: PASS
pytest: 30 passed
```

### 2. MSHNet 集成

已执行 MSHNet：

- forward；
- segmentation + domain-tail separation backward；
- optimizer step；
- checkpoint save/load roundtrip。

结果：PASS。

### 3. 两阶段 synthetic smoke training

已启动并完成 3 epoch：

```text
synthetic score records
→ grouped three-budget train/validation meta episodes
→ no-Reject monotone calibrator
→ query-risk-aligned optimisation
→ exact hard replay
→ best checkpoint selected by BSR/LogExcess/Pd
```

产物位于：

```text
validation/two_stage_smoke/
├── calibrator/best.pt
├── calibrator/best_hard_replay.json
├── calibrator/metrics.jsonl
├── smoke_summary.json
├── train_meta.npz
└── val_meta.npz
```

该运行证明软件闭环，不构成真实 IRSTD 结果。

## 未在当前环境运行

- 真实多数据集 MSHNet 长训练；
- 真实 nested LODO；
- external unseen-domain benchmark；
- 多 seed；
- GPU 性能与吞吐基准。

原因是当前交付环境没有完整的四个以上独立数据域和正式训练配置，且不能将 synthetic 数据或单一数据集替代为 claim-bearing 实验。

## 训练机上的首条命令

先执行软件自检：

```bash
bash scripts/validate_two_stage_release.sh /tmp/rc_irstd_validation
```

再执行真实数据一轮集成：

```bash
CUDA_VISIBLE_DEVICES=0 \
EPOCHS=1 PER_DOMAIN_BATCH=1 \
RUN_ROOT=outputs/integration/real_1ep \
bash scripts/start_training.sh detector \
  /data/NUAA-SIRST \
  /data/NUDT-SIRST \
  /data/IRSTD-1K
```

通过后按 `docs/TRAINING_RUNBOOK.md` 继续。
