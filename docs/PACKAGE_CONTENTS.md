# ZIP 内容索引

> **Reference snapshot / 非正式主线。** 本文件从 `RC-IRSTD_AAAI27_TwoStage_NoReject` 迁入，仅用于设计溯源与兼容审计；其中 `rc_irstd.*`、YAML、旧测试数和训练命令均不得作为当前 claim-bearing 入口。当前唯一权威实现为 flat v5（`data_ext/`、`model/`、`losses/`、`evaluation/`、`rc/`、`scripts.train_multisource_tail`），当前决策见仓库根目录 `RC-IRSTD_AAAI27_当前状态与下一步训练执行方案.md`。

## 开始阅读

1. `README.md`：方法、命令和当前验证结果；
2. `docs/AAAI27_TWO_STAGE_NO_REJECT_FINAL_PLAN.md`：定稿方法；
3. `docs/EXPERIMENT_PROTOCOL.md`：无泄漏协议和评价契约；
4. `docs/TRAINING_RUNBOOK.md`：分阶段真实训练步骤；
5. `docs/CODE_AUDIT_AND_IMPLEMENTATION.md`：源码审计和改动；
6. `docs/TRAINING_STATUS.md`：本次实际启动的训练与边界。

## 关键代码

```text
rc_irstd/losses/target_background_margin.py
rc_irstd/models/monotone_pixel_calibrator.py
rc_irstd/episodes/meta_dataset.py
rc_irstd/losses/calibrator.py
rc_irstd/evaluation/calibrator_replay.py
rc_irstd/pipelines/train_detector.py
rc_irstd/pipelines/build_source_reference.py
rc_irstd/pipelines/build_meta_dataset.py
rc_irstd/pipelines/train_calibrator.py
rc_irstd/pipelines/evaluate_calibrator.py
rc_irstd/pipelines/apply_calibrator.py
```

## 用户指定路径兼容层

```text
losses/target_background_margin.py
model/monotone_pixel_calibrator.py
rc/meta_dataset.py
rc/train_calibrator.py
```

## 启动器

```text
scripts/validate_two_stage_release.sh
scripts/start_training.sh
scripts/train_detector_mshnet.sh
scripts/build_meta_fold.sh
scripts/train_calibrator_risk_aligned.sh
scripts/deploy_no_reject.sh
scripts/smoke_two_stage_no_reject.sh
```

## 配置模板

```text
configs/aaai27_detector_tail_sep.yaml
configs/aaai27_calibrator_risk_aligned.yaml
configs/aaai27_pilot_matrix.yaml
configs/datasets.example.yaml
configs/budgets_aaai27.txt
```

## 验证证据

```text
validation/compileall.log
validation/shell_syntax.log
validation/pytest.log
validation/mshnet_integration.log
validation/two_stage_smoke.log
validation/two_stage_smoke/smoke_summary.json
validation/two_stage_smoke/calibrator/best.pt
validation/two_stage_smoke/calibrator/best_hard_replay.json
```

## 参考材料

`docs/reference/` 保存用户原始设计与评审建议。`docs/legacy/` 保存 0.3.x 历史工程文档；它们不是最终方法定义。
