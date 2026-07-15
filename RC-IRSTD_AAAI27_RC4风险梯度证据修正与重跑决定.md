# RC-IRSTD AAAI-27 RC4 风险梯度证据修正与重跑决定

日期：2026-07-15

## 结论

RC3 的 Stage-1 P3/D3 LODO 运行不得进入 G1。原因不是已观察到模型性能失败，而是冻结审计器要求首个风险 epoch 的隔离风险梯度严格大于零，而训练记录器只检查该 epoch 的第一个 batch。`D3_leave-IRSTD1K_s42` 的 epoch 5 首 batch 恰好满足 hinge，记录值因此为 0；同一 epoch 后续 batch 存在正 margin loss，但记录器不再探测。按 RC3 的 fail-closed 规则，该运行即使跑满也不可能通过 P3。

发现该问题时，official test 仍封存，三折 P3 尚未完成，G1 的 development-only 性能端点尚未计算。因此本次修正发生在任何 Stage-1 性能判定之前，不依据目标性能调参，也不放宽 G1 效应阈值。

## RC4 的唯一训练端修正

RC4 保留原字段 `risk_gradient_norm_first_active_step`，并新增：

- `risk_gradient_probe_count`；
- `risk_gradient_positive_found`；
- `risk_gradient_norm_first_positive_step`；
- `risk_gradient_first_positive_step`。

每个风险生效 epoch 仍探测首 batch。若首 batch 的隔离风险梯度为零，记录器只在后续辅助风险目标严格为正的 batch 上继续探测，直到得到首个严格正、有限的隔离风险梯度。探测使用 `torch.autograd.grad(..., retain_graph=True)`，不写入参数 `.grad`，不改变随后执行的 `loss.backward()`、梯度裁剪、优化器更新、数据顺序或随机数状态。

P1/P3 的首个风险 epoch 必须出现至少一个严格正的隔离风险梯度；首 batch 本身允许为零。P0/P2 的 D0 必须保持零探测、零正梯度证据。所有原有 loss 分解、有限性、固定最后 checkpoint、数据身份、GPU 身份和 official-test 封存条件保持不变。

## 验证证据

- 项目全量 CPU：246 passed，10 subtests passed；
- 独立 P0-P3 审计器：49 passed，34 subtests passed；
- 定向风险与审计测试：82 passed，38 subtests passed；
- 新增回归测试覆盖“首 batch 隔离梯度为 0、第二个 batch 为正”；
- GPU 0/1/2 三域 2-step engineering smoke：退出码 0，`risk_gradient_positive_found=true`，首个正梯度 step 为 0，范数为 10.830105781555176；logit、梯度和更新后参数均有限；
- smoke 仅使用冻结 `detector_fit` split，不使用 official test，也不构成性能证据。

## 重跑决定

为避免混用 release，RC3 的已完成 P0/P1/P2 和中断 P3 仅保留为工程与失效诊断证据。RC4 将从 epoch 0、seed 42、fixed-last 重新执行同一完整八项 Stage-1 矩阵：D0/D3 all-three，以及三折 D0/D3 LODO。矩阵的 G1 阈值、数据角色和 official-test 隔离不变。

长训练以脱离终端的受管服务运行，但每个服务必须保存 PID/启动命令/日志/退出码并由当前会话持续审计。没有完整退出码、工件审计和 development-only 配对诊断，不得宣称模型设计成功。
