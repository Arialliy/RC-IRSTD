# AAAI27 RC1 SLS implementation-invalid decision

Decision date: 2026-07-15 (Asia/Shanghai)

## Disposition

The run `outputs/stage1_pilot_30ep/D0_all-three_s42` is preserved only as
implementation-diagnostic evidence. It is classified as:

- `implementation-invalid`;
- `diagnostic-aborted`;
- not resumable;
- not eligible for Stage-1 Gate decisions, model selection, figures, tables,
  or paper claims.

The process was stopped deliberately in its attached PTY. Epochs 0--5 are the
only complete metric records; the epoch-5 checkpoint checksum remains valid.
Epoch 6 and `weights_last.pt` were not written.

## Root cause

The strict Stage-1 entry point imported the legacy `model.loss.SLSIoULoss`.
That implementation uses `smooth=0`. On an empty target crop it gives a
constant warm-up loss with no useful segmentation gradient, can reach `0/0`
after sigmoid underflow, and applies a location penalty with no physical
meaning after warm-up.

The intended implementation is the epsilon-guarded SLS loss that skips the
location term for empty targets. The mismatch affected 2,048 of 11,448
training samples across complete epochs 0--5 (17.89%), so it was not a rare
synthetic edge case.

This also invalidates the intended D0/D3 paired comparison: legacy D0 provides
an incorrect gradient path on empty crops, while D3's domain-tail term can
still provide a background gradient on those crops. A measured D3 advantage
could therefore be partly caused by repairing a defective baseline.

The epoch-6 loss increase was not itself the abort criterion. It coincided
with the preregistered `epoch > warm_epoch` SLS formula transition and merely
made the static implementation discrepancy visible in runtime behavior.

## Corrective release requirements

The successor release candidate must satisfy all of the following before a
new GPU run starts:

1. Bind the strict Stage-1 runtime to `losses.sls.SLSIoULoss(eps=1e-8)`.
2. Record the qualified implementation, revision, epsilon, and multiscale
   reduction in config, run contract, checkpoint, and per-epoch metrics.
3. Reject resume when the recorded segmentation-loss implementation differs.
4. Pass CPU tests for empty masks at extreme logits, mixed empty/non-empty
   batches, all five MSHNet heads, finite gradients, and strict runtime binding.
5. Create a new clean commit, tag, exact Git archive, checksum, matrix hash,
   and sealed preflight.
6. Restart both D0 and D3 from epoch 0; never resume the RC1 checkpoint.

No official-test predictions, labels, metrics, or performance values were
used to discover the issue or make this decision.
