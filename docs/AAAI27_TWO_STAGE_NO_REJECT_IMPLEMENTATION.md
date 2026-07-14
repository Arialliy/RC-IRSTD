# RC-IRSTD AAAI27 two-stage/no-Reject implementation

This document records what is implemented in this worktree. It is an
engineering contract, not a claim that the method already beats baselines.

## Final method path

Stage 1 uses the explicit D0–D3 identities in `scripts.train_multisource_tail`:
`segmentation-only`, `margin-background-only`, `margin-target-only`, and
`margin`. For
each source domain it forms the target-object lower tail and the background
local-peak upper tail after GT-neighbour exclusion and deterministic plateau
collapse. The hinge is applied after domain aggregation, followed by a
normalized smooth worst-domain reduction. D1/D2 retain the same forward hinge
as D3 and stop the target/background branch respectively. `separate` and
`legacy-image-margin` remain compatibility baselines outside D0–D3.

Stage 2 uses `rc.train_calibrator_risk_aligned`. One unlabeled context produces
the complete `[J]` inverse pixel-risk threshold curve. The architecture is
strictly monotone over the descending budget grid, computes tail logits in
float64, interpolates in log10-budget space, and forbids extrapolation. It has
no Reject head.

Meta-training accepts only schema-v4 episodes with:

- score manifest v3 and `role=official_train`;
- verified official train/test ID and image-byte disjointness;
- disjoint causal context/query IDs;
- a complete frozen pixel-budget grid per context/query group;
- a hash-bound curve manifest and independent label manifest;
- a global-exact curve or an audited event-exact high-tail suffix.

The risk-aligned objective combines query budget violation, Pd utility,
oracle-logit anchoring, curve smoothness, and exact-suffix coverage. It never
defaults to a uniform 65,536-pixel background sample. Validation reloads the
native-resolution query scores and independent labels, applies
`probability > threshold`, uses deterministic 8-connected one-to-one matching,
and selects the checkpoint lexicographically by BSR, LogExcess, then Pd.

## Split discipline with train/test-only datasets

The v2 effective development split first removes every official-train image
implicated by the frozen train/test near-duplicate audit. Raw data and official
split files remain untouched. The official test split is locked until final
evaluation. Pseudo-target
calibrator optimisation and model selection both use causal windows built
only inside each dataset's official training split. A pseudo-target held out
from calibrator optimisation is therefore a training-split meta-validation
domain; it is not the official test set. Any episode carrying `official_test`,
an unknown role, or legacy role-less provenance fails before fitting.

With only three datasets, a strict outer-target plus inner-pseudo-target fold
can leave only one detector source. Such a fold is a smoke/diagnostic result,
not strict multi-source nested-LODO evidence. Do not relabel it as an AAAI main
result.

## Launch

Use physical GPUs 0, 1, and 2 for Stage 1:

```bash
PYTHON_BIN=python ./scripts/train_rc_3gpu.sh \
  --source-dirs <SOURCE_1> <SOURCE_2> \
  --source-names <NAME_1> <NAME_2> \
  --outer-fold-id <FOLD> \
  --outer-target <TARGET> \
  --held-out-domains <TARGET> \
  --source-split-files <SOURCE_1_DETECTOR_FIT> <SOURCE_2_DETECTOR_FIT> \
  --risk-objective margin \
  --exclusion-radius 2 \
  --epochs 30 \
  --save-dir outputs/detectors \
  --run-name <FOLD>
```

For the paired D0 run, replace the objective with
`--risk-objective segmentation-only --lambda-margin 0`. The Stage-1 pilot Gate
uses only the frozen `detector_diagnostic` lists for evaluation; official test
is not opened to decide whether to expand seeds or run D1/D2.

The Stage-2 model is small. Run independent outer folds concurrently on GPUs
0/1/2 by setting `RC_CALIBRATOR_GPU` per process:

```bash
RC_CALIBRATOR_GPU=0 ./scripts/train_calibrator_risk_aligned.sh \
  --episodes <EPISODES.jsonl> \
  --val-pseudo-target <PSEUDO_TARGET> \
  --artifact-root <BUILD_SPEC_ROOT> \
  --output-dir outputs/rc/<FOLD> \
  --deployment-detector-checkpoint-sha <SHA256> \
  --deployment-detector-source-domain <SOURCE_1> \
  --deployment-detector-source-domain <SOURCE_2> \
  --deployment-source-reference <SOURCE_REFERENCE.npz> \
  --pixel-budget-grid 1e-4 1e-5 1e-6
```

Repeat with `RC_CALIBRATOR_GPU=1` and `2` for other independent folds. The
explicit `artifact-root` is the root against which episode metadata paths were
created; the loader never guesses it from cwd or the JSONL location.

The v5 online adapter emits a threshold and `no_reject=true`; it contains no
reject probability, cutoff, decision, or `p_min`. Final-target score manifests
must be `official_test`. The adapter reads and verifies context artifacts only
before freezing its decision; labels are attached later by
`evaluation.evaluate_adapter_output` for offline replay.

PyTorch checkpoints are trusted local training artifacts: loading a `.pt`
file may deserialize Python pickle before the v5 evidence contract is checked.
Do not expose this loader to untrusted third-party checkpoint uploads without
an authenticated or non-pickle serialization boundary.

## Evidence still required

Run the pre-registered real-data outer folds, direct/rolling/EVT baselines,
Stage-1 and Stage-2 ablations, seeds, contamination/context-size robustness,
and confidence intervals. Synthetic smoke tests prove execution and contracts
only; they must never be reported as method performance.
