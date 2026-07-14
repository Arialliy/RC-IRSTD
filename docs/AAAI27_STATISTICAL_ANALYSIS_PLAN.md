# RC-IRSTD AAAI-27 Statistical Analysis Plan

> Status: frozen protocol candidate; contains no observed model result.
> Machine-readable authority: `configs/aaai27_analysis_plan.json`.
> Any change to loss, tail definition, split, budget, seed, evaluation, or
> checkpoint policy requires a new release candidate and a new plan version.

## 1. Scope and claims

The method is Two-Stage No-Reject RC-IRSTD. MSHNet is the Stage-1 backbone,
not the full method identity.

This release authorizes only the Stage-1 development Gate after a clean,
tagged release has passed sealed preflight. It does not authorize:

- claim-bearing three-domain Stage 2;
- official-test-guided model, epoch, budget, feature, or method selection;
- paper performance claims;
- a causal or temporal generalization claim.

The current three datasets are static IID image benchmarks because no reliable
sequence/group sidecar is available. Canonical image ID is therefore the
bootstrap unit.

## 2. Data roles and leakage boundary

Development domains are NUAA-SIRST, NUDT-SIRST, and IRSTD-1K. Only their
official-train images may influence the Stage-1 Gate.

The frozen v2 development split applies a conservative image-only quarantine
before any random partition:

| Domain | Official train | Quarantined | Effective development | Fit | Diagnostic |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | 213 | 0 | 213 | 170 | 43 |
| NUDT-SIRST | 663 | 27 | 636 | 509 | 127 |
| IRSTD-1K | 800 | 3 | 797 | 638 | 159 |

The 30 quarantined training IDs remain in the untouched official split but are
excluded from every derived development role. The effective-development versus
official-test image-only audit has zero confirmed near-duplicate pairs under
the frozen pHash/cosine rule.

For Stage-1 pilot decisions:

- detector fitting uses `detector_fit` only;
- Gate metrics use `detector_diagnostic` only;
- diagnostic IDs never select a checkpoint because every run is fixed-last;
- official test remains sealed until the method, all D0–D3 definitions,
  budgets, three seeds, and the full Stage-1 expansion policy are frozen;
- official-test results cannot determine whether D1/D2 or additional seeds are
  run and cannot cause a return to method development.

This corrects the unsafe interpretation in which a 20–40 epoch official-test
pilot would become a development set.

## 3. Misc_111 geometry contract

NUAA-SIRST `Misc_111` is handled exactly as in BasicIRSTD:

- image shape: 220×325 (H×W);
- original mask shape: 400×592 (H×W);
- aspect-ratio relative error: approximately 0.001846;
- mask is resized to the image canvas with PIL NEAREST before any other spatial
  transform;
- binarization occurs after alignment;
- original-resolution evaluation is retained.

Any mismatch with aspect-ratio relative error above 1% fails closed.

## 4. Stage-1 estimands and variants

Let (R_d^-) denote the domain background upper-tail logit and (R_d^+) the
domain target lower-tail logit. The common forward hinge is

\[
V_d=\operatorname{ReLU}\left(m-(R_d^+-R_d^-)\right).
\]

The frozen variants are:

| ID | Optimized objective | Trainable tail | Stopped tail |
|---|---|---|---|
| D0 | multiscale SLS segmentation only | none | not applicable |
| D1 | common domain hinge | background (R_d^-) | target (R_d^+) |
| D2 | common domain hinge | target (R_d^+) | background (R_d^-) |
| D3 | common domain hinge | both | none |

D1–D3 have identical forward hinge values. Their only difference is the
pre-registered stop-gradient route. D0 computes detached tail diagnostics but
receives no auxiliary-risk gradient.

All paired comparisons use identical backbone, initialization seed,
domain-balanced sampling, augmentation, optimizer, learning-rate schedule,
steps, input geometry, checkpoint rule, threshold sweep, and matching rule.

## 5. Stage-1 single-seed Gate

The first Gate is fixed to:

- seed: 42;
- epochs: 30;
- checkpoint: fixed-last;
- budgets: (10^{-4},10^{-5},10^{-6});
- primary budget: (10^{-5});
- evaluation: original-resolution exact hard-threshold replay on frozen
  official-train diagnostic IDs.

Required runs are D0 and D3 for all-three, leave-NUAA, leave-NUDT, and
leave-IRSTD1K: eight runs in total. D1/D2 and seeds 123/3407 are not run before
this Gate is decided.

GPU order is frozen:

1. all-three D0/D3 use DataParallel on physical GPUs 0, 1, and 2;
2. after those processes finish, the three LODO folds use one process per GPU;
3. all-three and LODO processes never overlap;
4. D0/D3 within a fold use the same device mode and batch contract.

The single-seed Gate passes only if all required artifacts are complete and:

- D3−D0 Pd at the primary budget improves by at least 0.01 in the macro
  development estimand;
- at least two held-out domains improve their background tail or low-FA Pd;
- source/held-in IoU degradation is no worse than 0.01;
- the target lower tail does not show sustained degradation;
- the change is not explained only by a common logit down-shift;
- there is no empty-prediction shortcut, non-finite value, or persistent logit
  scale expansion;
- online threshold semantics and exact replay agree;
- official test is absent from every Gate input.

Failure of any required run is not imputed and makes the Gate false.

## 6. Post-Gate Stage-1 expansion

Only after the single-seed Gate passes:

- run D1 and D2 at seed 42 for mechanism attribution;
- run paired D0/D3 at seeds 42, 123, and 3407;
- compute image-level paired bootstrap confidence intervals;
- freeze failure cases and results before any official-test opening.

The primary confirmatory Stage-1 comparison is D3−D0. Report every seed and
domain, unweighted macro mean, worst-domain value, seed standard deviation,
and 95% paired bootstrap CI. Use 10,000 bootstrap resamples with bootstrap seed
2027. Pixels are never treated as independent CI samples. Secondary
comparisons use Holm family-wise correction at alpha 0.05.

## 7. Outcomes

Primary Stage-1 endpoint:

- macro held-out development Pd at native-resolution pixel budget (10^{-5}).

Ordered reporting set:

1. Pd at (10^{-4},10^{-5},10^{-6});
2. BSR;
3. LogExcess;
4. worst-domain Pd;
5. exact pixel false-alarm rate.

Supplementary outcomes are IoU/nIoU/hIoU, connected-component FA/MP as a
compatibility metric, source and oracle thresholds, (R_d^-), (R_d^+),
their gap, logit distribution diagnostics, and candidate/object counts.

At a budget where budget × total background pixels is below 20, the operating
point is marked inestimable for the primary Gate and only descriptive counts
are reported.

## 8. Missing runs, reruns, and stopping

There is no imputation. Reruns are allowed only for a documented implementation
or infrastructure failure identified without inspecting performance metrics.
The reason, failed artifact hash, and replacement run identity must be kept.

Stop the Stage-1 tail-separation claim if D3 only shifts all logits downward,
oracle Pd cannot recover target detection, the target lower tail degrades, most
domains lose low-FA Pd, or D3 and D0 show no stable difference. If target oracle
also fails, classify the limitation as representation/ranking rather than
calibration.

## 9. Stage-2 frozen specification and blockers

The threshold baseline family is T0–T9: fixed 0.5, pooled-source,
worst-source, nearest-source, rolling quantile, EVT/GPD, direct threshold MLP,
monotone oracle-threshold regression, proposed no-Reject calibrator, and target
oracle. Target oracle is diagnostic only.

The calibrator ablations are C0–C6 exactly as specified in the execution plan;
C3 is the main calibrator. Threshold baselines and calibrator ablations are
separate identifier spaces.

Claim-bearing Stage 2 remains blocked by incomplete baseline runners, the lack
of a fourth independent domain, the lack of external unseen targets, and only
one non-overlapping NUAA meta-train window. Three-domain Stage-2 runs may be
engineering diagnostics only and may not enter a main table, abstract,
significance statement, or final conclusion.

## 10. Evidence and release rule

Every run must bind the Git commit/tag, clean status, source archive SHA-256,
environment/GPU inventory, seed, source/held-out roles, dataset and split
hashes, checkpoint SHA-256, checkpoint rule, budget grid, threshold semantics,
matching rule, original-resolution status, and exact replay status.

No observed result may be inserted into this plan. Results belong in immutable
run artifacts and a separate result-freeze report.
