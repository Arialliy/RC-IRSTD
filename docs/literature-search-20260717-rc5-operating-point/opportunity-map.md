# RC5 Operating-Point Novelty Boundary and Opportunity Map

Date: 2026-07-17  
Target: AAAI-27  
Evidence status: prior-art report only; no RC5 experimental result is asserted

## Executive verdict

**No Direct paper was found among the 25 screened candidates, but the broad core idea is not new.** Public work already covers, separately and sometimes in close combinations:

- cross-domain/generalizable IRSTD;
- adapting detectors from unlabeled test samples;
- predicting or selecting thresholds from unlabeled test instances for a fixed representation model;
- source-free conformal threshold estimation; and
- Neyman–Pearson or conformal control of error criteria.

The strongest novelty position is therefore a **bounded composite deployment setting**, not a new calibration primitive:

> short unlabeled target context → frozen IRSTD detector → explicit native-pixel false-alarm-budget curve, structurally monotone across budgets, with no reject branch and an analytic target-tail anchor corrected by a source-trained model.

Confidence that no paper in the screened set matches the whole bundle: **medium**. Confidence that `unlabeled test-time threshold calibration` and `false-alarm-constrained decisions` are already covered concepts: **high**. A `first` claim is not supported.

## Closest-mechanism comparison

Legend: `Yes` = explicit in the cited work; `Partial` = related but materially different; `No` = absent; `Unclear` = not established from the allowed source.

| Work | Unlabeled target context | Frozen task predictor | Test-conditioned threshold | Explicit target level | IRSTD native-pixel extreme FA | Monotone budget curve | Analytic tail anchor + source correction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RC5 proposed contract | Yes, short prefix/context | Yes, detector and feature adaptors fixed | Yes | Pixel-FA budget grid | Yes | Yes | Yes |
| [OpenGCN](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_for_Transductive_Threshold_Calibration_in_Open-World_Recognition_CVPR_2024_paper.html) | Yes | Yes, DML embedding model | Yes; estimates TPR/TNR over thresholds | Target TPR or TNR | No; pairwise open-world recognition | Partial; performance functions are evaluated over thresholds, but no RC5-style constrained three-budget output | No; learned GNN/MLP with closed- and open-world calibration data |
| [Yilmaz–Heckel recalibration](https://arxiv.org/abs/2210.04166) | Yes | Yes, classifier | Yes | Conformal coverage level | No | Partial; levels imply threshold choices, not an IRSTD pixel-risk curve | No |
| [Source-Free Conformal Prediction](https://proceedings.mlr.press/v266/angelman25a.html) | Yes | Yes, source model used for pseudo-labels | Yes | Conformal coverage level | No | Partial | No; pseudo-label threshold estimation |
| [Fully TTOD](https://openaccess.thecvf.com/content/CVPR2024W/MAT/html/Ruan_Fully_Test-time_Adaptation_for_Object_Detection_CVPRW_2024_paper.html) | Yes, even one image | No; detector is optimized at test time | No; confidence/IoU filtering supports self-training | No operating-budget curve | No | No | No |
| [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg) | No; labeled exchangeable calibration is central | Yes | Yes, a calibrated monotone parameter | User risk level | No IRSTD/native-pixel study | Yes at the abstract monotone-loss level | No; formal labeled-calibration procedure |
| [Generalizable RealScene-ISTD](https://arxiv.org/abs/2504.16487) | Unclear from the allowed abstract | No; representation is adapted/trained | No | Evaluates Pd, Fa, IoU | Partial; IRSTD false alarms, not the RC5 extreme-budget contract | No | No |
| [Conformal FP segmentation](https://arxiv.org/abs/2511.15406) | No; labeled exchangeable calibration | Yes | Yes, one shrink parameter | False-positive tolerance | Partial; segmentation FP, not IRSTD/native-pixel extreme budgets | No multi-budget learned curve | No |

### What OpenGCN changes in the claim boundary

OpenGCN is the highest-priority citation risk. It already frames threshold choice as transductive inference, lets a learned calibrator inspect unlabeled test instances, predicts TPR/TNR as functions of distance threshold, and grid-searches a threshold for a desired operating metric. RC5 must not claim that any of those generic ideas is new.

The defensible differences are task and contract specific:

- pixel masks from an IRSTD detector rather than pairwise metric recognition;
- a detector that remains frozen, with no feature adaptor or prompt update;
- native-resolution pixel false-alarm budgets rather than pair-level TPR/TNR;
- a short, explicitly separated context rather than an unrestricted test dataset;
- simultaneous structurally monotone outputs for a frozen budget grid;
- no reject/abstain option;
- an exact analytic context-tail anchor plus source-only learned correction; and
- empirical held-out-domain evaluation without claiming distribution-free guarantees.

These are **differentiators to test**, not established superiority.

## Cluster opportunity map

| Cluster | Status | What is already covered | Remaining opening for RC5 | Evidence needed | Main risk |
| --- | --- | --- | --- | --- | --- |
| Cross-domain/generalizable IRSTD | Crowded but open | Domain alignment, invariant/noise-robust features, joint multi-domain detectors, real-scene/cross-dataset benchmarks | Adapt only the operating point while preserving a fixed detector and source-trained semantics | Strict nested LODO; identical detector weights across methods; context/query separation; cross-dataset error analysis | Calling target-context use `domain generalization` without qualification; under-citing IRSTD DA work outside the restricted source set |
| Recent strong IRSTD detectors | Crowded | SLS/MSHNet, multimodal prompting, learned noise suppression, invertible/reconstruction-guided encoders | Detector-agnostic post-hoc control may complement rather than replace representation improvements | At least one current strong detector; ideally more than one architecture; identical native-resolution replay | A weak Stage-1 detector can make calibration results unpersuasive; new CVPR 2026 baselines raise the bar |
| Unlabeled TTA/TTOD | Covered central concept, deployment gap | Single-image, batch, and continual unlabeled adaptation; student-teacher, adaptor, prompt, and normalization updates | A narrower non-updating alternative with predictable state, compute, and auditability | Wall-clock/state comparison; detector-parameter hashes before/after; robustness to context length and order | `Frozen detector` may be seen as a constraint rather than a contribution unless evidence shows a practical benefit |
| Dynamic threshold/recalibration | Covered central concept | Unlabeled transductive threshold inference, cutoff prediction under shift, pseudo-label conformal calibration, OOD temperature scaling | IRSTD-specific budget-indexed pixel operating curve with analytic-tail residual structure | OpenGCN/Yilmaz/SFCP discussion; independent-vs-monotone threshold ablation; anchor-only vs learned-only vs combined | Broad novelty is weak; analytic quantiles and monotonicity may be considered obvious unless the evidence package is strong |
| Neyman–Pearson/risk control | Theory substantially covered | Type-I-constrained optimization, transfer under shift, conformal monotone-risk calibration, high-probability risk control | Empirical extreme-tail operating-point adaptation under explicit non-guarantee language | Exact denominator and integerization; estimability rule; confidence intervals; held-out-domain stress tests | Any `guaranteed`, `certified`, `NP-optimal`, or `distribution-free` wording invites a fatal correctness objection |

## Cross-cutting benchmark opportunity

[Rethinking Evaluation of IRSTD](https://arxiv.org/abs/2509.16888) makes protocol design part of the scientific contribution. RC5 can gain reviewer value from a precise evaluation contract even if the calibrator mechanism is modest:

- define false-alarm numerator and native-resolution denominator exactly;
- report the entire frozen budget grid, not only the favorable operating point;
- declare when an extreme-budget cell is statistically inestimable;
- pair pixel-level budget compliance with target-level detection probability and segmentation quality;
- use exact hard-threshold replay, identical masks/scores, and paired resampling; and
- expose per-domain and macro summaries so a single easy domain cannot dominate.

This is a **benchmark/protocol strength**, not permission to claim a formal risk guarantee.

## Recommended recent strong baselines

### Detector-side baselines

| Priority | Baseline | Role | Fit and caveat |
| --- | --- | --- | --- |
| Required | [MSHNet/SLS](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | Strong simple IRSTD detector | High fit and already a natural backbone family; freeze the same checkpoint for all operating-point methods. |
| High | [NS-FPN](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | Current false-alarm-oriented detector/plugin | Strong topical comparator. Integrate only under a predeclared, reproducible protocol; do not add it after seeing target results. |
| High | [InvDet](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_Detection_CVPR_2026_paper.html) | Current cross-dataset representation baseline | Valuable if code/resources permit; it changes the detector and therefore answers a complementary question. |
| Conditional | [SAIST](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html) | Multimodal strong detector | Compare numerically only if image-text inputs and dataset protocol are genuinely available; otherwise cite as contextual work. |
| Conditional | [RealScene-ISTD](https://arxiv.org/abs/2504.16487) | Cross-domain method and real-scene dataset | Useful for generalization stress testing after verifying license, split, code, and preprint status. |
| Context only | [CoMoE](https://ojs.aaai.org/index.php/AAAI/article/view/37373) | Multi-domain moving-IRSTD detector | Important AAAI citation, but moving/multiframe assumptions and joint multi-domain training can make direct numbers incomparable. |

### Operating-point and adaptation controls

The minimum fair control family should include, without inventing results:

1. one fixed source-pooled threshold per budget;
2. a conservative worst-source threshold per budget;
3. the target-context analytic exact-order-statistic anchor alone;
4. an independent direct threshold predictor;
5. a monotone source-trained threshold predictor without the analytic anchor;
6. the shared-anchor plus source-trained correction;
7. an oracle target-label threshold for diagnosis only, never deployment; and
8. if feasible, one detector-updating TTA comparator from TeST, Fully TTOD, or continual TTOD, clearly separated because it violates the frozen-detector contract.

OpenGCN is a mandatory conceptual baseline/citation, but porting its pairwise graph construction to pixel masks would create a new method rather than a clean off-the-shelf baseline. Treat it as related work unless a task-faithful implementation is defined before results.

## Safe AAAI positioning language

### Recommended English formulations

> Prior work has separately studied cross-domain IRSTD, detector adaptation from unlabeled test data, and transductive threshold calibration. We study their intersection under a deployment contract in which the detector remains frozen and only a native-pixel operating-point curve is inferred from a short unlabeled target context.

> The calibrator outputs a structurally monotone family of thresholds indexed by explicit pixel false-alarm budgets and always emits a prediction mask; it has no rejection or abstention branch.

> Our parameterization combines an analytic context-tail anchor with a correction learned exclusively from source-domain episodes.

> The requested budgets specify empirical operating points evaluated by exact native-resolution replay. We do not claim distribution-free, certified, or Neyman–Pearson-optimal risk control.

> Among the works reviewed in our bounded search, we did not find an evaluation of this complete IRSTD deployment bundle; this observation motivates the setting but is not a claim of universal priority.

### Recommended Chinese interpretation

- 把贡献写成“IRSTD 特定部署问题 + 可审计协议 + 组合机制”，不要写成通用校准理论创新。
- 把 `calibration` 首次出现时限定为 `operating-point calibration`，明确不是 confidence calibration。
- 把 target context 称为 `unlabeled transductive context` 或 `unlabeled deployment context`，不要称为 zero-shot。
- 把预算称为 requested/nominal empirical budget；只有真实结果满足后才能说 achieved，且仍不能写 guarantee。
- 把 analytic anchor 称为 exact empirical tail/order-statistic anchor 或 inductive bias，不声称发明了分位数或 Neyman–Pearson 阈值思想。

## Unsafe claims and replacements

| Unsafe claim | Why unsafe | Safer replacement |
| --- | --- | --- |
| `the first unlabeled test-time threshold calibration method` | Contradicted conceptually by OpenGCN and unlabeled conformal recalibration | `an IRSTD-specific operating-point calibration method under a frozen-detector contract` |
| `the first test-time adaptation method for IRSTD` | IRSTD adaptation/generalization exists, and generic TTOD is mature | `a detector-preserving alternative to parameter-updating test-time adaptation` |
| `guaranteed false-alarm control` | RC5 has no matching finite-sample theorem under target shift | `targets and empirically evaluates specified false-alarm operating points` |
| `Neyman–Pearson optimal` | Requires an optimality theorem and assumptions not supplied by the empirical calibrator | `Neyman–Pearson-motivated asymmetric operating-point evaluation` |
| `domain generalization` | RC5 observes target-domain context at deployment | `unlabeled transductive operating-point adaptation on held-out domains` |
| `source-free method` | The calibrator is trained on source episodes | `source-data-free at deployment, with source-trained calibration parameters` |
| `zero-shot` | Target context is consumed | `label-free target-context adaptation` |
| `no supervision` | Source labels and detector training are used | `no target labels at deployment` |
| `calibrated probabilities` | RC5 chooses thresholds, not necessarily probability calibration | `calibrated operating thresholds` |
| `monotonicity is novel` | Structural monotonicity is a standard constraint and appears broadly in risk calibration | `monotonicity enforces budget-consistent decisions and is tested by ablation` |

## Reviewer-facing risk register

| Severity | Risk | Required mitigation before submission |
| --- | --- | --- |
| High | Reviewer identifies OpenGCN as the same high-level idea | Cite it prominently; state task/output/calibration-data differences; avoid generic novelty language |
| High | Empirical budget is interpreted as a guarantee | Repeat non-guarantee language in abstract/method/experiments; report violations and inestimable cells without imputation |
| High | Detector baseline set is stale after CVPR 2026 | Include or justify NS-FPN and InvDet; cite SAIST and CoMoE with protocol caveats |
| High | Target leakage through context, model selection, or threshold tuning | Freeze context/query construction, checkpoint rule, budget grid, and all gates before official-test access |
| Medium | Analytic anchor is viewed as a trivial quantile baseline | Make it an explicit baseline; isolate the source-trained correction and interaction by ablation |
| Medium | Three-point monotone curve is viewed as minor engineering | Demonstrate budget consistency, sample efficiency, cross-domain stability, and failure modes; do not oversell architecture |
| Medium | `no reject` seems like removal rather than contribution | Tie it to a concrete always-predict deployment requirement and compare utility/false-alarm trade-offs |
| Medium | Extreme budgets are not estimable | Predeclare the pixel-count estimability rule and mark cells inestimable rather than extrapolating or imputing |

## Evidence package that would support the bounded claim

No outcome is presumed. The minimum planned evidence is:

- strict held-out-domain evaluation with no official-test model selection;
- exact native-resolution pixel counting and frozen denominator semantics;
- fixed-source, analytic-anchor, independent-direct, monotone, and oracle-diagnostic controls;
- T6/T7/T8 ablations separating directness, monotonicity, loss, and anchor mixing;
- context-length and context-order sensitivity;
- per-domain and macro operating curves across all budgets;
- paired uncertainty intervals for the predeclared primary comparison;
- failure cases, budget violations, and inestimable cells shown explicitly; and
- detector-state hashes proving that deployment-time calibration does not update the detector.

## Bottom line

The idea remains **plausibly differentiated as an IRSTD deployment-and-evidence package**, but not as the invention of unlabeled threshold calibration, test-time adaptation, quantile anchoring, monotone risk curves, or false-alarm-constrained decisions. The paper should win on the precision of the problem contract, extreme-tail/native-pixel protocol, source/target separation, and empirical evidence—not on a `first` claim.
