# RC5+ Prior-Art Screening and Closest-Work Map

Date: 2026-07-19  
Screened: 30 deduplicated papers  
Final set: 15 papers  
Direct matches: 0

## Final 15-paper set

| # | Work | Cluster / type | Relation | I/C/N | Why retained |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Learning for Transductive Threshold Calibration in Open-World Recognition (OpenGCN)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_for_Transductive_Threshold_Calibration_in_Open-World_Recognition_CVPR_2024_paper.html), CVPR 2024 | threshold calibration / method | Conceptual | 5/4/5 | Closest mechanism: a learned calibrator consumes unlabeled test instances, estimates performance over thresholds, and chooses a target operating point for a frozen embedding model. |
| 2 | [ADA-IRSTD: Active Domain Adaptation for Cross-Domain Infrared Small Target Detection](https://doi.org/10.1109/JSTARS.2026.3702610), JSTARS 2026 accepted | cross-domain IRSTD / method | Conceptual | 4/4/4 | Learns a source-supervised false-alarm risk predictor and applies it to unlabeled target images, but uses it for active annotation and detector adaptation rather than budget-conditioned threshold inference. |
| 3 | [Rethinking Generalizable Infrared Small Target Detection](https://arxiv.org/abs/2504.16487), 2025 preprint | cross-domain IRSTD / method+benchmark | Contextual | 4/4/4 | Addresses IRSTD shift through cross-view/noise-robust representation learning and introduces RealScene-ISTD; it changes the detector rather than only its operating point. |
| 4 | [Rethinking Representations for Cross-Domain IRSTD: A Frequency-Domain Perspective](https://arxiv.org/abs/2604.01934), 2026 preprint | cross-domain IRSTD / method | Contextual | 4/4/3 | Current cross-domain competitor using phase rectification, orthogonal attention, and style recomposition; no target-context threshold function. |
| 5 | [Cross-domain Joint Learning with Prototype-guided Mixture-of-Experts for Infrared Moving Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37373), AAAI 2026 | cross-domain moving IRSTD / method | Contextual | 4/4/4 | Current AAAI evidence that cross-domain IR small-target learning is timely; it jointly trains a universal multi-domain detector. |
| 6 | [Rethinking Evaluation of Infrared Small Target Detection](https://arxiv.org/abs/2509.16888), NeurIPS 2025 D&B | IRSTD evaluation / benchmark | Contextual | 4/5/N/A | Makes cross-dataset, pixel/target, and error-type evaluation central; relevant to the native-pixel and held-out-domain evidence contract. |
| 7 | [Infrared Small Target Detection with Scale and Location Sensitivity](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html), CVPR 2024 | strong IRSTD / method | Contextual | 4/4/5 | Introduces SLS and MSHNet, the frozen primary backbone; it does not perform label-free target-context calibration. |
| 8 | [Seeing Through the Noise](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html), CVPR 2026 | false-alarm-oriented IRSTD / method | Contextual | 4/4/4 | A current detector-side false-alarm comparator based on noise suppression; inference changes representation rather than only threshold policy. |
| 9 | [Fully Test-time Adaptation for Object Detection](https://openaccess.thecvf.com/content/CVPR2024W/MAT/html/Ruan_Fully_Test-time_Adaptation_for_Object_Detection_CVPRW_2024_paper.html), CVPRW 2024 | unlabeled TTOD / method | Conceptual | 3/3/4 | Very close data regime, including one target image, but updates detector parameters using pseudo-label filtering. |
| 10 | [Test-time Recalibration of Conformal Predictors Under Distribution Shift Based on Unlabeled Examples](https://arxiv.org/abs/2210.04166), 2022/2023 preprint | unlabeled recalibration / method | Conceptual | 4/3/4 | Predicts a shifted cutoff from unlabeled examples; different output and assumptions, and no IRSTD/native-pixel curve. |
| 11 | [Calibrating Without Labels: Source-Free Conformal Prediction Using Pseudo-Labels](https://proceedings.mlr.press/v266/angelman25a.html), PMLR 2025 | unlabeled conformal / method | Conceptual | 4/4/5 | Estimates conformal thresholds from unlabeled target data and pseudo-labels; no analytic target-tail anchor or false-alarm-budget function. |
| 12 | [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg), ICLR 2024 | risk control / theory+method | Conceptual | 5/5/4 | Establishes calibrated monotone-risk control for a fixed predictor using labeled exchangeable calibration data; bounds RC5+ guarantee language. |
| 13 | [High Probability Risk Control Under Covariate Shift](https://proceedings.mlr.press/v266/almeida25a.html), PMLR 2025 | shifted risk control / theory+method | Conceptual | 4/4/4 | Controls FPR under covariate shift by importance-weighting labeled calibration losses; stronger formal assumptions and a different supervision regime. |
| 14 | [Controlling False Positives in Image Segmentation via Conformal Prediction](https://arxiv.org/abs/2511.15406), 2025 preprint | segmentation FP control / method | Conceptual | 4/3/3 | Post-hoc threshold/erosion family for a frozen segmenter with labeled exchangeable calibration; closest task/output risk-control neighbor. |
| 15 | [Neyman-Pearson Classification under Both Null and Alternative Distributions Shift](https://openreview.net/forum?id=pHckxhmBlI), ICLR 2026 | transfer NP / theory | Conceptual | 5/4/2 | Formal Type-I-constrained transfer under shifts in both classes; makes NP-optimal or guaranteed-control wording unsafe. |

## Closest-work table

| Closest work | What it already does | Overlap with RC5+ | Remaining novelty delta | Risk | Required differentiation |
| --- | --- | --- | --- | --- | --- |
| OpenGCN, CVPR 2024 | Learns transductive TPR/TNR-vs-threshold estimates from unlabeled test instances and selects a target threshold for a fixed embedding model | unlabeled target set, learned source-side calibrator, fixed base representation, target operating point | IRSTD pixel masks; short separated context; exact rational native-pixel FA budget; same-budget order-statistic anchor; continuous structurally monotone multi-budget function; no-reject sealed replay | high, non-fatal | Cite as the primary conceptual predecessor; never claim invention of transductive threshold calibration; compare exact input/output/supervision contracts. |
| ADA-IRSTD, JSTARS 2026 | Predicts false-alarm risk from source-supervised detector features on unlabeled target images and uses it for active selection | IRSTD, cross-domain target samples, source-supervised false-alarm statistic | It requests target annotations and adapts the detector; it predicts sample risk rather than a deployable budget-to-threshold curve; no analytic tail anchor or exact low-FA contract | high, non-fatal | Explicitly distinguish risk ranking for annotation from label-free operating-point inference and include it in Related Work. |
| Conformal FP segmentation, 2025 | Uses a nested threshold/erosion family on a frozen segmenter and selects one parameter to control image-level FP using labeled exchangeable calibration | frozen segmentation model, user error tolerance, post-hoc threshold | target context is unlabeled and shifted; native-pixel extreme FA; source OOF learned residual; simultaneous budget function; empirical non-guarantee | medium | State assumption and guarantee differences; report violations rather than borrow conformal guarantees. |
| Yilmaz--Heckel recalibration, 2022/23 | Predicts conformal cutoff under shift from unlabeled examples | label-free shifted cutoff prediction | classification-set output, no IRSTD pixel risk, no budget curve, no target-tail anchor | medium | Avoid generic `unlabeled recalibration is new` language. |
| Source-Free Conformal Prediction, PMLR 2025 | Pseudo-labels unlabeled targets to estimate conformal thresholds | unlabeled target threshold estimation | pseudo-label route, coverage rather than native-pixel FA, no short-context sealed contract | medium | Compare supervision and output semantics; do not call RC5+ source-free in training. |
| CRC, ICLR 2024 | Selects a monotone risk-control parameter with finite-sample guarantees | fixed predictor, requested risk level, monotone decision family | labeled exchangeability vs shifted unlabeled context; no learned source/analytic-target residual transport | medium | Use `empirical operating point`, not `certified`, `guaranteed`, or `distribution-free`. |
| High-probability risk control under shift, PMLR 2025 | Importance-weights labeled source calibration losses for target covariate shift and controls FPR | shifted target risk and false-positive constraint | explicit shift assumption/density ratios and labeled losses; no IRSTD short context or budget-conditioned neural function | medium | Present as formal boundary and not as an off-the-shelf equivalent baseline unless assumptions can be met. |
| RealScene-ISTD / S2CPNet / CoMoE | Improves cross-domain IRSTD via representation learning or multi-domain experts | held-out/cross-domain IRSTD motivation and datasets | RC5+ preserves the frozen detector and changes only the operating-point function | low-to-medium | Hash detector state, compare on identical score maps, and show when post-hoc adaptation complements representation changes. |

## Additional 15 screened candidates

| # | Candidate | Disposition |
| ---: | --- | --- |
| 16 | [Target-Aware Invertible Encoder with Reconstruction Guidance](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html), CVPR 2026 | Strong detector/cross-dataset baseline; redundant in the closest-mechanism table. |
| 17 | [Boosting IRSTD via Logit-Domain Contrast and Adaptive Shape Refinement](https://arxiv.org/abs/2607.01555), 2026 preprint | Very recent false-alarm-aware training loss; detector-side, not deployment calibration. |
| 18 | [Ivan-ISTD](https://arxiv.org/abs/2510.12241), 2025 preprint | Cross-domain/noise-invariance representation learning; no threshold policy. |
| 19 | [SAIST](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html), CVPR 2025 | Strong multimodal detector, but inputs make direct comparison conditional. |
| 20 | [ISNet](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html), CVPR 2022 | Older strong detector anchor, superseded in the compact set. |
| 21 | [TeST](https://openaccess.thecvf.com/content/WACV2023/html/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV_2023_paper.html), WACV 2023 | Unlabeled self-training that updates representations; retained as a baseline family but not a closest mechanism. |
| 22 | [What, How and When Should Object Detectors Update](https://openaccess.thecvf.com/content/CVPR2024/html/Yoo_What_How_and_When_Should_Object_Detectors_Update_in_Continually_CVPR_2024_paper.html), CVPR 2024 | Continual TTOD with adaptors; different stateful update regime. |
| 23 | [Tent](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021 | Foundational TTA, but parameter-updating and classification-centric. |
| 24 | [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a.html), ICML 2022 | Generic parameter-updating TTA; redundant for closest-work positioning. |
| 25 | [Consistency-Guided Temperature Scaling](https://ojs.aaai.org/index.php/AAAI/article/view/29041), AAAI 2024 | OOD probability calibration, not target-context operating-point calibration. |
| 26 | [A Generalized Neyman-Pearson Criterion for Optimal Domain Adaptation](https://proceedings.mlr.press/v98/scott19a.html), PMLR 2019 | Formal theory anchor; superseded by the 2026 transfer-NP entry in the compact table. |
| 27 | [Learn then Test](https://arxiv.org/abs/2110.01052), 2021 preprint | Foundational risk-control framework; represented by CRC and shifted risk control. |
| 28 | [Conformal Prediction Sets with Limited False Positives](https://arxiv.org/abs/2202.07650), 2022 preprint | User FP tolerance for multilabel sets; output and supervision differ. |
| 29 | [Unlabelled Data Improves Bayesian Uncertainty Calibration under Covariate Shift](https://proceedings.mlr.press/v119/chan20a.html), ICML 2020 | Supports unlabeled shifted calibration broadly, but not threshold curves. |
| 30 | [Test-Time Adaptive Object Detection with Foundation Model](https://openreview.net/forum?id=MO4U4mg0oT), NeurIPS 2025 | Prompt/teacher/memory-based TTOD; materially heavier and parameter-updating. |

## Safe novelty boundary

The defensible claim is not a new calibration primitive. It is an IRSTD-specific composite:

> Given a short unlabeled context from an unseen target domain and a frozen detector,
> RC5+ infers a continuous, structurally monotone threshold function indexed by exact
> native-pixel false-alarm budgets. The function transports an exact same-budget empirical
> target-tail anchor with a source-OOF learned correction and is executed through an
> endpoint-aware, no-reject sealed decision chain.

Unsafe: `first test-time threshold calibration`, `first false-alarm prediction`,
`guaranteed false-alarm control`, `Neyman--Pearson optimal`, `source-free training`, and
`zero-shot`.

## Novelty verdict

- Direct-prior fatal gate: PASS within the bounded 30-paper search; no Direct match found.
- Broad primitive novelty: FAIL by design; the primitives are known and must be cited.
- Composite novelty: 4/5, medium-high confidence, conditional on preserving every stated
  contract and on evidence that the combined mechanism changes extreme-low-FA behavior.
- Universal-priority or `first` claim: NOT AUTHORIZED.

