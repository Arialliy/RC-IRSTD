# RC5 Operating-Point Prior-Art Screening Table

Date: 2026-07-17  
Screened: 25 deduplicated papers  
Final set: 15 papers

## Reading key

- Relation is exactly one of **Direct**, **Conceptual**, or **Contextual**.
- `I/C/N` means insight, completeness, and experimental numeric-evidence scores on a 1–5 scale.
- `N/A benchmark` is used only for a pure benchmark/protocol paper.
- `Risk` means the paper must be discussed because it can weaken a broad novelty claim; it is not a quality judgment.
- Scores assess the cited work from its allowed primary page/full paper. They do not imply reproducibility was independently rerun.

## Final set: 15 papers

| # | Paper and verified status | Cluster | Type | Relation | I/C/N | Overall | Why it is in the final set |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Rethinking Generalizable Infrared Small Target Detection: A Real-scene Benchmark and Cross-view Representation Learning](https://arxiv.org/abs/2504.16487), arXiv 2025 preprint | Cross-domain/generalizable IRSTD | method + benchmark | Contextual | 4/4/4 | A | Directly studies IRSTD domain shift, cross-view alignment, noise-robust representation, and a real-scene dataset, but changes representation learning rather than calibrating a frozen detector's operating point. |
| 2 | [Cross-domain Joint Learning with Prototype-guided Mixture-of-Experts for Infrared Moving Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37373), AAAI 2026 | Cross-domain/generalizable IRSTD | method + benchmark | Contextual | 4/4/4 | A | A current AAAI reference for cross-dataset IR small-target generalization; jointly trains a universal detector over multiple domains and therefore defines a different deployment regime. |
| 3 | [Rethinking Evaluation of Infrared Small Target Detection](https://arxiv.org/abs/2509.16888), NeurIPS 2025 Datasets and Benchmarks paper, arXiv author record used | Cross-domain/generalizable IRSTD | pure benchmark | Contextual | 4/5/N/A benchmark | A | Challenges dataset-specific evaluation, combines pixel/target perspectives, adds error analysis, and explicitly promotes cross-dataset evaluation. It is central to protocol credibility, not a competing calibrator. |
| 4 | [Infrared Small Target Detection with Scale and Location Sensitivity](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html), CVPR 2024 | Recent strong IRSTD detector | pure method | Contextual | 4/4/5 | A | Introduces SLS loss and MSHNet and is a strong, simple detector baseline; it does not address unlabeled target-context operating-point adaptation. |
| 5 | [Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html), CVPR 2026 | Recent strong IRSTD detector | pure method | Contextual | 4/4/4 | A | NS-FPN explicitly targets false alarms through learned noise suppression and is a current detector-side comparator; its mechanism modifies feature extraction. |
| 6 | [Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html), CVPR 2026 | Recent strong IRSTD detector | pure method | Contextual | 4/4/5 | A | InvDet reports five-benchmark and cross-dataset evaluation and is important for showing that detector representation changes are a strong alternative to post-hoc calibration. |
| 7 | [TeST: Test-Time Self-Training Under Distribution Shift](https://openaccess.thecvf.com/content/WACV2023/html/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV_2023_paper.html), WACV 2023 | Unlabeled TTA/TTOD | pure method | Conceptual | 4/4/4 | A | Uses limited unlabeled test data and evaluates object detection and segmentation, but adapts model representations through student-teacher self-training. |
| 8 | [Fully Test-time Adaptation for Object Detection](https://openaccess.thecvf.com/content/CVPR2024W/MAT/html/Ruan_Fully_Test-time_Adaptation_for_Object_Detection_CVPRW_2024_paper.html), CVPR 2024 workshop | Unlabeled TTA/TTOD | pure method | Conceptual | 3/3/4 | B | Especially close in data regime because it adapts from a single unlabeled test image, but it updates the detector for several iterations and relies on pseudo-label filtering. |
| 9 | [What, How and When Should Object Detectors Update in Continually Changing Test Domains?](https://openaccess.thecvf.com/content/CVPR2024/html/Yoo_What_How_and_When_Should_Object_Detectors_Update_in_Continually_CVPR_2024_paper.html), CVPR 2024 | Unlabeled TTA/TTOD | pure method | Conceptual | 4/4/5 | A | A strong online-TTA comparator that freezes the backbone but updates added adaptors; it helps sharpen RC5's stricter claim that detector parameters and feature adaptors remain unchanged. |
| 10 | [Learning for Transductive Threshold Calibration in Open-World Recognition](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_for_Transductive_Threshold_Calibration_in_Open-World_Recognition_CVPR_2024_paper.html), CVPR 2024 | Dynamic threshold/recalibration | pure method | Conceptual | 5/4/5 | Risk | The closest mechanism-level precedent: a learned calibrator consumes unlabeled test instances, estimates TPR/TNR as functions of thresholds, and selects a threshold for a target metric while the embedding model is fixed. The task is pairwise metric recognition, not IRSTD pixel masking. |
| 11 | [Test-time Recalibration of Conformal Predictors Under Distribution Shift Based on Unlabeled Examples](https://arxiv.org/abs/2210.04166), arXiv 2022/2023 preprint | Dynamic threshold/recalibration | pure method | Conceptual | 4/3/4 | Risk | Explicitly predicts a cutoff for a shifted distribution from unlabeled examples. Its output is a conformal classification set, and the paper itself notes that general unlabeled recalibration cannot guarantee reliability. |
| 12 | [Calibrating Without Labels: Source-Free Conformal Prediction Using Pseudo-Labels](https://proceedings.mlr.press/v266/angelman25a.html), COPA/PMLR 2025 | Dynamic threshold/recalibration | pure method | Conceptual | 4/4/5 | Risk | Uses only unlabeled target data and pseudo-labels to estimate conformal thresholds across many shifts. It is not native-pixel false-alarm calibration and does not use an analytic tail anchor. |
| 13 | [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg), ICLR 2024 | Neyman–Pearson/risk control | theory/proof | Conceptual | 5/5/4 | A | Establishes calibrated control of monotone losses for a fixed predictor using labeled exchangeable calibration data. It prevents RC5 from implying a formal guarantee without matching assumptions and proof. |
| 14 | [A Generalized Neyman-Pearson Criterion for Optimal Domain Adaptation](https://proceedings.mlr.press/v98/scott19a.html), ALT/PMLR 2019 | Neyman–Pearson/risk control | theory/proof | Conceptual | 5/5/1 | A | Provides a formal NP-like domain-adaptation criterion with unlabeled target examples under explicit assumptions. It is a theory anchor, not a pixel-segmentation or short-context method. |
| 15 | [Neyman-Pearson Classification under Both Null and Alternative Distributions Shift](https://openreview.net/forum?id=pHckxhmBlI), ICLR 2026 | Neyman–Pearson/risk control | theory/proof | Conceptual | 5/4/2 | Risk | Recent transfer-NP theory controls Type-I error while adapting across shifts in both classes. It makes `NP-optimal` or guaranteed false-alarm language unsafe for an empirical calibrator. |

### Pure-benchmark quality note for paper 3

- Benchmark scope: IRSTD evaluation across pixel-, target-, error-type, and cross-dataset views.
- Task realism: explicitly challenges the prevailing same-dataset train/test convention.
- Metric validity: proposes a hybrid-level view rather than treating fragmented metrics as interchangeable.
- Baseline coverage: broad detector comparison and error analysis are reported by the official paper record.
- Adoption/reproducibility signal: an open-source evaluation toolkit is reported; adoption is necessarily recent.
- Known limitation for RC5: it does not by itself validate RC5's extreme-budget estimability or native-pixel denominator choices.

## Additional screened candidates: 10 papers

These were scored and retained as screening evidence but omitted from the 15-paper final set to avoid redundancy or protocol mismatch.

| # | Paper and status | Closest cluster | Relation | I/C/N | Overall | Disposition reason |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | [SAIST: Segment Any Infrared Small Target Model Guided by Contrastive Language-Image Pretraining](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html), CVPR 2025 | Strong IRSTD detector | Contextual | 4/4/4 | A | Baseline-only: strong and recent, but depends on multimodal prompting and a new image-text dataset, making a like-for-like RC5 comparison conditional on those inputs. |
| 17 | [ISNet: Shape Matters for Infrared Small Target Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html), CVPR 2022 | Strong IRSTD detector | Contextual | 4/4/4 | A | Older detector anchor; superseded in the compact final table by newer MSHNet/NS-FPN/InvDet references. |
| 18 | [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021 | Unlabeled TTA | Conceptual | 5/4/5 | A | Foundational TTA citation, but it updates normalization/affine parameters and is less task-specific than the selected TTOD papers. |
| 19 | [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a.html), ICML/PMLR 2022 | Unlabeled TTA | Conceptual | 4/5/5 | A | Strong generic TTA baseline; omitted because it is classification-centric and updates model parameters. |
| 20 | [Test-Time Adaptive Object Detection with Foundation Model](https://openreview.net/forum?id=MO4U4mg0oT), NeurIPS 2025 | Unlabeled TTA/TTOD | Conceptual | 4/4/5 | A | Current TTOD work, but its foundation-model prompt tuning, mean teacher, and dynamic memory are much heavier than RC5's frozen-detector post-processing. |
| 21 | [Unlabelled Data Improves Bayesian Uncertainty Calibration under Covariate Shift](https://proceedings.mlr.press/v119/chan20a.html), ICML/PMLR 2020 | Recalibration | Conceptual | 4/4/4 | B | Shows that unlabeled target data can aid uncertainty calibration, but does not output budget-indexed operating thresholds. |
| 22 | [Consistency-Guided Temperature Scaling Using Style and Content Information for Out-of-Domain Calibration](https://ojs.aaai.org/index.php/AAAI/article/view/29041), AAAI 2024 | Recalibration | Conceptual | 4/4/4 | A | Strong source-domain OOD confidence calibration; excluded because it does not consume short target context or control pixel false alarms. |
| 23 | [High Probability Risk Control Under Covariate Shift](https://proceedings.mlr.press/v266/almeida25a.html), COPA/PMLR 2025 | Risk control | Conceptual | 4/4/4 | B | Directly controls FPR in one application under covariate shift, but uses labeled calibration losses and importance weighting rather than label-free short-context calibration. |
| 24 | [Controlling False Positives in Image Segmentation via Conformal Prediction](https://arxiv.org/abs/2511.15406), arXiv 2025 preprint | Risk control/segmentation | Conceptual | 4/3/3 | B | Close error type and frozen-model post-processing, but selects one shrink parameter from a labeled exchangeable calibration set and is not IRSTD/extreme-budget work. |
| 25 | [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052), arXiv author record | Risk control | Conceptual | 5/5/4 | A | Foundational post-hoc risk-control framework; omitted from the final 15 only because CRC and the two NP papers already cover the main formal boundary. |

## Screening conclusion

- **Direct precedent found:** none in the bounded 25-paper search.
- **Closest conceptual precedent:** OpenGCN (paper 10), followed by unlabeled conformal recalibration (papers 11–12).
- **Closest application precedent:** generalizable/cross-domain IRSTD (papers 1–3), which changes detector learning or evaluation rather than only the operating point.
- **Strongest claim hazard:** presenting the broad ideas of test-time threshold calibration, label-free recalibration, or false-alarm-constrained decision-making as new.
