# RC5+ Budget-Conditioned Operating-Point Search Notes

Date: 2026-07-19  
Mode: standard, novelty-critical  
Target: AAAI-27  
Evidence status: public prior-art search only; no RC5+ result is asserted

## Public-safe search question

The search tests whether a public method already combines the following deployment and
algorithmic contract:

1. infrared small-target segmentation under held-out-domain shift;
2. a frozen detector and feature extractor;
3. a short, separated, unlabeled target-domain context;
4. an exact-rational native-pixel false-alarm budget as an explicit input;
5. a continuously queryable, structurally monotone budget-to-threshold function;
6. an exact same-budget target-tail/order-statistic anchor;
7. a correction trained only from source-domain OOF exact-event evidence;
8. EATC-v2 endpoint semantics and an always-predict, no-reject sealed decision chain.

The question is deliberately composite. Quantile estimation, monotone functions,
transductive calibration, false-positive control, domain adaptation, and IRSTD are treated
as established primitives unless the complete contract is matched.

## Query families

Only generic public terminology was queried. No private draft wording, local path, result,
or unreleased artifact was submitted to a search engine.

- `infrared small target detection adaptive threshold false alarm rate cross domain`
- `infrared small target detection test time adaptation domain generalization threshold`
- `infrared small target detection operating point calibration false alarm`
- `infrared small target detection dynamic threshold unlabeled`
- `ADA-IRSTD Active Domain Adaptation DOI`
- `FAPM infrared false alarm prediction module domain adaptation`
- `transductive threshold calibration 2025 2026`
- `test-time threshold calibration unlabeled distribution shift`
- `threshold prediction unlabeled target data distribution shift false positive`
- `operating point unlabeled target calibration threshold`
- `budget-conditioned threshold calibration false positive rate`
- `monotone operating point neural network threshold calibration`
- `false positive budget segmentation threshold calibration`
- `conformal false positive rate segmentation risk control`
- `Neyman-Pearson distribution shift classification false positive`
- `semantic domain adaptation infrared small target detection`

## Source and screening policy

- Preferred primary records: CVF Open Access, AAAI OJS, PMLR, OpenReview, arXiv author
  records, official publisher DOI pages, and official project/code pages when needed.
- Search snippets, ResearchGate, DBLP, blogs, and aggregators were discovery aids only and
  were not used as the final evidentiary record when a primary page was available.
- MDPI material was excluded from evidence and scoring.
- A Direct match must substantially share the task, deployment supervision, frozen-model
  constraint, output object, budget semantics, and evidence path. Ingredient overlap alone
  is Conceptual; application or benchmark overlap alone is Contextual.
- Paper type is recorded explicitly. Pure benchmarks are not given a fabricated method
  evidence score.
- `I/C/N` scores insight, completeness, and numerical experimental evidence on 1--5
  anchors. They assess the cited work, not RC5+.

## Breadth and update over the 2026-07-17 search

- Deduplicated candidates screened: 30.
- Final close/supporting set: 15.
- Final relation counts: 0 Direct, 9 Conceptual, 6 Contextual.
- Search period represented: 2019--2026.
- New high-priority records relative to the previous search include ADA-IRSTD (accepted
  2026), S2CPNet (2026 preprint), and the July 2026 AC-SLSIoU preprint.

ADA-IRSTD is the material new risk. Its source-supervised False Alarm Prediction Module
estimates false-alarm risk on unlabeled target images. It then uses the score for active
sample selection, obtains target labels, and updates an adaptation model. It does not infer
a false-alarm-budget threshold family, does not preserve a fully label-free target
deployment, and does not implement the RC5+ analytic-anchor/sealed-decision contract.
It therefore narrows broad wording but is not a Direct match.

## Search conclusion and confidence

- Direct precedent found in this bounded search: none.
- Highest mechanism risk: OpenGCN, because it already performs learned transductive
  threshold calibration from unlabeled test instances for a frozen representation model.
- Highest IRSTD-specific risk: ADA-IRSTD, because it already learns a source-supervised
  false-alarm-risk predictor for unlabeled target images.
- Highest formal-risk boundary: Conformal Risk Control, high-probability risk control under
  covariate shift, and false-positive conformal segmentation.
- Confidence that no screened paper matches the full contract: medium-high.
- Confidence that generic `unlabeled threshold calibration`, `false-alarm prediction`,
  `quantile anchor`, `monotonicity`, and `risk control` are not new: high.

The bounded absence of a Direct match does not prove worldwide priority and does not
authorize a `first` claim. A safe statement is limited to the reviewed clusters and the
complete IRSTD deployment bundle.

## Known limits

- This is not a patent search and does not exhaust every closed IEEE/ACM journal article.
- Several 2026 works are preprints or accepted manuscripts and can change before AAAI-27.
- Terminology remains fragmented across threshold calibration, operating-point prediction,
  risk control, CFAR, conformal calibration, and transfer Neyman--Pearson classification.
- Citation chaining must be repeated immediately before submission.

