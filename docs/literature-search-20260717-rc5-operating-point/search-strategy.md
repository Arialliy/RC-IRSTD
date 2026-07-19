# RC5 Operating-Point Prior-Art Search Strategy

Date: 2026-07-17  
Mode: standard  
Purpose: AAAI-27 novelty boundary, Related Work structure, and baseline selection  
Search cutoff: public records visible on 2026-07-17

## Question under review

The search tests whether public work already combines all of the following:

1. a short unlabeled context from an unseen target domain;
2. a frozen infrared-small-target detector;
3. a threshold family indexed by explicit false-alarm budgets rather than one accuracy-calibration scalar;
4. structural monotonicity across budgets;
5. native-resolution pixel false-alarm evaluation at extreme budgets;
6. no rejection or abstention branch;
7. an analytic target-tail/order-statistic anchor; and
8. a correction learned only from source domains.

This is a composite prior-art question. Individual ingredients are also searched separately so that a novel *combination* is not incorrectly presented as a novel primitive.

## Public queries used

No private draft sentence, result, local path, or unreleased artifact was placed in a query. Query families used only public task and method terms:

- `infrared small target detection domain adaptation generalization`
- `infrared small target detection cross-dataset generalization`
- `generalizable infrared small target detection benchmark`
- `infrared small target detection false alarm CVPR AAAI`
- `infrared small target detection test-time adaptation`
- `test-time adaptation object detection unlabeled target`
- `fully test-time adaptation object detection single image`
- `continual test-time adaptation object detector`
- `unlabeled test-time calibration distribution shift confidence threshold`
- `test-time recalibration unlabeled examples cutoff threshold`
- `transductive threshold calibration unseen test distribution`
- `source-free conformal prediction unlabeled target calibration`
- `out-of-domain temperature scaling calibration`
- `Neyman-Pearson classification domain shift false positive rate`
- `conformal risk control segmentation false positive`
- `learn then test risk control threshold`
- `native pixel false alarm infrared small target`
- `extreme false positive rate threshold calibration`

Queries were restricted to, or every candidate was subsequently verified at, one of these primary/stable source families:

- CVF Open Access;
- AAAI OJS proceedings;
- PMLR;
- official OpenReview forum/paper pages; and
- arXiv author-submitted records.

## Screening protocol

1. Broad discovery by public keywords.
2. Title deduplication.
3. Verification at an allowed primary/stable page.
4. Abstract/full-page inspection for task, supervision, deployment-time inputs, updated parameters, output type, risk semantics, and evaluation protocol.
5. Assignment to one closest-work cluster.
6. Similarity label:
   - **Direct**: substantially the same task, mechanism, and evidence path;
   - **Conceptual**: a central mechanism overlaps, but task, supervision, output, or guarantee differs;
   - **Contextual**: same application/benchmark pressure but no close operating-point mechanism.
7. Quality scoring on 1–5 anchors for insight, completeness, and experimental numeric evidence. Pure benchmarks use `N/A benchmark` and receive a benchmark-quality note.
8. Final inclusion based on closeness, source quality, recency, and value for positioning or baselines.

## Search breadth and result set

- Candidates screened: **25**.
- Final close/supporting set: **15**.
- Final relation counts: **0 Direct, 9 Conceptual, 6 Contextual**.
- Years represented in the final set: 2019–2026.
- Clusters covered: cross-domain/generalizable IRSTD; recent strong IRSTD detectors; unlabeled TTA/TTOD; dynamic threshold/recalibration; Neyman–Pearson/risk control.

The absence of a Direct match is evidence only within this bounded search. It is not proof of worldwide priority and does not authorize a `first` claim.

## Source-quality policy and exclusions

- MDPI material was excluded from search evidence, scoring, and citations.
- Search snippets, ResearchGate copies, aggregator summaries, DBLP-only records, inaccessible publisher pages, and low-signal venues were not used as evidence.
- Workshop papers and arXiv preprints were retained only when they were unusually close to the mechanism and are explicitly labeled as such.
- A paper's own `first` or `state-of-the-art` language was not adopted as an independent fact.
- Numerical claims are not imported into RC-IRSTD; the score records only the strength and breadth of evidence reported by the cited paper.

## Known coverage limits

- The user-requested source restriction excludes a systematic IEEE/ACM/full-journal and patent search. Several IRSTD domain-adaptation works are therefore acknowledged only as an unresolved citation-chaining risk, not used as final evidence.
- Terminology is fragmented: `operating point`, `threshold calibration`, `transductive calibration`, `risk control`, `false-alarm control`, and `Neyman–Pearson classification` can describe related mechanisms.
- Extreme native-pixel budgets may be discussed only inside supplements or under application-specific notation. This search inspected the closest official pages and selected full papers, but not every supplement exhaustively.
- RealScene-ISTD is retained as an arXiv preprint; no peer-reviewed venue was verified from the allowed sources.

## Decision rule for novelty wording

Use a bounded evidence statement:

> In the IRSTD, test-time adaptation, threshold-calibration, and risk-control works reviewed here, we did not find a paper evaluating the complete RC5 deployment bundle.

Do not convert it to:

> We are the first to perform unlabeled test-time threshold calibration or false-alarm control.

The latter is contradicted at the conceptual level by transductive threshold calibration, unlabeled conformal recalibration, and established Neyman–Pearson/risk-control work.
