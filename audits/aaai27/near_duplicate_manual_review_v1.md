# AAAI-27 near-duplicate review record

This review is an image-only integrity audit. It did not open masks, labels,
model scores, checkpoints, or metrics.

- Original audit: `near_duplicates_original_official_splits_v2.json`
- Candidate rule: pHash64 Hamming distance at most 4
- Confirmation rule: 64×64 mean-centred grayscale cosine at least 0.995
- Confirmed candidates: 31 pairs
- Visual artifact: `near_duplicate_pair_previews_v1.png`
- Pairwise decision: all 31 pairs are `same_scene_related`
- Conservative action: exclude every implicated official-train endpoint from
  every derived development role
- Unique excluded training IDs: 30 (NUDT-SIRST 27, IRSTD-1K 3, NUAA-SIRST 0)
- Raw data and official split files: unchanged
- Official-test labels: not read

The effective-development re-audit contains zero confirmed development/test
pairs. This supports use of the three datasets as Stage-1 development domains
under the frozen v2 split only. It does not establish temporal/causal
independence and does not provide the fourth independent outer domain required
for claim-bearing Stage 2.

The visual review was Codex-assisted; no human sign-off is claimed. The
quarantine is intentionally conservative, so no candidate is restored on the
basis of visual judgement.
