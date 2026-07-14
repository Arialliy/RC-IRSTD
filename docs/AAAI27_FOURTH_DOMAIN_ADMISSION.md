# AAAI27 fourth-domain admission record

Status: `G4_BLOCKED_PENDING_LICENSE_AND_DATA_AUDIT`

The local workspace contains no complete, licensed, independent fourth IRSTD
segmentation domain. Copies under other projects reduce to NUAA-SIRST,
NUDT-SIRST, or IRSTD-1K; three IRDST demo pairs and synthetic smoke fixtures
are not admissible research domains.

## Preferred acquisition route

### 1. RealScene-ISTD — preferred fourth development domain

- 739 real UAV infrared images at 540×420 with pixel masks.
- Official project: <https://github.com/luy0222/RealScene-ISTD>
- Paper: <https://arxiv.org/abs/2504.16487>
- The official repository supplies data and split links, but no explicit
  dataset-level license was found during the 2026-07-15 audit.

Admission requires written/authoritative research-use permission, original
archive SHA-256, official split preservation, sequence/source-group audit, and
cross-domain exact plus perceptual-near-duplicate checks against all three
current domains.

### 2. NUDT-SIRST-Sea — preferred external OOD stress domain

- 48 very large (10000×10000) satellite infrared images with pixel masks and
  17,598 annotated ships; official split 41/7.
- Official project: <https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection>
- Paper: <https://arxiv.org/abs/2209.13756>

This is better treated as a preregistered external pressure test. If tiled,
the 41/7 source-image split must be applied before tiling; tiles from one
source image may never cross roles. A data-level license still needs
confirmation.

### 3. SIRST-UAVB — conditional fallback

- 3,000 images with an official 4:1 split.
- Official project: <https://github.com/JN-Yang/PConv-SDloss-Data>
- Paper: <https://ojs.aaai.org/index.php/AAAI/article/view/32996>

UAVs have masks, but 1,742 bird instances are provided only as boxes. Treating
those birds as background creates systematic false-negative label noise for a
generic target detector. This domain is inadmissible unless the task is scoped
to UAV-only detection or the missing pixel labels are resolved without using
evaluation data.

## Explicit exclusions

- SIRST-v2 contains all SIRST-v1/NUAA material and is not independent.
- SIRST3 is the union of SIRST-v1, NUDT-SIRST, and IRSTD-1K.
- WideIRSTD aggregates several existing domains, including current data.
- NCHU-SIRST's first-party release provides Pascal/VOC boxes rather than
  official pixel masks.
- Local ISTDU-Net `Misc_1..20` images are byte-identical NUAA copies.
- Local IRDST contains only three demo pairs and no admissible train/dev split.

Until one candidate passes the admission contract, claim-bearing Stage 2,
strict nested LODO, and the G5/G6 success claims remain unauthorized.
