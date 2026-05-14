# WP1 Line-Audit Synthesis

Checked: 2026-05-12

This note merges:

- `assets/research/wp1-line-audit-chatgpt.md`
- `assets/research/wp1-line-audit-claude.md`
- `assets/research/wp1-line-audit-gemini.md`

## Scope Actually Covered

The files audit 15 existing high-risk rows:

Uni-OPD, MAD-OPD, VOLD, MiMo-V2-Flash, Gemma2 post-training, G-OPD, REOPOLD, PACED, SDPO, OPCD, OEL, SCOPE, KDRL, Lightning OPD, and DistiLLM-2.

They did not audit the new WP1 candidate queue:

vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens, DSKD v2, SelecTKD, ToDi, and HPD.

## Merge Decision

No main-table reclassification is needed. The current repo labels match the conservative synthesis of the three line audits.

| Method bucket | Final repo posture | Reason |
|---|---|---|
| Source-verified strict | Uni-OPD, MAD-OPD, VOLD Stage 2, MiMo-V2-Flash MOPD stage, G-OPD, REOPOLD, SDPO, OPCD | C1/C2/C3 are explicit in the audited source excerpts. |
| Borderline strict | Gemma2 post-training, SCOPE, KDRL | Gemma2 lacks an explicit post-training loss; SCOPE and KDRL have strict OPD subcomponents but method-level hybrids. |
| Partial OPD | PACED, OEL, Lightning OPD, DistiLLM-2 | Whole-method rollout provenance is mixed, staged, offline, or previous-epoch rather than clean current-policy OPD. |

## Notable Conflict Resolution

| Method | Audit disagreement | Repo decision |
|---|---|---|
| REOPOLD | One audit called it borderline because teacher supervision is expressed as clipped token-level reward. | Keep `strict_opd`: repo rules allow dense teacher log-ratio supervision on student rollouts, even when written as a policy-gradient objective. |
| PACED | Some audits call the reverse-KL self-distillation track strict. | Keep whole-method `partial_opd`; split a strict subrow only if the repo later wants substage-level rows. |
| OEL | Some audits call the OPCD consolidation substage strict. | Keep whole-method `partial_opd`; the full loop includes deployment collection and knowledge extraction. |
| SCOPE | Some audits call the incorrect branch strict. | Keep method-level `borderline_strict`; only the incorrect branch uses teacher KL, while the correct branch uses student-weighted MLE. |
| KDRL | Some audits call the KD-RKL subcomponent strict. | Keep method-level `borderline_strict`; full objective mixes KD-RKL with reward-only GRPO. |

## Implication For The Repo

This audit strengthened the existing table but did not close WP1 by itself. The candidate queue was later closed in [wp1-candidate-line-audit-synthesis.md](wp1-candidate-line-audit-synthesis.md).

Do not rerun the older 15-method spot-check list unless those rows change materially.
