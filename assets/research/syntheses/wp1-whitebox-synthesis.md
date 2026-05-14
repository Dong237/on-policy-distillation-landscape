# WP1 White-Box OPD Synthesis

Checked: 2026-05-13

This note merges the WP1 white-box OPD research outputs under `assets/research/`:

- `wp1-whitebox-chatgpt.md`
- `wp1-whitebox-claude.md`
- `wp1-whitebox-gemini.md`
- `wp1/wp1-white-box-divergence-audit.md`
- `wp1/wp1-expanded-audit.md`
- `wp1/wp1-candidate-rows.md`
- `section3_taxonomy_papers_comparison.md`
- `wp1-whitebox-redirect-spotcheck-chatgpt.md`
- `wp1-whitebox-redirect-spotcheck-claude.md`
- `wp1-whitebox-redirect-spotcheck-gemini.md`

## Decision

WP1 confirms the repo's white-box backbone but should not trigger a bulk strict-row merge. The verified core remains GKD, MiniLLM, DistillSpec's draft-generated variant, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, and Fast OPD. DistiLLM, DistiLLM-2, Speculative KD, PACED, AdaSwitch, and Lightning OPD stay below strict because their rollout provenance is stale, mixed, interleaved, or offline.

The new 2026 white-box methods are important, but they are not yet merged into `tables/opd_papers.md` as strict rows. They first need line-level C1/C2/C3 audit.

## What WP1 Adds To The Landscape

WP1 shifts the white-box story from "which KL?" to four implementation questions:

| Question | Methods or examples | Why it matters |
|---|---|---|
| Which divergence or target? | GKD, MiniLLM, DistiLLM, Entropy-Aware OPD, Veto, CaOPD | The objective controls mode coverage, mode seeking, uncertainty preservation, and calibration. |
| Which tokens are worth training on? | TIP, Rock Tokens, SOD, SelecTKD, REOPOLD masking | OPD cost is dominated by teacher scoring; token selection can reduce cost or instability. |
| How is the policy-gradient view stabilized? | G-OPD, REOPOLD, vOPD, AOPD, StableOPD | Reverse-KL OPD behaves like dense teacher-guided policy optimization and inherits variance and collapse risks. |
| Can OPD cross tokenizer boundaries? | SimCT, DSKD, ULD, MultiLevelOT, DWA-KD, byte-level methods | Strict OPD needs teacher distributions on student-generated states even when tokenizers differ. |

## Confirmed Keep-Strict Rows

These stay as strict seeds, with the caveats already recorded in the main table:

| Method | WP1 action | Caveat |
|---|---|---|
| GKD | Keep strict for pure on-policy lambda setting. | Mixed lambda variants should be separate partial rows if added. |
| MiniLLM | Keep strict. | Teacher-mixed sampling is a stabilizer; record it as an implementation caveat. |
| DistillSpec | Keep strict only for draft-generated/on-policy drafter training. | Other data-generation variants should not inherit strictness automatically. |
| Entropy-Aware OPD | Keep strict, medium confidence. | Needs line audit for exact entropy and top-k implementation. |
| G-OPD | Keep strict. | RL-style notation is acceptable because the signal is dense teacher log-ratio supervision. |
| REOPOLD | Keep strict. | Near-policy rollout convention and clipping/masking should stay in notes. |
| Veto | Keep strict, medium confidence. | Author/code metadata and bridge-target algebra need line extraction. |
| Fast OPD | Keep strict for retained prefixes. | Tokenizer assumptions and FLOP claims need line audit. |

## Confirmed Downgrades

| Method | Label | Reason |
|---|---|---|
| DistiLLM | `partial_opd` | Replay/stale or adaptive off-policy rollout freshness weakens C1. |
| DistiLLM-2 | `partial_opd` | Batched previous-policy student outputs and teacher outputs weaken C1. |
| Speculative KD | `partial_opd` | Student proposals can be replaced by teacher tokens, creating a hybrid trace. |
| PACED | `partial_opd` | Full recipe mixes teacher-sequence forward KL and student-sequence reverse KL. |
| AdaSwitch | `partial_opd` | Student prefix plus teacher suffix is not exact student rollout supervision. |
| Lightning OPD | `partial_opd` | Teacher log-probs are cached over reference/SFT rollouts, so current-policy C1 fails. |

## Candidate Queue, Resolved By Line Audit

The candidate queue was later audited and synthesized in [wp1-candidate-line-audit-synthesis.md](wp1-candidate-line-audit-synthesis.md).

| Candidate | Final repo action | Rationale |
|---|---|---|
| vOPD | `strict_opd` | Reverse-KL OPD with a detached control variate baseline. |
| TIP | `strict_opd` | Token selection on top of same-rollout teacher KL. |
| AOPD | `strict_opd` | Asymmetric teacher-logit objective on student trajectories. |
| StableOPD | `partial_opd` | Strict OPD core exists, but the full stabilization recipe mixes on-policy and off-policy branches. |
| CaOPD | `strict_opd` | Privileged self-teacher and calibration-corrected KL on student rollouts. |
| SOD | `strict_opd` | Step-wise teacher-logit OPD term remains explicit alongside GRPO. |
| SimCT | `strict_opd` | Cross-tokenizer teacher distribution on student text states via aligned continuation units. |
| OPSDL | `strict_opd` | Short-context self-teacher supervises long-context student outputs. |
| DP-OPD | `strict_opd` | DP-SGD wraps a GKD-style OPD loop; mixed lambda variants should be split if needed. |
| Rock Tokens | `strict_opd` | Token masking/freezing on top of reverse-KL OPD. |
| DSKD v2 | `partial_opd` | On-policy mode exists, but method-level evidence is mixed or optional-mode. |
| SelecTKD | `adjacent` | Propose-and-verify token selection lacks a clear dense same-rollout OPD loop. |
| ToDi | `adjacent` | Fixed-data divergence innovation, not OPD. |
| HPD | `partial_opd` | Off-policy-first with approximate on-policy next-token sampling. |

## Adjacent Work Added To Ledger

WP1 also identifies methods worth tracking as adjacent because they are objective, tokenizer, or efficiency innovations rather than strict OPD by default:

- CSD
- AMiD
- BiLD
- RSKD
- OKD
- EGAD
- ToDi
- HPD
- ULD
- MultiLevelOT
- DWA-KD
- DynSDPB

These should not be upgraded without explicit student-rollout evidence.

## WP2 Redirect Queue, Resolved

The WP2 secondary cleanup routed PAD, ADPA, GAKD, daDPO, and CTPD back into WP1 because they require white-box probabilities or projected teacher distributions. The redirect spot-check is synthesized in [wp1-whitebox-redirect-spotcheck-synthesis.md](wp1-whitebox-redirect-spotcheck-synthesis.md).

| Method | Final repo action | Rationale |
|---|---|---|
| PAD | `partial_opd` | Strong OPD-like preference-distribution loss, but default sampling is one-shot and iterative sampling is only epoch-level fresh. |
| ADPA | `partial_opd` | White-box advantage supervision exists, but states and advantages are frozen or precomputed in the main recipe. |
| daDPO | `partial_opd` | Teacher distributions enter a DPO-style loss on sampled teacher/student pairs, but the pairs are static for training. |
| GAKD | `adjacent` | Reverse-KL/adversarial losses use corpus or teacher-generated sequences, not current student autoregressive rollouts. |
| CTPD | `adjacent` | Cross-tokenizer teacher log-prob projection over static preference data, not current-rollout OPD. |

## Open Research Gaps

- No published Renyi or alpha-divergence OPD study was found.
- No Bregman-divergence OPD study was found.
- No systematic divergence-scheduling paper was found.
- Cross-tokenizer OPD at industrial scale remains unvalidated.
- Token selection and variance reduction are now major practical frontiers for white-box OPD.

## Current Router State

WP1 and WP2 are sufficiently closed for a first version. The next external task should move to a new work package, not another pass over the WP1/WP2 queues.
