# WP2 Black-Box/API OPD Synthesis

Checked: 2026-05-13

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) downgrades PRISM from `borderline_strict` to `adjacent` because its response-level MoE discriminator score is consumed as scalar reward/advantage rather than a distillation loss.

This note merges:

- `assets/research/wp2-blackbox-claude.md`
- `assets/research/wp2-blackbox-gemini.md`
- `assets/research/wp2-line-audit-chatgpt.md`
- `assets/research/wp2-secondary-spotcheck-chatgpt.md`
- `assets/research/wp1-whitebox-redirect-spotcheck-chatgpt.md`

## Core Finding

WP2 did not find a source-verified `strict_opd` black-box/API method. The current black-box frontier mostly satisfies C1 and C2 but fails strict C3 because the teacher/API/discriminator signal is consumed as a response-level reward, score, or preference pair rather than dense teacher-style distributional supervision.

In practice, black-box "OPD" usually collapses into one of three forms:

| Form | Typical objective | Strictness implication |
|---|---|---|
| Response-level discriminator | GRPO/PPO reward from discriminator score | `partial_opd` or `borderline_strict`, not strict by default. |
| Verbal/API score | Ordinal score, critique, or scalar judge reward | Usually `partial_opd` or adjacent. |
| Teacher/student preference pair | DPO/ORPO/ranking loss | Partial if online/mixed, adjacent if static. |

## Merge Decision

| Method | WP2 decision | Repo action |
|---|---|---|
| GAD | `partial_opd` | Update to `verified/source_verified/high`; keep below strict because discriminator is response-level reward. |
| OVD | `partial_opd` | Update to `verified/source_verified/high`; verbal score is scalar/ordinal, not dense OPD. |
| PRISM | `adjacent` after post-WP8 WP9 | Current-policy response rollouts and MoE discriminator supervision exist, but the policy consumes scalar reward/advantages rather than a distillation loss. |
| SODA | `partial_opd` | Keep as source-verified partial; semi-on-policy q0 snapshot plus DPO preference objective. |
| ORPO-Distill | `partial_opd` | Keep as source-verified partial; mixed-policy teacher-positive/student-negative ORPO. |
| Lion | `adjacent` | Keep adjacent; student outputs drive curriculum, but training is SFT on teacher text. |
| CGD | `adjacent` | Add adjacent row; student output is context, teacher refined answer is target. |
| SuperCorrect | `adjacent` | Add adjacent row; teacher correction traces and cross-model DPO. |
| D2PO | `adjacent` | Add adjacent row; on-policy preferences, not distillation. |
| PLaD | `adjacent` | Add adjacent row; ranking loss and fixed or independent teacher outputs. |
| LLMR | `adjacent` | Add adjacent row; reward-based and not black-box in the strict sense. |
| RL-KD with Judge | `adjacent` | Add adjacent row; judge scalar reward via PPO. |
| GAKD | Out of WP2 scope | Keep adjacent; WP1 redirect audit found corpus or teacher-generated sequences rather than current student rollouts. |
| BOND | `adjacent` | Keep adjacent; target is reward-model Best-of-N distribution, not teacher-on-rollout OPD. |

## Why No Strict Black-Box OPD Yet

Strict OPD needs an objective that consumes teacher-style supervision on the student's own generated states. White-box OPD can do this with token logits or log-probs. Black-box methods usually only see text responses or scores, so the supervision becomes:

- scalar reward;
- response-level discriminator score;
- verbal score;
- chosen/rejected pair;
- teacher-generated correction text.

These are useful but do not directly match a teacher distribution over the student's visited states. They should not be promoted to strict without evidence of non-scalar, same-rollout distillation supervision.

## Resolved Primary Queue

The WP2 line audit resolved the primary black-box/API queue:

- GAD: current student responses are present, but discriminator supervision is consumed as response-level GRPO reward.
- PRISM: current policy rollouts are scored by an MoE discriminator, but policy updates consume reward/advantages rather than dense teacher-style supervision.
- OVD: teacher or environment agents provide verbal scores on rollouts, but training uses rejection sampling and GRPO over accepted or teacher-replaced trajectories.
- SODA: student negatives are sampled once from the base q0 snapshot before fine-tuning; DPO consumes static preference pairs.
- ORPO-Distill: mixed-policy student negatives are contrasted with teacher positives through ORPO; no teacher supervision on exact current rollouts is shown.

## Resolved Secondary Spot-Check

The secondary source-level triage is complete:

- GAKD, ADPA, daDPO, PAD, and CTPD were redirected out of WP2 because they require white-box teacher distributions, logits, token probabilities, or projected log-probs.
- PAD, ADPA, and daDPO are tracked as `partial_opd` method-level rows.
- GAKD and CTPD stay in the adjacent ledger because current student rollout evidence is missing.
- RLKD and GAR remain adjacent reward-RL neighbors.
- RLTF, FCP, and ALT remain verbal-feedback neighbors because their objectives compile feedback into self-distilled generations, feedback prediction, or conditional SFT rather than dense same-rollout teacher supervision.

Do not reroute this queue unless a new primary source changes the training loop evidence.
