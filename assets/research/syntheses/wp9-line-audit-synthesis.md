# WP9 Line-Audit Synthesis

Last checked: 2026-05-12

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) supersedes the PRISM label and downgrades it to `adjacent` under the strict C3 rule.

Inputs:

- `assets/research/wp9-line-audit-chatgpt.md`
- `assets/research/wp9-line-audit-claude.md`
- `assets/research/wp9-line-audit-gemini.md`

This pass asked external models to extract line-level primary-source evidence for the previous `human_spotcheck_needed` queue. The merge rule remains conservative: the table label describes the whole method or named stage recorded in the row, not the most favorable subcomponent.

## Resolved Strict

| Method | Decision | Scope note |
|---|---|---|
| Uni-OPD | `strict_opd`, `source_verified` | Line audits found current-policy student trajectories, teacher token guidance, and reverse-KL / margin-calibrated objective evidence. |
| MAD-OPD | `strict_opd`, `source_verified` | Strict for multi-teacher debate supervision on student on-policy states. |
| VOLD | `strict_opd`, `source_verified` | Strict only for Stage 2 unified RL plus OPD, not the SFT stage. |
| MiMo-V2-Flash | `strict_opd`, `source_verified` | Strict only for Stage 3 MOPD; the whole industrial pipeline remains multi-stage. |
| G-OPD | `strict_opd`, `source_verified` | Student-generated trajectories plus teacher logit distribution and generalized OPD objective. |
| REOPOLD | `strict_opd`, `source_verified` | Teacher log-ratio is consumed as dense token-level supervision despite RL-style wording. |
| SDPO | `strict_opd`, `source_verified` | Feedback-conditioned self-teacher counts as privileged-context teacher. |
| OPCD | `strict_opd`, `source_verified` | Context-conditioned teacher supervises student trajectories with reverse KL. |

## Resolved Borderline Or Partial

| Method | Decision | Reason |
|---|---|---|
| Gemma2 post-training | `borderline_strict`, `source_verified` | Public report states distillation on the student's distribution and cites GKD/MiniLLM, but does not expose full rollout/objective details. |
| PACED | `partial_opd`, `source_verified` | Reverse-KL self-distillation subcomponent is strict-like, but the full method also uses teacher-generated sequences and fixed pass-rate weights. |
| OEL | `partial_opd`, `source_verified` | OPCD consolidation substage is strict-like; the full method also includes deployment collection and knowledge extraction. |
| SCOPE | `borderline_strict`, `source_verified` | Incorrect-trajectory branch is token-level OPD; correct branch uses student-weighted MLE without teacher supervision. |
| KDRL | `borderline_strict`, `source_verified` | KD-RKL subcomponent is strict-like; full method is KD+GRPO hybrid. |
| Lightning OPD | `partial_opd`, `source_verified` | Offline approximation with precomputed reference/SFT rollouts, so current-policy freshness fails. |
| DistiLLM-2 | `partial_opd`, `source_verified` | Batch/previous-epoch student outputs mixed with teacher outputs; not fresh every-iteration OPD. |

## Final Narrow Audit: VLA-OPD And PRISM

Follow-up inputs:

- `assets/research/wp9-line-audit-vla-prism-chatgpt.md`
- `assets/research/wp9-line-audit-vla-prism-claude.md`
- `assets/research/wp9-line-audit-vla-prism-gemini.md`

| Method | Decision | Reason |
|---|---|---|
| VLA-OPD | `strict_opd`, `source_verified` | All three audits found student VLA rollouts, expert teacher action logits on the same visited states, and reverse-KL/action-token objective evidence. |
| PRISM | `adjacent`, `source_verified` after post-WP8 update | All three line audits found current-policy response rollouts and MoE discriminator supervision, but post-WP8 WP9 applies the stricter C3 rule: the consumed signal is response-level scalar/advantage-style feedback rather than token/logit distillation. |

The first WP9 spot-check queue is now closed. Future work should move to new research packages rather than re-auditing the same two entries.
