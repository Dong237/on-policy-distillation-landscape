# WP3 Teacher-Free And Self-Play OPD Synthesis

Checked: 2026-05-13

This note merges:

- `assets/research/wp3-teacher-free-selfplay-chatgpt.md`
- `assets/research/wp3-teacher-free-selfplay-claude.md`
- `assets/research/wp3-teacher-free-selfplay-gemini.md`

## Core Finding

WP3 confirms a real but narrow strict teacher-free OPD core. The strict cases use a same-model teacher substitute with privileged context, feedback, demonstrations, concise instructions, or short context, and the objective consumes token-level KL/JSD-style supervision on the current student's own rollout.

The broad self-play preference family remains adjacent. SPIN, SPPO, DNO, Self-Rewarding LMs, Iterative DPO, IPO, SimPO, and OAIF can be online or self-play, but their consumed signal is a preference label, scalar judge score, sequence-level game loss, or reference-free margin rather than distillation-style supervision on exact student rollouts.

## Merge Decision

| Method | Final repo action | Reason |
|---|---|---|
| OPSD | Keep `strict_opd`; upgrade to `source_verified` high confidence. | Current student reasoning traces receive privileged self-teacher token supervision through KL-style distillation. |
| CRISP / OPSDC | Keep `strict_opd`; normalize authors and upgrade to `source_verified` high confidence. | The paper states that a concise-instruction-conditioned same model provides teacher logits and reverse KL on the student's own rollouts; periodic teacher refresh is an implementation detail, not a strictness failure. |
| SDPO | Keep `strict_opd`. | Feedback-conditioned self-teacher supervises current policy samples through token/logit self-distillation. |
| OPCD | Keep `strict_opd`. | Student samples without context; context-conditioned same model supervises the same output prefixes through reverse KL. |
| SDFT 2026 | Add `strict_opd` row. | Demonstration-conditioned EMA self-teacher supplies reverse-KL supervision on student-sampled responses. |
| OPSDL | Keep `strict_opd`. | Long-context student outputs are supervised by a short-context self-teacher through pointwise reverse KL. |
| CaOPD | Keep `strict_opd` with medium confidence. | Privileged self-teacher supervision is strict, but calibration target replacement remains worth future code-level checking. |
| OEL | Keep whole-method `partial_opd`. | OPCD-style consolidation substage is strict, but the full method also includes deployment collection and knowledge extraction. |
| GATES | Keep whole-method `partial_opd`; upgrade to `source_verified` high confidence. | Main method distills consensus-gated tutor trajectories; an on-policy arm is not enough to label the full method strict. |
| HDPO | Keep whole-method `partial_opd`; upgrade to `source_verified` high confidence. | GRPO branch is on-policy but reward-only; distillation branch uses privileged teacher rollouts, so C1 fails for the distillation arm. |
| pi-Distill joint objective | Track adjacent/not OPD, not as an OPD table row. | Privileged teacher generates the supervised traces; the OPSD subroutine is the strict companion. |
| SDFT 2024 | Track adjacent/not OPD. | One-shot self-rewritten data plus SFT, not same-rollout distillation. |

## Self-Play Preference Boundary

Previous checkpoints, self-play opponents, or self-judges can satisfy the teacher-substitute role only when they provide distillation-style supervision on the current student's rollout. They do not satisfy strict OPD when they only:

- generate negatives;
- rank or judge response pairs;
- provide scalar rewards;
- act as DPO/IPO/SPO reference policies;
- rewrite a static dataset.

This keeps SPIN and related preference/game objectives in the adjacent ledger rather than the OPD table.

## Table Changes

- Added `sdft-continual-2026` to `tables/opd_papers.md`.
- Updated `opsd-2026` and `opsdc-crisp-2026` to `source_verified` high confidence.
- Updated `gates-2026` and `hdpo-2026` to `source_verified` high confidence while keeping them `partial_opd`.
- Added WP3 adjacent rows for SPIN, SPPO, Self-Rewarding LMs, DNO, IPO, SimPO, Iterative DPO, OAIF, pi-Distill, and SDFT 2024.

## Remaining Gaps

| Gap | Current action | Needed next |
|---|---|---|
| MTP-SD and DAIL were mentioned in earlier planning but not resolved by this WP3 batch. | Keep out of tables or leave existing adjacent wording cautious. | Route a narrow follow-up only if they matter for the first release. |
| CaOPD calibration target may alter the teacher signal. | Keep strict with medium confidence. | Recheck when code or a clearer appendix is available. |
| Previous-checkpoint token-level OPD without privileged context is still underexplored. | Record as an open gap. | Search specifically for EMA/previous-checkpoint logits on current student rollouts. |
| Substage vs method-level labels remain important. | Keep OEL/GATES/HDPO partial at method level. | Split substage rows only if the repo adopts a general substage-row policy. |

## Next Router State

WP3 is complete for the current pass. Later WP4-WP7 passes have been merged separately; current routing state is tracked in [deep-research-plan.md](deep-research-plan.md).
