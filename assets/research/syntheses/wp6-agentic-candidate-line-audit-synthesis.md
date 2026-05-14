# WP6 Agentic Candidate Line Audit Synthesis

Checked: 2026-05-13

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) downgrades OEC to `adjacent` because expert continuations are trained with rejection-sampled SFT/NLL rather than teacher-distribution supervision on student tokens.

This note merges:

- `assets/research/wp6-agentic-candidate-line-audit-chatgpt.md`
- `assets/research/wp6-agentic-candidate-line-audit-claude.md`
- `assets/research/wp6-agentic-candidate-line-audit-gemini.md`

## Merge Decision

The candidate queue produced three kinds of outcomes:

- Stage/component strict evidence: LiteGUI guided OPD stage.
- Borderline or partial method-level evidence: TCOD, Skill-SD, OpenClaw-RL OPD component, RLSD.
- Adjacent or unresolved: OEC, DGPO, RISE, SAD, GLM-5 cross-stage OPD.

The repo remains conservative at method level. If a strict-looking subcomponent is embedded in GRPO/RLVR, hard-label SFT, PPO reward shaping, teacher-prefix rollouts, or industrial reporting without visible loss details, the row is not promoted as clean method-level strict OPD.

## Accepted Table Updates

| Method | Repo action | Reason |
|---|---|---|
| TCOD | Added `borderline_strict` to `tables/opd_papers.md`. | F2B is strong OPD evidence, but the method also includes B2F teacher-prefix states before the student suffix; conservative method-level label avoids overclaiming. |
| LiteGUI | Added `strict_opd` stage row to `tables/opd_papers.md` and `tables/vlm_opd_papers.md`. | Stage-1 guided OPD/GKD has student GUI completions scored by a teacher on the same sampled sequence with token-level reverse-KL style supervision. Stage-2 GRPO is separable. |
| Skill-SD | Added `borderline_strict` to `tables/opd_papers.md`. | The skill-conditioned self-distillation term is token-level and same-rollout, but the full recipe is GRPO-heavy and the SDL coefficient is small. |
| OEC | Superseded by post-WP8 WP9: now `adjacent`. | Student prefixes put the expert at visited states, but the objective is hard-label SFT/rejection sampling rather than teacher-distribution OPD. |
| OpenClaw-RL OPD component | Added `borderline_strict` to `tables/opd_papers.md`. | Hindsight teacher log-prob gaps are token-level and same-rollout, but consumed as a policy-gradient advantage and mixed with scalar PRM reward. |
| RLSD / Self-Distilled RLVR | Added `partial_opd` to `tables/opd_papers.md`. | Privileged self-teacher signal affects token-level update magnitude while RLVR reward determines direction; strict C3 fails. |
| DGPO | Added adjacent row to `tables/adjacent_work.md`. | Cold-start KD is offline and online teacher KL is folded into PPO guidance or advantage shaping. |
| RISE | Added adjacent row to `tables/adjacent_work.md`. | Current-model unrolls exist, but training is reward-weighted regression/SFT on improved responses, not distribution matching. |
| SAD | Kept adjacent false-positive row. | Structured Agent Distillation remains offline/pre-collected teacher-trajectory distillation. |

## Held Out

| Method | Status | Why |
|---|---|---|
| GLM-5 cross-stage OPD | Not merged. | ChatGPT and Claude found plausible stage-strict industrial evidence, but Gemini could not locate C2/C3 in line-visible public text. It should not be promoted until a section-level primary quote shows teacher specification, rollout freshness, and loss form. |
| GUI-SD | Kept existing strict row, later upgraded by WP7. | ChatGPT and Gemini supported the row while Claude could not find the exact name; WP7 later verified the GUI grounding row with high confidence. |

## Boundary Notes

- DAgger-style hard expert continuation on a student prefix is not clean strict OPD unless the source uses a richer distributional or distillation-style objective.
- Teacher log-prob gaps inside PPO/GRPO can be borderline or partial, but reward shaping alone is not strict.
- Privileged self-teachers count only when they score the same student-generated tokens, actions, coordinates, or states.
- Stage-scoped strict rows are allowed when the stage is separable and explicitly named in the row rationale.

## Router State

WP6 is complete. WP7 has since rechecked GUI-SD and LiteGUI where they overlap the multimodal scope.
