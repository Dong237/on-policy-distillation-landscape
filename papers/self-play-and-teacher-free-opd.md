# Self-Play and Teacher-Free OPD

Teacher-free OPD replaces an external teacher with a previous checkpoint, privileged self-view, feedback-conditioned self-view, or other teacher substitute.

## Current Status

The self-play audit provenance is archived under [assets/research](../assets/research/). The strict core is smaller than the surface vocabulary suggests: strict methods need current student rollouts plus token/logit-style supervision from the teacher substitute on those same rollouts.

## Strict Core

| Method | Current label | Why |
|---|---|---|
| OPSD | `strict_opd` | Privileged self-teacher supervises current student reasoning traces through per-token KL-style distillation. |
| CRISP / OPSDC | `strict_opd` | Concise-instruction-conditioned self-teacher provides logits on student rollouts; reverse KL compresses reasoning. |
| SDPO | `strict_opd` | Feedback-conditioned self-teacher supervises current-policy attempts through logit/KL self-distillation. |
| OPCD | `strict_opd` | Context-free student samples outputs; context-conditioned same model supervises the same prefixes. |
| SDFT 2026 | `strict_opd` | Demonstration-conditioned EMA self-teacher provides reverse-KL supervision on student-sampled responses. |
| OPSDL | `strict_opd` | Long-context student outputs are supervised by a short-context self-teacher. |

## Partial And Mixed Methods

| Method | Current label | Why |
|---|---|---|
| OEL | `partial_opd` | OPCD-like consolidation is strict, but the whole loop also includes deployment collection and knowledge extraction. |
| GATES | `partial_opd` | Main training distills consensus-gated tutor trajectories; an on-policy arm is not enough to label the whole method strict. |
| HDPO | `partial_opd` | GRPO branch is on-policy but reward-only; distillation branch uses privileged teacher rollouts rather than student rollouts. |
| pi-Distill joint objective | adjacent / not OPD | Privileged teacher generates the supervised traces; OPSD is the strict companion. |

## Key Boundary

Self-play preference optimization is not automatically OPD. SPIN, SPPO, Self-Rewarding LMs, DNO, Iterative DPO, IPO, SimPO, and OAIF stay adjacent because they use preference labels, scalar self-judge scores, sequence-level game losses, or reference-free margins rather than teacher-style supervision on exact current rollouts.

Previous checkpoints can count as a teacher substitute only when their token/logit distribution supervises the current student's rollout through a distillation objective.
