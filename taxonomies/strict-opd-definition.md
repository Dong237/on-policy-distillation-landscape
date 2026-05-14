# Strict OPD Definition

## Definition

A method is **Strict On-Policy Distillation (OPD)** only if all three conditions are explicitly supported:

1. **C1: Student rollout.** The current student or policy generates its own trajectories, completions, rationales, actions, tool calls, or intermediate states during training.
2. **C2: Teacher-style supervision.** A teacher, expert, discriminator, privileged-context model, reference model, previous checkpoint, or equivalent feedback source supervises those exact student-generated states.
3. **C3: Objective consumption.** The training objective directly consumes that supervision signal.

## Strictness checklist

A method is `strict_opd` only if all are true:

- [ ] Student generates trajectories during training.
- [ ] Supervision is applied to those student-generated trajectories.
- [ ] The feedback source is teacher, expert, discriminator, privileged model, reference model, or previous checkpoint.
- [ ] The supervision is distillation-style, not only sparse reward.
- [ ] The objective consumes the teacher-style supervision signal, not merely an outcome reward.
- [ ] Rollout freshness is recorded: current-policy, per-iteration, per-epoch, replay-buffer-stale, previous-policy, precomputed, static, or unclear.
- [ ] Teacher kind is recorded: larger LLM, multimodal teacher, expert policy, discriminator, reference model, privileged self, previous checkpoint, multi-teacher, verifier-only, or none.

If one or more are unclear, mark as `partial_opd` or `unclear`.

## Verifier Interpretation

The verifier pass made the strict boundary more explicit:

- Token/logit KL, teacher log-probs, privileged self-teacher KL, and action-token VLA teacher supervision can all satisfy C3.
- A same-weight model can be the teacher when it has privileged context, answers, memory, demonstrations, or other information the student does not see.
- A VLA action-token rollout is still an OPD rollout if the student policy generates the action trajectory and an expert teacher supervises those same action tokens.
- Response-level discriminator, verbal-score, or reward-model feedback is not locked strict by default. Use `borderline_strict` or `partial_opd` unless the source shows a non-scalar distillation objective.
- Stage-specific industrial claims must be scoped to the qualifying phase. Do not label a whole training pipeline strict because one phase uses OPD.
- The Thinking Machines Lab OPD blog is an operational definition/background source, not a normal paper row.

## Strictness labels

| Label | Meaning |
|---|---|
| `strict_opd` | Full OPD loop with student on-policy rollouts and distillation-style supervision. |
| `borderline_strict` | OPD principle is explicit, but feedback is black-box, response-level, action-level, frontier multimodal, or not yet red-team verified as classic token/logit OPD. |
| `partial_opd` | Some but not all OPD requirements are met. |
| `adjacent` | Useful for the landscape, but missing OPD supervision on student rollouts. |
| `not_opd` | Does not use student on-policy training states. |
| `unclear` | Public evidence is insufficient. |

## Canonical Evidence Field

Rows marked `strict_opd` or `borderline_strict` must fill `strictness_evidence` with all three markers:

`C1 ...; C2 ...; C3 ...`

Rows marked `adjacent` or `not_opd` must use the same field to state the missing condition, such as `Missing C2 and C3`.

## Common downgrades

- Reward-only GRPO/RLVR: `adjacent`.
- SFT on teacher-generated CoT: `not_opd` or `adjacent`.
- Offline KD from a larger model: `not_opd`.
- DPO on static preference pairs: `adjacent`.
- Industrial "online RL" without teacher-on-rollout supervision: `adjacent`.
- VLM post-training report with no OPD details: `unclear` or `adjacent`.

## Examples

| Work | Classification | Reason |
|---|---|---|
| GKD | `strict_opd` | Student self-generated outputs receive teacher feedback. |
| MiniLLM | `strict_opd` | Reverse-KL objective is optimized on student on-policy generation. |
| VOLD | `strict_opd` | VLM student traces are guided by a text teacher during online training. |
| Video-OPD | `strict_opd` | Current-policy video grounding trajectories receive dense teacher supervision. |
| PRISM | `adjacent` | Response-level MoE discriminator reward is consumed as policy-gradient advantage, not a distillation loss. |
| Uni-OPD | `strict_opd` | Line audit found student trajectories, teacher token guidance, and reverse-KL / margin-calibrated objective evidence. |
| VLA-OPD | `strict_opd` | Line audit confirms action-token teacher supervision on self-generated VLA trajectories with reverse KL. |
| MAD-OPD | `strict_opd` | Multi-teacher debate supplies token-level supervision on student on-policy states. |
| LiteGUI | `strict_opd` stage | Guided OPD/GKD stage has teacher supervision on the same sampled GUI sequence; later GRPO is separate. |
| OEC | `adjacent` | Expert completes from student-visited SWE states, but training is rejection-sampled SFT/NLL rather than distributional OPD. |
| RLSD | `partial_opd` | Privileged self-teacher affects token-level update magnitude while RLVR reward determines direction. |
| KEPO | `borderline_strict` | Medical VLM teacher divergence is quality-gated and not fully defined. |
| D-OPSD / Flow-OPD | `borderline_strict` | Diffusion/flow velocity-field supervision is same-rollout but not clean token/logit KL. |
| GTR-Turbo | `adjacent` | Reverse KL is collapsed into scalar PPO reward shaping. |
| VLM-R1 | `adjacent` | On-policy GRPO with reward-only supervision. |
| LLaVA-KD | `not_opd` | Offline MLLM distillation without student rollouts. |
