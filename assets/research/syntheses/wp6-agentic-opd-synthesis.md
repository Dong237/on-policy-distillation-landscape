# WP6 Agentic And Interactive OPD Synthesis

Checked: 2026-05-13

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) downgrades OEC to `adjacent` because expert continuations are trained with rejection-sampled SFT/NLL rather than teacher-distribution supervision on student tokens.

This note merges:

- `assets/research/wp6-agentic-opd-chatgpt.md`
- `assets/research/wp6-agentic-opd-claude.md`
- `assets/research/wp6-agentic-opd-gemini.md`

## Core Finding

Agentic OPD is real but narrow. Strict cases require more than an online agent trajectory: a teacher, expert, privileged branch, or debate ensemble must supervise the exact student-visited state, token, action, coordinate, or trajectory segment, and the objective must consume that signal as distillation rather than scalar reward.

Most agentic papers are false positives under the repo definition. Browser RL, SWE/code RL, self-correction, search agents, tool-use RL, and GUI RL often satisfy C1 because the agent acts online, but fail C2/C3 because the consumed signal is reward, pass/fail tests, ORM scores, DPO preferences, verbal memory, or SFT on static traces.

## Merge Decision

| Method | Final repo action | Reason |
|---|---|---|
| MAD-OPD / OPAD | Keep `strict_opd`. | Student samples on-policy actions or tokens; debate teachers force-decode or score that same state; JSD/RKL consumes token-level teacher supervision. |
| SOD | Keep `strict_opd`. | Student tool/reasoning trajectories receive step-weighted teacher-logit OPD alongside GRPO. |
| OPCD | Keep `strict_opd`. | Context-free student rollouts are scored by a privileged context-conditioned self-teacher through reverse KL. |
| OEL | Keep method-level `partial_opd`. | OPCD-style consolidation substage is strict, but the full method also includes deployment collection and experience extraction. |
| VLA-OPD | Keep `strict_opd`. | Student VLA rollouts visit states; a frozen expert teacher labels those same states; action-token reverse KL drives the update. |
| GUI-SD | Add `strict_opd` with grounding caveat; later WP7 upgraded confidence to high. | Student coordinate outputs are supervised by a privileged-visual self-teacher with weighted reverse KL; strict only for GUI grounding, not long-horizon GUI control. |
| SCoRe | Add adjacent. | Multi-turn online self-correction RL uses scalar reward and base-policy KL, not teacher distillation. |
| Reflexion | Add not-OPD adjacent ledger row. | Inference-time verbal memory with no weight update. |
| WebRL, ReTool, Search-R1, Agent-R1 | Add adjacent. | Online agent trajectories are trained with reward/ORM/task success, not teacher KL or correction distributions. |
| T3-Agent / MM-Traj, Visual Program Distillation, SWE-agent | Add/keep adjacent or not-OPD. | Offline synthetic traces or system/evaluation papers, not student-on-policy distillation. |
| SAD | Add false-positive adjacent row. | Structured Agent Distillation uses teacher-forced/pre-collected trajectories; C1 fails. |

## Candidate Queue Resolution

The follow-up line audit is synthesized in [wp6-agentic-candidate-line-audit-synthesis.md](wp6-agentic-candidate-line-audit-synthesis.md).

Accepted updates:

- `litegui-2026` as a stage-scoped `strict_opd` row in both main and VLM tables.
- `tcod-2026`, `skill-sd-2026`, and `openclaw-rl-opd-2026` as conservative `borderline_strict` rows.
- `oec-2025` was initially added as borderline, then later downgraded to `adjacent` by the post-WP8 WP9 verifier.
- `rlsd-2026` as `partial_opd`.
- `dgpo-2025` and `rise-2024` as adjacent rows.
- `sad-agent-2025` remains adjacent/offline.

Held out:

- GLM-5 cross-stage OPD. Public evidence is plausible but too terse for a table row until a section-level primary quote exposes teacher specification, rollout freshness, and loss form.
- GUI-SD remains in the table as limited strict GUI grounding; WP7 later resolved the confidence/status to high and source-verified.

## Evidence Ledger

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | state_or_action_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAD-OPD / OPAD | https://arxiv.org/abs/2605.01347 | Sec. 4.1-4.4 | "samples an on-policy action" | "Teachers then force-decode" | "confidence-weighted divergence" | multi-teacher debate ensemble | token/action token | JSD or reverse KL | current per-step | strict_opd | none | high |
| SOD | https://arxiv.org/abs/2605.07725 | Sec. 3.3; Sec. 4.2 | "student-generated trajectories" | "token-level supervision from a teacher" | "step-wise OPD objective" | white-box teacher logits | tool/reasoning step tokens | step-weighted reverse KL plus GRPO | current sampled trajectory | strict_opd | GRPO is auxiliary, but OPD component is strict | high |
| OPCD | https://arxiv.org/abs/2602.12275 | Sec. 3; Algorithm 1 | "y is sampled from the student model" | "teacher model that conditions on the context" | "reverse KL divergence" | privileged-context self-teacher | token | reverse KL | current student rollout | strict_opd | none | high |
| OEL | https://arxiv.org/abs/2603.16856 | Sec. 3.1-3.3 | "generates a response y" | "knowledge-conditioned output of a teacher" | "token-level reverse KL divergence" | experiential knowledge-conditioned self-teacher | token | OPCD-style reverse KL | mixed deployment plus consolidation | partial_opd | strict only for consolidation substage | high |
| VLA-OPD | https://arxiv.org/abs/2603.26666 | Algorithm 1 | "student VLA policy interacts" | "frozen expert teacher" | "Reverse-KL objective" | expert VLA teacher | action token | reverse KL action supervision | current student trajectories | strict_opd | none | high |
| GUI-SD | https://arxiv.org/abs/2605.00642 | Sec. 4.1-4.2 | "student generates an on-policy trajectory" | "privileged information" | "weighted reverse-KL objective" | privileged-visual self-teacher | coordinate tokens | entropy-guided weighted reverse KL | current policy | strict_opd | strict for GUI grounding only; WP7 later source-verified | high |
| SCoRe | https://arxiv.org/abs/2409.12917 | Abstract | "self-generated correction traces" | not found | "multi-turn online reinforcement learning" | none; base-policy KL only | correction trace | reward RL | current policy | adjacent | C2/C3 fail: scalar reward, not teacher distillation | high |
| Reflexion | https://arxiv.org/abs/2303.11366 | Abstract; Algorithm 1 | "Generate initial trajectory" | "linguistic feedback" | "not by updating weights" | verbal memory | episode memory | inference-time reflection | current/past trials | not_opd | no training objective | high |
| WebRL | https://arxiv.org/abs/2411.02337 | method | browser trajectories | outcome-supervised reward model | RL reward | scalar ORM | trajectory | reward-only browser RL | current policy | adjacent | C2/C3 fail | high |
| ReTool | https://arxiv.org/abs/2504.11536 | method | multi-turn code execution rollouts | outcome feedback | task rewards | scalar reward | trajectory | reward-only tool/code RL | current policy | adjacent | C2/C3 fail | high |
| Search-R1 | https://arxiv.org/abs/2503.09516 | method | multi-turn search interactions | outcome reward | RL reward | scalar reward | trajectory | reward-only search RL | current policy | adjacent | C2/C3 fail | high |
| Agent-R1 | https://arxiv.org/abs/2511.14460 | README/framework | agent environment interaction | scalar environment reward | reward-only RL | scalar reward | trajectory | agent RL framework | current policy | adjacent | no strict distillation signal found | medium |
| T3-Agent / MM-Traj | https://arxiv.org/abs/2412.15606 | data construction | not found for trained student | GPT-4o mini trajectories | SFT on MM-Traj | offline synthetic teacher | static trajectory | offline imitation | static | not_opd | C1 fails | high |
| Visual Program Distillation | https://arxiv.org/abs/2312.03052 | method | not found for trained student | verified LLM programs | offline distillation | offline program teacher | static program trace | offline distillation | static | not_opd | C1 fails | high |
| SWE-agent | https://arxiv.org/abs/2405.15793 | system paper | not applicable | not applicable | not applicable | none | system/evaluation | not training | not applicable | not_opd | no training objective | high |

## Boundary

Agentic online interaction is not enough. Unit-test reward, web-task success, GUI task completion, search answer correctness, and environment reward are reward-only unless a primary source shows teacher-style supervision on the exact visited state and a distillation objective consuming it.

Privileged context can satisfy C2 only when the privileged branch scores the same student-generated trajectory. If the privileged model generates separate trajectories for SFT, the method is offline imitation or partial at most.

## Systems And Safety Notes

- Strict agentic OPD adds environment/tool execution cost on top of teacher forward passes.
- MAD-OPD/OPAD is especially costly because debate teachers may force-decode per step.
- SOD reduces teacher harm under tool-induced drift by step weighting rather than earliest-error correction.
- GUI-SD reports a grounding setting, not deployed browser control.
- Safety protocols for on-policy exploration are mostly absent from public sources; strict rows usually rely on simulators, static benchmarks, or sandboxed tool tasks.

## Remaining Gaps

| Gap | Current action | Needed next |
|---|---|---|
| New single-agent candidates | Line audit merged. | Carry the conservative labels forward; GLM-5 remains held out, while GUI-SD was resolved by WP7. |
| Long-horizon browser strict OPD | WebRL/Search-R1 are reward-only. | Look for teacher action/logit correction on exact browser trajectories. |
| Earliest-error tool correction | SOD reweights steps but does not provide explicit corrective demonstrations. | Search for DAgger-style tool-call correction with distillation-style objective. |
| VLA safety and systems cost | VLA-OPD is strict, but operational cost/safety details remain sparse. | Audit full paper/code for resets, teacher calls, and safety shields. |
| Human/expert demonstrator OPD | Not resolved beyond VLA-OPD expert policy. | Add only if expert labels supervise states visited by the current student and the objective is distillation-style. |

## Next Router State

WP6 priority and candidate merges are complete. WP7 has since rechecked GUI-SD and LiteGUI because they overlap the GUI/VLM scope.
