# WP6 Audit: Agentic and Interactive OPD

Date: 2026-05-13  
Agent: ChatGPT  
Scope: multi-turn, tool-using, browser/GUI, code, robotics/environment, and trajectory-level training where a student visits states and another signal supervises those exact states.

## Concise Memo

Strict WP6 OPD is real but still narrow. The strongest rows are MAD-OPD/OPAD, SOD, OPCD, VLA-OPD, and GUI-SD. OEL is method-level partial: its consolidation stage is strict OPD, but the full loop also contains deployment trajectory collection and experience extraction before the on-policy context-distillation update. These methods satisfy the important WP6 distinction: the student creates the state or action being trained on, and the teacher, expert, self-teacher, or privileged-context branch supervises that same visited state with a token/action-level distillation loss.

The biggest false-positive cluster is agentic RL. SCoRe, WebRL, ReTool, VTool-R1, Search-R1, and Agent-R1 all have on-policy or online environment interaction, but the consumed learning signal is reward, outcome-supervised reward model output, or RL advantage. They do not expose teacher logits, token/action corrections, KL targets, trajectory corrections, or a distillation-style signal on the current visited state. Reflexion is even farther away: it operates through verbal memory and explicitly avoids weight updates.

Offline tool-agent tuning is also not OPD. T3-Agent and Visual Program Distillation use generated or verified tool trajectories as static training data. They are useful adjacent evidence for tool-use imitation, but C1 fails because the trained student is not generating the supervised trajectory. SWE-agent is an important code-agent system, not a training/distillation method in the paper audited here.

Safety handling is mostly implicit. Strict rows tend to avoid live high-risk execution by using simulators, static GUI grounding datasets, tool-integrated math/code environments, or deployment logs with server-side training. Explicit exploration safety constraints are rarely specified. The systems cost pattern is clear: strict OPD replaces sparse delayed rewards with dense teacher calls, but pays for student rollouts plus teacher forward passes, privileged-context branches, debate ensembles, tool/environment execution, and sometimes per-step force decoding.

## Evidence Ledger

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | state_or_action_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAD-OPD / OPAD | https://arxiv.org/abs/2605.01347 | Abstract; Sec. 4.1-4.4; App. C | "samples an on-policy action" | "Teachers then force-decode" | "confidence-weighted divergence" | Multi-teacher debate ensemble over the student's on-policy state; OPAD adds per-step environment observations. | Token/action tokens at each trajectory step. | JSD for agents or reverse KL for code, weighted by debate confidence. | Current on-policy; per-step debate on actual trajectory observations. | strict_opd | None. Cost is high because each step can require rollout, environment transition, multiple teacher debate calls, and force decoding. | high |
| SOD | https://arxiv.org/abs/2605.07725 | Abstract; Sec. 3.3; Sec. 4.2-4.3; App. C.1 | "student-generated trajectories" | "token-level supervision from a teacher" | "step-wise OPD objective" | Qwen3 teacher model; reliability weighted by student-teacher divergence under tool-induced state drift. | Model-generated tokens grouped by tool/reasoning steps; tool observations excluded. | Step-weighted reverse-KL OPD plus GRPO outcome reward. | Current sampled trajectory; step weights computed on visited states. | strict_opd_hybrid | Hybrid objective includes GRPO reward, but the OPD component is strict. | high |
| OPCD | https://arxiv.org/abs/2602.12275 | Abstract; Sec. 3; Algorithm 1 | "y is sampled from the student model" | "teacher model that conditions on the context" | "reverse KL divergence" | Privileged context-conditioned teacher; student lacks context. | Token-level distribution along the student's sampled response. | Reverse KL to context-conditioned teacher. | Current student rollout for each update. | strict_opd | Not environment-interactive by itself, but it is a core strict OPD primitive used in OEL. | high |
| OEL | https://arxiv.org/abs/2603.16856 | Abstract; Sec. 3.1-3.3; Fig. 3; App. B.4 | "generates a response y" | "knowledge-conditioned output of a teacher" | "token-level reverse KL divergence" | Teacher is conditioned on extracted experiential knowledge from user-side trajectories; often a frozen prior checkpoint. | Token-level response from partial rollout prefixes. | OPCD-style reverse KL; reward-free. | Partial: deployment trajectories are collected per OEL round, but training rollouts are server-side single-turn responses from prefixes. | partial_opd | Strict only in the OPCD consolidation stage; full method also uses extracted knowledge from prior deployment trajectories. | high |
| VLA-OPD | https://arxiv.org/abs/2603.26666 | Abstract | "student's self-generated trajectories" | "expert teacher" | "Reverse-KL objective" | Expert VLA teacher in robotic manipulation benchmarks. | Dense token-level supervision over action trajectories. | Reverse KL action-token distillation. | On-policy student trajectories. | strict_opd | Full paper details were not pulled beyond the arXiv abstract in this audit; classification is supported by the abstract but systems details remain sparse. | medium-high |
| GUI-SD | https://arxiv.org/abs/2605.00642 and https://arxiv.org/html/2605.00642 | Abstract v3; HTML v2 Sec. 2; Sec. 4.1-4.2; App. B | "student generates an on-policy trajectory" | "privileged information" | "weighted reverse-KL objective" | Same model as self-teacher under visual privileged context: target bounding box, soft mask, and hint. | Coordinate output tokens, especially higher-order digits. | Entropy-guided weighted reverse KL. | Single rollout from current policy; no live GUI action sequence. | strict_opd_limited | Strict for GUI grounding, but not long-horizon browser/GUI control. | high |
| SCoRe | https://arxiv.org/abs/2409.12917 | Abstract | "self-generated correction traces" | not found | "multi-turn online reinforcement learning" | No teacher or privileged distillation signal found; uses self-generated data and reward shaping. | Whole correction attempts. | Online RL with reward bonus. | Current model distribution. | adjacent_reward_rl | C1 holds, but C2 and C3 fail: reward-only RL, not teacher distillation. | high |
| Reflexion | https://arxiv.org/abs/2303.11366 | Abstract; Sec. 1; Algorithm 1 | "Generate initial trajectory" | "linguistic feedback" | "not by updating weights" | Environment feedback, heuristics, or self-evaluation converted to memory. | Episode/trial level memory hints. | Verbal reinforcement in context, no gradient update. | Current and past trials in episodic memory. | adjacent_no_weight_update | No distillation objective and no model training update. | high |

## Candidate Table Rows / Updates

| id | title | year | area | candidate_status | recommended_label | evidence_summary | notes |
|---|---|---:|---|---|---|---|---|
| mad-opd-2026 | MAD-OPD / OPAD | 2026 | agentic/code OPD | keep | strict_opd | Student samples action; teachers force-decode same action; divergence loss trains student. | OPAD is the most WP6-relevant substage because it includes per-step environment observations. |
| sod-2026 | SOD: Step-wise On-policy Distillation | 2026 | tool-integrated reasoning | keep | strict_opd_hybrid | Student-generated tool trajectories receive teacher token supervision with step-wise OPD loss. | Hybrid with GRPO; retain strict label because the dense OPD component is real. |
| opcd-2026 | On-Policy Context Distillation | 2026 | context/privileged teacher | keep | strict_opd | Student rollout is matched to context-conditioned teacher via reverse KL. | Add cross-reference as OEL's strict consolidation primitive. |
| oel-2026 | Online Experience Learning | 2026 | environment feedback / deployment learning | keep | partial_opd | User-side experience is extracted, then OPCD trains current model responses against knowledge-conditioned teacher. | Method-level label should remain partial, not fully strict. |
| vla-opd-2026 | VLA-OPD | 2026 | robotics / VLA | keep | strict_opd | Expert teacher provides dense token supervision on student self-generated robotic trajectories via RKL. | Needs full-paper cost and safety notes if this row is expanded later. |
| gui-sd-2026 | GUI-SD | 2026 | GUI grounding | add | strict_opd_limited | Student on-policy coordinate trajectory is supervised by privileged visual self-teacher with weighted RKL. | Limit label to grounding; not multi-step GUI automation. |
| score-2024 | SCoRe | 2024 | self-correction RL | add to adjacent | adjacent_reward_rl | Online self-generated correction traces are optimized by RL rewards, not teacher distillation. | Useful false-positive anchor for "multi-turn online" claims. |
| reflexion-2023 | Reflexion | 2023 | verbal self-correction | add to adjacent | adjacent_no_weight_update | Agent trajectories are reflected into memory, explicitly without weight updates. | Do not list as OPD. |
| webrl-2024 | WebRL | 2024 | browser/web agents | add to adjacent | adjacent_reward_rl | Web curriculum RL uses unsuccessful attempts and outcome-supervised reward model. | Important browser-agent false positive. |
| retool-2025 | ReTool | 2025 | code/tool RL | add to adjacent | adjacent_reward_rl | Multi-turn code execution rollouts learn from outcome feedback/task rewards. | Tool interaction is real, but C2/C3 fail. |
| vtool-r1-2025 | VTool-R1 | 2025 | multimodal tool use | add to adjacent | adjacent_reward_rl | Python visual tools are in the rollout, but training uses outcome rewards and no process supervision. | Explicitly states no process-based supervision. |
| search-r1-2025 | Search-R1 | 2025 | search agents | add to adjacent | adjacent_reward_rl | Multi-turn search interactions are optimized with an outcome reward. | No teacher correction of exact query/action found. |
| t3-agent-2024 | T3-Agent / MM-Traj | 2024 | multimodal tool imitation | add to adjacent | offline_imitation | GPT-4o mini generates trajectories; VLM is tuned on MM-Traj. | Static synthetic tool traces, not current-policy trajectories. |
| vpd-2023 | Visual Program Distillation | 2023 | tool/program distillation | add to adjacent | offline_distillation | LLM samples programs; correct programs are translated and distilled into a VLM. | Distillation yes, but off-policy/static, no student-visited states. |
| swe-agent-2024 | SWE-agent | 2024 | code agents | add to adjacent | agent_system_not_training | Paper studies agent-computer interface, not post-training. | Useful systems reference, not OPD. |
| agent-r1-2025 | Agent-R1 | 2025 | generic agent RL framework | add to adjacent | adjacent_reward_rl_framework | Defines RL-based LLM agents for interactive environments. | No teacher-distillation signal found in abstract or official README snippets. |

## Adjacent / False-Positive Ledger

| method | source_url | primary-source signal | why not strict OPD | final_label | confidence |
|---|---|---|---|---|---|
| SCoRe | https://arxiv.org/abs/2409.12917 | Multi-turn online RL under model distribution with reward bonus. | No teacher/expert logits or correction signal on exact visited states; C2 and C3 fail. | adjacent_reward_rl | high |
| Reflexion | https://arxiv.org/abs/2303.11366 | Verbal feedback stored in episodic memory; no weight update. | No training objective consuming distillation signal. | adjacent_no_weight_update | high |
| WebRL | https://arxiv.org/abs/2411.02337 | Self-evolving curriculum plus outcome-supervised reward model. | Online browser tasks, but supervision is reward/ORM, not token/action distillation. | adjacent_reward_rl | high |
| ReTool | https://arxiv.org/abs/2504.11536 | Multi-turn real-time code execution with outcome feedback and task rewards. | Tool calls are student-generated, but learning is reward-only. | adjacent_reward_rl | high |
| VTool-R1 | https://arxiv.org/abs/2505.19255 | Visual editing tools integrated into RFT; outcome-based task rewards. | Explicitly no process-based supervision, so no teacher correction/distillation. | adjacent_reward_rl | high |
| Search-R1 | https://arxiv.org/abs/2503.09516 | Multi-turn search interactions and outcome-based reward. | No teacher scoring or correcting exact search actions beyond scalar outcome. | adjacent_reward_rl | high |
| Agent-R1 | https://arxiv.org/abs/2511.14460 and https://github.com/AgentR1/Agent-R1 | RL framework for agents with active environment interaction. | Framework is RL-based; no strict distillation signal found. | adjacent_reward_rl_framework | medium-high |
| T3-Agent / MM-Traj | https://arxiv.org/abs/2412.15606 | GPT-4o mini generates queries, files, and trajectories; VLM is trajectory-tuned. | Supervised data is generated before student training; C1 fails. | offline_imitation | high |
| Visual Program Distillation | https://arxiv.org/abs/2312.03052 | LLM programs are sampled, executed, verified, translated, and distilled into a VLM. | Distillation is from static verified programs, not student-visited states. | offline_distillation | high |
| SWE-agent | https://arxiv.org/abs/2405.15793 | Agent-computer interface for autonomous code editing and tests. | System/evaluation paper; no training objective found. | agent_system_not_training | high |

## Focus Question Answers

| question | audit answer |
|---|---|
| Does the student agent generate the trajectory being supervised? | Yes for MAD-OPD/OPAD, SOD, OPCD, OEL's consolidation rollout, VLA-OPD, and GUI-SD. No for T3-Agent and VPD. |
| Does a teacher/expert correct or score the exact visited state/action? | Yes for strict rows: debate teachers, teacher models, expert policies, or privileged-context self-teachers evaluate the same sampled tokens/actions. For reward-RL rows, only success/reward signals were found. |
| Is the consumed signal distillation-style or scalar reward? | Strict rows consume KL/JSD/token-level distillation. SCoRe, WebRL, ReTool, VTool-R1, Search-R1, and Agent-R1 consume RL rewards or reward model signals. |
| Are corrections online/current-policy, replay-buffer stale, or static? | MAD-OPD/OPAD, SOD, OPCD, VLA-OPD, and GUI-SD are current-policy. OEL is iterative: deployment trajectories become per-round experience, while the OPCD training rollout is current. WebRL uses online curriculum/reuse but reward-only. Static rows include T3-Agent and VPD. |
| How are safety constraints handled during exploration? | Mostly not explicit. Robotics and GUI rows rely on simulators/static benchmarks; SOD uses TIR/code/math settings; MAD-OPD discusses inherited safety/refusal behavior but still does not provide a robust exploration-safety protocol. |
| What is the systems cost? | Strict OPD shifts cost from reward-only sparse learning to dense teacher calls. Costs include environment resets/steps, tool execution, privileged branch forward passes, teacher/debate calls, force decoding, and top-k KL approximations. GUI-SD reports about 4.2 hours per epoch versus roughly 16.7-16.9 for GRPO baselines; OEL reports 20 or 100 OPCD steps with 64 samples per step. |

## Gap Map

| gap | current evidence | consequence for landscape |
|---|---|---|
| Long-horizon browser strict OPD | WebRL/Search-R1 style rows are reward-based; no strict browser OPD found in this scan. | Keep browser-agent training mostly adjacent unless a source shows teacher correction/logits on the same live web trajectory. |
| Code-agent strict OPD | MAD-OPD covers code generation and OPAD-style agentic settings; ReTool and SWE-agent do not satisfy strict OPD. | Separate "code agent systems/RL" from strict code-token OPD. |
| Tool-call correction at earliest-error state | SOD handles unreliable tool-induced drift by reweighting, not by explicit corrective demonstrations at earliest error. | Add a future row only if a source proves online tool-call correction or action-level teacher labels. |
| Privileged-context self-teachers | OPCD, OEL, and GUI-SD are strong evidence that asymmetric context can satisfy strict OPD. | Mark as strict only when the same student-sampled trajectory is scored under the privileged context. |
| Robotics/VLA systems cost and safety | VLA-OPD abstract proves strict criteria but not full operational details. | Needs a full-paper follow-up for reset counts, rollout budget, teacher-call budget, and safety constraints. |
| Replay-buffer freshness | OEL and WebRL mix fresh interaction with per-round or replayed experience. | Landscape rows should explicitly track "current rollout", "per-round stale", and "static". |
| Human/expert demonstrator OPD | VLA-OPD uses an expert teacher; no DAgger-like human-in-the-loop correction row was verified here. | Add if source shows expert supervision on states visited by the current student and a distillation objective. |
| Safety-constrained online exploration | Primary sources rarely specify more than benchmark/simulator scope. | Add a safety column for WP6 rows; downgrade deployment claims when exploration safeguards are not found. |
