# WP4 Audit: Reasoning OPD and OPD+RL Hybrids

Date: 2026-05-13

Scope: math, code, long-CoT, and reasoning-heavy LLM training methods that look like on-policy distillation, RLVR, or KD/RL hybrids.

Source rule: primary sources only. Evidence below uses arXiv papers or official project pages/reports. When a criterion was not found in the primary source, the row is downgraded.

Strict OPD criteria used here:

- C1: the current student/policy generates the training rollout, reasoning trace, response, state, or prefix.
- C2: a teacher-style source supervises that exact student-generated state.
- C3: the objective consumes that signal as distillation-style supervision, not only scalar reward, verifier reward, static DPO, or SFT on teacher traces.

## Concise Memo

The strongest strict WP4 rows are G-OPD, REOPOLD, Fast OPD, SOD, MiMo-V2-Flash MOPD, Nemotron-Cascade2 MOPD, and the Qwen3 on-policy distillation stage. They all put teacher logits, teacher log-probs, teacher/student log-ratios, or dense distillation advantages on student-sampled tokens or prefixes and optimize a KL/RKL/MOPD-style objective. SOD, KDRL, MiMo-V2-Flash, and Nemotron-Cascade2 are the clearest RL-hybrid cases because they combine dense teacher token/log-prob supervision with GRPO or outcome-reward RL.

Reward-only RLVR remains adjacent even when rollouts are on-policy. RLKD is the cleanest false positive in this pass: it uses a generation-structure reward model plus outcome reward inside GRPO, but the teacher-derived signal is consumed as scalar reward rather than token/logit distillation. Pure GRPO/RLVR baselines in KDRL, SOD, and Qwen3 are also adjacent, not strict OPD.

Several methods contain strict subcomponents but should be labeled conservatively at method level. PACED mixes teacher-side forward-KL preparation with a reverse-KL self-distillation arm; only the reverse-KL student-rollout branch is strict. KDRL has a strict KD-RKL arm but the complete recipe is GRPO plus KD, so `borderline_strict` is appropriate. SCOPE has a strict incorrect-trajectory KL branch, but its correct-trajectory branch is weighted MLE without teacher KL, so method-level `borderline_strict` or `partial_opd` is safer than unconditional strict. Lightning OPD preserves dense per-token supervision but precomputes teacher log-probs on fixed SFT/reference rollouts, so it is an offline approximation rather than strict current-student OPD. HDPO adds privileged self-distillation to GRPO, but its teacher-style signal is on privileged rollouts rather than the exact failed student rollout; keep it partial.

Failure-mode handling is a major theme. REOPOLD clips and masks teacher/student log-ratio rewards to avoid heavy-tailed negative rewards and entropy collapse. KDRL reports that pure reward shaping collapses and that KD-only RKL inflates response length; the joint GRPO+KD objective moderates length and stability. SCOPE routes correct and incorrect trajectories differently to avoid over-imitation and to preserve valid unconventional solutions. SOD adds step-wise OPD to GRPO to reduce error propagation and entropy collapse in multi-turn/tool reasoning. PACED uses pass-rate weighting to focus on learnable prompts and reports less catastrophic forgetting. MiMo-V2-Flash and Nemotron-Cascade2 use MOPD to recover or rebalance capabilities after RL stages. Qwen3 reports that OPD expands exploration where direct RL plateaus. Fast OPD and Lightning OPD mainly attack serving and training cost.

Teacher scoring cost is real. Strict OPD generally requires teacher log-probs on student-sampled tokens during training. G-OPD may also require a reference model log-prob. REOPOLD and Nemotron-Cascade2 reduce memory/compute by using sampled-token log-probs instead of full vocabulary distributions. SCOPE reports extra time from teacher queries. KDRL reports GRPO is fastest because it avoids teacher inference; KD-RKL and KDRL are slower. SOD says its step-wise OPD adds little beyond normal OPD but remains slower than GRPO-only. Fast OPD reduces training FLOPs by supervising prefixes. Lightning OPD removes the live teacher server by caching teacher log-probs. Qwen3 reports OPD reaching stronger results with about one tenth of direct RL GPU hours in the audited comparison.

## Evidence Ledger

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G-OPD | https://arxiv.org/abs/2602.12125 | Abstract; Sec. 3.1; Sec. 4.1 | "student-generated trajectories" | "teacher's logit distribution" | "minimize the reverse KL divergence" | teacher logits/log-probs plus optional reference log-probs | dense token logit/log-prob | reverse KL as dense RL / generalized OPD | current student rollouts | strict_opd | none | high |
| REOPOLD | https://arxiv.org/abs/2603.11137 | Abstract; Sec. 3.1-3.3 | "sampled tokens" | "teacher-student log-likelihood ratio" | "minimizing the RKL" | teacher log-probs on sampled tokens | token log-ratio, clipped/masked | relaxed reverse KL / token reward view | on-policy or recent behavior samples | strict_opd | none; quote-level evidence supports sampled-token OPD | high |
| Fast OPD | https://arxiv.org/abs/2602.15260 | Abstract; Sec. 2; Sec. 3; Sec. 6 | "prefixes of student-generated outputs" | "a teacher at token level" | "supervising only short prefixes" | teacher token log-probs on prefixes | dense prefix-token | prefix OPD / reverse KL approximation | current student prefixes with early termination | strict_opd_prefix | strict only over prefixes, not full responses | high |
| PACED | https://arxiv.org/abs/2603.11178 | Abstract; Sec. 3.1-3.3; Algorithm 1 | "sample K rollouts from the student" | "teacher-forced distillation" | "L(theta;x)=w(p)*Ldistill(theta;x)" | frozen teacher for FKL; student/self distribution for RKL | problem-weighted KL, mixed granularity | pass-rate weighted FKL then RKL/self-distillation | mixed: current pass-rate samples plus teacher-side targets | partial_opd | whole recipe mixes teacher-side/off-policy FKL with a strict RKL branch | medium-high |
| KDRL | https://arxiv.org/abs/2506.02208 | Abstract; Sec. 3.1-3.3; Sec. 4.3 | "on-policy rollouts" | "teacher model offers token-level supervision" | "integrates GRPO and KD" | teacher logits/log-probs for KD plus verifier rewards | token KL plus sequence reward | GRPO + reverse KL KD | current student rollouts | borderline_strict | strict KD-RKL arm exists, but method-level objective also contains reward-only GRPO | high |
| RLKD | https://arxiv.org/abs/2505.16142 | Abstract; Sec. 3.2-3.3; Sec. 4 | not found as exact student-rollout statement | "alignment between the reasoning structures of student and teacher" | "as the total reward for GRPO" | learned structure reward model from teacher traces | scalar structure reward plus outcome reward | GRPO / reward modeling | online RL implied, but C1 wording not found | adjacent_not_opd | teacher signal is consumed as scalar reward, not token/logit distillation | high |
| SCOPE | https://arxiv.org/abs/2604.10688 | Abstract; Sec. 3.1-3.3; Sec. 4.3 | "student self-sampled rollouts" | "token-level KL divergence supervision from a teacher" | "minimizes forward KL" | teacher log-probs / perplexity weighting | token KL on incorrect traces; weighted MLE on correct traces | routed KL + MLE policy optimization | current student rollouts | borderline_strict | strict for incorrect-trajectory branch; full method has a non-teacher MLE branch | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | Abstract; Sec. 2.2-3; Appendix | "over SFT rollouts" | "precompute teacher's log-probabilities" | "preserves dense per-token supervision" | cached teacher log-probs; no live teacher | dense token log-prob, offline cached | offline OPD approximation | fixed SFT/reference rollouts | partial_opd | C1 fails for current-student rollouts; teacher scores are cached on fixed rollouts | high |
| HDPO | https://arxiv.org/abs/2603.23871 | Abstract; Sec. 2.2; Sec. 3 | "all rollouts fail" | "distills the teacher token distribution" | "combines LGRPO with a LJSD distillation term" | privileged-context self-teacher with answer access | token distribution over privileged rollouts | GRPO + JSD privileged self-distillation | current failed rollouts plus separate privileged rollouts | partial_opd | teacher supervises privileged generated rollouts, not the exact failed student states | high |
| SOD | https://arxiv.org/abs/2605.07725 | Abstract; Sec. 3.1-3.2; Appendix G.4 | "student-generated trajectory" | "teacher model provides token-level supervision" | "L = L_GRPO + L_step_OPD" | teacher log-probs on student states | dense token or step-level OPD plus sparse reward | GRPO + step-wise OPD | current on-policy tool/reasoning rollouts | strict_opd_hybrid | none | high |
| MiMo-V2-Flash MOPD | https://arxiv.org/abs/2601.02780 | Abstract; Sec. 3.3; Appendix B.2 | "student samples from its own evolving distribution" | "specialized teachers" | "surrogate loss of MOPD" | multiple teacher logits/log-probs plus outcome rewards | dense token-level reward/log-ratio | MOPD + outcome reward models / GRPO | current student samples | strict_opd_stage | strict for MOPD stage; broader pipeline also has SFT and separate RL stages | high |
| Nemotron-Cascade2 MOPD | https://arxiv.org/abs/2603.19220; https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/ | Sec. 3.2; Sec. 4.4; official project page | "For each prompt x, we sample a response" | "select a domain teacher" | "dense token-level distillation advantage" | selected domain teacher checkpoints; sampled-token log-probs | token-level distillation advantage on sampled tokens | MOPD inside Cascade RL | current inference/train policy samples | strict_opd_stage | strict for MOPD stages; full Cascade2 recipe also includes SFT and GRPO stages | high |
| Qwen3 OPD stage | https://arxiv.org/abs/2505.09388 | Sec. 2.3.3; Sec. 4.3; Table 21 | "student model first generates on-policy sequences" | "teacher model Qwen3-32B" | "aligning its logits ... to minimize the KL divergence" | teacher logits from larger Qwen3 models | dense token logits/KL | on-policy distillation stage | current student sequences during stage | strict_opd_stage | strict for OPD stage only; full Qwen3 training is multi-stage | high |

## Candidate Markdown Rows For Source-Supported Table Changes

These are candidate row edits or confirmations, not a request to add duplicate entries where the repository already has matching rows.

| candidate_id | proposed_label | source_url | table_action | source-supported note |
|---|---|---|---|---|
| `g-opd-2026` | strict_opd | https://arxiv.org/abs/2602.12125 | keep/confirm | Student-generated trajectories receive teacher logit distribution supervision through reverse KL. |
| `reopold-2026` | strict_opd | https://arxiv.org/abs/2603.11137 | keep/confirm | Teacher-student log-likelihood ratio is token-level distillation signal on sampled tokens; clipped/masked for stability. |
| `fast-opd-2026` | strict_opd_prefix | https://arxiv.org/abs/2602.15260 | refine if table supports sublabel | Strict only for student-generated prefixes; paper reports 2x-47x FLOP reduction. |
| `paced-2026` | partial_opd | https://arxiv.org/abs/2603.11178 | keep conservative | Reverse-KL branch is strict, but whole method includes teacher-forced/off-policy forward-KL target preparation. |
| `kdrl-2025` | borderline_strict | https://arxiv.org/abs/2506.02208 | keep conservative | Strict KD-RKL arm is combined with GRPO reward optimization. |
| `rlkd-2025` | adjacent_not_opd | https://arxiv.org/abs/2505.16142 | keep outside strict table | Teacher-structure signal is used as scalar reward for GRPO, not distillation supervision. |
| `scope-2026` | borderline_strict or partial_opd | https://arxiv.org/abs/2604.10688 | consider method-level downgrade if table requires whole-method labels | Incorrect-trajectory branch is strict KL on student rollouts; correct branch is weighted MLE. |
| `lightning-opd-2026` | partial_opd | https://arxiv.org/abs/2604.13010 | keep conservative | Cached teacher log-probs preserve dense supervision but C1 current-student rollout freshness is missing. |
| `hdpo-2026` | partial_opd | https://arxiv.org/abs/2603.23871 | keep conservative | Adds JSD privileged self-distillation to GRPO, but teacher supervises privileged rollouts rather than the exact failed rollouts. |
| `sod-2026` | strict_opd_hybrid | https://arxiv.org/abs/2605.07725 | keep/confirm | Explicit combined objective `L_GRPO + L_step_OPD` on student-generated tool trajectories. |
| `mimo-v2-flash-2026` | strict_opd_stage | https://arxiv.org/abs/2601.02780 | keep/confirm | MOPD stage uses teacher dense token rewards/log-ratios on student samples and can combine with outcome rewards including GRPO. |
| `nemotron-cascade2-2026` | strict_opd_stage | https://arxiv.org/abs/2603.19220 | keep/confirm | MOPD uses selected domain teachers and dense token-level distillation advantage on sampled student tokens. |
| `qwen3-opd-2025` | strict_opd_stage | https://arxiv.org/abs/2505.09388 | add compute note | OPD stage aligns student logits to Qwen3-32B/235B teachers; audited comparison reports 1,800 GPU hours for OPD vs 17,920 for direct RL on Qwen3-8B. |

## Adjacent And False-Positive Ledger

| method | source_url | adjacent_type | why_close | why_not_strict | confidence |
|---|---|---|---|---|---|
| RLKD | https://arxiv.org/abs/2505.16142 | reward-only RLVR with teacher-derived reward model | Uses teacher reasoning traces to train a Generation Structure Reward Model and combines that with outcome reward in GRPO. | C3 fails: the teacher-style signal is a scalar reward, not dense token/logit/log-prob distillation on the exact student state. | high |
| KDRL GRPO-only baseline | https://arxiv.org/abs/2506.02208 | reward-only RLVR baseline | Same on-policy rollout setup and rule-based verifier reward family as KDRL. | No teacher KL or token-level KD in the GRPO-only variant. | high |
| SOD GRPO-only baseline | https://arxiv.org/abs/2605.07725 | reward-only RLVR baseline | Uses student on-policy tool/reasoning trajectories and sparse outcome rewards. | C2/C3 fail when the step-wise OPD term is absent. | high |
| Qwen3 direct RL comparison | https://arxiv.org/abs/2505.09388 | reward-only RL comparison | Same off-policy checkpoint is trained with direct RL and compared to OPD. | Direct RL lacks teacher-logit KL supervision; paper reports lower pass@1/pass@64 improvements than OPD. | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | offline OPD approximation | Keeps dense per-token teacher log-prob supervision while avoiding live teacher serving. | C1 fails under strict current-student definition because rollouts/log-probs are precomputed on fixed SFT/reference outputs. | high |
| HDPO | https://arxiv.org/abs/2603.23871 | privileged self-distillation plus RL | Adds teacher-style JSD token distribution distillation to GRPO. | The supervised trajectory is privileged-context generated, not the exact failed current-student rollout. | high |
| DeepSeek-R1 distilled checkpoints | https://arxiv.org/abs/2501.12948 | SFT on teacher traces | Common false positive for reasoning distillation because smaller models learn long CoT traces from a stronger teacher. | Static SFT on teacher-generated traces fails C1 and is not OPD. Needs separate line-level audit if table changes are planned. | medium |
| Static DPO/preference optimization family | primary source required per method | static preference optimization | Often uses model samples and teacher/judge preferences. | Static DPO does not supervise the exact current student rollout with dense teacher distillation unless the paper adds an explicit on-policy KL/logit arm. | medium |

## Gap Map

| gap | status | next audit step |
|---|---|---|
| SCOPE method-level label | Evidence supports a strict incorrect-trajectory KL branch and a non-teacher correct-trajectory MLE branch. | Decide whether the main table can express branch-level `borderline_strict`; otherwise downgrade whole-method label to `partial_opd`. |
| PACED split rows | Evidence supports strict OPD only for the reverse-KL/self-distillation subtrack. | If the table supports subcomponents, split FKL curriculum and RKL OPD branch; otherwise keep method-level `partial_opd`. |
| KDRL objective label | Evidence supports strict KD-RKL plus GRPO. | Keep `borderline_strict`; do not upgrade to unconditional strict unless table semantics allow mixed-objective strict arms. |
| HDPO strictness | Evidence supports privileged self-distillation but not supervision of the exact failed rollout. | Keep partial unless authors/code show JSD is computed on the same student-generated states. |
| Lightning OPD rollout freshness | Evidence supports dense cached teacher supervision but fixed/reference rollouts. | Keep partial/offline; a strict row would require current-student refresh evidence. |
| Nemotron-Cascade2 implementation details | arXiv and official page support MOPD. | Optional follow-up: inspect released code/docs for teacher routing, number of teacher forward passes, and whether `pi_inf` and `pi_train` ever diverge in a way that affects C1 freshness. |
| Qwen3 OPD serving cost | Paper gives OPD vs direct-RL GPU-hour comparison. | Optional follow-up: find exact teacher-serving batch/throughput assumptions, if any are released in technical docs. |
| VOLD / Video-OPD | Outside this WP4 reasoning pass; existing repo has WP3/WP5-style video rows. | Line-audit if the table wants cross-modal OPD+RL hybrids in this WP4 slice. |
| Other alternating KD/RL methods | Names such as ReLIFT/scaffolded GRPO appear adjacent in HDPO context but were not line-audited here. | Audit only if repository scope expands beyond the prioritized list. |
| Static SFT-on-CoT and DPO false positives | Ledger includes general warning, but not every repo row was re-audited in this pass. | For any candidate upgrade, require direct C1/C2/C3 quotes from the method's primary source. |

