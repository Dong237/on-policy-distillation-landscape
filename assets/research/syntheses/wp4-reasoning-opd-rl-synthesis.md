# WP4 Reasoning And OPD+RL Synthesis

Checked: 2026-05-13

This note merges:

- `assets/research/wp4-reasoning-opd-rl-chatgpt.md`
- `assets/research/wp4-reasoning-opd-rl-claude.md`
- `assets/research/wp4-reasoning-opd-rl-gemini.md`

## Core Finding

WP4 confirms the repo's existing reasoning/OPD+RL boundary. RL notation does not downgrade a method by itself: dense teacher logits, log-probs, or teacher/student log-ratios on current student rollouts can satisfy strict OPD even when the update is written as a policy-gradient or GRPO-style surrogate.

The hard boundary is scalarization. If the teacher signal is collapsed into a scalar reward, verbal score, structure reward, verifier reward, or static preference pair before optimization, the method is adjacent or partial rather than strict.

## Merge Decision

| Method | Final repo action | Reason |
|---|---|---|
| G-OPD | Keep `strict_opd`. | Student-generated trajectories receive dense teacher-logit reward extrapolation through a KL-constrained OPD objective. |
| REOPOLD | Keep `strict_opd`. | Teacher/student log-likelihood ratios are consumed per token on sampled student rollouts; clipping and entropy masking are stabilizers, not scalar reward replacement. |
| Fast OPD | Keep `strict_opd` with prefix caveat. | The supervised window is a student-generated reasoning prefix; prefix truncation is a systems optimization, not an off-policy data source. |
| SOD | Keep `strict_opd`. | The full objective preserves a step-wise OPD teacher-logit term alongside GRPO. |
| MiMo-V2-Flash MOPD | Keep stage-scoped `strict_opd`. | The MOPD stage uses domain-teacher token rewards/log-ratios on student samples; the whole training pipeline remains multi-stage. |
| Nemotron-Cascade2 MOPD | Keep stage-scoped `strict_opd`. | Multi-domain OPD uses selected domain teachers and dense token-level distillation advantage inside the cascade. |
| Qwen3 OPD stage | Keep stage-scoped `strict_opd`, but do not label the whole Qwen3 pipeline OPD. | Official evidence supports a distinct on-policy logit-distillation phase; the exact loss and serving mechanics should be rechecked in WP5. |
| KDRL | Keep `borderline_strict`. | KD-RKL on student rollouts is strict-like, but the method-level objective is a GRPO plus KD hybrid with reward-only components. |
| SCOPE | Keep `borderline_strict`. | Incorrect trajectories receive teacher KL; correct trajectories use student-weighted MLE without teacher supervision. |
| PACED | Keep `partial_opd`. | Reverse-KL/self-distillation branch is strict-like, but the method mixes teacher-forced/off-policy forward-KL data. |
| Lightning OPD | Keep `partial_opd`. | Dense teacher log-probs are cached on fixed SFT/reference rollouts, so strict current-policy freshness fails. |
| HDPO | Keep `partial_opd`. | GRPO branch is current-policy but reward-only; distillation branch uses privileged teacher rollouts rather than exact failed student states. |
| RLKD | Keep adjacent/not OPD. | Teacher reasoning structure becomes scalar GSRM reward for GRPO; no dense same-rollout distillation signal is consumed. |

No main-table label changes are required from this WP4 merge. The table already encodes the conservative method-level labels for the priority queue. Future edits should be narrow confidence or compute-note updates, not broad promotion.

## Evidence Ledger

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G-OPD | https://arxiv.org/abs/2602.12125 | Sec. 3.1; Sec. 4.1 | "student-generated trajectories" | "teacher's logit distribution" | "minimize the reverse KL divergence" | white-box teacher logits plus reference log-probs | token | KL-constrained reward extrapolation | current student rollouts | strict_opd | none | high |
| REOPOLD | https://arxiv.org/abs/2603.11137 | Sec. 3.1-3.3 | "sampled tokens" | "teacher-student log-likelihood ratio" | "minimizing the RKL" | white-box teacher log-probs | token | relaxed/clipped reverse KL | near-current refreshed rollouts | strict_opd | none; clipping is a stabilizer | high |
| Fast OPD | https://arxiv.org/abs/2602.15260 | Sec. 3 | "prefixes of student-generated outputs" | "teacher gives token-level supervision on those prefixes" | "prefix-limited reverse KL" | white-box teacher logits | prefix-token | prefix reverse KL | current student prefixes | strict_opd | strict only over scheduled prefixes | medium-high |
| PACED | https://arxiv.org/abs/2603.11178 | Sec. 3.1-3.3; Algorithm 1 | "sample K rollouts from the student" | "teacher-forced distillation" | "L(theta;x)=w(p)*Ldistill(theta;x)" | white-box teacher or self-teacher | token | pass-rate weighted FKL/RKL | mixed current and teacher-forced data | partial_opd | whole method mixes strict-like RKL with off-policy/teacher-forced FKL | high |
| KDRL | https://arxiv.org/abs/2506.02208 | Sec. 3.1-3.4 | "on-policy rollouts" | "teacher model offers token-level supervision" | "integrates GRPO and KD" | white-box frozen teacher | token KL plus scalar reward | reverse-KL KD plus GRPO | current student rollouts | borderline_strict | strict KD-RKL arm exists, but full method also includes reward-only GRPO | high |
| RLKD | https://arxiv.org/abs/2505.16142 | Sec. 3.2-3.3 | "Student rollouts via GRPO" | "GSRM scores structural alignment" | "total reward for GRPO" | teacher-derived scalar reward model | scalar sequence reward | GRPO with structure reward | current-policy RL implied | adjacent | C3 fails: teacher signal is scalar reward, not distillation | high |
| SCOPE | https://arxiv.org/abs/2604.10688 | Sec. 3.1-3.3 | "student model generates a group of N responses" | "teacher policy ... external guidance" | "L_OPD" on incorrect branch | white-box teacher logits on incorrect branch | token KL plus MLE | correctness-routed KL/MLE | current student rollouts | borderline_strict | only the incorrect branch has teacher KL; correct branch lacks teacher supervision | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | Sec. 3.1-3.2 | "rollouts are sampled from pi_ref" | "precompute teacher's log-probabilities" | "A_t = log pi_T - log pi_theta" | cached white-box teacher log-probs | token | offline OPD approximation | precomputed SFT/reference rollouts | partial_opd | C1 current-policy freshness fails by design | high |
| HDPO | https://arxiv.org/abs/2603.23871 | Sec. 3.2 | "all rollouts fail" | "distills the teacher token distribution" | "L_HDPO = L_GRPO + lambda * L_JSD" | privileged-context self-teacher | token JSD plus scalar reward | JSD plus GRPO | mixed failed rollouts and privileged rollouts | partial_opd | distillation supervises privileged rollouts, not exact failed student states | high |
| SOD | https://arxiv.org/abs/2605.07725 | Sec. 3.3; Algorithm 1 | "sample trajectory" | "teacher supervision" | "L = L_GRPO + L_OPD_step" | white-box teacher logits | token, step-reweighted | step-wise OPD plus GRPO | current student trajectories | strict_opd | none | high |
| MiMo-V2-Flash MOPD | https://arxiv.org/abs/2601.02780 | Sec. 4.1; Sec. 4.4 | "samples from its own evolving distribution" | "specialized teachers" | "L_MOPD" with teacher/student log-ratio advantage | multi-teacher log-probs | token plus outcome reward | MOPD plus reward optimization | current student samples | strict_opd_stage | strict only for MOPD stage | high |
| Nemotron-Cascade2 MOPD | https://arxiv.org/abs/2603.19220 | MOPD section | "sample a response" | "domain teachers" | "dense token-level distillation advantage" | multi-teacher domain checkpoints | token | dense distillation advantage | current student samples | strict_opd_stage | strict only for MOPD stage | high |
| Qwen3 OPD stage | https://arxiv.org/abs/2505.09388 | Post-training report | "on-policy sequences" | "teacher model Qwen3-32B" | "aligning its logits ... KL divergence" | larger Qwen teacher logits | token | teacher-logit KL | documented OPD stage | strict_opd_stage | public report is stage-level; exact full-pipeline details remain out of scope | medium-high |

## RL Boundary

Dense teacher log-probability in the loss is the key. G-OPD, REOPOLD, SOD, MiMo-V2-Flash MOPD, Nemotron-Cascade2, and the strict arm of KDRL remain OPD because teacher probabilities or teacher/student log-ratios are consumed per token on student rollouts.

Reward-only RLVR remains adjacent even when the rollouts are fresh. RLKD is the cleanest WP4 false positive because the teacher-derived structure signal is consumed as scalar reward by GRPO.

## Systems Notes

- Live strict OPD usually requires teacher scoring on student-sampled tokens.
- Fast OPD reduces cost by supervising only informative reasoning prefixes.
- Lightning OPD removes live teacher serving by precomputing teacher log-probs, but that is exactly why it stays partial under the strict current-policy definition.
- REOPOLD and Nemotron-Cascade2 use sampled-token or dense-advantage formulations that avoid requiring full-vocabulary teacher distributions at every step.
- Multi-teacher industrial OPD creates routing and serving questions that should move to WP5.

## Remaining Gaps

| Gap | Current action | Needed next |
|---|---|---|
| Industrial stage mechanics | Keep Qwen3, MiMo, and Nemotron labels stage-scoped. | WP5 should verify official training-system details, teacher routing, and serving-cost evidence. |
| Fast OPD compute claims | Keep strict prefix OPD but avoid overclaiming speedups. | Optional line audit for exact FLOP setup and prefix schedule. |
| KDRL/SCOPE subcomponent labels | Keep method-level `borderline_strict`. | Split substage rows only if the repo adopts a general substage policy. |
| Lightning OPD equivalence claims | Keep partial despite dense cached signal. | Use as systems-adjacent evidence in WP5, not as strict current-policy OPD. |
| Reward-only reasoning methods | Keep in adjacent ledger. | Require direct C1/C2/C3 quotes before promoting any GRPO/RLVR method. |

## Next Router State

WP4 is complete for the current pass. Later WP5-WP7 passes have been merged separately; current routing state is tracked in [deep-research-plan.md](deep-research-plan.md).
