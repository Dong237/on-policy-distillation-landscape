# Reasoning and OPD + RL Hybrids

This page tracks OPD for math, code, long-CoT, and reasoning-heavy tasks, especially where dense teacher supervision is combined with sparse rewards.

## Current Status

The current labels are conservative. The deciding question is whether teacher probabilities, logits, or log-ratios remain dense loss terms on current student rollouts. RL notation is acceptable; scalar reward compression is not.

## Strict Or Stage-Strict Rows

- G-OPD and REOPOLD: teacher log-ratios are used as dense token-level OPD supervision in an RL-style objective.
- Fast OPD: strict over student-generated reasoning prefixes; prefix truncation is a cost optimization.
- SOD: step-wise OPD teacher supervision is combined with GRPO.
- MiMo-V2-Flash MOPD and Nemotron-Cascade2 MOPD: strict for the documented MOPD stages, not for the entire industrial pipelines.
- Qwen3 OPD stage: strict only for the documented on-policy logit-distillation phase; keep medium confidence/source-weak because the public report is terse on mechanics.
- SDPO and OPCD remain strict teacher-free/privileged-context reasoning rows, covered primarily under the self-play page.

## Borderline And Partial Rows

- KDRL: KD-RKL subcomponent is strict-like, but the method-level recipe mixes dense KD with reward-only GRPO.
- SCOPE: incorrect trajectories receive teacher KL; correct trajectories use student-weighted MLE without teacher supervision.
- PACED: reverse-KL/self-distillation branch is strict-like, but the full recipe mixes teacher-forced/off-policy forward-KL data.
- HDPO: privileged self-distillation plus GRPO, but the distillation arm uses privileged teacher rollouts rather than exact failed student states.
- Lightning OPD: dense cached teacher log-probs on SFT/reference rollouts; partial because current-policy freshness fails.

## Boundary

Pure RLVR is adjacent. A hybrid method must show teacher-style supervision on student trajectories in addition to outcome reward.

RLKD is the main false positive in this bucket: it uses teacher-derived reasoning structure as scalar reward inside GRPO, not as dense same-rollout distillation.

The archived evidence ledger is under [assets/research](../assets/research/).
