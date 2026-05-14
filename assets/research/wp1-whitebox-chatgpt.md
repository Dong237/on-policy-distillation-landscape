# WP1: White-Box OPD and Divergence/Objectives Audit

Checked: 2026-05-12

Scope: LLM white-box on-policy distillation (OPD) and objective/divergence methods. This audit uses primary papers and source TeX where available. VLM, RLVR-only, offline KD, DPO, and industrial reports are excluded except where a method is a useful downgrade or contrast case.

Classification follows `taxonomies/strict-opd-definition.md`:

- C1: current student/policy rollout generates the supervised state.
- C2: teacher-style supervision is evaluated on that exact state.
- C3: the objective directly consumes the teacher-style signal.

## Memo: Families and Objective Taxonomy

White-box OPD in this slice is mostly distribution matching on student-generated text states. The key divide is not "KL vs reward wording"; it is whether the supervised prefix is fresh student policy data and whether the teacher distribution/log-prob is consumed by the update.

| Family | Methods | Objective pattern | WP1 classification tendency |
|---|---|---|---|
| Canonical on-policy distribution matching | GKD; MiniLLM | KL-family divergence on student-generated completions | Strict when pure student rollout is used |
| Stabilized/adaptive divergences | DistiLLM; DistiLLM-2; Entropy-Aware OPD; Veto | Skew KL, reverse KL, entropy-gated FKL/RKL, adaptive bridge targets | Strict only if rollout freshness holds; otherwise partial |
| RL-equivalent dense log-ratio OPD | G-OPD; REOPOLD; Lightning OPD | Dense teacher log-ratio advantage, often equivalent to sequence reverse KL | Strict for fresh student rollouts; Lightning is partial/offline |
| Systems/efficiency OPD | DistillSpec; Fast OPD; Lightning OPD | Draft-model or prefix-limited teacher scoring, cached/offline approximations | Split by rollout source; systems goal alone does not change C1/C2/C3 |
| Hybrid/interleaved rollout methods | Speculative KD; PACED; DistiLLM variants | Teacher-student mixed traces, replay buffers, pass-rate curricula | Whole-method label is usually partial; subcomponents may be strict |
| Cross-tokenizer white-box KD | ULD, ALM, BLD candidates | Tokenizer-aligned teacher distribution matching | No clearly strict OPD candidate found in this pass; treat as CTD/offline KD until C1 is explicit |

Objective taxonomy:

- Forward KL: `KL(pi_T(.|s_t) || pi_theta(.|s_t))`; mode-covering, used in GKD variants, DistillSpec variants, and PACED forward track.
- Reverse KL / sequence RKL: `KL(pi_theta(.|q) || pi_T(.|q))`, or sampled-token policy-gradient surrogate with advantage `log pi_T(a_t|s_t) - log pi_theta(a_t|s_t)`; core for MiniLLM, Fast OPD, REOPOLD, Lightning baseline, and several OPD reasoning papers.
- Generalized JSD: GKD uses beta-JSD as a tunable alternative to KL.
- Skew KL / skew reverse KL: DistiLLM and DistiLLM-2 smooth the teacher/student target distribution with a mixture.
- Adaptive target KL: Veto defines a target `Q(.|s_t) proportional to pi_T(.|s_t) * pi_theta(.|s_t)^beta`, then minimizes FKL or RKL to `Q`.
- Entropy-aware hybrid KL: Entropy-Aware OPD adds teacher-top-k forward KL only for high-teacher-entropy tokens on top of an OPD/RKL surrogate.
- Dense log-ratio reward: G-OPD and REOPOLD express OPD as policy optimization with token reward/advantage derived from teacher/student/reference log-probs. This remains OPD when the signal is teacher distributional supervision, not reward-only RLVR.
- Cached/offline OPD approximation: Lightning OPD uses the same dense advantage as OPD but fixes the rollout distribution to an SFT/reference model and caches teacher log-probs, so C1 freshness fails.

## Method Classification Matrix

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Rollout freshness | Teacher access and kind |
|---|---|---|---|---|---|---|---|
| GKD | Yes for lambda=1; mixed for lambda<1 | Yes | Yes | `strict_opd` for pure on-policy; mixed variants `partial_opd` | `L_GKD=(1-lambda) E_data[D(T||S)] + lambda E_{y~S}[D(T||S)]`; `D` can FKL/RKL/JSD | Current student samples for on-policy term | White-box teacher probabilities/log-probs; larger LM |
| MiniLLM | Yes, with teacher-mixed sampling caveat | Yes | Yes | `strict_opd` | Sequence reverse KL `KL(q_theta(.|x) || p_T(.|x))`; policy-gradient surrogate with dense log-ratio returns | Fresh student or teacher-mixed behavior samples | White-box teacher logits/probs; larger LM |
| DistiLLM | Weak/mixed | Yes | Yes | `partial_opd` | Skew KL `KL(p_T || alpha p_T + (1-alpha) q_theta)` and skew RKL `KL(q_theta || (1-alpha)p_T + alpha q_theta)` | Replay-buffer stale and fixed-data mixture | White-box teacher logits; larger LM |
| DistiLLM-2 | Weak/mixed | Yes | Yes | `partial_opd` | CALD contrastive loss mixing SKL on teacher responses and SRKL on student responses | Per-epoch batched SGO plus teacher-generated responses | White-box teacher logits; larger LM |
| Entropy-Aware OPD | Yes | Yes | Yes | `strict_opd` | OPD clipped/RKL surrogate plus high-entropy teacher-top-k FKL: `L_EOPD = L_OPD + I[H_T>tau] L_FKL` | Current/previous-policy PPO-style behavior rollout per iteration | White-box teacher log-probs, entropy, top-k logits; larger reasoning LM |
| G-OPD | Yes | Yes | Yes | `strict_opd` | `max E_{y~pi_theta}[lambda log(pi_T/pi_ref) - KL(pi_theta||pi_ref)]`; equivalent dense teacher/reference log-ratio objective | Current student rollout | White-box teacher and reference log-probs; larger teacher/reference LM |
| REOPOLD | Yes | Yes | Yes | `strict_opd` | Relaxed/clipped reverse-KL policy objective with `R_t=log pi_T(a_t|s_t)-log pi_theta(a_t|s_t)`, token masks, and clipped advantages | Current/near-current behavior rollout | White-box teacher logits/log-probs; larger reasoning LM |
| Veto | Yes | Yes | Yes | `strict_opd` | Adaptive target `Q(.|s_t) proportional to exp(z_T + beta z_S)`; optimize `KL(Q||pi_theta)` or `KL(pi_theta||Q)` | Current student trajectories | White-box teacher logits; larger LM |
| Fast OPD | Yes for retained prefixes | Yes | Yes | `strict_opd` | Prefix-limited reverse KL estimate `E_{x~pi_s} sum_t log pi_s(a_t|s_t)/pi_T(a_t|s_t)` | Current student prefix rollouts, truncated/scheduled length | White-box teacher log-probs; larger reasoning LM |
| DistillSpec | Yes for draft-generated-data variant | Yes | Yes | `strict_opd` for draft/on-policy variant; otherwise split | Token divergence between target and draft on sampled draft states; paper studies FKL, RKL, JSD, TVD | Draft-current for strict variant; mixed/offline in other variants | White-box target/teacher LM distribution; target model for speculative decoding |
| Speculative KD | Mixed/interleaved | Yes | Yes | `partial_opd` | Token-level distillation on sequences generated by student proposals with teacher top-k accept/reject/replacement | Fresh but hybrid student/teacher token trace | White-box teacher logits/top-k; target/reference LM |
| PACED | Mixed | Yes | Yes | `partial_opd` whole method; strict only for reverse-KL self-distill subtrack | Weight `w(p)=p^alpha(1-p)^beta`; forward KL on teacher sequence in one track; reverse KL on student sequence in self-distill track | Pass-rate estimates may be fixed/staged; forward track teacher-forced, reverse track student rollout | White-box teacher logits for KL; expert/teacher-generated solutions in some tracks |
| Lightning OPD | No fresh current rollout | Yes | Yes | `partial_opd` / offline approximation | `J_on=E_{x~pi_theta} sum_t A_t`; Lightning uses `J_off=E_{x~pi_ref} sum_t A_t`, `A_t=log pi_T - log pi_theta` | Precomputed SFT/reference rollouts, cached teacher log-probs | White-box teacher log-probs cached once; larger reasoning LM |
| Cross-tokenizer OPD candidates | Not established | White-box KD yes | KD objective yes | `not_opd` or `adjacent` until C1 is proven | ULD/ALM/BLD align distributions across tokenizer spaces | Public evidence found here is offline/static or unspecified, not current student rollout | White-box teacher distribution transformed/aligned across tokenizers |

## Evidence Ledger

| Method | Primary source | Evidence used | Conservative note |
|---|---|---|---|
| GKD | [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) | Source defines discrepancy `D(T||S)(y|x)` as token average and `L_OD=E_x E_{y~S(.|x)} KL(T||S)(y|x)`. General GKD mixes fixed data and student samples by lambda and supports FKL/RKL/JSD. | Mark strict only for the pure on-policy term/setting. |
| MiniLLM | [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) | Source minimizes sequence reverse KL from student distribution to teacher distribution and derives an on-policy policy-gradient estimator with teacher log-prob rewards. | Teacher-mixed sampling is a stabilizer; do not overstate as purely current-student at every token. |
| DistiLLM | [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) | Source defines skew KL and skew reverse KL and uses an adaptive off-policy/replay-buffer loop with student-generated outputs sometimes reused. | Downgrade to partial because current-policy freshness is weakened by replay/stale and fixed-data mixture. |
| DistiLLM-2 | [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://arxiv.org/abs/2503.07067) | Source uses batched SGO collected ahead of epochs and a contrastive loss over teacher-generated and student-generated responses with SKL/SRKL terms. | Whole method is partial; a strict subrow would need a fresh-SGO-only variant. |
| DistillSpec | [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) | Source defines target-vs-draft token divergence over generated sequences and analyzes draft-generated data with divergences including FKL, RKL, JSD, TVD. | Strict only for draft-generated/on-policy data; it is a drafter-training systems method, not generic post-training. |
| Entropy-Aware OPD | [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079) | Source starts from on-policy reverse-KL/OPD surrogate and adds teacher-entropy-gated top-k forward KL when teacher uncertainty is high. | Top-k FKL is approximate support matching; keep exact top-k implementation in notes. |
| G-OPD | [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) | Source formulates OPD as KL-constrained RL with dense `log(pi_T/pi_ref)` teacher/reference signal and extends it with reward extrapolation lambda. | Strict because teacher logit/log-prob supervision is dense on student trajectories; not RLVR-only. |
| REOPOLD | [Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) | Source samples outputs from the current/old student, computes token rewards `log pi_T - log pi_theta`, clips/masks them, and updates with an RKL-style policy surrogate. | Includes broad experiments; WP1 classification here is for the LLM white-box OPD core only. |
| Veto | [Stable On-Policy Distillation through Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) | Source defines on-policy KD over `y~P_S(.|x)` and an adaptive target `Q proportional to P_T * P_S^beta`, optimized with forward or reverse KL. | Preprint/source formatting still needs author/code normalization; objective evidence is strong enough for strict. |
| Fast OPD | [Fast and Effective On-Policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) | Source optimizes reverse KL on student samples but truncates student generation to reasoning prefixes and schedules the prefix length. | Strict only for prefix states; note same-vocabulary/tokenization assumptions. |
| Speculative KD | [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) | Source lets student propose tokens, teacher accepts/rejects/replaces using top-k, then minimizes teacher-student token divergence on the resulting trace. | Mixed trace means not exact student-generated states after replacement; partial. |
| PACED | [PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) | Source computes student pass-rate weights `w(p)=p^alpha(1-p)^beta`; one track uses forward KL on teacher sequences, another self-distill track uses reverse KL on student sequences. | Whole method partial; split the reverse-KL self-distill track only if the table needs a strict subrow. |
| Lightning OPD | [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010) | Abstract and source state teacher log-probs are precomputed once over SFT/reference rollouts; method defines `J_off` by replacing `x~pi_theta` with `x~pi_ref`. | C1 fails current-policy freshness; classify as offline OPD approximation, not strict OPD. |
| Cross-tokenizer candidates | [ULD](https://arxiv.org/abs/2402.12030), [ALM](https://arxiv.org/abs/2503.20083), [BLD](https://arxiv.org/abs/2604.07466) | These primary sources solve teacher-student tokenizer mismatch by optimal transport, likelihood matching, or byte-level interfaces. | Useful for white-box cross-tokenizer KD, but this pass found no clear current-student-rollout OPD loop. |

## Suggested Markdown Row Fragments

These fragments are intended to update or validate the relevant fields in `tables/opd_papers.md`; they are not a replacement for the full table schema.

| id | method_family | opd_strictness | divergence_or_objective | rollout_source | rollout_freshness | teacher_access | teacher_kind | strictness_evidence | notes |
|---|---|---|---|---|---|---|---|---|---|
| gkd-2024 | white_box_opd | strict_opd | generalized_kl_fkl_rkl_jsd | student_on_policy | current_policy | teacher_logprobs | larger_llm | C1 pure lambda=1 samples student outputs; C2 teacher scores those outputs; C3 generalized KL consumes teacher distribution. | Mixed lambda variants are partial. |
| minillm-2024 | reverse_kl_opd | strict_opd | sequence_reverse_kl | student_on_policy | current_or_teacher_mixed_behavior | teacher_logits | larger_llm | C1 student sequence distribution is optimized; C2 teacher distribution supplies log-ratio; C3 reverse-KL/PG objective consumes it. | Record teacher-mixed sampler caveat. |
| distillm-2024 | adaptive_off_policy_kd | partial_opd | skew_kl_and_skew_reverse_kl | mixed_policy | replay_buffer_stale | teacher_logits | larger_llm | C1 weakened by replay/static mixture; C2 teacher logits exist; C3 skew KL consumes them. | Do not list as strict. |
| distillm2-2025 | contrastive_kd | partial_opd | contrastive_skl_srkl | mixed_policy | per_epoch_batched_stale | teacher_logits | larger_llm | C1 weakened by teacher response plus stale SGO mixture; C2/C3 hold for KL terms. | Split only if a fresh SGO-only variant is documented. |
| distillspec-2024 | speculative_drafter_opd | strict_opd | f_divergence_fkl_rkl_jsd_tvd | draft_student_on_policy | current_policy_for_draft_variant | target_model_logits | reference_target_lm | C1 draft-generated sequences in on-policy variant; C2 target model scores draft states; C3 divergence objective trains draft. | Systems/inference acceleration row, not capability post-training. |
| entropy-aware-opd-2026 | entropy_aware_opd | strict_opd | reverse_kl_plus_entropy_gated_topk_forward_kl | student_on_policy | current_or_behavior_policy | teacher_logits_entropy_topk | larger_reasoning_llm | C1 student trajectories; C2 teacher log-probs/entropy/top-k; C3 EOPD objective consumes both. | Top-k FKL is approximate teacher-support matching. |
| g-opd-2026 | generalized_opd_reward_extrapolation | strict_opd | kl_constrained_log_ratio_objective | student_on_policy | current_policy | teacher_and_reference_logprobs | larger_llm_plus_reference | C1 student trajectories; C2 teacher/reference dense log-ratio; C3 G-OPD objective consumes signal. | Not reward-only RLVR. |
| reopold-2026 | relaxed_opd | strict_opd | clipped_relaxed_reverse_kl | student_on_policy | current_or_old_policy_iteration | teacher_logprobs | larger_reasoning_llm | C1 student/old-policy rollouts; C2 teacher log-ratio on tokens; C3 relaxed RKL surrogate consumes it. | Keep LLM core separate from broader experiments. |
| veto-2026 | adaptive_target_opd | strict_opd | kl_to_teacher_student_bridge_target | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student trajectories; C2 adaptive target combines teacher and student logits; C3 FKL/RKL to target updates student. | Normalize authors/code later. |
| fast-opd-2026 | prefix_opd | strict_opd | prefix_reverse_kl | student_on_policy_prefix | current_policy_truncated | teacher_logprobs | larger_reasoning_llm | C1 student prefix rollout; C2 teacher scores retained prefix; C3 reverse KL consumes it. | Strict only for prefix states. |
| speculative-kd-2024 | hybrid_speculative_kd | partial_opd | interleaved_token_kl | mixed_policy | fresh_hybrid_trace | teacher_logits_topk | target_reference_lm | C1 weakened by teacher replacement; C2 teacher logits; C3 token KL. | Keep separate from DistillSpec. |
| paced-2026 | curriculum_opd | partial_opd | beta_weighted_forward_or_reverse_kl | mixed_policy | fixed_or_staged_passrate_and_mixed_sequences | teacher_logits | larger_llm_or_self_teacher | C1 mixed by track; C2 teacher/self-teacher logits; C3 weighted KL objective. | Whole method partial; reverse-KL self-distill subtrack can be strict. |
| lightning-opd-2026 | offline_opd_approximation | partial_opd | cached_reverse_kl_logratio_surrogate | off_policy | precomputed_reference_rollouts | cached_teacher_logprobs | larger_reasoning_llm | Missing C1: rollouts fixed from pi_ref/SFT; C2 cached teacher log-probs; C3 offline objective consumes them. | Useful efficiency contrast. |
| cross-tokenizer-opd-candidates | cross_tokenizer_kd | not_opd_or_adjacent | uld_alm_bld_distribution_alignment | off_policy_or_unclear | static_or_unclear | teacher_distribution_alignment | larger_llm | Missing C1 in audited primary evidence; C2/C3 are KD-style only. | Do not add strict OPD row without current rollout proof. |

## False Positives and Downgrade Notes

- DistiLLM is not strict under this repo's current policy because the paper's streamlined algorithm deliberately reuses generated outputs and mixes replay/static data. Teacher logits and skew KL are real, but rollout freshness is not.
- DistiLLM-2 should not inherit strictness from the phrase "on-policy" alone. Its batch approach collects SGO before an epoch and mixes teacher-generated responses with student-generated responses.
- Speculative KD is fresh but not cleanly student-on-policy after teacher replacement. The exact supervised sequence can include teacher tokens, so it is partial.
- PACED has a strict reverse-KL self-distillation subcomponent, but the full method combines pass-rate weighting, forward KL on teacher sequences, and student-sequence self-distillation. Whole-method label should remain partial unless rows are split.
- Lightning OPD is intentionally offline: its contribution is to approximate OPD by fixing rollouts to `pi_ref` and caching teacher log-probs. It should be included as a white-box objective/efficiency method but downgraded from strict OPD.
- DistillSpec is strict only for draft-generated/on-policy drafter training. Other data-generation choices in the same paper should be tracked separately.
- Cross-tokenizer ULD/ALM/BLD are white-box KD enablers, not OPD by default. They become WP1 strict only if paired with current student rollouts and teacher distribution evaluation on those exact states.
- G-OPD and REOPOLD use RL-style notation but should not be downgraded merely for policy-gradient wording; the supervision is dense teacher log-prob/logit guidance on student-generated states.

## Open Gaps Requiring Line-Level Audit

- Lightning OPD: verify whether any experimental ablation uses current student rollouts; the main algorithm is offline/reference-rollout only.
- Veto: normalize author list, code status, and final equation numbering; source evidence for the adaptive target is sufficient but metadata is rough.
- Entropy-Aware OPD: audit the exact top-k renormalization and whether teacher entropy is computed over full vocabulary or cached/top-k support in all experiments.
- PACED: decide whether to split into two rows: forward-KL teacher-sequence PACED (`not_opd`/partial) and reverse-KL self-distillation PACED (`strict_opd` substage).
- DistiLLM/DistiLLM-2: inspect whether any reported experimental setting disables replay/batching enough to warrant a separate strict row. Current whole-method labels should stay partial.
- Cross-tokenizer OPD: line-audit ULD/ALM/BLD TeX for any student-rollout loop. Do not add a strict row until C1/C2/C3 are explicit.
- Fast OPD: audit tokenizer/vocabulary assumptions and special-token handling; same-vocab limitation matters for cross-tokenizer claims.
- DistillSpec: split rows by data source if the table needs precision: draft-generated strict, target/generated offline variants not strict.
- G-OPD: record exact reference model choices by experiment; lambda/reference access affects teacher-access cost and reproducibility.
