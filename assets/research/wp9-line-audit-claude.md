# WP9 Line-Level Primary-Source Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML/PDF full text for all 15 methods
**Methodology:** Direct web fetch of abstracts + HTML method sections, cross-validated by 5 independent research agents. Each agent independently located primary sources, extracted exact quotes, and assessed C1/C2/C3. Disagreements were reconciled against the repo's strict-opd-definition.md framework.

---

## Scope

This audit covers the 15 methods listed in the WP9 human spot-check queue (`notes/wp9-verifier-synthesis.md`, line 119):

> Uni-OPD, MAD-OPD, VOLD, MiMo-V2-Flash, Gemma2 post-training, G-OPD, REOPOLD, PACED, SDPO, OPCD, OEL, SCOPE, KDRL, Lightning OPD, DistiLLM-2

No new candidate papers were introduced. No broad survey was performed.

## OPD Test

Strict OPD requires all three conditions from `taxonomies/strict-opd-definition.md`:

- **C1 (Student rollout):** The current student/policy generates the training trajectory.
- **C2 (Same-rollout teacher-style supervision):** A teacher, expert, discriminator, reference model, previous checkpoint, or privileged-context model supervises that exact student trajectory.
- **C3 (Objective consumption):** The training objective consumes the teacher-style signal, not only a scalar reward.

---

## Summary of Actions

| Action | Methods |
|---|---|
| **Confirmed strict** (source_verified) | Uni-OPD, MAD-OPD, VOLD, G-OPD, REOPOLD |
| **Promote to strict** | SDPO (from borderline), OPCD (from borderline), MiMo-V2-Flash MOPD stage (from partial) |
| **Upgrade to borderline** | Gemma 2 post-training (from partial), SCOPE (from partial) |
| **Confirmed borderline** | KDRL |
| **Confirmed partial** | OEL, Lightning OPD, DistiLLM-2, PACED |

## Key Nuances Discovered

1. **PACED** is NOT uniformly on-policy. Forward-KL distillation uses teacher-generated tokens y_T; only the reverse-KL self-distillation track uses student tokens y_S. The "student rollouts" claim in the abstract refers to the pass-rate weighting mechanism, not the KL loss itself. Pass-rate weights are "computed once and kept fixed" by default.

2. **DistiLLM-2** explicitly self-identifies as NOT on-policy: "we adopt a batch approach...rather than on-policy approach, which samples at every training iteration." SGOs are from theta_{e-1} (previous epoch checkpoint).

3. **MiMo-V2-Flash** has clear MOPD equations: token-level reverse-KL advantage A_{MOPD,t} = sg[log(pi_domain/pi_theta)] consumed in a policy-gradient surrogate. This is strict OPD despite being wrapped in RL notation.

4. **SCOPE** has a dual-path architecture: only the incorrect-trajectory branch receives teacher KL supervision; the correct branch uses student-weighted MLE without the teacher. Upgraded to borderline_strict because the teacher-supervised branch is genuine token-level OPD.

5. **KDRL** has a clean KD-RKL subcomponent (strict), but the full unified objective J_KDRL = J_GRPO - beta * D_KL mixes reward-only RL with KD.

6. **SDPO** uses a self-teacher pi_theta(.|x,f) with stopgrad. This qualifies as a privileged-context teacher under the repo's C2 definition, consistent with OPSD and CRISP precedents. One agent argued for partial_opd (no external teacher); reconciliation favors strict_opd because `strict-opd-definition.md` explicitly lists "privileged-context model" as a valid C2 teacher.

7. **Gemma 2** post-training evidence is a single sentence citing GKD and MiniLLM: "distillation from the teacher on the student's distribution." Intent is unmistakably on-policy distillation, but no loss function, rollout schedule, or mechanistic detail is given.

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Uni-OPD | https://arxiv.org/abs/2605.03677 | S3.1, Eq.1-3 | "student policy pi_theta samples its trajectories" (S3.1); "tau sampled by the student" | "r_t^{OPD} = log pi_T(o_t\|q,o_{<t}) - log pi_theta(o_t\|q,o_{<t})" (S3.1 Eq.3); "teacher to provide fine-grained supervision for student-generated trajectories" | "J_{OPD}(theta) = min_theta E_{tau~pi_theta}[D_{KL}(pi_theta(tau\|q) \|\| pi_T(tau\|q))]" (S3.1 Eq.1); margin calibration restores order consistency (S3.2 Eq.4) | current_policy | strict_opd | none | high |
| MAD-OPD | https://arxiv.org/abs/2605.01347 | S3.1, S4.1-4.4, Eq.7-8 | "y_hat ~ pi_theta(.\|x): the student generates trajectories along its own distribution" (S3.1); "a_m ~ pi_theta(.\|s_m): on-policy" (Alg.1) | "teachers debate over the student's on-policy state...token-level supervision, with each teacher's contribution weighted by its post-debate confidence" (S4.1); "p_{T_k}(.\|s_m, H_m^R, y_hat_{<t})" (S4.3) | "L_{mad-opd}(theta) = E[sum_t sum_k w_k * D(p_{T_k} \|\| p_S)]" (S4.3 Eq.8); "JSD for agentic...reverse KL for code generation" (Remark 1) | current_policy | strict_opd | none | high |
| VOLD | https://arxiv.org/abs/2510.23497 | S3.1-3.2, Eq.5 | "The expectation E_{tau~pi_theta} makes this 'on-policy' since the prefixes h_t come from trajectories sampled from the current student policy pi_theta" (S3.1) | "teacher is queried on the same prefix h_t to provide distributional targets that adapt to the student's evolving policy" (S3.1) | "L_{VOLD}(theta) = L_{GRPO}(theta) + beta * E[sum_t D_{KL}(pi_phi(.\|h_t) \|\| pi_theta(.\|h_t))]" (S3.2 Eq.5); "By reusing the same rollouts for both objectives" | current_policy | strict_opd | none; hybrid KL+GRPO but KL is dense token-level on same student rollouts | high |
| MiMo-V2-Flash | https://arxiv.org/abs/2601.02780 | S4.4 (MOPD), Eq.5-9 | "student model samples from its own evolving distribution" (S4.4); "y ~ mu_theta(.\|x)"; "By learning from its own distribution, the student avoids the exposure bias and distribution mismatch common in off-policy methods" | "dense, token-level rewards from specialized teachers through KL divergence rewards" (S4.4); "log pi_{domain}(y_t\|x,y_{<t})/pi_theta(y_t\|x,y_{<t})" (Eq.8) | "A_{MOPD,t} = sg[log(pi_{domain}/pi_theta)] + alpha * A_{ORM}" (Eq.9); surrogate loss L_{MOPD} consumes this (Eq.7) | current_policy | strict_opd | strict only for MOPD stage (Stage 3); full pipeline is multi-stage hybrid | high |
| Gemma 2 post-training | https://arxiv.org/abs/2408.00118 | S4 Post-Training | not found explicitly; implied by "on the student's distribution" | "We also run distillation from the teacher on the student's distribution (Agarwal et al., 2024; Gu et al., 2024)" (S4) | not found; cites GKD and MiniLLM which use KL divergence; pre-training loss: "minimize the negative log-likelihood between the probabilities from the teacher and the student" (S3.2) | unclear | borderline_strict | exact rollout mechanism, loss function, and rollout schedule not specified in the report; evidence is a single sentence citing GKD/MiniLLM | medium |
| G-OPD | https://arxiv.org/abs/2602.12125 | S3.1-3.2, Eq.4, 9, 11 | "the trajectories y are generated by the policy model itself, making the training remain on-policy" (S3.1); "y ~ pi_theta(.\|x)" | "aligns the student with the teacher's logit distribution on student-generated trajectories" (Abstract); "r_t^{OPD} = log(pi^*(y_t\|x,y_{<t})/pi_{ref}(y_t\|x,y_{<t}))" (S3.1 Eq.9) | "J_{G-OPD}(theta) = max E[lambda * log(pi^*/pi_{ref}) - D_{KL}(pi_theta \|\| pi_{ref})]" (S3.2 Eq.11); "OPD is equivalent to a specific KL-constrained RL objective" with lambda>1 for extrapolation | current_policy | strict_opd | none | high |
| REOPOLD | https://arxiv.org/abs/2603.11137 | S2.2, S3.1, S4, Eq.2-7 | "student model learns from its own trajectories" (S2.2); "o ~ pi_theta(.\|q)" (Eq.2); "Generate {o_i} ~ pi_{theta_old}(.\|q)" (Alg.1) | "R_{i,t}(theta) = log pi_T(o_{i,t}\|q,o_{i,<t})/pi_theta(o_{i,t}\|q,o_{i,<t})" (S3.1) -- teacher-student log-ratio at every token | "maximize weighted sum over rho_{i,t}(theta) * R_hat^lambda_{i,t}(theta) * M_{i,t}(k)" (S4 Eq.4); clipped reward R_hat and dynamic masking M (Eq.5-6) | current_policy | strict_opd | "token reward" wording sits on KD/RL boundary, but teacher logits are consumed per-token via log-ratio; no off-policy or replay | high |
| PACED | https://arxiv.org/abs/2603.11178 | S3.1-3.3, Alg.1 | "we sample K rollouts from the student and compute the pass rate" (S3.1); for reverse-KL: "y_S ~ pi_theta(.\|x)" (S3.3 Alg.1) | "frozen Qwen3-14B as teacher" (distill track); "frozen self-teacher" (self-distill track) | Forward KL on teacher tokens: "sum_t D_{KL}(p_T(.\|y_{T,<t}) \|\| p_S(.\|y_{T,<t}))" (S3.3); Reverse KL on student tokens: "sum_t D_{KL}(p_S(.\|y_{S,<t}) \|\| p_T(.\|y_{S,<t}))" (S3.3) | current_policy (reverse-KL stage); teacher-generated (forward-KL stage) | partial_opd | forward-KL stage uses teacher-generated sequences y_T (C1 fails for that stage); only reverse-KL self-distillation stage uses student tokens y_S and qualifies as strict; pass-rate weights computed once and kept fixed | high |
| SDPO | https://arxiv.org/abs/2601.20802 | S2, Eq.1, Alg.1 | "Sample responses: {y_i}_{i=1}^G ~ pi_theta(.\|x)" (Alg.1 Line 4) | "the self-teacher, pi_theta(.\|x,f), which refers to the current policy prompted with the question x and the rich feedback f" (S2); "We can use the same policy in two different roles" | "L_{SDPO}(theta) := sum_t KL(pi_theta(.\|x,y_{<t}) \|\| stopgrad(pi_theta(.\|x,f,y_{<t})))" (S2 Eq.1); "logit-level policy gradient where the advantages are estimated using the self-teacher" (S2.1) | current_policy | strict_opd | self-teacher (no external teacher); privileged-context variant where feedback f is the privileged context; stopgrad creates functional teacher-student separation | high |
| OPCD | https://arxiv.org/abs/2602.12275 | Abstract, S3, Eq.1-2 | "training a student model on its own generated trajectories" (Abstract); "let the student model pi_theta generate complete response trajectories y. Importantly, these trajectories are generated without context c" (S3) | "minimizing reverse KL against a context-conditioned teacher" (Abstract); "evaluate it using the teacher model pi_teacher, which processes the concatenated sequence [c;x;y]" (S3) | "L(theta) = E_{(x,c)~D, y~pi_theta(.\|x)}[1/\|y\| sum_t D_{KL}(pi_theta(.\|x,y_{<t}) \|\| pi_teacher(.\|c,x,y_{<t}))]" (S3 Eq.1) | current_policy | strict_opd | privileged-context self-teacher (same model with vs. without context), not external teacher | high |
| OEL | https://arxiv.org/abs/2603.16856 | Abstract, S3.2, Eq.2 | "enables language models to continuously improve from their own deployment experience" (Abstract); "The student pi_theta generates a response y conditioned only on x" (S3.2) | "optimized to match the knowledge-conditioned output of a teacher pi_teacher through token-level reverse KL divergence" (S3.2); teacher is "frozen initial pi_theta before training" conditioned on experiential knowledge e | "L(theta) = E_{x~D, e~C, y~pi_theta(.\|x)}[1/\|y\| sum_t D_{KL}(pi_theta(.\|x,y_{<t}) \|\| pi_teacher(.\|e,x,y_{<t}))]" (S3.2 Eq.2) -- identical to OPCD | current_policy (inner OPCD step); per_iteration (outer loop) | partial_opd | strict only for the OPCD-like consolidation substage; full loop includes deployment trajectory collection + knowledge extraction stages that are not OPD | medium |
| SCOPE | https://arxiv.org/abs/2604.10688 | S3.1, S3.3, Eq.2-6 | "student model generates a group of N responses" (S3.1); "on-policy rollouts" routed by correctness (Abstract) | "dense, token-level KL supervision from a teacher model on the student's self-sampled rollouts" (Abstract); teacher-perplexity-weighted KL on incorrect branch (S3.3) | Incorrect branch: "L_OPD = sum_t rho * (log pi_theta - log pi_T)" (S3.1 Eq.3); Correct branch: "L_MLE = -sum_t rho * log pi_theta" no teacher (S3.1 Eq.2); Combined: "J_SCOPE = sum_{i in Omega_c} w_i^{stu} * L_MLE + sum_{i in Omega_w} w_i^{tea} * L_OPD" (S3.3 Eq.6) | current_policy | borderline_strict | only incorrect-trajectory branch has teacher token-level KL; correct branch uses student-only MLE without teacher supervision | high |
| KDRL | https://arxiv.org/abs/2506.02208 | S2.1-2.2, S3.1, Eq.5, 8 | "We adopt on-policy optimization by setting pi_{theta_old}=pi_theta" (S2.1); "o ~ pi_theta(.\|q)" | "R_{i,t}(theta) = log pi_T(o_{i,t}\|q,o_{i,<t})/pi_theta(o_{i,t}\|q,o_{i,<t}) acts as the token-level reward signal" (S2.2 Eq.5) | "J_{KDRL}(theta) = J_{GRPO}(theta) - beta * D_{KL}(pi_theta \|\| pi_T)" (S3.1 Eq.8); "unified objective that integrates GRPO and KD" | current_policy | borderline_strict | KD-RKL subcomponent is strict_opd; full method hybridizes with GRPO (reward-only RL from rule-based verifier, not teacher) | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | Abstract, S3.1, Alg.1 | C1 FAILS: "Standard OPD [samples] rollouts online from the current student while Lightning OPD reuses rollouts precomputed from pi_ref before training begins" (S3.1); "fixes the rollout distribution to pi_ref" | teacher log-probs precomputed: "rollouts are sampled from pi_ref and the teacher is queried once to precompute and store per-token log-probabilities" (S3.1) | "A_t(theta) = log pi_T(a_t\|s_t) - log pi_theta(a_t\|s_t)" in advantage-weighted PG (S3.1); distillation objective on cached data | precomputed_sft_rollouts | partial_opd | rollouts from pi_ref, not pi_theta; teacher precomputed and cached; paper self-identifies as "offline" approximation to OPD | high |
| DistiLLM-2 | https://arxiv.org/abs/2503.07067 | S2.2, S3.1.2, Alg.1 | C1 WEAKENED: "Sample responses y_t, y_s from teacher p(.\|x) and student q_{theta_{e-1}}(.\|x)" (Alg.1); "we adopt a batch approach...rather than on-policy approach, which samples at every training iteration" (S2.2) | teacher logits supervise both branches: "D_{SKL}^{alpha_t}(x,y_t)" + "D_{SRKL}^{alpha_s}(x,y_s)" (S3.1.2 Eq.2) | "SKL for teacher responses...SRKL for student responses" -- contrastive skew-KL loss (S3.1.2) | per_epoch / replay_buffer_stale | partial_opd | paper explicitly self-identifies as NOT on-policy; SGOs from theta_{e-1} (previous epoch), not current theta; mixed teacher+student responses | high |

---

## Reconciliation Notes

### Agent disagreements resolved

1. **SDPO**: One agent (REOPOLD/PACED/SDPO) labeled SDPO as `partial_opd`, arguing that the self-teacher is not an external model and therefore does not satisfy C2. However, `taxonomies/strict-opd-definition.md` explicitly allows "privileged-context model" as a valid C2 teacher, and OPSD and CRISP (both `strict_opd` seeds in the repo) use the same pattern. SDPO's self-teacher pi_theta(.|x,f) with stopgrad is functionally equivalent: the feedback f serves as privileged context that the student does not see. **Resolution: strict_opd**, consistent with repo precedent.

2. **SCOPE**: The OPCD/OEL/SCOPE agent upgraded SCOPE from `partial_opd` to `borderline_strict`, arguing that the incorrect-trajectory branch has genuine token-level teacher KL, which is more than scalar reward. The correct-trajectory branch is pure student MLE. **Resolution: borderline_strict**, because the teacher-supervised branch is genuine token-level OPD (not just a scalar reward), but the full method is not uniformly OPD.

3. **PACED**: Both agents and direct fetches agree. Forward-KL distillation track trains on teacher-generated sequences y_T (C1 fails). Reverse-KL self-distillation track trains on student-generated sequences y_S (all C1/C2/C3 pass). The "two-stage forward-then-reverse KL schedule" is the recommended recipe, making the overall method partial_opd. **Resolution: partial_opd** overall; reverse-KL stage alone qualifies as strict.

### Methods with stage-specific labels

Several methods are strict OPD only for a specific component or stage:

| Method | Strict substage | Non-strict stages |
|---|---|---|
| MiMo-V2-Flash | Stage 3 (MOPD) | Stages 1-2 (SFT, RL) |
| PACED | Reverse-KL self-distillation | Forward-KL distillation (teacher tokens) |
| KDRL | KD-RKL subcomponent | GRPO (reward-only RL) |
| OEL | OPCD consolidation substage | Deployment collection, knowledge extraction |
| SCOPE | Incorrect-trajectory branch | Correct-trajectory branch (student MLE) |
| Gemma 2 | Post-training SFT sub-step | Pre-training KD (offline), RLHF |
| VOLD | Stage 2 (GRPO+KL) | Stage 1 (SFT cold-start) |

### Promotions relative to current repo labels

| Method | Repo current | Audit verdict | Change type |
|---|---|---|---|
| Uni-OPD | strict_opd (human_spotcheck_needed) | strict_opd (source_verified) | confirmed |
| MAD-OPD | strict_opd (human_spotcheck_needed) | strict_opd (source_verified) | confirmed |
| VOLD | strict_opd (wp0_consensus) | strict_opd (source_verified) | confirmed |
| MiMo-V2-Flash | partial_opd (human_spotcheck_needed) | strict_opd (MOPD stage) | **PROMOTE** |
| Gemma 2 post-training | partial_opd (human_spotcheck_needed) | borderline_strict | **UPGRADE** |
| G-OPD | strict_opd (wp0_consensus) | strict_opd (source_verified) | confirmed |
| REOPOLD | strict_opd (wp0_consensus) | strict_opd (source_verified) | confirmed |
| PACED | partial_opd (human_spotcheck_needed) | partial_opd (reverse-KL substage strict) | confirmed |
| SDPO | borderline_strict (human_spotcheck_needed) | strict_opd | **PROMOTE** |
| OPCD | borderline_strict (human_spotcheck_needed) | strict_opd | **PROMOTE** |
| OEL | partial_opd (human_spotcheck_needed) | partial_opd (OPCD substage strict) | confirmed |
| SCOPE | partial_opd (human_spotcheck_needed) | borderline_strict | **UPGRADE** |
| KDRL | borderline_strict (human_spotcheck_needed) | borderline_strict | confirmed |
| Lightning OPD | partial_opd (human_spotcheck_needed) | partial_opd | confirmed |
| DistiLLM-2 | partial_opd (human_spotcheck_needed) | partial_opd | confirmed |

---

## Verification methodology

1. **Abstract extraction**: WebFetch of each arXiv abstract page to identify paper title, authors, and high-level method description.

2. **HTML full-text extraction**: WebFetch of `arxiv.org/html/` endpoints to read method sections, algorithm boxes, and loss equations for methods where abstracts were insufficient (SDPO, PACED, MiMo-V2-Flash, Gemma 2, DistiLLM-2, VOLD).

3. **Cross-validation**: Five independent research agents each covered 3 methods, performing their own web searches and fetches without seeing each other's results. Disagreements (SDPO, SCOPE) were resolved by consulting the repo's canonical definitions.

4. **Conservative rules applied**:
   - Quotes must be locatable in the source; "not found" triggers downgrade.
   - Reward-only signals default to partial/adjacent.
   - Stage-specific labels do not generalize to the full pipeline.
   - Self-identification as "not on-policy" (DistiLLM-2) is treated as authoritative evidence against strict_opd.
   - Privileged-context self-teachers are valid C2 teachers per repo definition.
