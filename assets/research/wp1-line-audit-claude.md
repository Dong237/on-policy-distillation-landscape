# WP1 Line-Level Primary-Source Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML full text for all 15 methods
**Methodology:** Five independent research agents each fetched primary-source HTML/abstracts for 3 methods, extracted exact quotes, and assessed C1/C2/C3. Results were reconciled against `taxonomies/strict-opd-definition.md`.

---

## Scope

This audit covers the 15 priority methods from the WP1 spot-check queue:

> Uni-OPD, MAD-OPD, VOLD, MiMo-V2-Flash, Gemma2 post-training, G-OPD, REOPOLD, PACED, SDPO, OPCD, OEL, SCOPE, KDRL, Lightning OPD, DistiLLM-2

No new candidate papers were introduced. No broad survey was performed.

## OPD Test

Strict OPD requires all three conditions from `taxonomies/strict-opd-definition.md`:

- **C1 (Student rollout):** The current student/policy generates the training trajectory.
- **C2 (Same-rollout teacher-style supervision):** A teacher, expert, discriminator, reference model, previous checkpoint, or privileged-context model supervises that exact student trajectory.
- **C3 (Objective consumption):** The training objective consumes the teacher-style signal, not only a scalar reward.

---

## Memo

### Action summary

| Action | Methods |
|---|---|
| **Confirmed strict** (source_verified) | Uni-OPD, MAD-OPD, VOLD, G-OPD, REOPOLD, SDPO, OPCD, MiMo-V2-Flash (MOPD stage) |
| **Confirmed borderline** | Gemma 2 post-training, SCOPE, KDRL |
| **Confirmed partial** | PACED, OEL, Lightning OPD, DistiLLM-2 |

### Key nuances

1. **PACED** has two distinct training tracks. Forward-KL distillation uses teacher-generated tokens y_T (C1 fails). Only the reverse-KL self-distillation track uses student tokens y_S and qualifies as strict. The paper's "student rollouts" claim in the abstract refers to the pass-rate weighting mechanism, not the forward-KL loss itself.

2. **DistiLLM-2** explicitly self-identifies as NOT on-policy: "we adopt a batch approach...rather than on-policy approach, which samples at every training iteration." SGOs are from theta_{e-1} (previous epoch checkpoint).

3. **MiMo-V2-Flash** has clear MOPD equations: token-level KL advantage A_{MOPD,t} = sg[log(pi_domain/pi_theta)] consumed in a PPO-style surrogate. Strict OPD despite RL notation. Strict only for MOPD stage (Stage 3).

4. **SCOPE** has a dual-path architecture: only the incorrect-trajectory branch receives teacher KL supervision; the correct branch uses student-weighted MLE without the teacher. Borderline because the teacher-supervised branch is genuine token-level OPD, but the full method is not uniformly OPD.

5. **KDRL** has a clean KD-RKL subcomponent (strict), but the full unified objective J_KDRL = J_GRPO - beta * D_KL mixes reward-only RL with KD.

6. **SDPO** uses a self-teacher pi_theta(.|x,f) with stopgrad. This qualifies as a privileged-context teacher under the repo's C2 definition, consistent with OPSD and CRISP precedents. The feedback f serves as privileged context that the student does not see.

7. **Gemma 2** post-training evidence is a single sentence citing GKD and MiniLLM: "distillation from the teacher on the student's distribution." No loss function, rollout schedule, or mechanistic detail is given in the report.

8. **Lightning OPD** explicitly replaces student rollouts with reference/SFT rollouts. C1 fails by design. The paper self-identifies as "offline" and measures the gradient gap from true OPD (Theorem 3.5).

9. **OEL** wraps OPCD-style consolidation inside a multi-stage loop where experiential knowledge comes from prior deployment rounds. Consolidation substage alone is strict; full pipeline is partial.

10. **OPCD** is textbook strict OPD: student generates without context, teacher evaluates with context [c;x;y], reverse-KL objective. Same model with vs. without context constitutes a valid privileged-context teacher.

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Uni-OPD | https://arxiv.org/abs/2605.03677 | S3.1, Eq.1-2 | "the student policy pi_theta samples its trajectories and is optimized by minimizing the reverse KL divergence to the teacher" (S3.1) | "OPD relies on the teacher to provide fine-grained supervision for student-generated trajectories" (S3.1); r_t^OPD = log pi_T - log pi_theta (S3.4) | "J_OPD(theta) = min E_{tau~pi_theta}[D_KL(pi_theta(tau\|q) \|\| pi_T(tau\|q))]" (Eq.1); Uni-OPD extends with margin calibration (Eq.4) | current_policy | strict_opd | none | high |
| MAD-OPD | https://arxiv.org/abs/2605.01347 | S4.4, Eq.1,8, Alg.1 | "student generates an on-policy trajectory...at step m, it samples action a_m ~ pi_theta(.\|s_m)" (S4.4); y_hat ~ pi_theta (Eq.1) | "Each teacher processes the student's on-policy sample y_hat with the debate transcript H_m^R appended to its context" (S4.3); token-level supervision weighted by post-debate confidence | "L_{mad-opd}(theta) = E[sum_t sum_k w_k D(p_{T_k}(.\|s_m,H_m^R,y_hat_{<t}) \|\| p_S)]" (Eq.8); divergence D is distributional, not scalar | current_policy | strict_opd | none | high |
| VOLD | https://arxiv.org/abs/2510.23497 | S3.1-3.2, Eq.3,5 | "on-policy distillation provides student supervision on its own sampled trajectories" (S3.1); "both GRPO and on-policy KD require sampling trajectories from the student policy pi_theta" (S3.2) | "the teacher is queried on the same prefix h_t to provide distributional targets that adapt to the student's evolving policy" (S3.1) | "L_VOLD = L_GRPO + beta E_{tau~pi_theta}[sum_t D_KL(pi_phi(.\|h_t) \|\| pi_theta(.\|h_t))]" (Eq.5); by reusing same rollouts for both objectives | current_policy | strict_opd | none; hybrid KL+GRPO but KL is dense token-level on same student rollouts | high |
| MiMo-V2-Flash | https://arxiv.org/abs/2601.02780 | S4.1, S4.4, Eq.7-9 | "The student model samples from its own evolving distribution and receives token-level supervision" (S4.1); "y ~ mu_theta(.\|x)" (S4.4) | "dense, token-level rewards from specialized teachers through KL divergence rewards" (S4.4); A_MOPD,t = sg[log pi_domain/pi_theta] (Eq.8) | "L_MOPD(theta) = -E[1/\|y\| sum w_t A_MOPD,t log pi_theta]" (Eq.7); combined advantage feeds teacher KL + outcome reward into surrogate loss (Eq.9) | current_policy | strict_opd | strict only for MOPD stage (Stage 3); full pipeline is multi-stage hybrid | high |
| Gemma 2 post-training | https://arxiv.org/abs/2408.00118 | S4 Post-Training | C1 implied: "on the student's distribution" (S4) | "We also run distillation from the teacher on the student's distribution (Agarwal et al., 2024; Gu et al., 2024)" (S4) | not found; cites GKD and MiniLLM which use KL divergence; pre-training loss in S3.2 is forward-KL, not the post-training objective | unclear | borderline_strict | exact rollout mechanism, loss function, and rollout schedule not specified; evidence is a single sentence citing GKD/MiniLLM; C3 not found in post-training context | medium |
| G-OPD | https://arxiv.org/abs/2602.12125 | S3.1-3.2, Eq.4,11 | "The main idea of OPD is to let the student generate its own trajectories" (S3.1); "the trajectories y are generated by the policy model itself, resulting in the on-policy training" (S3.1) | "aligning the student with the teacher's logit distribution on each token of these student-generated trajectories" (S3.1) | "J_{G-OPD}(theta) = max E[lambda log(pi*/pi_ref) - D_KL(pi_theta \|\| pi_ref)]" (Eq.11); lambda>1 extrapolates beyond teacher | current_policy | strict_opd | none | high |
| REOPOLD | https://arxiv.org/abs/2603.11137 | S4, Alg.1, Eq.4-5 | "Generate {o_i} ~ pi_theta_old(.\|q) for each q in B" (Alg.1 Line 7); theta_old <- theta each step (Line 21) | "R_{i,t}(theta) = log pi_T(o_{i,t}\|q,o_{i,<t}) / pi_theta(o_{i,t}\|q,o_{i,<t})" (Alg.1 Line 8) -- teacher log-ratio per token | "rho_{i,t}(theta) * R_hat^lambda_{i,t}(theta) * M_{i,t}" (Eq.4); R_hat is clipped teacher reward max(sg(R_t), log(lambda/(1-lambda))) with dynamic masking | current_policy | strict_opd | none; RL-style wording but supervision is token-level teacher logit comparison, not sparse outcome reward | high |
| PACED | https://arxiv.org/abs/2603.11178 | S3.1-3.3, Alg.1 | Forward-KL: C1 FAILS -- "L_distill(theta; y_{T,i}, x_i) {Teacher-forced distillation}" (Alg.1 Line 19); Reverse-KL: C1 PASS -- "Sample y_{S,i} ~ pi_theta(.\|x_i)" (Alg.1 Lines 21-22) | Forward: "sum_t D_KL(p_T(.\|y_{T,<t}) \|\| p_S(.\|y_{T,<t}))" on teacher sequences; Reverse: "sum_t D_KL(p_S(.\|y_{S,<t}) \|\| p_T(.\|y_{S,<t}))" on student sequences (S3.3) | "L(theta;x) = w(p) * L_distill(theta;x) where w(p) = p^alpha(1-p)^beta" (S3.2 Eq.3); beta-kernel weighting from pass-rate | mixed_freshness | partial_opd | forward-KL track uses teacher-generated y_T (C1 fails); only reverse-KL self-distillation track uses student y_S; whole method must be partial | high |
| SDPO | https://arxiv.org/abs/2601.20802 | S2, Eq.1, Alg.1 | "Sample responses: {y_i}_{i=1}^G ~ pi_theta(.\|x)" (Alg.1 Line 4) | "pi_theta(.\|x,f), which refers to the current policy prompted with the question x and the rich feedback f" (S2); stopgrad blocks gradient flow to prevent teacher regression | "L_SDPO(theta) := sum_t KL(pi_theta(.\|x,y_{<t}) \|\| stopgrad(pi_theta(.\|x,f,y_{<t})))" (Eq.1) -- token-level KL to feedback-conditioned self-teacher | current_policy | strict_opd | privileged-context self-teacher (no external teacher); stopgrad creates functional teacher-student separation; valid C2 per repo definition | high |
| OPCD | https://arxiv.org/abs/2602.12275 | S3, Eq.1, Alg.1 | "the student model pi_theta generates complete response trajectories y...generated without context c" (S3); "Sample response y ~ pi_theta(.\|x)" (Alg.1) | "we evaluate it using the teacher model pi_teacher, which processes the concatenated sequence [c;x;y]" (S3); teacher sees context, student does not | "L(theta) = E_{y~pi_theta}[1/\|y\| sum_t D_KL(pi_theta(.\|x,y_{<t}) \|\| pi_teacher(.\|c,x,y_{<t}))]" (Eq.1) -- token-level reverse KL | current_policy | strict_opd | privileged-context self-teacher (same model with/without context); valid C2 per repo definition | high |
| OEL | https://arxiv.org/abs/2603.16856 | S3.1-3.2, Eq.2 | "The student pi_theta generates a response y conditioned only on x" (S3.2) -- consolidation substage only | "experiential-knowledge-conditioned teacher provides dense, token-level training signal" (S3.2); teacher sees (e,x,y_{<t}), student sees (x,y_{<t}) | "L(theta) = E[1/\|y\| sum_t D_KL(pi_theta(.\|x,y_{<t}) \|\| pi_teacher(.\|e,x,y_{<t}))]" (Eq.2) -- identical structure to OPCD | current_policy (consolidation); per_round (full pipeline) | partial_opd | strict only for OPCD-like consolidation substage; full loop includes deployment collection + knowledge extraction stages that are not OPD; experiential knowledge e from prior deployment round | medium |
| SCOPE | https://arxiv.org/abs/2604.10688 | S3.1, S3.3, Eq.2-3,6 | "for each input prompt x, the student model generates a group of N responses" (S3.1) -- routed by correctness | Incorrect branch only: "we leverage the teacher policy pi_T to provide external guidance" (S3.1); Correct branch: "Rather than relying on teacher guidance, we directly leverage these successful attempts" -- NO teacher | Incorrect: "L_OPD = sum_t rho(log pi_theta - log pi_T)" (Eq.3); Correct: "L_MLE = -sum_t rho log pi_theta" (Eq.2) NO teacher; Combined: "J_SCOPE = sum w_i^stu L_MLE + sum w_i^tea L_OPD" (Eq.6) | per_iteration | borderline_strict | only incorrect-trajectory branch has teacher token-level KL; correct branch is student-only MLE without teacher supervision | high |
| KDRL | https://arxiv.org/abs/2506.02208 | S2.1-2.2, S3.1, Eq.5,8 | "KD-RKL performs on-policy sampling" (S2.2); "on-policy optimization by setting pi_theta_old = pi_theta" (S2.1) | "R_{i,t}(theta) = log pi_T(o_{i,t}\|q,o_{i,<t}) / pi_theta(o_{i,t}\|q,o_{i,<t}) acts as the token-level reward signal" (Eq.5) -- teacher log-ratio per token | "J_KDRL(theta) = J_GRPO(theta) - beta * D_KL^{k2}(pi_theta \|\| pi_T)" (Eq.8); KD-RKL sub-objective is strict; GRPO uses scalar rule-based reward, not teacher signal | current_policy | borderline_strict | KD-RKL subcomponent is strict_opd; full method hybridizes with GRPO (reward-only RL from rule-based verifier, not teacher); GRPO violates C2/C3 | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | S3.1-3.2, Alg.1, Thm.3.5 | C1 FAILS: "rollouts are sampled from pi_ref" (S3.2); "x^j ~ pi_ref(.\|q^j)" (Alg.1); paper says "we investigate whether OPD can be performed offline" (Abstract) | C2 FAILS: "the teacher is queried once to precompute and store per-token log-probabilities" (S3.2) -- on pi_ref rollouts, not student rollouts | "A_t(theta) = log pi_T(a_t\|s_t) - log pi_theta(a_t\|s_t)" advantage on cached data (S3.1); Thm.3.5 bounds gradient gap from true OPD | precomputed_sft_rollouts | partial_opd | C1 and C2 fail by design; rollouts from pi_ref, teacher precomputed; paper self-identifies as "offline" approximation to OPD | high |
| DistiLLM-2 | https://arxiv.org/abs/2503.07067 | S2.2, S3.1.2, Alg.1 | C1 WEAKENED: "Sample responses y_t, y_s from teacher p(.\|x) and student q_{theta_{e-1}}(.\|x)" (Alg.1) -- previous epoch checkpoint, not current theta | "we adopt a batch approach...rather than on-policy approach, which samples at every training iteration" (S2.2); "adaptive off-policy approach" (S2.2) | "L_CALD = (1/2)(2-beta)*SKL(teacher_data) + beta*SRKL(student_data)" (S3.1.2 Eq.5); contrastive skew-KL consumes teacher distributions | per_epoch / replay_buffer_stale | partial_opd | paper explicitly self-identifies as NOT on-policy; SGOs from theta_{e-1} (previous epoch), not current theta; mixed teacher+student responses | high |

---

## Methods with stage-specific labels

| Method | Strict substage | Non-strict stages |
|---|---|---|
| MiMo-V2-Flash | Stage 3 (MOPD) | Stages 1-2 (SFT, RL) |
| PACED | Reverse-KL self-distillation | Forward-KL distillation (teacher tokens) |
| KDRL | KD-RKL subcomponent | GRPO (reward-only RL) |
| OEL | OPCD consolidation substage | Deployment collection, knowledge extraction |
| SCOPE | Incorrect-trajectory branch | Correct-trajectory branch (student MLE) |
| Gemma 2 | Post-training SFT sub-step (implied) | Pre-training KD (offline), RLHF |
| VOLD | Stage 2 (GRPO+KL) | Stage 1 (SFT cold-start) |

---

## Cross-validation with existing repo labels

| Method | Repo current | Audit verdict | Change type |
|---|---|---|---|
| Uni-OPD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| MAD-OPD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| VOLD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| MiMo-V2-Flash | strict_opd (source_verified) | strict_opd (MOPD stage, source_verified) | confirmed |
| Gemma 2 post-training | borderline_strict (source_verified) | borderline_strict (source_verified) | confirmed |
| G-OPD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| REOPOLD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| PACED | partial_opd (source_verified) | partial_opd (source_verified) | confirmed |
| SDPO | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| OPCD | strict_opd (source_verified) | strict_opd (source_verified) | confirmed |
| OEL | partial_opd (source_verified) | partial_opd (source_verified) | confirmed |
| SCOPE | borderline_strict (source_verified) | borderline_strict (source_verified) | confirmed |
| KDRL | borderline_strict (source_verified) | borderline_strict (source_verified) | confirmed |
| Lightning OPD | partial_opd (source_verified) | partial_opd (source_verified) | confirmed |
| DistiLLM-2 | partial_opd (source_verified) | partial_opd (source_verified) | confirmed |

**Result: All 15 labels confirmed. No reclassifications needed.**

---

## Verification methodology

1. **Primary-source fetching:** Five independent agents each fetched arXiv HTML full text for 3 methods (Uni-OPD/MAD-OPD/VOLD, MiMo-V2-Flash/Gemma2/G-OPD, REOPOLD/PACED/SDPO, OPCD/OEL/SCOPE, KDRL/Lightning-OPD/DistiLLM-2).

2. **Quote extraction:** Each agent located exact section numbers, equation numbers, and algorithm line numbers for C1/C2/C3 evidence.

3. **Conservative rules applied:**
   - Quotes must be locatable in the source; "not found" triggers downgrade (applied to Gemma 2 C3).
   - Reward-only signals default to partial/adjacent (applied to GRPO component of KDRL).
   - Stage-specific labels do not generalize to the full pipeline (applied to MiMo-V2-Flash, PACED, OEL, VOLD).
   - Self-identification as "not on-policy" is authoritative evidence against strict_opd (applied to DistiLLM-2, Lightning OPD).
   - Privileged-context self-teachers are valid C2 teachers per repo definition (applied to SDPO, OPCD, OEL).
   - RL-style notation does not warrant downgrade when the signal is dense teacher logit comparison (applied to REOPOLD, G-OPD).
