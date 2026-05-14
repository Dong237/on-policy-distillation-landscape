# WP1 Candidate Queue: Line-Level Primary-Source Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML full text for all 14 methods
**Methodology:** Four independent research agents fetched primary-source HTML for 3-4 methods each, extracted exact quotes, assessed C1/C2/C3. Results reconciled against `taxonomies/strict-opd-definition.md`.

---

## Scope

This audit covers the 14 WP1 white-box OPD candidates pending line audit:

> vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens, DSKD v2, SelecTKD, ToDi, HPD

Methods already audited in `wp1-line-audit-claude.md` or `wp9-line-audit-claude.md` are excluded.

## OPD Test

Strict OPD requires all three conditions from `taxonomies/strict-opd-definition.md`:

- **C1 (Student rollout):** The current student/policy generates the training trajectory or prefix.
- **C2 (Same-rollout teacher-style supervision):** A teacher, reference model, privileged self-view, or projected teacher distribution supervises that exact student-generated state.
- **C3 (Objective consumption):** The training objective consumes the teacher-style signal as a distributional distillation objective, not just a scalar reward.

---

## Memo

### Action summary

| Action | Methods |
|---|---|
| **Confirm strict_opd** (10) | vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens |
| **Confirm partial_opd** (2) | DSKD v2, HPD |
| **Confirm adjacent** (2) | SelecTKD, ToDi |

### Key findings

1. **vOPD** is textbook reverse-KL OPD. The control variate V(c_t) = -D_KL(pi_theta || pi_T) is purely a variance-reduction technique on a standard OPD objective; it requires no additional critic or inference.

2. **TIP** applies a Soft-OR token selection layer (s_t = h_hat + delta_hat - h_hat*delta_hat) ON TOP of an unmodified reverse-KL OPD base. Selection does not alter the OPD nature of the underlying loss. 50% token selection matches full training performance.

3. **AOPD** splits tokens by advantage sign: reverse-KL policy gradient for positive-advantage tokens, forward-KL for non-positive-advantage (where teacher assigns higher probability than student). Both branches are distributional objectives from teacher logits on student states.

4. **StableOPD** diagnoses truncation collapse in OPD and stabilizes via golden-data mixture + reference KL(pi_theta||pi_ref) regularization. Rollout freshness is technically "mixed" (student rollouts + golden SFT data in same minibatch), but the on-policy distillation core is standard.

5. **CaOPD** is a privileged-context self-teacher OPD. The teacher pi_theta(.|x,z) uses privileged context z (ground truth, verifier feedback); the student sees only x. Calibration replaces raw teacher confidence with empirical target mu_hat, preserving the KL structure.

6. **SOD** uses step-level divergence reweighting w_k on top of reverse-KL OPD, combined additively with GRPO. The OPD component independently satisfies C1/C2/C3; GRPO is additive, not replacing the distributional loss.

7. **SimCT** is the first strict cross-tokenizer OPD. Minimal aligned units U_SimCT = (V_T intersect V_S) union A ensure teacher distributions are evaluated on student-generated text states after alignment. Theorem 1 guarantees finest boundary-consistent partition.

8. **OPSDL** is a privileged-self OPD for long-context models. Short-context self-teacher pi_theta(.|C_S,Q) supervises long-context student pi_theta(.|C_L,Q) on the same generated tokens. Point-wise reverse KL via policy gradient.

9. **DP-OPD** wraps standard GKD-style OPD in DP-SGD. The privacy noise is applied to student gradients only; the frozen teacher provides logits on fresh student rollouts. The OPD loop is unaffected by the privacy mechanism. Best config uses lambda=0.5 (50% on-policy, 50% off-policy).

10. **Rock Tokens** applies gradient freeze (w=0) on persistent high-loss tokens ON TOP of a standard reverse-KL OPD loop. Rock score R(v) = mean_loss * freq identifies ~18% persistent outliers. Causal knockout analysis validates the mechanism. The base OPD loop is clean.

11. **DSKD v2** is primarily a cross-tokenizer distillation framework. An on-policy mode exists (Section III-C, Eq. 18: y-hat ~ q_theta) but is described as an optional extension. All main experiments use fixed offline datasets (~11k samples). Label partial_opd because on-policy mode is formally described but secondary.

12. **SelecTKD** claims on/off-policy compatibility but ALL experiments use fixed offline datasets (UltraChat200k, MetaMathQA). No student rollouts are generated. The propose-and-verify mechanism operates on dataset tokens. Label adjacent.

13. **ToDi** is purely off-policy. The per-token sigmoid-weighted FKL/RKL blend is a divergence-design innovation, not an on-policy innovation. Training uses fixed databricks/dolly-15k dataset. The paper never mentions "on-policy" in relation to its own method. Label adjacent.

14. **HPD** explicitly self-identifies as off-policy-first with "lightweight approximation of on-policy next-token sampling" (single token per offline prefix position, not full rollouts). Appendix A states: "full rollouts... are necessary to further enhance distilled model performance." HPD + OPD experiments (Section 5.4) use HPD as initialization for a separate OPD stage, not as OPD itself. Label partial_opd.

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | rollout_freshness | objective_family | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vOPD | https://arxiv.org/abs/2605.07865 | S2.1 Eq.1-3, S3.1 Eq.9-10 | "OPD samples from the student during generation to obtain an unbiased estimator of the KL" (S2.1) | "per-token reward r_t = log pi_T(y_t\|c_t) - log pi_theta(y_t\|c_t)" (S2.1 Eq.3) | "trains the student by minimizing the reverse KL divergence between the student and the teacher" (S2.1); V(c_t) = -D_KL is closed-form control variate (Eq.9) | white_box | current_policy | reverse_kl | strict_opd | none | high |
| TIP | https://arxiv.org/abs/2604.14084 | S3 Eq.1, S5.1 Eq.5, S6 Eq.7 | "the student generates a rollout y ~ S_theta(.\|x), and the teacher scores each position" (S3) | "delta_t = D_KL(P_S(.\|c_t) \|\| P_T(.\|c_t))" -- teacher distributions score each student-generated position (S3) | "L_TIP = 1/\|T\| sum_{t in T} D_KL(P_S(.\|c_t) \|\| P_T(.\|c_t))" (Eq.7); "All experiments use reverse KL supervision" (S8) | white_box | current_policy | reverse_kl | strict_opd | none; token selection (Soft-OR s_t) is applied ON TOP of OPD, does not alter base loop | high |
| AOPD | https://arxiv.org/abs/2605.06387 | S3 Eq.1-2, S5.1 Eq.7,9-10, S5.2 Eq.11 | "OPD trains a student on its own trajectories with token-level teacher feedback" (S1) | "A_t = sg[log P_T(y_t\|c_t) - log P_S(y_t\|c_t)]" -- teacher log-probs on student-sampled tokens (S3 Eq.1) | "L_AOPD = E[1/\|y\| sum_t (G_t * L_t^FKL + (1-G_t) * L_t^OPD)]" (Eq.10); both branches are distributional objectives from teacher logits | white_box | current_policy | hybrid_kl | strict_opd | none; RL notation but reward is computed from teacher logits on student tokens | high |
| StableOPD | https://arxiv.org/abs/2604.08527 | S2.2 Eq.2, S3.1, S4.1 Eq.5, S4.2 Eq.6 | "OPD addresses this issue by training on student-generated responses y_hat ~ pi_S^theta" (S2.2) | "r_{i,t}^KL = log pi_T(y_hat_{i,t}\|...) - log pi_theta(y_hat_{i,t}\|...)" (S3.1) | "L_Stable-OPD = L_mix + beta_KL * E[KL(s_t)]" where KL(s_t) = D_KL(pi_theta \|\| pi_ref) (Eq.6) | white_box | mixed | reverse_kl | strict_opd | none; golden-data mixture is additive stabilizer, not replacement for on-policy core | high |
| CaOPD | https://arxiv.org/abs/2604.16830 | S2.3 Eq.2, S4.1 Eq.6, S4.2 Eq.7 | "we sample K independent trajectories (a_k, c_k) ~ pi_theta(.\|x)" (S4.1) | "Teacher policy pi_theta(.\|x,z): the same model, but additionally conditioned on a privileged context z" (S2.1) | "L_CaOPD = ... sum D_KL(pi_theta(.\|y_{<t},x) \|\| pi_theta(.\|y_{<t},x,z_tilde))" (Eq.7) | privileged_self | current_policy | forward_kl | strict_opd | none | high |
| SOD | https://arxiv.org/abs/2605.07725 | S3.3, S4.2 Eq.6-7, S4.3 Eq.9-10, Alg.1 | "sample trajectory tau_i from pi_theta_old by interacting with E" (Alg.1) | "The score d_k serves as a lightweight indicator of the local reliability of teacher supervision" (S4.2) | "This objective corresponds to a sampled estimator of the reverse KL divergence between student and teacher" (S3.3); L = L_GRPO + L_OPD_step (Eq.10) | white_box | per_iteration | hybrid_kl | strict_opd | none; OPD component independently satisfies C1/C2/C3; GRPO is additive | high |
| SimCT | https://arxiv.org/abs/2605.07711 | S3.1-3.3, S3.3 Eq.8-9, S3.4 Thm.1 | "at each step the student samples a prefix under its current policy, the teacher is queried on this prefix" (S4.1) | "we target the white-box OPD setting in which teacher next-token distributions are available" (S5) | "L_SimCT = D(q_S^SimCT(.\|x_{<t}), q_T^SimCT(.\|x_{<t}))" with D = reverse KL (Eq.9); "applies the standard reverse-KL OPD loss" | white_box | current_policy | cross_tokenizer_kl | strict_opd | none; Theorem 1 guarantees aligned units are valid supervision interface on student states | high |
| OPSDL | https://arxiv.org/abs/2604.17535 | S3.3 Eq.2-3 | "for a response y sampled from the long-context student policy pi_theta(.\|C_L,Q)" (S3.3) | "the same model parameters theta are used under the short context, forming a self-referential teacher-student structure" (S3.3) | "unbiased estimator for the gradient of the point-wise reverse KL divergence" (S3.3); A_t = log(pi_Teacher/pi_theta) (Eq.2-3) | privileged_self | current_policy | reverse_kl | strict_opd | none; short-context self-teacher supervises same generated tokens from long-context student | high |
| DP-OPD | https://arxiv.org/abs/2604.04461 | S3.2 Eq.1, S3.3 Eq.2-4, Alg.1 | "With probability lambda, we perform an on-policy step by sampling student continuations: y^i ~ S_theta(.\|x_i)" (S3.2 Eq.1) | "the student is trained to agree with the teacher on the trajectories it actually visits" (S3.3) | "l_i(theta) = L_GKD(p_S, p_T; beta)" (S3.3 Eq.4); GKD divergence parameterized by beta | white_box | current_policy | jsd | strict_opd | none; DP-SGD noise on student gradients does not alter OPD loop | high |
| Rock Tokens | https://arxiv.org/abs/2605.09253 | S2.1 Eq.1-2, S2.2 Eq.4, S4.2, S5.1 | "x_{1:T} ~ pi_theta" (Eq.1); "the student samples 4 rollouts each prompt" (S5.1 Stage 2) | "aligning a student model with a superior teacher through trajectories sampled from its own policy" (S1) | "L_OPD = E_{x~pi_theta}[sum_t D_KL(pi_theta(.\|x_{<t}) \|\| pi_T(.\|x_{<t}))]" (S2.1 Eq.1); "pure reverse KL (kd_ratio=1.0)" (S5.1) | white_box | per_iteration | reverse_kl | strict_opd | none; gradient freeze w=0 on rock tokens is applied ON TOP of standard OPD | high |
| DSKD v2 | https://arxiv.org/abs/2504.11426 | S III-A Eq.3-13, S III-B Eq.14, S III-C Eq.18 | "y-hat is sampled from q_theta(.\|x)" (S III-C Eq.18) -- on-policy mode described as optional extension | "use a linear projector W^{t->s} to transform the hidden states of the teacher model into the representation space of the student" (S III-A) | "L^op_kd = 1/\|y-hat\| sum D(p(y_i\|y-hat_{<i},x) \|\| q_theta(y_i\|y-hat_{<i},x))" (S III-C Eq.18) | white_box | unclear | cross_tokenizer_kl | partial_opd | on-policy mode is optional extension; all main experiments use fixed offline datasets (~11k); primary contribution is cross-tokenizer alignment, not on-policy loop | medium |
| SelecTKD | https://arxiv.org/abs/2510.24021 | S4 Eq.6, S4.1.2 Eq.9, S4.2 Eq.13 | not found -- paper claims on/off-policy compatibility but ALL experiments use fixed offline datasets (UltraChat200k, MetaMathQA) | "the student proposes its most likely token... We then check whether this token is among the teacher's top-k candidates" (S4.1.2) -- on offline sequences only | "L_SelecTKD = sum_t V_t * D(p_t \|\| q_t)" (Eq.6); weighted KL-family divergence | white_box | precomputed | generic_kl | adjacent | C1 fails empirically: no student rollouts in any experiment; propose-and-verify on dataset tokens only | high |
| ToDi | https://arxiv.org/abs/2505.16297 | S4 Eq.7-11 | not found -- uses fixed databricks/dolly-15k dataset; paper never mentions "on-policy" for its own method | not found for OPD purposes -- teacher supervises dataset states, not student-generated states | "D_ToDi = alpha * D_FKL + (1-alpha) * D_RKL" (Eq.7); per-token sigmoid-weighted blend | white_box | precomputed | sigmoid_blend | adjacent | C1 and C2 both fail: purely off-policy distillation on fixed dataset; novel divergence design, not on-policy innovation | high |
| HPD | https://arxiv.org/abs/2604.20244 | S4.1 Eq.9, S4.2 Eq.11-15, Alg.1, App.A | C1 WEAK: "we let the student sample a different token: a_t ~ q_theta(.\|s_t), given the ground-truth offline prefix s_t" (S4.2) -- single token per position, not full rollouts | teacher evaluates p(a_t\|s_t) on student-sampled token via k_1 estimator (S4.2) | "L_HPD = E[-w_t^e log q_theta(a_t*\|s_t) - w_t^s log q_theta(a_t\|s_t)]" (Eq.15); weights from teacher-student log-ratio | white_box | mixed | hybrid_kl | partial_opd | self-identifies as off-policy-first; Appendix A: "full rollouts are necessary to further enhance"; single-token sampling is not full on-policy generation | high |

---

## Classification summary by objective family

| Objective family | Strict methods | Partial/Adjacent |
|---|---|---|
| Reverse KL (standard or point-wise) | vOPD, TIP, Rock Tokens, OPSDL | -- |
| Hybrid KL (asymmetric / step-weighted / + GRPO) | AOPD, SOD, StableOPD | HPD |
| Forward KL (calibrated) | CaOPD | -- |
| JSD (GKD-style) | DP-OPD | -- |
| Cross-tokenizer KL | SimCT | DSKD v2 |
| Sigmoid blend / generic | -- | ToDi, SelecTKD |

---

## Cross-validation with WP1 whitebox ledger predictions

| Method | WP1 prediction | This audit | Match? |
|---|---|---|---|
| vOPD | strict_opd | strict_opd | yes |
| TIP | strict_opd | strict_opd | yes |
| AOPD | strict_opd | strict_opd | yes |
| StableOPD | strict_opd | strict_opd | yes |
| CaOPD | strict_opd | strict_opd | yes |
| SOD | strict_opd | strict_opd | yes |
| SimCT | strict_opd | strict_opd | yes |
| OPSDL | strict_opd | strict_opd | yes |
| DP-OPD | strict_opd | strict_opd | yes |
| Rock Tokens | strict_opd | strict_opd | yes |
| DSKD v2 | partial_opd to strict_opd (mode-dependent) | partial_opd | yes (conservative) |
| SelecTKD | partial_opd to strict_opd | adjacent | DOWNGRADE -- no empirical on-policy evidence |
| ToDi | adjacent until C1 confirmed | adjacent | yes |
| HPD | partial_opd | partial_opd | yes |

**One reclassification:** SelecTKD downgraded from "partial_opd to strict_opd" to adjacent. The paper claims data-regime agnosticism but no experiment generates student rollouts. The propose-and-verify mechanism is tested exclusively on fixed datasets.

---

## Verification methodology

1. **Primary-source fetching:** Four independent agents fetched arXiv HTML for 3-4 methods each (vOPD/TIP/AOPD/StableOPD, CaOPD/SOD/SimCT/OPSDL, DP-OPD/Rock Tokens/DSKD v2, SelecTKD/ToDi/HPD).

2. **Quote extraction:** Each agent located exact section numbers, equation numbers, and algorithm references for C1/C2/C3 evidence.

3. **Decision rules applied:**
   - RL notation accepted when reward/advantage computed from teacher logits on student tokens (vOPD, AOPD, SOD, OPSDL).
   - Token selection/weighting does not prove OPD by itself; base loop verified (TIP, Rock Tokens).
   - Cross-tokenizer alignment satisfies C2 when teacher distribution evaluated on student-generated states after alignment (SimCT passes; DSKD v2 partial because on-policy mode is secondary).
   - Privileged self-teacher satisfies C2 when it supervises same generated tokens (CaOPD, OPSDL pass).
   - Privacy noise does not affect strictness (DP-OPD passes).
   - Downgrade applied when source lacks explicit student rollouts + same-rollout teacher supervision (SelecTKD, ToDi, HPD).
