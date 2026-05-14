# WP6 Agentic Candidate Line-Level Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML/PDF only. No secondary summaries.

---

## Evidence Ledger

### 1. TCOD — Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents

| Field | Value |
|---|---|
| **method** | TCOD |
| **source_url** | https://arxiv.org/abs/2604.24005 |
| **exact_section** | Eq. 2 (vanilla OPD), Eq. 3 (TCOD-F2B), Eq. 5 (TCOD-B2F), Eq. 4 (curriculum schedule), Section 4.3 (experience replay) |
| **C1 quote** | "Given a teacher policy pi_phi and a student policy pi_theta, the goal of on-policy distillation is to align the student with the teacher under the student's own state distribution." Eq. 2 expectation is over E[tau ~ pi_theta]. Algorithm 1: "Sample a_t ~ pi_theta(.\|h_t)" — student samples actions. |
| **C2 quote** | D_KL(pi_phi(a_t\|h_t) \|\| pi_theta(a_t\|h_t)) computed at every student-visited state h_t. Teacher provides full output distributions at student-visited states. |
| **C3 quote** | Eq. 3 (TCOD-F2B): L = E_{tau~pi_theta}[sum_{t=0}^{k-1} D_KL(pi_phi(a_t\|h_t) \|\| pi_theta(a_t\|h_t))]. **Pure forward-KL distillation. No reward component whatsoever.** No GRPO, no PPO, no scalar reward in student objective. GRPO used only to pre-train the ALFWorld teacher. |
| **teacher_access** | white_box (teacher logits); separate larger model (Qwen2.5-7B teacher for 3B student; Qwen3-30B-A3B for 1.7B/4B students) |
| **state_or_action_granularity** | token-level (per-token KL at each trajectory step) |
| **objective_family** | forward_kl (D_KL(teacher \|\| student)) |
| **rollout_freshness** | current_policy with bounded staleness: sub-trajectory experience replay, Delta_max=2 (experiences up to 2 policy updates old may be reused). Paper: "Delta_max=2 provides an optimal balance between sample efficiency and the strictness of the on-policy constraint." |
| **final_label** | **strict_opd** (TCOD-F2B variant); **strict_opd** (TCOD-B2F variant with caveat) |
| **reason_to_downgrade** | TCOD-B2F uses teacher-generated prefix states (first L-k steps) to initialize student rollouts. However: (a) KL is NOT computed on teacher-prefix steps, (b) gradient does not flow through them, (c) teacher steps "serve only to place the student on the doorstep of success." Replay staleness Delta_max=2 is mild. Neither warrants downgrade from strict. |
| **confidence** | high |

**Benchmarks:** ALFWorld (Valid Seen/Unseen/Hard splits), WebShop, ScienceWorld. TCOD-B2F 7B student achieves 20.66 on ALFWorld Hard vs teacher's 6.61 (surpasses teacher by +14 points). TCOD-F2B Qwen3-4B achieves 29.54 avg across 3 benchmarks vs teacher's 30.28.

**Models:** Teachers: Qwen2.5-7B (GRPO-tuned), Qwen3-30B-A3B-Instruct. Students: Qwen2.5-3B, Qwen2.5-7B, Qwen3-1.7B, Qwen3-4B.

**Key distinction:** TCOD is the only agentic OPD method that uses **no reward signal at all** in the student's training objective. Pure distillation.

---

### 2. LiteGUI — Distilling Compact GUI Agents with Reinforcement Learning

| Field | Value |
|---|---|
| **method** | LiteGUI |
| **source_url** | https://arxiv.org/abs/2605.07505 |
| **exact_section** | Section 3.2 (Guided On-policy Distillation), Eq. 10–12, Appendix A.5 |
| **C1 quote** | "the student first samples an output a_hat_t conditioned on the current input x_t, and the teacher then computes token-level likelihoods for the sampled sequence" (S3.2). "Unlike offline distillation from fixed target responses, our GKD stage is fully on-policy: the student always generates the response that is used for distillation." (A.5). "we set the GKD on-policy probability to lambda=1.0" — 100% on-policy, zero teacher-generated sequences. |
| **C2 quote** | "the teacher distribution is conditioned on pi_T(.\|x_t, g_t) whereas the student policy remains pi_theta(.\|x_t)" (S3.2). Teacher receives oracle ground-truth action set A*_t as privileged context. Three guidance variants: Single-GT, Multi-GT, Most-Matched-GT (best; dynamically selects ground truth closest to student's output via phi_gui matching). |
| **C3 quote** | Eq. 12: L_OPD = E_{y_hat_t ~ pi_theta(.\|x_t)}[sum_j (log pi_theta(y_hat_{t,j}\|x_t, y_hat_{t,<j}) - log pi_T(y_hat_{t,j}\|x_tilde_t, y_hat_{t,<j}))]. "Following GKD (Agarwal et al., 2024), we optimize the student using a reverse-KL-style objective" (S3.2). **Stage 1 only.** |
| **teacher_access** | white_box + privileged_context. Teacher: Qwen3-VL-32B-Instruct with oracle ground-truth valid actions A*_t appended to prompt. Student never sees A*_t. |
| **state_or_action_granularity** | token-level (per-token reverse-KL) |
| **objective_family** | reverse_kl (GKD-style, Stage 1); grpo_reward_only (Stage 2, separate) |
| **rollout_freshness** | current_policy (fresh each iteration, lambda=1.0) |
| **final_label** | **strict_opd** (Stage 1 GKD); Stage 2 GRPO is reward-only RL (adjacent), cleanly separated |
| **reason_to_downgrade** | Stage 2 uses same Qwen3-VL-32B as VLM judge for R_sub reward, but provides only scalar scores (0.0/0.3/0.5/0.8/1.0), not distributional guidance. Stages are strictly sequential (Stage 2 initializes from Stage 1 checkpoint). No method-level downgrade needed because stages are cleanly separable. |
| **confidence** | high |

**Benchmarks:** ScreenSpot-Pro (46.86% 2B, 58.95% A3B), OS-World (13.24% 2B, 22.7% A3B), Lite-Bench (61.76% 2B, 89.26% A3B). LiteGUI-2B surpasses UI-TARS-72B on ScreenSpot-Pro (46.86 vs 38.1).

**Models:** Teacher: Qwen3-VL-32B-Instruct. Students: Qwen3-VL-2B-Instruct, Qwen3-VL-30B-A3B.

**Key distinction:** First paper to apply GKD-style on-policy distillation to GUI agents. Claims "first attempt to apply distillation to the GUI agent domain."

---

### 3. Skill-SD — Skill-Conditioned Self-Distillation for Multi-turn LLM Agents

| Field | Value |
|---|---|
| **method** | Skill-SD |
| **source_url** | https://arxiv.org/abs/2604.10674 |
| **exact_section** | Algorithm 1, Section 3.1–3.3, Eq. 10–14, Appendix F |
| **C1 quote** | Algorithm 1 line 6: "Sample {tau_i}_{i=1}^G ~ pi_{theta_old}^{stu}(.\|x); record log pi_{theta_old}^{stu}". "in Skill-SD, the student generates trajectories under the plain prompt and the teacher re-scores them" (S4.3). |
| **C2 quote** | "the same token sequence is re-scored under the student prompt and the skill-augmented teacher prompt" (Figure 1 caption). Teacher: pi_{theta_bar}^{tea}(.\|x + S(x), y_{<t}). theta_bar synced from theta every iteration (Algorithm 1 line 2). Teacher **never generates its own trajectories** — only re-scores student tokens. |
| **C3 quote** | Eq. 14: L_total = L_GRPO + lambda * L_SDL. Eq. 12 (SDL): L_SDL = (1/N) sum_i sum_t rho_{i,t}^{on} * (e^{-ell_{i,t}} - 1 + ell_{i,t}). Eq. 10: ell_{i,t} = log pi_theta^{stu} - log pi_{theta_bar}^{tea} (log-ratio). **Importance-weighted reverse-KL (k3 estimator).** But: **lambda = 0.001** (at lambda=0.01, paper reports over-regularization). |
| **teacher_access** | privileged_self (same weights theta_bar synced each iteration, skill-augmented prompt). Skills extracted from student trajectories, but summarized by external LLM (Seed1.8, Appendix F). |
| **state_or_action_granularity** | token-level (per-token importance-weighted reverse-KL) |
| **objective_family** | importance_weighted_reverse_kl + grpo (combined) |
| **rollout_freshness** | per_iteration (theta_old refreshed every iteration; theta_bar also synced each iteration) |
| **final_label** | **borderline_strict** |
| **reason_to_downgrade** | The SDL distillation coefficient lambda=0.001 makes distillation a **small regularizer** alongside the dominant GRPO (reward-only RL). The paper's own ablation shows lambda=0.01 causes over-regularization, confirming the distillation signal is secondary. This inverts the SOD pattern (where OPD is the primary signal and GRPO is auxiliary). Additionally, skills are summarized by an external LLM (Seed1.8), adding a mild external dependency. |
| **confidence** | high |

**Benchmarks:** AppWorld (64.9% accuracy, +14.0 over vanilla GRPO), Sokoban (62.5%, +10.9 over vanilla GRPO).

**Models:** Base: Qwen3-4B-Instruct-2507. Self-teacher: same weights + skill-augmented prompt.

**Ablation (Table 2):** On-policy + dynamic teacher = 64.9% (best). Off-policy + dynamic = 42.1%. Off-policy + frozen = 45.6%. This confirms on-policy rollouts and dynamic teacher syncing both matter, even at lambda=0.001.

---

### 4. GUI-SD

| Field | Value |
|---|---|
| **method** | GUI-SD |
| **source_url** | not found |
| **final_label** | **not found in primary search** |
| **reason_to_downgrade** | No paper with this exact name was found on arXiv or OpenReview. If this is a variant name for LiteGUI or another GUI-agent distillation method, it should be resolved via the requesting party. |
| **confidence** | n/a |

---

### 5. OEC — On-policy Expert Corrections

| Field | Value |
|---|---|
| **method** | OEC |
| **source_url** | https://arxiv.org/abs/2512.14895 |
| **exact_section** | Algorithm 1 (p.4), Section 3.1–3.2 |
| **C1 quote** | Algorithm 1: "M <- M^S # Start rollout with student" then "for step in 1...h do: if step = switch then M <- M^E". "OECs are generated by rolling out trajectories using the student model and then, part way through a trajectory, swapping the underlying model to the expert." (S3.1). Student generates turns 1 through (switch-1). |
| **C2 quote** | "On-policy expert corrections (OECs) trajectories mitigate this by querying for expert demonstrations beginning part way through a student trajectory, at which point the history is necessarily on-policy." (S3.1). "When the underlying model...is switched from the student to the expert, we preserve the history of the trajectory" (S3.1). Expert operates on student's history/state. |
| **C3 quote** | Loss is standard NLL/cross-entropy (SFT) on expert tokens only. "We also mask the on-policy (i.e., student) portion of each trajectory so that training is only performed on the expert portion" (S3.2). Plus rejection sampling filtering based on environment reward (unit tests). **No KL divergence, no distributional matching, no teacher logits.** |
| **teacher_access** | expert_policy (separate, larger model) |
| **state_or_action_granularity** | action-level (hard expert labels, not soft distributions) |
| **objective_family** | cross_entropy_on_expert_completions (SFT) |
| **rollout_freshness** | current_policy (student generates fresh each round) |
| **final_label** | **borderline_strict** (DAgger-style imitation learning) |
| **reason_to_downgrade** | C3 fails strict: loss is NLL/SFT on expert completions, not a distributional/KL loss. Student turns are explicitly masked from the loss. The paper self-identifies as "imitation learning" (abstract, S5 conclusion), cites DAgger as central motivation, and explicitly states: "We introduce a novel, partially on-policy data generation technique...to address the problem of covariate shift in imitation learning." Hard action labels (DAgger-style) rather than soft distribution matching (OPD-style). |
| **confidence** | high |

**Benchmarks:** SWE-bench Verified (14% improvement over pure behavioral cloning). Switch distribution: U(0,30).

**DAgger connection:** Explicit and central. "Our data generation method...adapts the principle behind DAgger to LM agents" (S1). Figure 2 labels their method as "(1) DAgger (ours)."

---

### 6. OpenClaw-RL (HG-OPD component)

| Field | Value |
|---|---|
| **method** | OpenClaw-RL (OPD component) |
| **source_url** | https://arxiv.org/abs/2603.10165 |
| **exact_section** | Eq. 1 (OPD loss), hybrid objective formula, Appendix C (KL equivalence claim) |
| **C1 quote** | S_i^q = top-k{pi_old(.\|s_t, y_{<i})} — student's top-k vocabulary at each token position. Importance ratio rho_v = exp(l_cur(v) - l_old(v)) confirms standard on-policy PPO-style rollout-then-update. |
| **C2 quote** | "a teacher distribution pi_T(.\|s_t^h) is obtained by querying the same model under the hint-augmented prompt." Hint h extracted by PRM from next-state s_{t+1}. Teacher evaluates same tokens the student generated: pi_T(.\|s_t^h, y_{<i}). Overlap metric O[h,i] = \|S_i^q ∩ S_{i,h}^p\| compares teacher/student top-k at same positions. |
| **C3 quote** | Eq. 1: L_i^{OPD} = sum_{v in S_i} max(-A_v * rho_v, -A_v * clip(rho_v, 1-eps_lo, 1+eps_hi)), where A_v = Delta_v * w_v, Delta_v = clip(l_{T,h*}(v) - l_old(v), -C, +C). **This is a PPO-style clipped surrogate where the teacher-student log-prob gap is consumed as the advantage signal.** Appendix C titled "OPD Objective is Token-level KL" claims mathematical equivalence but proof was inaccessible in HTML rendering. |
| **teacher_access** | privileged_self (same model weights + PRM-derived textual hint from next state) |
| **state_or_action_granularity** | token-level (per-token per-vocabulary-item advantage) |
| **objective_family** | clipped_surrogate_with_teacher_log_ratio_advantage |
| **rollout_freshness** | current_policy (standard PPO rollout loop) |
| **final_label** | **borderline_strict** (OPD component) |
| **reason_to_downgrade** | (1) The teacher log-prob gap is consumed as an RL-style advantage in a PPO clipped surrogate, not as a direct KL divergence term. This is the same structural pattern as G-OPD/REOPOLD (already classified strict or stage-strict in the repo), but the KL-equivalence proof (Appendix C) could not be verified. (2) Full method also combines with Binary RL (scalar PRM reward): L_hybrid = w_RL * L_GRPO + w_OPD * L_OPD (w_RL=w_OPD=1). The Binary RL and OPD components are additive and separable. (3) Self-teacher (same model + hint), not a separate stronger model. (4) The term "HG-OPD" does not appear in the paper — it uses "OPD" for the distillation component and "OpenClaw-RL" for the overall method. |
| **confidence** | medium |

**Note on precedent:** The repo already classifies G-OPD (arXiv 2602.12125) and REOPOLD (arXiv 2603.11137) as strict_opd, where teacher log-ratios serve as dense token-level supervision in an RL-style objective. If the same precedent applies, OpenClaw-RL's OPD component could be upgraded to strict. However, verifying Appendix C's KL-equivalence proof is needed first.

---

### 7. GLM-5 Cross-Stage OPD

| Field | Value |
|---|---|
| **method** | GLM-5 cross-stage OPD |
| **source_url** | https://arxiv.org/abs/2602.15763 |
| **exact_section** | Section 3.5 |
| **C1 quote** | Method named "on-policy cross-stage distillation." Advantage formula uses pi_{theta_train} as current student. Structural parallel with GRPO rollout loop (Eq. 1) implies student generates on-policy. **No explicit sentence stating "the student generates on-policy rollouts" — inferred from method name, cited algorithm, and formula structure.** |
| **C2 quote** | "the final checkpoints from the preceding training stages serve as teacher models." Token-level log-probability ratio: A^(i,t) = sg[log(pi_{theta_teacher_infer}(y_{i,t}\|x, y_{i,<t}) / pi_{theta_train}(y_{i,t}\|x, y_{i,<t}))]. Teacher provides distributional supervision at token level. |
| **C3 quote** | "The training loss can be obtained by replacing the advantage term in Eq. 1 with the following formula" — teacher log-ratio replaces advantage in GRPO clipped surrogate (same pattern as OpenClaw-RL). Cited reference [28] is likely Agarwal et al. "On-Policy Distillation" (2024), classified as strict_opd in the repo. |
| **teacher_access** | white_box (prior-stage checkpoints as frozen teachers); multi-teacher (Reasoning RL checkpoint + General RL checkpoint) |
| **state_or_action_granularity** | token-level (per-token log-ratio) |
| **objective_family** | teacher_log_ratio_advantage_in_grpo_surrogate |
| **rollout_freshness** | current_policy (implied by "on-policy" naming and GRPO structure); group_size=1 |
| **final_label** | **stage-strict** (industrial, cross-stage distillation phase only) |
| **reason_to_downgrade** | (1) Terse industrial description — no ablation, limited detail. (2) Same RL-advantage consumption pattern as OpenClaw-RL. (3) Not specifically applied to the agentic RL stage — recovers "skills acquired in earlier SFT and RL stages (Reasoning RL and General RL)." (4) No explicit "student generates on-policy" sentence. Parallel to Qwen3 OPD stage classification. |
| **confidence** | medium |

**Context:** "In our multi-stage RL pipeline, sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities. To mitigate this issue, we perform on-policy cross-stage distillation as the final stage." Hyperparameters: group_size=1, batch_size=1024, prompts sampled from corresponding teachers' RL training sets.

---

### 8. RLSD / Self-Distilled RLVR

| Field | Value |
|---|---|
| **method** | RLSD |
| **source_url** | https://arxiv.org/abs/2604.03128 |
| **exact_section** | Eq. 5 (self-teacher), Eq. 14 (per-token weight), Eq. 15 (token advantage), Eq. 16 (RLSD loss), Algorithm 1 |
| **C1 quote** | Algorithm 1 line 6: "Sample G responses {y^(1),...,y^(G)} ~ pi_theta(.\|x)". Unambiguous on-policy generation from current policy. |
| **C2 quote** | Eq. 5: P_T(.\|y_{<t}) := pi_theta(.\|x, r, y_{<t}). Self-teacher = same model + privileged reference answers r. "The ratio P_T(y_t)/P_S(y_t) is therefore an evidence ratio: the factor by which the privileged information revises the model's belief about each token." |
| **C3 quote** | **FAILS.** The teacher signal enters ONLY as a stop-gradient magnitude weight, NOT as a loss term. Eq. 14: w_t = exp(sign(A) · Delta_t) = (P_T(y_t)/P_S(y_t))^{sign(A)}. Eq. 15: A_hat_t = A · ((1-lambda) + lambda · clip(w_t, ...)). Delta_t = sg(log P_T - log P_S) where **sg = stop-gradient**. "The stop-gradient ensures that Delta_t serves purely as a weighting signal and does not introduce auxiliary gradient pathways." |
| **teacher_access** | privileged_self (same weights + reference answers) |
| **state_or_action_granularity** | token-level (magnitude modulation) |
| **objective_family** | reward_rl_with_teacher_magnitude_modulation |
| **rollout_freshness** | current_policy |
| **final_label** | **adjacent** |
| **reason_to_downgrade** | C3 fails: teacher evidence ratio is a stop-gradient weight on reward-based advantage, not a direct loss term. "the environment reward determines the direction of each token's update (reinforcement or penalization), while the teacher's evidence ratio modulates only the magnitude." Paper explicitly rejects direct KL (OPSD) because it causes "information leakage": "L_OPSD = L* + I(Y_t; R \| X, Y_{<t})" where I is conditional mutual information. The teacher is deliberately NOT a distillation target. |
| **confidence** | high |

---

### 9. DGPO — Distillation-Guided Policy Optimization

| Field | Value |
|---|---|
| **method** | DGPO |
| **source_url** | https://arxiv.org/abs/2508.20324 |
| **exact_section** | Eq. 1 (Phase 1 offline KD), Eq. 4 (Phase 2 reward), PPO objective |
| **C1 quote** | Phase 2: E_{x~D, y~pi_old(.\|x; R)} — outputs sampled from student's old policy. "we transition to PPO-based RL using the distilled student as the initial policy." |
| **C2 quote** | "the teacher pi_g guides the student to mimic teacher behavior through KL regularization" — teacher provides distributional evaluation. |
| **C3 quote** | **FAILS.** Phase 2 reward (Eq. 4): r_phi(x,y) = {1 if y=y*; -beta * D_KL[pi_theta(y\|x;R) \|\| pi_g(y\|x;R)] otherwise}. The KL is **converted to scalar reward**, consumed via PPO's GAE → advantages → clipped surrogate. "This can be seen as a form of targeted regularization which allows free exploration during correct predictions but applies corrective guidance through KL penalties when the student fails." No direct KL loss in Phase 2 objective. Phase 1 (Eq. 1: L_distill = L_CE + lambda * D_KL) IS direct distillation but is offline SFT (C1 fails). |
| **teacher_access** | white_box (frozen larger model pi_g) |
| **state_or_action_granularity** | sequence-level (KL computed as scalar reward for PPO) |
| **objective_family** | ppo_with_teacher_kl_reward_shaping |
| **rollout_freshness** | current_policy (Phase 2 PPO) |
| **final_label** | **adjacent** |
| **reason_to_downgrade** | Phase 2 teacher KL is reward shaping (scalar reward consumed by GAE), not a direct distillation loss. Phase 1 is offline KD (C1 fails). Sequential KD-then-RL pipeline with teacher-shaped rewards. |
| **confidence** | high |

---

### 10. RISE — Recursive Introspection for Self-Improvement

| Field | Value |
|---|---|
| **method** | RISE |
| **source_url** | https://arxiv.org/abs/2407.18219 |
| **exact_section** | Eq. 4.6 (training loss), Section describing distillation variant |
| **C1 quote** | "In a given round k, for a given problem x_i, we unroll the current model pi_{theta_k}(.\|.) to produce multiple sequential attempts, denoted by y^i_t ~ pi_{theta_k}(.\|s^i_t)." Explicitly: "D_{on-policy} := {(s^i_t, y^i_t, f^i_t, r^i_t)}." |
| **C2 quote** | Distillation variant: "Perhaps the most straightforward approach is to query an off-the-shelf more capable model to provide a correct response given the prompt x_i, the previous response y^i_t, and an optional external feedback f^i_t." Teacher provides response TEXT at student-visited states. Self-distillation variant: best-of-N from student itself (no external teacher). |
| **C3 quote** | **FAILS.** Eq. 4.6: max_theta E[sum_t log pi_theta(y_tilde^i_t \| s^i_t) · exp(r^i_t / tau)]. This is **reward-weighted SFT** (cross-entropy weighted by exponential reward). No KL divergence term, no distributional matching. "We refer to this as the distillation variant...(**note that this is different from the classic notion of knowledge distillation**)" — paper's own disclaimer. |
| **teacher_access** | expert_policy (off-the-shelf capable model, distillation variant) or none (self-distillation variant) |
| **state_or_action_granularity** | sequence-level (whole response replacement at student states) |
| **objective_family** | reward_weighted_sft |
| **rollout_freshness** | per_iteration (iterative rounds of rollout + fine-tune) |
| **final_label** | **adjacent** |
| **reason_to_downgrade** | C3 fails: reward-weighted SFT is not distributional distillation. Teacher provides target text tokens (hard labels), not distributional signal. Paper explicitly disclaims "different from classic knowledge distillation." Self-distillation variant further fails C2 (no external teacher, just best-of-N student samples). |
| **confidence** | high |

---

### 11. SAD — Structured Agent Distillation

| Field | Value |
|---|---|
| **method** | SAD |
| **source_url** | https://arxiv.org/abs/2505.13820 |
| **exact_section** | Section 3.2 (Trajectory Segmentation), Section 3.3 Eq. 4–6 (losses), Appendix E.2 (Teacher Trajectory Collection), Appendix H |
| **C1 quote** | **FAILS.** Appendix E.2: "To supervise student agents via Structured Agent Distillation, we first collect high-quality teacher trajectories using GPT-2. **All trajectories are generated offline before student training** to avoid dependence on external API calls during optimization." Section 3.2: "Given a **teacher-generated** trajectory tau, we decompose it into two disjoint spans." Appendix H: "the student is trained via **teacher-forced decoding** using the same token sequence as the teacher." |
| **C2 quote** | Teacher supervises its OWN generated states, not student states. Figure 6 caption: "The input to the student model is a trajectory consisting of both reasoning and action segments **generated by the teacher model**." |
| **C3 quote** | Eq. 4: L_CoT = sum_t m_r(t) KL(p_T \|\| p_S). Eq. 5: L_Act = sum_t m_a(t) KL(p_T \|\| p_S). Eq. 6: L_total = lambda_r * L_CoT + lambda_a * L_Act. Span-level KL is a proper distributional loss, but applied to **teacher-generated trajectories**, not student-generated ones. |
| **teacher_access** | white_box (teacher logits on teacher-generated sequences) |
| **state_or_action_granularity** | token-level with span masking (reasoning vs. action spans) |
| **objective_family** | forward_kl_on_teacher_trajectories |
| **rollout_freshness** | static (pre-collected offline) |
| **final_label** | **not_opd** |
| **reason_to_downgrade** | C1 fails completely. All trajectories are teacher-generated and pre-collected offline. Student never generates its own rollouts. The paper's use of "online distillation" (Appendix B) means online teacher logit queries, NOT on-policy student generation: "offline trajectories become stale as the student deviates from the teacher's rollout policy...In practice, cached teacher logits make online distillation efficient." Teacher-forced decoding throughout. This is offline KD with structured span-aware losses, not OPD. |
| **confidence** | high |

---

## Summary Classification Table

| # | Method | C1 (Student rollout) | C2 (Teacher on student states) | C3 (Distillation objective) | Final Label | Confidence |
|---|---|---|---|---|---|---|
| 1 | **TCOD** | PASS (student samples actions; E[tau~pi_theta]) | PASS (token-level forward-KL at student-visited states) | PASS (pure forward-KL, no reward) | **strict_opd** | high |
| 2 | **LiteGUI** (Stage 1) | PASS (lambda=1.0, fully on-policy) | PASS (Qwen3-VL-32B + oracle actions on student sequences) | PASS (reverse-KL, GKD-style) | **strict_opd** | high |
| 3 | **Skill-SD** | PASS (per-iteration theta_old) | PASS (same weights + skills, re-scores student tokens) | PASS but WEAK (importance-weighted reverse-KL at lambda=0.001 — small regularizer alongside dominant GRPO) | **borderline_strict** | high |
| 4 | **GUI-SD** | — | — | — | **not found** | n/a |
| 5 | **OEC** | PASS (student generates first K turns) | PASS (expert completes from student state) | FAIL (NLL/SFT on expert tokens, student turns masked; no KL) | **borderline_strict** (DAgger-style IL) | high |
| 6 | **OpenClaw-RL** (OPD component) | PASS (PPO-style on-policy) | PASS (self-teacher = same model + hint, re-scores student tokens) | BORDERLINE (teacher log-ratio as advantage in PPO surrogate; Appendix C claims KL equiv., unverified) | **borderline_strict** | medium |
| 7 | **GLM-5** (cross-stage OPD) | PASS (implied by method name + formula) | PASS (token-level teacher log-ratio) | BORDERLINE (same pattern as OpenClaw-RL; cites Agarwal et al. OPD algorithm) | **stage-strict** (industrial) | medium |
| 8 | **RLSD** | PASS (current policy) | PASS (self-teacher = same model + reference answers) | FAIL (stop-gradient magnitude weight, not loss term; "teacher is magnitude evaluator, not distillation target") | **adjacent** | high |
| 9 | **DGPO** | PASS (Phase 2 PPO) | PASS (teacher KL evaluation) | FAIL (teacher KL → scalar reward → GAE → PPO advantage; not direct loss) | **adjacent** | high |
| 10 | **RISE** | PASS (multi-turn student rollouts) | PARTIAL (teacher provides text, not distributions) | FAIL (reward-weighted SFT; paper disclaims "different from classic KD") | **adjacent** | high |
| 11 | **SAD** | FAIL ("All trajectories generated offline before student training"; teacher-forced) | FAIL (teacher supervises own states) | C3 form OK (span KL) but on teacher data | **not_opd** | high |

---

## Promotion / Downgrade Recommendations

### Promote to strict_opd (needs table row):
1. **TCOD** — Pure forward-KL distillation for multi-turn agents. No reward. Strongest agentic OPD seed. Add to `tables/opd_papers.md`.
2. **LiteGUI** (Stage 1) — GKD-style reverse-KL for GUI agents. First GUI OPD. Add to `tables/opd_papers.md`.

### Keep as borderline_strict:
3. **Skill-SD** — Distillation signal exists but lambda=0.001 makes it a regularizer. Reward-RL dominant. Do not promote to strict unless distillation coefficient impact is reanalyzed.
4. **OEC** — DAgger-style IL with hard action labels, not distributional OPD. Important structural ancestor.
5. **OpenClaw-RL** (OPD component) — Same teacher-log-ratio-as-advantage pattern as G-OPD/REOPOLD. Could follow their precedent to strict, but Appendix C proof needed first.

### Keep as stage-strict (industrial):
6. **GLM-5** — Cross-stage OPD for general/reasoning recovery. Terse but consistent with existing industrial OPD stage classifications.

### Confirm adjacent:
7. **RLSD** — Conflict resolved: teacher is stop-gradient magnitude modulator, not distillation target.
8. **DGPO** — Teacher KL is reward shaping via GAE, not direct loss.
9. **RISE** — Reward-weighted SFT, not distributional distillation.

### Confirm not_opd:
10. **SAD** — Definitively offline teacher-forced KD. Legacy misclassification corrected.

### Unresolved:
11. **GUI-SD** — Name not found. Clarify if this is an alias for LiteGUI or a separate method.
