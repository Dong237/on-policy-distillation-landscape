# WP7 Candidate Line-Level Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML/PDF only. No secondary summaries.

---

## 1. Memo: Promote / Keep Candidate / Adjacent / Not Found

### Promote to strict: NONE
No Priority A candidate has enough verified C1/C2/C3 evidence to promote to strict_opd.

### Keep as borderline_strict candidate (need code/author confirmation):

**KEPO** — Mathematical structure of Eq. 7 strongly implies token-level KL distillation on student-generated medical VLM trajectories, but the core divergence D(π_T ‖ π_θ) is never formally defined (only referenced to GKD). Quality-gated: distillation fires only on reward-positive trajectories. Title says "Preference Optimization" but no preference mechanism exists — the method is GRPO + gated KL distillation. **Promote only when code confirms token-level KL on student sequences.**

**D-OPSD** — Velocity-field regression on student-generated denoising trajectories with stop-gradient EMA teacher. Paper honestly admits "this objective is not a token-level KL divergence." On-policy in data-generation sense (student generates its own x_t states), but stop-gradient prevents distributional optimization. **Keep as borderline_strict for diffusion-model OPD extension; do not promote without differentiable-trajectory evidence.**

**Flow-OPD** — Per-step KL→L2 derivation is mathematically correct for local Gaussian transitions, but the derived reward is consumed via PPO (stop-gradient on student velocity, gradients only through policy ratio). This is RL with a teacher-derived dense reward, architecturally in the DDPO/DPOK family. **Keep as borderline_strict; the KL-equivalence is valid per-step but the RL consumption pattern makes it structurally different from direct KL minimization.**

### Downgrade to adjacent:

**GTR-Turbo** — Definitively resolved. The KL variant computes RevKL(student‖teacher) on student-generated thought tokens but averages it into a scalar and stores it as an "auxiliary reward" in the PPO reward buffer: `B ← B ∪ (o_t, a_t, r_t − β·RevKL(...), o_{t+1})`. No KL loss term appears in the PPO objective. The SFT variant trains on teacher-generated thoughts (not student-generated). **Downgrade from borderline_strict to adjacent.**

### Priority B metadata confirmations:

**Video-OPD** — Teacher confirmed as Qwen3-VL-32B (GRPO post-trained). Loss confirmed as reverse-KL D_KL(student‖teacher) via Eq.5 and Eq.11. TVDF filters trajectories before distillation. No changes needed.

**X-OPD** — White-box token-level supervision confirmed. Two advantages: in-modal A_im (Eq.1) and cross-modal A_cm (Eq.2), combined via λ weighting (Eq.5). Teacher: Qwen3-A3B; student: Qwen3-Omni-A3B. No tokenizer mismatch (shared output vocabulary). No changes needed.

**Uni-OPD** — Margin calibration confirmed: margin mask (Eq.10, discards groups with m(q) < δ) and margin shift (Eq.11, additive correction λ(q) = δ − m(q) to positive trajectories). Multi-teacher via Eq.4: J = Σ wᵢ D_KL(π_θ‖π_Tᵢ). 9 MLLM benchmarks confirmed: MathVision, DynaMath, WeMath, LogicVista, VisuLogic, AI2D, ChartQA, DocVQA, InfoVQA. No changes needed.

**PRISM** — Discriminator output confirmed as scalar reward → GRPO advantages. Eq.1: r(x,y) = α·D_v + (1−α)·D_r. Eq.3: normalized into GRPO advantages. No distributional signal in the loss. Existing borderline_strict classification is generous but defensible (adversarial OPD claim is well-grounded, but mechanism is reward-shaped). No changes needed.

**Qwen3-VL** — **CRITICAL FINDING.** On-policy distillation IS confirmed in Section 4.3, including "minimizing the KL divergence" between student and teacher logits on student-generated sequences. HOWEVER, the paper explicitly states: "Crucially, we perform this distillation using **text-only data** to fine-tune the **LLM backbone**." The multimodal components (vision encoder, MLP merger, multimodal tasks) receive only reward-only RL (SAPO). **Upgrade confidence from medium to high. Classification remains adjacent for the VLM — strict OPD exists only for the text backbone.**

---

## 2. Evidence Ledger

### Priority A Candidates

| method | source_url | exact_section | modality | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence | suggested_repo_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KEPO | https://arxiv.org/abs/2602.00400 | Eq. 7, Algorithm 1, S3.2.1 | Medical VLM | Algorithm 1: "Sample {y_i}_{i=1}^G ~ pi_theta(.\|x)". Student generates all trajectories on-policy. | Eq. 7: D(pi_T \|\| pi_theta)(y_i \| x) term structurally requires teacher distributions on student-generated trajectory y_i. Referenced to GKD [1]. Quality gate: I_{r_i >= tau} selects which trajectories receive teacher supervision. | Eq. 7: J_KEPO = E[sum w_i A_hat_i − I_{r_i>=tau} D(pi_T \|\| pi_theta)(y_i\|x) − beta D_KL(pi_theta \|\| pi_ref)]. D is **never formally defined** — only referenced to GKD. No equation for D. No mention of "logits", "softmax", or "temperature" in teacher-student interaction. | white_box (implied by GKD reference; Qwen3-VL-32B teacher, Qwen3-VL-2B student) | likely token (by GKD analogy) | grpo + gated_kl_distillation (by reference) | current_policy | **borderline_strict** | C3 divergence D formally undefined — only referenced to GKD. No explicit confirmation of teacher logit computation. Title says "Preference Optimization" but no preference mechanism exists. | medium | Keep in candidate queue. Promote to strict only when code confirms token-level KL on student sequences. First medical VLM OPD candidate. |
| D-OPSD | https://arxiv.org/abs/2605.05204 | Eq. 7, S2.2, Algorithm 1 | Text-to-image diffusion | Algorithm 1: x_{tK}^s ~ N(0,I), then student runs ODE solver for K steps. "optimization is always performed on the student's actual roll-outs" (S2.2). | Teacher = EMA copy with privileged multimodal conditioning c_t = f_mm(y, x_0). Evaluates student's EXACT states: u_k^t = v_{theta_bar}(x_{tk}^s, t_k, c_t). | Eq. 7: L = E[(1/K) sum \|\|u_k^s − sg(u_k^t)\|\|_2^2]. Paper explicitly states: "Although this objective is **not a token-level KL divergence**, it serves the same role in our setting." **No KL proof.** Stop-gradient on teacher AND on trajectory states. | privileged_self (same model + target image via EMA; momentum ~0.9999) | step-level (denoising steps) | mse_velocity_field (NOT kl) | current_policy (student generates own x_t) | **borderline_strict** | (1) Paper admits loss is NOT KL. (2) Stop-gradient on trajectory prevents distributional optimization — functionally regression on on-policy data. (3) No formal MSE↔KL proof. | medium | Keep in candidate queue for diffusion OPD extension. Do not promote without differentiable-trajectory evidence or formal KL equivalence. |
| Flow-OPD | https://arxiv.org/abs/2605.08063 | Eq. 5, 8-12, S5.1.1 | Text-to-image flow matching | SDE (Eq.5): student generates stochastic denoising paths with noise injection. "The fundamental premise of Flow-OPD requires the student to expose its own specific distribution shifts" (S5.1.1). | Four domain-specialized teachers (GenEval, OCR, DeQA, PickScore) provide velocity fields v_{phi_k} at student-generated states. Hard routing: one teacher per sample via k=R(c). | Per-step KL→L2 (Eq.8-9): D_KL = \|\|mu_theta − mu_target\|\|^2 / (2σ²Δt) — **correct for local Gaussian transitions**. BUT consumed via PPO: r_t^OPD = −w(t)\|\|v̄_θ − v_target\|\|² is stop-gradiented and fed as dense reward into clipped surrogate (Eq.11). + MAR L2 regression regularizer (Eq.12). | white_box (multi-teacher velocity fields; 4 specialized teachers + aesthetic anchor) | step-level (per-denoising-step) | ppo_with_dense_kl_derived_reward + l2_regression_regularizer | current_policy (SDE stochastic paths) | **borderline_strict** | (1) Per-step KL→L2 is correct but trajectory-level KL not proven. (2) KL-derived reward consumed via PPO (RL-style), not direct loss. (3) Stop-gradient on student velocity. (4) Hybrid: PPO + L2 regression mixes paradigms. | medium | Keep in candidate queue. Same teacher-log-ratio-as-advantage pattern as OpenClaw-RL but in diffusion domain. |
| GTR-Turbo | https://arxiv.org/abs/2512.13043 | Algorithm 2, Eq. 6, S3.4 | VLM agent | Algorithm 2 line 8: "Generate (th_t, a_t) using pi_theta_k given o_t". Student generates thoughts + actions in visual environments (Points24, ALFWorld, Android-in-the-Wild). | Teacher = merged checkpoint via TIES merging of historical RL checkpoints. RevKL(pi_theta, pi_merged; th) computed on student-generated thought tokens (Eq.6). | **KL consumed as scalar reward in PPO.** S3.4: "we compute the reverse KL...average over all tokens, and take its negative value as an auxiliary reward for PPO updates." Algorithm 2 line 11: B ← B ∪ (o_t, a_t, **r_t − β·RevKL(...)**, o_{t+1}). No KL loss term in objective. PPO advantage: A' = A^pi − RevKL. | privileged_self (merged RL checkpoints via TIES) | token-level KL computed but collapsed to scalar per step | kl_as_ppo_reward_shaping | current_policy | **adjacent** | KL is averaged into scalar reward and stored in PPO reward buffer. Paper calls it "auxiliary reward." Same pattern as DGPO (KL→scalar→GAE→PPO). SFT variant trains on teacher-generated thoughts (not student-generated). | high | **Downgrade** from borderline_strict to adjacent. Add to adjacent_work.md as "kl_reward_shaped_agentic_rl". |

### Priority B Metadata Verification

| method | source_url | exact_section | field_checked | verified_value | previous_value | change_needed | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Video-OPD | https://arxiv.org/abs/2602.02994 | S4.1, Eq. 5, Eq. 11 | teacher_identity | "Qwen3-VL-32B post-trained with GRPO is employed as the teacher model" | frontier teacher | Yes — update teacher_model field | high |
| Video-OPD | https://arxiv.org/abs/2602.02994 | Eq. 5, Eq. 11 (Appendix A) | exact_loss | Eq. 5: r_t = −(log π_θ(a_t\|s_t) − log π_tea(a_t\|s_t)). Eq. 11 proves: E[r_t · ∇log π_θ] = ∇D_KL(π_θ \|\| π_tea). **Reverse-KL: D_KL(student \|\| teacher).** | reverse_kl | No (confirmed) | high |
| Video-OPD | https://arxiv.org/abs/2602.02994 | S3.3 | tvdf_mechanism | Stage 1: Teacher reliability pre-validation via mean IoU. Stage 2: Disagreement prioritization via δ_i = τ_i − σ_i. Filters trajectories BEFORE distillation. | not documented | Yes — add to notes | high |
| X-OPD | https://arxiv.org/abs/2603.24596 | Eq. 1-5 | teacher_access | White-box. Teacher: Qwen3-A3B-Instruct (text-only). Student: Qwen3-Omni-A3B-Instruct (speech). Shared output vocabulary. | white_box | No (confirmed) | high |
| X-OPD | https://arxiv.org/abs/2603.24596 | Eq. 1-5 | token_level_loss | Token-level. A_im(y_t) = log π_φ(y_t\|T,y_{<t}) − log π_θ(y_t\|T,y_{<t}). A_cm(y_t) = log π_φ(y_t\|T,y_{<t}) − log π_θ(y_t\|S,y_{<t}). Combined: L = λ·L_im + (1−λ)·L_cm. | token | No (confirmed) | high |
| Uni-OPD | https://arxiv.org/abs/2605.03677 | Eq. 4, 5, 9-11 | margin_calibration | Eq.9: m(q) = min_{τ∈S+} G_OPD − max_{τ∈S−} G_OPD. Eq.10: margin mask keeps groups with m(q) ≥ δ. Eq.11: margin shift λ(q) = δ − m(q), applied to positive trajectories only. | not documented in detail | Yes — add equations to notes | high |
| Uni-OPD | https://arxiv.org/abs/2605.03677 | Eq. 4, Table 2 | multi_teacher | Eq.4: J = Σ wᵢ D_KL(π_θ \|\| π_Tᵢ). Supports single/multi/cross-modal. MLLM teachers: Qwen3-VL-4B-Instruct-Code-RL and Qwen3-VL-4B-Instruct-Math-RL. | multi_teacher | No (confirmed) | high |
| Uni-OPD | https://arxiv.org/abs/2605.03677 | Table 2 | mllm_benchmarks | 9 benchmarks: MathVision, DynaMath, WeMath (math); LogicVista, VisuLogic (logic); AI2D, ChartQA, DocVQA, InfoVQA (document). Students: Qwen3-VL-2B/4B. | 5 domains; 16 benchmarks (generic) | Yes — enumerate MLLM benchmarks explicitly | high |
| PRISM | https://arxiv.org/abs/2604.28123 | Eq. 1-4 | discriminator_mapping | Eq.1: r(x,y) = α·D_v + (1−α)·D_r → scalar. Eq.2: Bradley-Terry ranking loss for discriminator training. Eq.3: Normalized into GRPO advantages A_i. Eq.4: Minimax with policy consuming discriminator as reward. **Confirmed: scalar reward → GRPO advantages. No distributional signal in the loss.** | discriminator → response-level | No (confirmed — already documented) | high |
| Qwen3-VL | https://arxiv.org/abs/2511.21631 | S4.3, p.9, p.11 | multimodal_opd_evidence | **On-policy distillation confirmed but explicitly text-only on LLM backbone.** S4.3: "On-policy Distillation: the student model generates responses...align logits by minimizing KL divergence." p.9: "Crucially, we perform this distillation using **text-only data** to fine-tune the **LLM backbone**." Multimodal tasks get reward-only RL (SAPO, S4.4). | "Public evidence does not disclose strict OPD training loop" (medium confidence) | Yes — upgrade to "OPD confirmed for text backbone only; multimodal is reward-only RL" (high confidence) | high |

---

## 3. Candidate Table-Row Suggestions

### KEPO (add to `tables/vlm_opd_papers.md` as borderline_strict candidate)

```
id: kepo-2026
title: KEPO: Knowledge-Enhanced Preference Optimization for Medical VLM
year: 2026
paper_url: https://arxiv.org/abs/2602.00400
modality: VLM
domain: medical_visual_reasoning
opd_strictness: borderline_strict
teacher_kind: larger_vlm (Qwen3-VL-32B)
student_model: Qwen3-VL-2B
rollout_source: student_on_policy
supervision_granularity: likely token (by GKD analogy)
loss_objective: grpo_plus_gated_kl_distillation
strictness_evidence: C1 student generates on-policy; C2 D(pi_T||pi_theta) on student trajectories (by GKD reference); C3 distillation term in Eq.7 but D formally undefined
benchmarks: OmniMedVQA (8 imaging modalities: MRI, CT, X-Ray, US, Dermoscopy, Fundus, OCT, Microscopy)
confidence: medium
notes: First medical VLM OPD candidate. D never formally defined. Promote when code confirms token-level KL.
vision_task: medical_visual_reasoning
visual_input_type: medical_images
```

### D-OPSD and Flow-OPD — NOT ready for table rows

Both need resolution of MSE↔KL equivalence question and stop-gradient impact. Keep in candidate queue only. These represent a genuinely new modality axis (text-to-image generation) but the OPD taxonomy needs a decision on whether velocity-field MSE qualifies as distributional distillation.

---

## 4. Downgrade Notes

### GTR-Turbo → adjacent

**Add to `tables/adjacent_work.md`:**
```
id: gtr-turbo-2025
title: GTR-Turbo: Merged Checkpoint as Free Teacher for Agentic VLM Training
year: 2025
link: https://arxiv.org/abs/2512.13043
category: kl_reward_shaped_agentic_rl
why_adjacent: KL variant computes RevKL on student thought tokens but averages into scalar and stores as PPO "auxiliary reward." SFT variant trains on teacher-generated (not student-generated) thoughts.
why_not_strict_opd: KL is consumed as reward shaping inside PPO, not as a direct loss term. Same pattern as DGPO. No separate KL loss in the objective.
evidence_url: https://arxiv.org/abs/2512.13043
confidence: high
notes: Merged-checkpoint teacher is architecturally interesting (TIES merging). Downgraded after line audit confirmed KL→scalar→PPO pathway.
```

### D-OPSD and Flow-OPD — keep as candidates, do NOT add to adjacent table

These are genuine on-policy methods (student generates denoising trajectories). They are NOT adjacent in the traditional sense (reward-only RL or offline KD). They represent a boundary case where:
- The teacher signal IS distributional (velocity fields, not scalar rewards)
- The consumption mechanism is either regression with stop-gradient (D-OPSD) or RL with dense KL-derived reward (Flow-OPD)
- Neither uses an explicit KL loss term in the autoregressive sense

**Recommended taxonomy decision:** Create a "diffusion/flow OPD" subsection in `papers/multimodal-frontiers.md` noting this is an active boundary question. The velocity-field MSE ↔ distributional KL connection is mathematically grounded but not equivalent to the token-level KL that defines strict OPD in the LLM setting.

### Qwen3-VL — update metadata, keep as adjacent

Update `tables/adjacent_work.md` entry:
- Change `why_adjacent` to: "On-policy distillation with KL loss is confirmed for the text-only LLM backbone (Section 4.3). However, the paper explicitly states: 'we perform this distillation using text-only data to fine-tune the LLM backbone.' Multimodal components receive reward-only RL (SAPO)."
- Change `confidence` from `medium` to `high`.
