# WP9 Post-WP8 Verifier Pass

**Verifier:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Scope:** Red-team audit of all strict_opd and borderline_strict rows after WP6/WP7/WP8 merges.

---

## 1. Memo

### Downgrades Recommended (3)

**PRISM** (borderline_strict → **adjacent**): The MoE discriminator outputs a scalar score r(x,y) = α·D_v + (1−α)·D_r, consumed as GRPO advantages via group normalization (Eq.3). The paper explicitly disables KL regularization (KL coefficient = 0.0). This is GAN-like adversarial reward-based RL. No token-level distributional signal exists in the loss. The "on-policy distillation" framing describes reward-shaped policy improvement toward teacher-like behavior, not distributional divergence minimization.

**Qwen3 OPD stage** (strict_opd → **insufficient_evidence**): The Qwen3 technical report contains approximately 2-3 sentences about "distilling output logits from teacher models into lightweight student models." There is no dedicated OPD section, no mention of "on-policy," no KL equation, no description of student generating rollouts for distillation, and no loss function. The three-stage flagship pipeline (Long-CoT Cold Start → Reasoning RL → Thinking Mode Fusion) does not include a named OPD stage. C1/C2/C3 are all unverifiable from the public report. The mention of "logit distillation" is consistent with standard offline KD. Downgrade to `unclear` with low confidence.

**OEC** (borderline_strict → **adjacent**): The expert generates its OWN completions from student-reached states — it does NOT score student tokens with a distributional signal (C2 fails). The loss is NLL/SFT on expert completions with student turns explicitly masked (C3 fails — no KL, reverse-KL, JSD, or any distributional divergence). This is DAgger-style imitation learning, not OPD. The visited-state structure (C1 partial) gives structural similarity to OPD, but the supervision mechanism and objective are categorically different.

### All Other Rows Pass

**Strict rows confirmed (no changes):**
- GUI-SD, MAD-OPD/OPAD, SOD, OPCD, VLA-OPD, VOLD, Video-OPD, X-OPD, Uni-OPD, LiteGUI (Stage 1), MiMo-V2-Flash MOPD, Nemotron-Cascade2 MOPD

**Borderline rows confirmed (no changes):**
- TCOD, Skill-SD, OpenClaw-RL OPD component, KEPO, D-OPSD, Flow-OPD, KDRL, SCOPE, Gemma 2 post-training

### Notable Red-Team Findings

1. **SOD's OPD component is the PRIMARY signal** — ablation shows removing OPD causes a 40% performance drop vs only 9.2% for removing GRPO. This strengthens, not weakens, the strict classification.

2. **GUI-SD's "Distillation-to-SFT Collapse" is a strength** — the authors identified that textual privileged context collapses teacher entropy to 0.17 (near-deterministic), degrading distillation to SFT. Their visual privilege solution maintains teacher entropy at 0.50. This careful engineering supports strict classification.

3. **VOLD needs a caveat** — Stage 2 trains on text-only data. The teacher never sees images. Visual reasoning transfer is emergent at inference, not supervised during training. Still strict for the text modality, but the VLM claim needs qualification.

4. **OpenClaw-RL's Appendix C KL-equivalence claim was not found** — the appendix contains prompt templates, not a formal proof. Borderline is appropriate.

5. **Gemma 2 post-training is the weakest borderline** — entire evidence is a single sentence plus GKD/MiniLLM citations. No loss equation. Retained only because citation-intent is strong.

---

## 2. Verification Table

| method | current_label | verified_label | primary_source_url | exact_section | C1 quote | C2 quote | C3 quote | rollout_freshness | teacher_access | supervision_granularity | objective_family | pass_or_fail | downgrade_reason | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GUI-SD | strict_opd | strict_opd | https://arxiv.org/abs/2605.00642 | S4.1-4.2, Eq.3 | "generates an on-policy trajectory under x" | "privileged information...produces step-wise target distributions along the same trajectory" | L = E[1/\|y\| sum w(t) D_KL(P_S \|\| P_T)]; weighted reverse-KL; no GRPO | current_policy | privileged_visual_self (red bbox + Gaussian mask) | token (coordinate digits) | weighted_reverse_kl | PASS | none | high |
| TCOD | borderline_strict | borderline_strict | https://arxiv.org/abs/2604.24005 | Eq.2-5, S4.3 | "Sample a_t ~ pi_theta(.\|h_t)" (F2B); B2F uses teacher prefix | "D_KL(pi_phi(a_t\|h_t) \|\| pi_theta(a_t\|h_t)) at every student-visited state" | Pure forward-KL; no reward signal | current_policy (Delta_max=2 replay) | white_box (separate larger teacher) | token | forward_kl | PASS | B2F teacher prefix + replay justify borderline, not strict | high |
| Skill-SD | borderline_strict | borderline_strict | https://arxiv.org/abs/2604.10674 | Eq.10-14, Algorithm 1 | "Sample {tau_i} ~ pi_{theta_old}^stu" | "same token sequence re-scored under skill-augmented teacher prompt" | L_SDL = importance-weighted reverse-KL (k3 estimator); lambda=0.001 | per_iteration | privileged_self (skill-conditioned) | token | importance_weighted_reverse_kl + grpo | PASS | lambda=0.001 makes SDL a regularizer; GRPO dominant | high |
| OEC | borderline_strict | **adjacent** | https://arxiv.org/abs/2512.14895 | Algorithm 1, S3.1-3.2 | "Start rollout with student" (partial — expert takes over) | Expert GENERATES own completions, does NOT score student tokens | NLL on expert completions; student turns masked; no KL/distributional loss | current_policy (student prefix) | expert_policy (external LLM) | action (hard labels) | cross_entropy_sft | **FAIL** | C2 fails (expert generates, not scores); C3 fails (NLL/SFT, not distributional) | high |
| OpenClaw-RL OPD | borderline_strict | borderline_strict | https://arxiv.org/abs/2603.10165 | Eq.1, hybrid objective | "This rollout is then scored by both student and hint-conditioned teacher, fixing the same token sequence" | "hint-augmented self-teacher pi_T rescores already-sampled student tokens" | L_OPD uses teacher log-prob gap as PPO advantage; Appendix C KL proof not found | current_policy | privileged_self (hindsight hint) | token | clipped_surrogate_teacher_advantage + grpo | PASS | PPO-advantage consumption; KL proof claim unsubstantiated | medium |
| MAD-OPD / OPAD | strict_opd | strict_opd | https://arxiv.org/abs/2605.01347 | S4.1-4.3, Eq.7-8 | "y_hat ~ pi_theta: student generates trajectories" | "teachers force-decode the student's on-policy tokens with debate transcript" | L = E[sum_t sum_k w_k D(p_Tk \|\| p_S)]; JSD (agentic) / RKL (code) | current_policy (per-step) | multi_teacher_debate | token | confidence_weighted_jsd_or_rkl | PASS | none | high |
| SOD | strict_opd | strict_opd | https://arxiv.org/abs/2605.07725 | Eq.9-10, Table 2 ablation | "student-generated trajectories" | "token-level supervision from teacher" | L_OPD^step + L_GRPO; removing OPD = −40% vs removing GRPO = −9.2% | current_policy | white_box (teacher logits) | token + step-level reweighting | step_weighted_reverse_kl + grpo | PASS | none; OPD is the PRIMARY signal per ablation | high |
| OPCD | strict_opd | strict_opd | https://arxiv.org/abs/2602.12275 | S3, Eq.1, Algorithm 1 | "Sample y ~ pi_theta(.\|x)" without context | "teacher pi_teacher processes [c;x;y]" on same tokens | L = E[D_KL(pi_theta \|\| pi_teacher)]; reverse-KL | current_policy (no replay) | privileged_self (context-conditioned) | token | reverse_kl | PASS | none; strictly on-policy, fresh each step | high |
| VOLD | strict_opd | strict_opd (Stage 2, text-only caveat) | https://arxiv.org/abs/2510.23497 | Stage 2, KL loss term | "VLM student generates on-policy reasoning traces" | "text-only teacher provides token-level KL on same traces" | L_VOLD = L_GRPO + beta * E[D_KL(pi_teacher \|\| pi_student)] | current_policy | white_box (text LLM on VLM rollouts) | token | kl_plus_grpo | PASS | none; but caveat: Stage 2 is text-only training, visual transfer is emergent | high |
| Video-OPD | strict_opd | strict_opd | https://arxiv.org/abs/2602.02994 | Eq.5, Eq.11, S3.3 | "student generates current-policy video grounding trajectories" | "Qwen3-VL-32B (GRPO) evaluates log-probs on student actions" | r_t = −(log pi_theta − log pi_tea); equivalent to reverse-KL per Eq.11 | current_policy | white_box (multimodal teacher) | token | reverse_kl_equivalent | PASS | none; TVDF filtering preserves on-policy property | high |
| X-OPD | strict_opd | strict_opd | https://arxiv.org/abs/2603.24596 | Eq.1-5 | "speech student generates rollouts from speech input" | "text teacher scores same output tokens via log-probs under text input" | L = lambda*L_im + (1−lambda)*L_cm; token-level cross-modal KL | current_policy | white_box (text teacher on speech student) | token | cross_modal_kl | PASS | none; knowledge gap is practical, not structural | high |
| Uni-OPD | strict_opd | strict_opd | https://arxiv.org/abs/2605.03677 | Eq.4, 9-11 | "student generates on-policy trajectories with data balancing" | "domain teachers provide token-level KL guidance" | J = sum w_i D_KL(pi_theta \|\| pi_Ti); margin mask (m(q)>=delta) + margin shift | current_policy | white_box (domain-specific multi-teacher) | token | reverse_kl_margin_calibrated | PASS | none; MLLM experiments verified on Qwen3-VL-2B/4B | high |
| VLA-OPD | strict_opd | strict_opd | https://arxiv.org/abs/2603.26666 | Algorithm 1 | "current student policy collects trajectories in environment" | "frozen expert teacher labels those same visited states" | Reverse-KL on action tokens; zero-forcing handles OOD states | current_policy | white_box (expert VLA teacher) | action-token | reverse_kl | PASS | none; reverse-KL design handles teacher quality on OOD states | high |
| LiteGUI (Stage 1) | strict_opd | strict_opd (stage-scoped) | https://arxiv.org/abs/2605.07505 | S3.2, Eq.12, A.5 | "student always generates the response used for distillation"; lambda=1.0 | "Qwen3-VL-32B with oracle actions scores same student sequence" | Eq.12: reverse-KL GKD-style; Stage 2 GRPO is separate | current_policy | white_box + privileged (oracle actions) | token | reverse_kl_gkd | PASS | none; stages cleanly separated; same teacher in Stage 2 provides only scalar reward | high |
| PRISM | borderline_strict | **adjacent** | https://arxiv.org/abs/2604.28123 | Eq.1-4 | "current-policy responses are sampled" | "MoE discriminator scores responses" | r(x,y) = α·D_v + (1−α)·D_r → scalar → GRPO advantages; **KL coefficient = 0.0** | current_policy | black_box (MoE discriminator) | sequence (scalar) | adversarial_grpo_reward | **FAIL** | Discriminator is scalar reward via GRPO. KL explicitly disabled. GAN-like adversarial RL, not distributional distillation. | high |
| Qwen3 OPD stage | strict_opd | **unclear** | https://arxiv.org/abs/2505.09388 | ~2-3 sentences in report | not found ("distilling output logits" only) | not found | not found (no loss equation) | unclear | unclear | unclear | unclear | **FAIL** | Report is too terse: no on-policy mention, no KL equation, no mechanism details. Consistent with offline KD. C1/C2/C3 all unverifiable. | low |
| MiMo-V2-Flash MOPD | strict_opd | strict_opd (stage-scoped) | https://arxiv.org/abs/2601.02780 | Eq.3-4 | "student samples from its own evolving distribution" | "domain teachers provide token-level advantage: log pi_domain / pi_student" | L_MOPD with truncated importance weighting; token-level log-ratio KL | current_policy | white_box (domain teachers) | token | token_level_log_ratio_kl | PASS | none; MOPD is distinct Stage 3, clearly separated from GRPO | high |
| Nemotron-Cascade2 MOPD | strict_opd | strict_opd (stage-scoped) | https://arxiv.org/abs/2603.19220 | S4.4, Eq.3-4 | "strict on-policy student training" | "domain teachers provide token-level advantage" | L_MOPD: a_t^MOPD = sg[log pi_domain − log pi_train]; truncated importance weighting | current_policy | white_box (multi-domain teachers) | token | token_level_advantage_kl | PASS | none; exact loss equations provided; more detail than most industrial reports | high |
| KEPO | borderline_strict | borderline_strict | https://arxiv.org/abs/2602.00400 | Eq.7, Algorithm 1 | "Sample {y_i} ~ pi_theta(.\|x)"; student generates on-policy | "D(pi_T \|\| pi_theta) on student trajectories" (by GKD reference) | Quality-gated: I_{r>=tau} * D(pi_T\|\|pi_theta); D never formally defined | current_policy | white_box (Qwen3-VL-32B, implied) | likely token (by GKD analogy) | grpo + gated_kl | PASS | D undefined; quality gate limits distillation scope; structural OPD loop present | medium |
| D-OPSD | borderline_strict | borderline_strict | https://arxiv.org/abs/2605.05204 | Eq.7, S2.2 | "optimization on student's actual roll-outs" | "EMA privileged self-teacher with target-image context on same states" | L = E[\|\|u_s − sg(u_t)\|\|²]; paper admits "not a token-level KL divergence" | current_policy | privileged_self (EMA + target image) | step (denoising) | mse_velocity_field | PASS | Paper itself says "not KL"; stop-gradient prevents distributional optimization | medium |
| Flow-OPD | borderline_strict | borderline_strict | https://arxiv.org/abs/2605.08063 | Eq.8-12, S5.1.1 | "student exposes its own distribution shifts" via SDE | "multi-teacher velocity fields at student states" | Per-step KL→L2 correct; consumed via PPO as dense reward | current_policy (SDE stochastic) | white_box (4 specialized teachers) | step (denoising) | ppo_dense_kl_reward | PASS | KL derivation correct per-step but consumed via PPO, not direct loss | medium |
| KDRL | borderline_strict | borderline_strict | https://arxiv.org/abs/2506.02208 | main method | "using on-policy rollouts" | "teacher offers token-level supervision through KL" | J_KDRL = J_GRPO − beta * D_KL(pi_theta\|\|pi_T); KD subcomponent is strict | current_policy | white_box (frozen teacher) | token | reverse_kl + grpo | PASS | Full method is hybrid; KD subcomponent alone is strict | high |
| SCOPE | borderline_strict | borderline_strict | https://arxiv.org/abs/2604.10688 | main method | "student generates a group" of rollouts | "teacher pi_T" on incorrect trajectories only; correct branch has NO teacher | Forward KL on incorrect branch; student-weighted MLE on correct branch | current_policy | white_box (selective) | token (incorrect only) | selective_forward_kl | PASS | Half-OPD: teacher only on incorrect trajectories | high |
| Gemma 2 post-training | borderline_strict | borderline_strict | https://arxiv.org/abs/2408.00118 | ~1 sentence | "distillation from the teacher on the student's distribution" (implied) | stated but not described | no loss equation; cites GKD + MiniLLM | unclear | unclear (implied white_box) | unclear | unclear (by GKD/MiniLLM citation) | PASS (barely) | Weakest borderline: single-sentence evidence, citation-only | medium |

---

## 3. Recommended Table Edits

### Downgrades

| Row | Current Table | Current Label | New Label | Action |
|---|---|---|---|---|
| **prism-2026** | tables/opd_papers.md, tables/vlm_opd_papers.md | borderline_strict | **adjacent** | Move to tables/adjacent_work.md. Reason: scalar discriminator reward via GRPO; KL=0.0. |
| **qwen3-opd-2025** | tables/opd_papers.md | strict_opd | **unclear** | Downgrade opd_strictness to unclear; reduce confidence to low. Reason: insufficient public evidence. |
| **oec-2025** | tables/opd_papers.md | borderline_strict | **adjacent** | Move to tables/adjacent_work.md. Reason: NLL/SFT on expert completions; DAgger IL, not distributional OPD. |

### Metadata Updates (No Label Changes)

| Row | Field | Old Value | New Value | Reason |
|---|---|---|---|---|
| vold-2025 | notes | Line audit confirms strict only for Stage 2 unified RL plus OPD. | Stage 2 trains on text-only data (orz-57k math). Teacher never sees images. Visual transfer is emergent, not supervised. | VOLD caveat from red-team |
| sod-2026 | notes | SOD is strict for its dense step-wise teacher-logit OPD component; GRPO is auxiliary. | Ablation confirms OPD is PRIMARY signal: removing OPD = −40% vs removing GRPO = −9.2%. | SOD ablation evidence strengthens strict |
| gui-sd-2026 | notes | WP6 merge treats GUI-SD as strict... | "Distillation-to-SFT Collapse" finding: textual privilege collapses teacher entropy (0.17); visual privilege preserves it (0.50). | GUI-SD design rationale |
| gemma2-posttraining-2024 | notes | (existing) | Weakest borderline in the table. Single-sentence evidence plus GKD/MiniLLM citation. No loss equation. | Red-team caveat |

### Summary Statistics After Edits

| Category | Before | After | Delta |
|---|---|---|---|
| strict_opd rows (main table) | 29 | 28 | −1 (Qwen3 → unclear) |
| borderline_strict rows (main table) | 13 | 11 | −2 (PRISM → adjacent, OEC → adjacent) |
| Adjacent rows | 45+ | 47+ | +2 |
| Unclear rows | 0 | 1 | +1 |

---

*End of WP9 post-WP8 verifier pass.*
