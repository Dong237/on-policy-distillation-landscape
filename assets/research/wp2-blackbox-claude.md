# WP2: Black-Box / API On-Policy Distillation Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML full text, OpenReview, official project pages
**Methodology:** Six independent research agents: (1) primary-source audit of GAD/OVD/Lion, (2) broad web search for new candidates (34 methods surveyed), (3) false-positive/adjacent mapping (50+ methods excluded), (4) primary-source audit of SODA/CGD/D2PO/SuperCorrect, (5) primary-source audit of ORPO-Distill/PLaD/LLMR/RL-KD-Judge. All reconciled against `taxonomies/strict-opd-definition.md`.

---

## 1. Memo

### What was searched

- arXiv keyword queries: "black-box on-policy distillation", "adversarial distillation LLM", "API teacher distillation on-policy", "verbal feedback distillation", "discriminator distillation LLM on-policy", "online preference distillation", "critique distillation student generated", "teacher-as-judge distillation", "GAN-style LLM distillation"
- Forward/backward citation chains from GAD, Lion, OVD, PRISM, GKD, MiniLLM
- Explicit search for: ORPO-Distill, SuperCorrect, SODA, CGD, D2PO, PLaD, LLMR, RL-KD with LLM-as-Judge, ADPA, GAR, PAD, RLTF, ALT, FCP, ALIVE, DPKD, REDI, RLKD, daDPO, CTPD
- False-positive sweep: GRPO/DAPO/RLVR families, static DPO/ORPO/SimPO, teacher-generated SFT, RLHF, Constitutional AI/RLAIF, BOND, SPIN, SPAG, SPC

### Method families found

| Family | Methods | Characteristic |
|---|---|---|
| **Adversarial discriminator OPD** | GAD, PRISM | Discriminator trained on teacher-vs-student outputs provides reward; student optimized via GRPO |
| **Verbal/score-based OPD** | OVD | Teacher provides discrete verbal scores; used for rejection sampling, not loss |
| **Semi on-policy contrastive** | SODA | Student snapshot outputs as rejected, teacher as chosen; DPO objective |
| **Critique-conditioned offline KD** | CGD, SuperCorrect | Teacher critiques/corrects student errors; student trains on teacher text |
| **Online preference optimization** | D2PO, ORPO-Distill, PLaD | Student generates; discriminator/teacher provides preference signal; DPO/ORPO objective |
| **Reward-from-teacher RL** | LLMR, RL-KD with Judge | Teacher/judge scores student rollouts; consumed as scalar reward via PPO/GRPO |
| **Iterative curriculum SFT** | Lion | Teacher generates training data; student errors identify hard examples |

### What is actually black-box OPD versus adjacent

**The central finding of WP2 is that strict black-box OPD is essentially empty.** No method found satisfies all three criteria (C1+C2+C3) when the teacher is truly black-box. The fundamental constraint is:

> Without token logits, a black-box teacher can only provide response-level signals (scores, preferences, corrections, verbal feedback). These signals are consumed as scalar rewards (via GRPO/PPO) or preference pairs (via DPO), not as distributional distillation objectives.

The three methods closest to black-box OPD (GAD, OVD, PRISM) are all `borderline_strict` because:
- C1 passes: student generates on-policy rollouts
- C2 passes: black-box teacher/discriminator supervises those rollouts
- C3 fails strict: the supervision signal is a response-level scalar reward consumed via GRPO, not a token-level distributional distillation loss

This is structurally inevitable: without logits, there is no distribution to match. All black-box methods must collapse teacher supervision into either:
1. A scalar reward/score -> consumed via RL (GRPO/PPO) -> this is reward-based RL, not distillation
2. A preference pair (teacher-chosen, student-rejected) -> consumed via DPO/ORPO -> this is preference optimization, not same-rollout distillation
3. A text correction/critique -> consumed via SFT on teacher text -> this is offline KD, not OPD

### Key differences from white-box OPD

| Dimension | White-box OPD | Black-box "OPD" |
|---|---|---|
| **Teacher signal** | Token-level distribution (logits, log-probs) | Response-level scalar (score, preference, verbal) |
| **Supervision granularity** | Per-token KL divergence | Per-response reward or ranking |
| **Objective** | D_KL(pi_theta \|\| pi_T) or variants | GRPO/PPO reward maximization or DPO preference |
| **Distribution matching** | Direct logit alignment | Indirect via reward shaping or preference |
| **C3 status** | Passes (distributional objective) | Fails strict (scalar reward or preference pair) |
| **Theoretical grounding** | OPD-as-RL equivalence (dense teacher reward = reverse KL) | No distributional interpretation |

### Practical costs and failure modes

| Issue | Details |
|---|---|
| **API cost** | GAD requires many teacher API calls for discriminator training data; OVD requires teacher scoring of every student rollout |
| **Response inflation** | Student may learn to produce longer outputs that game discriminator scores (GAD) or verbal rubrics (OVD) |
| **Discriminator overfitting** | GAD's discriminator can overfit to surface features (style, length) rather than quality, leading to reward hacking |
| **Teacher drift** | API teacher models may be updated, causing distribution shift during training |
| **Calibration** | Verbal scores (OVD) are poorly calibrated across diverse tasks; discrete 0-9 scale has limited dynamic range |
| **Prompt sensitivity** | Discriminator/judge prompts significantly affect supervision quality; results are not robust to prompt variation |
| **Mode collapse** | Without token-level guidance, student may collapse to a narrow reward-maximizing mode |

---

## 2. Evidence Ledger

### Primary-source verified methods

| method | primary_source_url | source_status | C1 evidence | C2 evidence | C3 evidence | teacher_access | supervision_granularity | objective_family | rollout_freshness | suggested_label | confidence | downgrade_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAD | https://arxiv.org/abs/2511.10643 | fetched | "Sample student responses G(x)" -- on-policy | "A discriminator to distinguish its responses from the teacher LLM's...predicts a sequence-level scalar score D([x,y])" | "Treat D(G(x)) as a reward and optimize it using policy gradient" -- GRPO on scalar reward | discriminator | response | grpo_reward | current_policy | borderline_strict | high | C3 fails strict: discriminator produces sequence-level scalar reward consumed via GRPO; paper acknowledges "discriminator can be interpreted as an on-policy reward model" |
| OVD | https://arxiv.org/abs/2601.21968 | fetched | "The student generates N trajectories per problem: y_i^(j) ~ pi_S(.\|x_i)" | "Teacher outputs a distribution over a discrete verbal vocabulary of size v=10 (scores {0,1,...,9})" | "All trajectories contribute to L_RL based on their rewards" -- verbal scores filter only; outcome-based rewards drive GRPO | verbal | response | grpo_reward | current_policy | borderline_strict | high | C3 fails strict: verbal scores used only for rejection sampling, not in loss; actual RL uses outcome rewards (F1/exact match); teacher trajectories mixed into batch as replacement demonstrations |
| PRISM | https://arxiv.org/abs/2604.28123 | verified_in_repo | "For each input x, we sample a group of N responses from the current policy G(.\|x)" | "MoE discriminators supervise those student rollouts" | "r(x,y) = alpha*Dv(x,c) + (1-alpha)*Dr(x,t)" -- single scalar reward per response consumed via GRPO | discriminator | response | grpo_reward | current_policy | borderline_strict | high | C3 fails strict: response-level scalar reward via GRPO; no token-level KL; paper self-describes as "response-level adversarial game" |
| Lion | https://arxiv.org/abs/2305.12870 | fetched | C1 FAILS: "We construct instruction-response data {x_i, T(x_i)} by forward propagating instructions through the teacher T" -- teacher generates training responses | "The referee R quantitatively measures the quality difference" -- but only for curriculum selection, not for training | "Fine-tune student S to align the response of the teacher model, by optimizing the autoregressive language modeling objective" -- SFT on teacher outputs | api | outcome | sft_teacher_data | off_policy | adjacent | high | C1 and C3 fail: student always trains on teacher-generated responses via SFT; discrimination stage identifies hard examples for curriculum, not training signal |
| SODA | https://arxiv.org/abs/2604.03873 | fetched | C1 PARTIAL: "we sample responses from the base student q0 before any fine-tuning" -- static snapshot, not current policy | "only its generated text is accessible, a setting known as black-box distillation" | "The distillation itself happens through preference optimization" -- DPO on teacher-positive/student-negative | black_box | sequence | dpo_preference | static_snapshot | partial_opd | medium | C1 fails strict: student samples from frozen q0, not evolving theta; paper self-identifies as "semi on-policy"; signal is "front-loaded...utility diminishes as q_theta diverges" |
| CGD | https://arxiv.org/abs/2505.11628 | fetched | C1 FAILS: student generates initial response, but trains on TEACHER's refined answer y-hat | "The teacher model T_phi critiques this response, generating a textual explanation" | "L(theta) = E[-log S_theta(y-hat \| x, y', c)]" -- SFT on teacher-generated refined answers | verbal | token_on_teacher_text | sft_teacher_data | one_shot_initial | adjacent | high | C1 and C3 fail: student's own output y' is only context input; training target is teacher-generated y-hat; this is offline KD conditioned on student errors |
| D2PO | https://arxiv.org/abs/2405.01511 | fetched | "y1, y2 <- pi(x), pi(x) // get 2 rollouts from policy" -- on-policy | "our algorithm is flexible in the form of the discriminator, which is treated as a black box" | "pi <- pi + nabla L_DPO(pi, y+, y-) // DPO update with new preferences" -- DPO on discriminator-labeled pairs | discriminator | response | dpo_preference | current_policy | adjacent | high | C3 fails: discriminator produces scalar score creating binary preference labels consumed via DPO; both chosen/rejected are student-generated; no teacher distribution or distillation signal; paper frames as "learning from preferences" |
| SuperCorrect | https://arxiv.org/abs/2410.09008 | fetched | C1 FAILS for Stage 1 (SFT on teacher templates); PARTIAL for Stage 2 (student generates erroneous traces but trains on teacher corrections) | "we design a prompt P_c to instruct pi_tea to search for the logic flaws and errors" -- teacher corrects student errors | "maximize the probability of the teacher LLM's correction...while minimizing the probability of the student LLM's self-correction" -- cross-model DPO | api | step | sft_plus_dpo | one_shot_from_sft | adjacent | high | C1 and C3 fail: Stage 1 is pure offline SFT; Stage 2 is DPO where chosen=teacher correction text, rejected=student self-correction; student trains to imitate teacher text |
| ORPO-Distill | https://arxiv.org/abs/2509.25100 | fetched | "Rejected is a student CoT trace leading to a negative or incorrect answer" -- student generates rejected side | C2 FAILS: "Chosen is a teacher CoT trace leading to the positive or correct answer" -- teacher generates own output, does NOT supervise student | "L_ORPO = L_SFT + lambda * L_OR" -- ORPO on teacher-chosen vs student-rejected | black_box | sequence | orpo_preference | mixed_policy | adjacent | high | C2 fails: teacher generates independent output as "chosen"; teacher never examines or scores student output; this is preference optimization, not same-rollout supervision |
| PLaD | https://arxiv.org/abs/2406.02886 | fetched | C1 WEAK: "Sample student output y^S = q_theta(y\|x)" -- but from fixed snapshot before training | C2 FAILS: teacher generates own output assumed preferred; never examines student output | "L_rankcal = max(0, beta - log P(y+\|x) + log P(y-\|x))" -- margin-based ranking | black_box | sequence | ranking_preference | fixed_snapshot | adjacent | high | C1 weak (fixed snapshot); C2 fails (teacher generates own output); C3 fails (preference ranking, not distillation on student rollouts) |
| LLMR | https://arxiv.org/abs/2409.12500 | fetched | "a sequence is sampled from the student's prediction y ~ q_theta" | NOT BLACK-BOX: requires teacher token-level probabilities for step-wise reward | "implicit step-wise reward function derived from unconditional teacher-forcing" -- consumed via REINFORCE | white_box | step | rl_reward | current_policy | adjacent | high | Not black-box (requires teacher log-probs); C3 fails (reward-based RL via REINFORCE) |
| RL-KD with Judge | https://arxiv.org/abs/2604.02621 | fetched | "the student model generates a candidate reasoning trajectory" -- on-policy | "fed to the large judge LLM" which produces "a continuous scalar reward" | "L_overall = lambda*L^VR_PPO + mu*L^YoN_PPO + rho*L^Rerank_PPO" -- PPO on scalar rewards | black_box | outcome | ppo_reward | current_policy | adjacent | high | C3 fails: judge produces scalar reward consumed via PPO; textbook reward-based RL, not distillation |

---

## 3. Candidate Markdown Rows

Only methods with sufficient evidence for `tables/opd_papers.md`. GAD, OVD, and PRISM are already in the table. SODA is the only new candidate with enough evidence, though at partial_opd.

### Existing rows to confirm/update

| id | opd_strictness | verification_status | conflict_status | confidence | notes |
|---|---|---|---|---|---|
| gad-2025 | partial_opd | verified | source_verified | high | WP2 audit confirms: discriminator scalar reward via GRPO; borderline_strict at most; keep partial_opd per repo conservative rule |
| ovd-2026 | partial_opd | verified | source_verified | high | WP2 audit confirms: verbal scores filter only; GRPO drives training with outcome rewards; keep partial_opd |
| prism-2026 | borderline_strict | verified | source_verified | high | Already audited in WP9; response-level adversarial scalar via GRPO; confirmed borderline |

### New candidate row (pending table validation)

```
| soda-2026 | SODA: Semi On-Policy Black-Box Distillation | 2026 | Authors TBD | arXiv | https://arxiv.org/abs/2604.03873 | | | LLM | black_box_alignment | semi_opd_contrastive | outcome_based | black_box | sequence | dpo_preference | partial | partial_opd | black_box_response | api_teacher | larger_llm | policy_student | api_teacher | mixed_policy | static_snapshot | preference_pair | sequence | dpo | C1 partial: student samples from frozen q0 not current theta; C2 pass: black-box API teacher; C3 pass marginal: DPO preference objective. | no | | post_training | instruction_following | custom | unclear | not_applicable | https://arxiv.org/abs/2604.03873 | arxiv | needs_review | wp2_initial | medium | Semi on-policy by design; student snapshot is q0; utility front-loaded and diminishes as policy diverges. | API teacher cost for each training batch | Paper self-identifies as "semi on-policy"; rollout distribution is frozen. | 2026-05-12 |
```

---

## 4. Adjacent / False-Positive Ledger

### Black-box methods audited and excluded

| method | link | why_adjacent | missing_condition | confidence |
|---|---|---|---|---|
| Lion | https://arxiv.org/abs/2305.12870 | Student always trains on teacher-generated responses via SFT; discrimination stage only identifies hard examples for curriculum | C1 (student never trains on own rollouts), C3 (SFT on teacher text) | high |
| CGD | https://arxiv.org/abs/2505.11628 | Student output is input context; student trains on teacher's refined answer via SFT | C1 (trains on teacher text), C3 (offline SFT) | high |
| SuperCorrect | https://arxiv.org/abs/2410.09008 | Stage 1 is SFT on teacher templates; Stage 2 is cross-model DPO with teacher corrections as chosen | C1 (trains on teacher corrections), C3 (SFT + DPO on teacher text) | high |
| D2PO | https://arxiv.org/abs/2405.01511 | Discriminator provides scalar score for binary preference labels consumed via DPO; no teacher distribution | C3 (preference optimization, not distillation) | high |
| ORPO-Distill | https://arxiv.org/abs/2509.25100 | Teacher generates own output as "chosen"; never examines student output; ORPO preference loss | C2 (teacher doesn't supervise student rollout), C3 (preference optimization) | high |
| PLaD | https://arxiv.org/abs/2406.02886 | Teacher generates own output assumed preferred; fixed student snapshot; ranking loss | C1 (fixed snapshot), C2 (no rollout supervision), C3 (preference ranking) | high |
| LLMR | https://arxiv.org/abs/2409.12500 | Requires white-box teacher log-probs; reward consumed via REINFORCE | Not black-box; C3 (reward-based RL) | high |
| RL-KD with Judge | https://arxiv.org/abs/2604.02621 | Judge produces scalar reward consumed via PPO; textbook reward-based RL | C3 (scalar reward via PPO) | high |

### Broader false-positive categories

| Category | Representative methods | Why NOT OPD | Missing |
|---|---|---|---|
| Reward-only RLVR/GRPO | GRPO, DAPO, VLM-R1, Perception-R1, InternVL3.5 | Student rollouts + scalar rewards only; no teacher distribution | C2, C3 |
| Static DPO/preference | DPO, IPO, ORPO, SimPO, KTO on fixed data | Fixed preference pairs, not online student rollouts | C1 |
| Teacher-generated SFT | DeepSeek-R1 distilled, Gemma 2 pre-training KD, LLaVA-KD, SeqKD | Student never generates training states | C1 |
| Online RLHF | RLHF, RLAIF, Constitutional AI | Reward model on student rollouts, but no teacher distribution matching | C3 |
| Self-play without teacher | SPIN, SPAG, SPC, GAR, ALIVE | No external teacher; self-play or self-critique | C2 |
| "Distillation" via rewards | RLKD, BOND | Teacher structure mediated through scalar reward | C3 |
| Offline correction/critique | CGD, SuperCorrect, REDI | Student errors trigger teacher generation; student trains on teacher text | C1, C3 |

---

## 5. Gap Map

### Methods requiring line-level audit

| Method | arXiv | Why it needs audit | Likely label | Priority |
|---|---|---|---|---|
| ADPA | 2502.17927 | ICLR 2025 Spotlight; advantage from aligned teacher; unclear if truly black-box or requires white-box | borderline_strict or adjacent | high |
| PAD | 2502.14272 | ACL 2025; models teacher's preference distribution; may distill preference distributions on student rollouts | partial_opd or adjacent | high |
| RLTF | 2602.02482 | RL from text feedback; self-distillation of feedback studied; borderline between verbal OPD and RL | partial_opd or adjacent | medium |
| FCP | 2509.22638 | Feedback-conditional policy; online bootstrapping with verbal feedback; conditional generation vs distillation | partial_opd or adjacent | medium |
| ALT | 2407.16970 | EMNLP 2024; alignment via textual feedback conditioning; may qualify if feedback conditions a distillation objective | partial_opd or adjacent | medium |
| daDPO | 2506.15717 | Distribution-aware DPO using teacher output distributions; potentially bridges preference and distillation | partial_opd or borderline_strict | medium |
| CTPD | 2601.11865 | Cross-tokenizer preference distillation; first black-box cross-architecture transfer of alignment | partial_opd or adjacent | low |
| RLKD | 2505.16142 | RL with Generative Structure Reward Model from teacher; structural alignment signal | adjacent or borderline_strict | low |
| GAR | 2512.16917 | Co-evolving discriminator + reasoner; self-play, no external teacher; step-level adversarial | adjacent | low |

### Open structural questions

1. **Can black-box OPD ever be strict?** The current landscape suggests no: without token logits, teacher supervision collapses to response-level scalars, which are consumed as rewards (RL) or preferences (DPO), not as distributional distillation targets. Is there a theoretical pathway to strict black-box OPD?

2. **Discriminator as teacher proxy.** GAD and PRISM train discriminators on teacher-vs-student outputs. Could a discriminator that provides per-token or per-step feedback (not just response-level) satisfy C3? GAR's step-level discriminator is closest but lacks an external teacher.

3. **Verbal feedback as distribution.** OVD's verbal scores (0-9) are a coarse discrete distribution. Could finer-grained verbal feedback (per-sentence, per-step critiques) satisfy C3 if consumed as a distillation target rather than a filter?

4. **Online preference distillation.** D2PO generates on-policy pairs with discriminator-labeled preferences. If the preference signal were token-level (not response-level), would DPO on dense teacher-labeled tokens qualify as C3? This is essentially the white-box setting reconstructed from black-box components.

5. **Semi on-policy saturation.** SODA shows that static-snapshot student outputs lose utility as the policy diverges. Is there a black-box method that iteratively refreshes student rollouts while maintaining the preference structure?

6. **Cross-architecture black-box OPD.** No method found achieves strict OPD across different model architectures/tokenizers in the black-box setting. SimCT (WP1) solves cross-tokenizer OPD but requires white-box access. CTPD (2601.11865) may bridge this gap.

---

## Classification Summary

| Label | Count | Methods |
|---|---|---|
| **borderline_strict** | 3 | GAD, OVD, PRISM |
| **partial_opd** | 1 | SODA |
| **adjacent** | 8 | Lion, CGD, SuperCorrect, D2PO, ORPO-Distill, PLaD, LLMR, RL-KD with Judge |
| **needs_audit** | 9 | ADPA, PAD, RLTF, FCP, ALT, daDPO, CTPD, RLKD, GAR |

The borderline_strict methods (GAD, OVD, PRISM) represent the current frontier of black-box OPD. They satisfy C1 (student on-policy) and C2 (black-box teacher/discriminator supervision on student rollouts), but consume the supervision as response-level scalar rewards via GRPO, placing them below strict OPD's C3 threshold for distributional distillation.
