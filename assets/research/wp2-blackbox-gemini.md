# WP2 — Black-Box / API On-Policy Distillation Landscape (Gemini)

**Researcher:** Gemini (deep research agent)
**Date:** 2026-05-12
**Scope:** WP2 — Black-box / API teachers, response-level discriminators, verbal/critique distillation, preference-based and adversarial OPD for LLMs and adjacent multimodal cases.
**Strict OPD test:** C1 (current-policy student rollout) ∧ C2 (black-box teacher / API / discriminator / judge supervises that exact state) ∧ C3 (distillation-style consumption, not scalar-reward RLVR).

---

## 1. Memo

### 1.1 What I searched
- Repo state: [papers/black-box-and-api-opd.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/papers/black-box-and-api-opd.md), [papers/black-box-opd.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/papers/black-box-opd.md), [tables/opd_papers.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/tables/opd_papers.md), [tables/adjacent_work.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/tables/adjacent_work.md).
- Primary sources (arXiv abstract + HTML where available) for the seed list: GAD (arXiv 2511.10643), Lion (arXiv 2305.12870), OVD (arXiv 2601.21968), PRISM (arXiv 2604.28123), ORPO-Distill (arXiv 2509.25100), SuperCorrect (arXiv 2410.09008), GAKD (OpenReview ICLR 2026 submission 10374), BOND (arXiv 2407.14622), cross-lingual reward-model transfer (arXiv 2404.12318) as a false-positive probe.
- Secondary scans: [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD) cross-listing of black-box entries and [thunlp/OPD](https://github.com/thunlp/OPD) for any black-box additions.

### 1.2 Method families found
1. **Black-box GAN-style adversarial OPD on student rollouts** — GAD, GAKD, PRISM. Discriminator is trained on student rollouts vs teacher rollouts and supplies per-rollout feedback that the student consumes.
2. **Verbal-score / API-score critique distillation** — OVD (verbal scores 0–9 from a teacher). Trajectory-level scalar feedback consumed via PG.
3. **Black-box preference / contrastive distillation on student rollouts** — ORPO-Distill (mixed-policy ORPO over <chosen=teacher trace, rejected=student trace>). The student-generated negative trace satisfies C1 partially; the loss is preference-based, not a teacher distribution.
4. **Best-of-N distillation** — BOND. Distribution matching to a Best-of-N policy using Jeffreys divergence; technically uses RLHF reward model rather than a black-box teacher LLM, but the method is on-policy distribution-matching.
5. **Curriculum / "hard instruction" distillation that *uses* student outputs only for difficulty mining** — Lion. Student outputs drive curriculum, but the actual SFT/training targets are teacher-generated — so this is offline KD, not OPD.
6. **Teacher-corrected preference distillation** — SuperCorrect. Cross-model DPO over teacher-generated correction traces; still offline-pair DPO.

### 1.3 Black-box OPD vs adjacent — boundary calls
| Method | Status here | Why |
|---|---|---|
| GAD ([arXiv 2511.10643](https://arxiv.org/abs/2511.10643)) | `partial_opd` (kept, do not promote) | Title says "Black-Box On-Policy Distillation" but the discriminator is explicitly described as "an on-policy reward model that co-evolves with the student"; abstract gives no non-scalar token-level distillation loss. The repo already keeps it `partial_opd`. |
| PRISM ([arXiv 2604.28123](https://arxiv.org/abs/2604.28123)) | `borderline_strict` (keep) | "Casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator … providing disentangled corrective signals" — disentangled (non-scalar) but still response-level; objective is adversarial distribution alignment, plausibly stronger than reward-only RLVR. |
| OVD ([arXiv 2601.21968](https://arxiv.org/abs/2601.21968)) | `partial_opd` (keep) | "Replaces token-level probability matching with trajectory matching using discrete verbal scores (0–9)". Scores are ordinal scalars per rollout; on-policy trajectories are explicit but C3 collapses to verbal-score optimization (RLVR-flavored). |
| GAKD ([OpenReview 9pfWYxYHAn](https://openreview.net/forum?id=9pfWYxYHAn)) | `borderline_strict` candidate (new) | White-box logit access to teacher — strictly speaking *not* black-box — but the contribution is sequence-level adversarial discriminator on per-token logits, plus reverse KL. Track because it adds the missing piece of "non-scalar discriminator signal + token-level KL". |
| ORPO-Distill ([arXiv 2509.25100](https://arxiv.org/abs/2509.25100)) | `partial_opd` (new candidate) | Mixed-policy: "rejected = student-generated CoT", "chosen = teacher-generated CoT". ORPO is a static-pair preference loss, not a same-rollout teacher distribution, so promote only to partial. |
| BOND ([arXiv 2407.14622](https://arxiv.org/abs/2407.14622)) | `adjacent` | "BOND … forces the distribution of generations from the policy to get closer to the Best-of-N distribution" using Jeffreys divergence. The supervision is BoN of the same policy under a reward model — so it is on-policy distribution matching against a *reward-induced* target, not a black-box teacher LLM. Track as adjacent RLHF/distillation hybrid. |
| Lion ([arXiv 2305.12870](https://arxiv.org/abs/2305.12870)) | `adjacent` (already in [adjacent_work.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/tables/adjacent_work.md)) | Student outputs feed *curriculum mining*, but the SFT targets are teacher-generated responses on hard instructions. Fails C1 + C2 simultaneously for the training step. |
| SuperCorrect ([arXiv 2410.09008](https://arxiv.org/abs/2410.09008)) | `adjacent` (new) | "Cross-model collaborative DPO" on teacher correction traces — static preference pairs from the teacher. Fails C1: training pairs are teacher-generated. |

### 1.4 Key differences from white-box OPD
- **No teacher logits / hidden states.** Supervision is at most a sequence- or rollout-level signal: discriminator probability, verbal score, preference label, or reward.
- **Granularity drops from per-token KL to per-rollout adversarial / preference / verbal.** This is why response-level black-box methods sit at `borderline_strict` or `partial_opd` in this repo's taxonomy.
- **Optimization is GAN/PG-flavored, not direct KL.** GAD, PRISM, OVD all combine on-policy generation with policy-gradient updates (often GRPO / minimax) instead of a closed-form distillation loss.
- **Co-evolving teacher.** The "teacher" in GAD/PRISM is a learned discriminator that updates with the student; this is closer to RLHF reward-model-as-teacher than to a frozen distillation target.
- **Calibration risk shifts.** Without per-token teacher distributions, dense calibration evidence is lost; only rollout-level fidelity remains.

### 1.5 Practical costs and failure modes
| Risk | Where it bites | Notes |
|---|---|---|
| API cost | OVD (verbal score per rollout), BoN-style | Each on-policy rollout requires one or more teacher API calls; budget scales with rollout length × N batch × rounds. |
| Response inflation | OVD, PRISM | Discrete verbal scores and response-level discriminators reward longer / more "polished" responses — well-known length bias. StableOPD-style length controls don't apply. |
| Discriminator overfitting | GAD, PRISM, GAKD | Discriminator learns dataset shortcuts; student exploits them rather than matching the teacher distribution. PRISM's MoE discriminator with disentangled perception/reasoning experts is a partial mitigation. |
| Reward hacking | OVD, BOND, GAD | Verbal scores and reward-model proxies are gameable; classical RLHF failure mode. |
| Teacher drift | GAD, PRISM (co-evolving discriminator), Lion (curriculum updates) | Discriminator/curriculum changes during training can chase the student into pathological modes. |
| Calibration loss | All response-level methods | Teacher uncertainty is *not* preserved; a known failure relative to logit-level OPD. |
| Prompt sensitivity | OVD, SuperCorrect, ORPO-Distill | Verbal-score rubric, correction-trace prompt, and preference-label prompt all materially change supervision quality. |
| Cost vs static SFT | Lion, SuperCorrect | Once supervision collapses to a static teacher dataset, the user is paying OPD-grade rollout cost without OPD-grade gradient signal. |

### 1.6 Self-claimed-OPD that are really not
- **Lion** — markets as "adversarial distillation" and uses student outputs, but the actual training step is SFT on teacher responses on hard instructions.
- **SuperCorrect** — DPO on teacher correction traces; preference optimization on static pairs.
- **ORPO-Distill** — explicitly *mixed-policy* not on-policy; ORPO objective is preference-pair loss.
- **BOND** — distribution matching but the target is BoN of the same policy under an RLHF reward model; teacher LLM is not in the loop.
- **Cross-lingual reward-model transfer** ([arXiv 2404.12318](https://arxiv.org/abs/2404.12318)) — reward-only PPO, no distillation objective. Pure RLHF.

---

## 2. Evidence Ledger

| method | primary_source_url | source_status | C1 evidence | C2 evidence | C3 evidence | teacher_access | supervision_granularity | objective_family | rollout_freshness | suggested_label | confidence | downgrade_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GAD | https://arxiv.org/abs/2511.10643 | arxiv_v3_2026_01 | "GAD frames the student LLM as a generator" — student rollouts are explicit | "trains a discriminator to distinguish its responses from the teacher LLM's" — black-box teacher only via text outputs | "The discriminator acts as an on-policy reward model that co-evolves with the student, providing stable, adaptive feedback" — minimax + reward; abstract gives no non-scalar token-level distillation loss | black_box_text_outputs (GPT-5-Chat) + learned_discriminator | sequence (response-level) | minimax_grpo_with_discriminator_reward | current_policy | `partial_opd` | high | C3 collapses to a learned reward signal; not non-scalar teacher-style supervision per WP9 rule |
| PRISM | https://arxiv.org/abs/2604.28123 | arxiv_v1_2026_04 | "casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator" — student rollouts under current policy | "MoE discriminator with dedicated perception and reasoning experts, providing disentangled corrective signals that steer the policy toward the supervision distribution without requiring access to teacher logits" — Gemini 3 Flash provides supervision distribution; discriminator scores student rollouts | "PRISM casts alignment as a black-box, response-level adversarial game … providing disentangled corrective signals" — disentangled (non-scalar) but response-level adversarial objective | black_box_api_teacher (Gemini 3 Flash) + MoE_discriminator | sequence | adversarial_distribution_alignment | current_policy | `borderline_strict` | high | Disentangled corrective signal is non-scalar in spirit, but evaluation remains response-level and is followed by RLVR, not token-level KL |
| OVD | https://arxiv.org/abs/2601.21968 | arxiv_v1_2026_01 | "trajectory matching using discrete verbal scores (0–9) from teacher models … allowing the student model to freely explore the output space" — student rollouts | "discrete verbal scores (0–9) from teacher models" — black-box API teacher | "memory-efficient framework that replaces token-level probability matching with trajectory matching using discrete verbal scores" — supervision is ordinal scalar per rollout | black_box_api_teacher | rollout (verbal score) | trajectory_verbal_score_optimization | current_policy | `partial_opd` | high | Verbal score is a scalar; C3 collapses to RLVR-style PG against an ordinal teacher score |
| GAKD | https://openreview.net/forum?id=9pfWYxYHAn | openreview_iclr2026_under_review | "trains: (1) a generator (student) to align with the teacher's distribution via a combination of sequence-level adversarial loss and reverse KLD loss" — student-as-generator implies on-policy | "(2) a discriminator to distinguish whether per-token logits are from the teacher or student" — discriminator over logits is per-token; teacher is white-box | "By jointly minimizing the token-level reverse KLD and sequence-level adversarial losses" — combined token-KL + adversarial | white_box_teacher_logits + learned_discriminator | token + sequence | reverse_kl_plus_sequence_gan | abstract does not commit (reviewer flagged this; "feasibility of optimizing reverse KLD loss on teacher-generated sequences" suggests off-policy in proof) | `borderline_strict` candidate | medium | (a) Strictly white-box, not black-box — does not belong in WP2 final table; (b) abstract's RKL feasibility proof is for *teacher-generated* sequences, weakening C1 |
| ORPO-Distill | https://arxiv.org/abs/2509.25100 | arxiv_v1_2025_09 (NeurIPS 2025 Efficient Reasoning workshop) | "adopts a mixed-policy strategy for utilizing student-generated outputs, outperforming both off- and on-policy alternatives" — student traces are used, but explicitly *mixed-policy* | "teacher and student model generate diverse positive and negative reasoning traces forming the preference dataset" — black-box teacher provides text traces, no logits | "Odds-Ratio Preference Optimization objective that contrasts teacher and student traces" — ORPO is preference-pair, not teacher-distribution | black_box_teacher_traces | preference_pair (chosen=teacher, rejected=student) | orpo_contrastive_preference | mixed_policy (mixed_freshness) | `partial_opd` | high | Mixed-policy by design; ORPO loss is contrastive preference, not same-rollout teacher distribution |
| Lion | https://arxiv.org/abs/2305.12870 | arxiv_v2_2023_10 (EMNLP 2023) | "prompt the teacher model to identify 'hard' instructions" — student outputs feed curriculum, not training | "teacher model … generate new 'hard' instructions for the student model" — teacher provides text instructions and target responses | "three-stage adversarial loop of imitation, discrimination, and generation" — imitation is SFT on teacher responses to hard instructions | black_box_teacher_responses | sequence (SFT targets) | iterative_curriculum_sft | mostly off_policy (student outputs only mine difficulty) | `adjacent` | high | not found C1+C2 jointly: training step is SFT on teacher-generated text, not teacher supervision on student rollouts |
| SuperCorrect | https://arxiv.org/abs/2410.09008 | arxiv_v3_2025_02 (ICLR 2025) | "we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training" — training pairs are teacher correction traces | "uses a large teacher model to supervise and correct both the reasoning and reflection processes" — supervision via correction traces, not on-policy student rollouts | "cross-model DPO approach teaches the student model to … with error-driven insights from the teacher" — DPO on teacher correction traces | black_box_teacher_correction_traces (GPT-4 / DeepSeek-R1) | preference_pair | cross_model_dpo_on_teacher_traces | off_policy (teacher-generated) | `adjacent` | high | C1 fails: training pairs are teacher-generated correction traces; static DPO not OPD |
| BOND | https://arxiv.org/abs/2407.14622 | arxiv_v1_2024_07 | "forces the distribution of generations from the policy to get closer to the Best-of-N distribution" — on-policy generations | "Best-of-N distribution" is the target — induced by reward model over the same policy, not a separate teacher LLM | "We use the Jeffreys divergence (a linear combination of forward and backward KL) to balance between mode-covering and mode-seeking … iterative formulation that utilizes a moving anchor" | rlhf_reward_model (no teacher LLM) | sequence | jeffreys_divergence_to_bon_self_distribution | current_policy | `adjacent` | high | C2 missing in the WP2 sense: target is BoN of *same* policy via reward model, not a teacher; this is RLHF distribution matching |

---

## 3. Candidate Markdown Rows for `tables/opd_papers.md`

> Only ORPO-Distill is a net-new candidate from this WP. GAD, PRISM, OVD already exist in the table and should remain at their current labels (verified above). GAKD is a strong borderline addition but is white-box, so it belongs in WP1 follow-up rather than WP2. Suggested row for ORPO-Distill, schema-aligned with [tables/opd_papers.md](file:///Users/bytedance/Desktop/Work/on-policy-distillation-landscape/tables/opd_papers.md):

```markdown
| orpo-distill-2025 | ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation | 2025 | Singh et al. | NeurIPS 2025 Workshop on Efficient Reasoning | https://arxiv.org/abs/2509.25100 |  |  | LLM | cross_architecture_distillation | black_box_preference_distillation | preference_pair | black_box | sequence | orpo_contrastive_preference | partial | partial_opd | preference_pair | black_box_teacher_traces | larger_llm | smaller_language_model | larger_language_model | mixed_policy | mixed_freshness | preference_pair | sequence | orpo_loss_with_mixed_policy_negatives | C1 partial: student traces serve only as ORPO negatives under a mixed-policy schedule; C2 black-box teacher provides positive traces; C3 ORPO preference loss is not same-rollout teacher distribution. | no |  | post_training | five_qa_benchmarks | custom | unclear | not_applicable | https://arxiv.org/abs/2509.25100 | arxiv | needs_review | wp2_new | medium | Mixed-policy ORPO with student-as-rejected and teacher-as-chosen; preference loss not strict OPD. | one teacher-generation pass plus periodic student re-sampling; cheaper than full OPD | Confirms ORPO-Distill belongs partial_opd, not strict; cross-architecture/black-box useful for WP2. | 2026-05-12 |
```

> No row proposed for GAKD until a line audit confirms whether the RKL term is computed on student rollouts or only on teacher-generated sequences (the OpenReview reviewer notes the latter for the proof step).

---

## 4. Adjacent / False-Positive Ledger

| method | link | why_adjacent | missing_condition | confidence |
|---|---|---|---|---|
| Lion | https://arxiv.org/abs/2305.12870 | Adversarial *curriculum* over student-identified hard instructions; training step is SFT on teacher-generated responses | C1 (training data is teacher-generated, not student rollouts at the supervised step) | high |
| SuperCorrect | https://arxiv.org/abs/2410.09008 | Two-stage thought-template + cross-model DPO on teacher correction traces; static preference pairs | C1 (teacher-generated correction traces); C3 (DPO on static pairs is not same-rollout distillation) | high |
| BOND | https://arxiv.org/abs/2407.14622 | Best-of-N distribution matching with Jeffreys divergence — on-policy distribution matching but the target is BoN of the same policy under an RLHF reward model | C2 in the WP2 sense (no separate black-box teacher LLM; the "teacher distribution" is BoN of the same policy) | high |
| Cross-Lingual RM Transfer | https://arxiv.org/abs/2404.12318 | Pure on-policy RLHF with a transferred reward model; no distillation objective | C2 + C3 (reward-only RLHF, not distillation) | high |
| GAKD | https://openreview.net/forum?id=9pfWYxYHAn | Strong adversarial-KD method, but **white-box** (teacher logits) — does not belong in WP2 black-box scope; reviewer flagged that the RKL feasibility proof uses teacher-generated sequences | Out of WP2 scope; for WP1, C1 needs §-level confirmation that RKL is on student rollouts | medium |

---

## 5. Gap Map — Open Questions Requiring Line-Level Audit

1. **GAD §3 / Algorithm 1** — Confirm whether the discriminator emits any non-scalar signal (e.g., per-token discrimination logits) consumed by the student loss, or only a scalar reward. Current label `partial_opd` rests on the abstract; PDF audit could promote or lock it.
2. **PRISM §4 (MoE discriminator)** — Identify whether the perception/reasoning expert outputs are (a) two scalars, (b) per-step rationale-vs-perception scores, or (c) per-token credit. Promotion to `strict_opd` requires (b) or (c). Code in [github.com/XIAO4579/PRISM](https://github.com/XIAO4579/PRISM) should resolve.
3. **OVD §3 (verbal score → loss)** — Confirm whether the verbal score is consumed via REINFORCE/GRPO (scalar) or interpolated into a per-token weight. Project page [ovd.github.io](https://ovd.github.io/) needed.
4. **ORPO-Distill mixed-policy schedule** — Quantify the on-policy fraction across training; if the schedule asymptotically approaches 100% on-policy, label could move from `partial_opd` to `borderline_strict`.
5. **GAKD §3 + Corollary 1** — Confirm whether the implemented RKL term is on student rollouts or on teacher-generated sequences (importance-sampled). White-box, but relevant to WP1's adversarial-KD subfamily.
6. **PRISM "supervision distribution"** — Whether the 113K Gemini 3 Flash demonstrations are used as classifier targets only or also as direct SFT data. The latter would weaken C1 for the demonstration component.
7. **Verbal-score rubric stability** — For OVD-style methods, audit prompt rubric drift across rounds; report whether the abstract's "0–9" scale is fixed or evolving.
8. **GAD discriminator co-evolution dynamics** — Whether the discriminator is reset, frozen, or continuously updated. Determines reward-hacking surface.

---

## Sources

- [GAD — Black-Box On-Policy Distillation of Large Language Models (arXiv 2511.10643)](https://arxiv.org/abs/2511.10643)
- [Lion — Adversarial Distillation of Proprietary LLMs (arXiv 2305.12870, EMNLP 2023)](https://arxiv.org/abs/2305.12870)
- [OVD — On-Policy Verbal Distillation (arXiv 2601.21968)](https://arxiv.org/abs/2601.21968)
- [PRISM — Pre-alignment via Black-Box OPD for Multimodal RL (arXiv 2604.28123)](https://arxiv.org/abs/2604.28123)
- [ORPO-Distill — Mixed-Policy Preference Optimization (arXiv 2509.25100)](https://arxiv.org/abs/2509.25100)
- [SuperCorrect — Thought Template Distillation + Cross-Model DPO (arXiv 2410.09008, ICLR 2025)](https://arxiv.org/abs/2410.09008)
- [GAKD — Generative Adversarial Knowledge Distillation (OpenReview ICLR 2026 sub. 10374)](https://openreview.net/forum?id=9pfWYxYHAn)
- [BOND — Best-of-N Distillation (arXiv 2407.14622)](https://arxiv.org/abs/2407.14622)
- [Cross-Lingual Reward-Model Transfer (arXiv 2404.12318)](https://arxiv.org/abs/2404.12318)
- [thinkwee/AwesomeOPD — Black-Box section](https://github.com/thinkwee/AwesomeOPD)
- [thunlp/OPD — Tsinghua OPD repo](https://github.com/thunlp/OPD)
