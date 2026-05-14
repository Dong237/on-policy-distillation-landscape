# WP9 Line-Level Primary-Source Audit — VLA-OPD & PRISM

**Auditor:** Gemini (line-level primary-source pass)
**Date:** 2026-05-12
**Scope:** VLA-OPD (arXiv:2603.26666) and PRISM (arXiv:2604.28123) only. No survey, no new candidates.

---

## Memo

**VLA-OPD** — satisfies strict OPD. The paper explicitly describes a three-phase loop (Algorithm 1) in which:
- the student π_θ generates G action trajectories (C1),
- a frozen expert teacher π_tea is queried on the exact same student-visited states to produce per-token action logits (C2),
- the optimization objective consumes the teacher's log-probability per token via a Reverse-KL-based intrinsic reward `r_t = −(log π_θ − log π_tea)` plugged into a per-token policy-gradient update (C3).

Granularity is token-level (action-token). This is a textbook action-token OPD pattern, matching the decision rule that permits `strict_opd` for VLA action-token supervision when the student generates the action trajectory and an expert teacher supervises those same action tokens with a KL/distillation objective.

**PRISM** — does NOT satisfy strict OPD; keep below strict. The paper explicitly self-describes as "black-box, response-level adversarial" with a MoE discriminator, and is explicitly "logit-free" ("without requiring access to teacher logits"):
- C1 holds (policy rollouts from current policy).
- C2 is a discriminator, not a teacher providing same-rollout distributional targets.
- C3 is a minimax/adversarial loss where the policy is optimized via a response-level corrective signal from the MoE discriminator — a policy-gradient-style reward rather than non-scalar same-rollout distillation.

By the stated decision rule ("A response-level discriminator, verbal score, or policy-gradient reward should be borderline_strict or partial_opd, not strict_opd"), the verdict is **borderline_strict**: the MoE discriminator provides disentangled perception/reasoning signals (richer than a single scalar verifier), but the supervision is response-level and logit-free, so it falls short of strict token-level distillation. Confidence on the disqualifying facts (logit-free, response-level, adversarial) is high; the exact loss algebra from §3.2.4 was not retrievable from the public HTML, so the label is downgraded conservatively.

---

## Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | supervision_granularity | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|
| VLA-OPD | https://arxiv.org/abs/2603.26666 | Abstract; §3.1 Algorithm 1 (Phases 1–3); §3.3 "Dense Teacher Supervision" | "Phase 1 (Student Sampling): The student VLA policy interacts with the environment to collect on-policy trajectory rollouts" (Fig.1 caption / §3.1); "Generate G trajectories {τ_1,…,τ_G} using student π_θ(·\|o)" (Alg.1, line 8); "dense, token-level supervision on the student's self-generated trajectories" (Abstract) | "Phase 2 (Teacher Labeling): For each state visited by the student, a frozen expert teacher provides dense, token-level action labels" (Fig.1 caption); "Query student logits π_θ(·\|s_{t,i}) and teacher logits π_tea(·\|s_{t,i})" (Alg.1, line 12); "VLA-OPD leverages an expert teacher to provide dense, token-level supervision on the student's self-generated trajectories" (Abstract) | "r_t = −(log π_θ(a_{t,i}\|s_{t,i}) − log π_tea(a_{t,i}\|s_{t,i}))" — Negative Reverse-KL intrinsic reward (Alg.1, line 14); "∇J ≈ (1/(B·G)) Σ_j Σ_i Σ_t ∇_θ log π_θ(a_{t,i}\|s_{t,i}) · r_t" (Alg.1, line 20); "we formulate VLA-OPD via a Reverse-KL objective" (Abstract) | token-level (action-token) | strict_opd | none | high |
| PRISM | https://arxiv.org/abs/2604.28123 | Abstract; §1; §2.2; §3 overview; §3.2 "Distribution Alignment via On-Policy Distillation" (incl. §3.2.4 "Adversarial On-Policy Distillation") | "By optimizing on rollouts sampled from its current policy, on-policy distillation (OPD) mitigates exposure bias and encourages more faithful policy refinement" (§1); "The discriminator learns to separate policy rollouts from the supervision pool … while the policy is optimized to generate responses that increasingly resemble the supervision distribution" (§1) | "black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator with dedicated perception and reasoning experts, providing disentangled corrective signals that steer the policy toward the supervision distribution without requiring access to teacher logits" (Abstract); "it operates without teacher logits via adversarial discrimination, and it provides decoupled feedback through dedicated perception and reasoning experts" (§2.2) | "we formulate alignment as a minimax game … between the policy and a Mixture-of-Experts (MoE) discriminator" (§1); exact §3.2.4 loss algebra: not found in retrieved HTML | response-level (adversarial; MoE-decoupled into perception + reasoning experts, still response-level) | borderline_strict | Supervision is a response-level, logit-free MoE discriminator (adversarial/policy-gradient-style reward), not a non-scalar same-rollout token-level distillation signal; paper self-identifies as "black-box" and "without teacher logits"; fails the strict-OPD requirement that C3 consume a teacher distributional target rather than a scalar/vector reward from a discriminator | high |

---

## Decision-rule application

- **VLA-OPD → strict_opd.** The student generates the action trajectory, an expert teacher supervises those same action tokens, and the training objective is a Reverse-KL signal consumed per token. All three of C1/C2/C3 are met with direct, line-level quotes from Algorithm 1.
- **PRISM → borderline_strict.** C1 met. C2/C3 use a response-level, logit-free MoE adversarial discriminator. Per the stated rule, response-level discriminators / policy-gradient rewards must remain below strict; held at borderline (rather than partial) because the MoE provides disentangled perception+reasoning signals beyond a single scalar verifier. The §3.2.4 loss expression was not retrievable from the public HTML during this audit and is therefore not quoted; this is the conservative-downgrade trigger.

## Notes on unavailable evidence

- For PRISM §3.2.4 ("Adversarial On-Policy Distillation"), the exact loss equation (e.g., the minimax form, any discriminator-output-to-reward mapping, and any token-level vs. response-level decomposition) was not surfaced by the HTML retrieval used here. The label was set conservatively per the audit rule "If a quote is unavailable, write 'not found' and downgrade conservatively."
