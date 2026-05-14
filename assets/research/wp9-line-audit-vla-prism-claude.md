# WP9 Line-Level Primary-Source Audit: VLA-OPD & PRISM

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML full text for both methods
**Methodology:** Direct web fetch of arXiv abstract pages + HTML method sections (`arxiv.org/html/`). Quotes extracted from method sections, equations, and algorithm boxes. Two rounds of extraction per paper: (1) abstract + section map, (2) targeted verbatim quote extraction for C1/C2/C3.

---

## Scope

Two methods only, per task specification:

- **VLA-OPD** (arXiv:2603.26666) — VLA on-policy distillation for robotics
- **PRISM** (arXiv:2604.28123) — Pre-alignment via black-box on-policy distillation for multimodal RL

No new papers introduced. No broad survey performed. No modifications to existing repo entries.

## OPD Test

Strict OPD requires all three conditions from `taxonomies/strict-opd-definition.md`:

- **C1 (Student rollout):** The current student/policy generates the training trajectory.
- **C2 (Same-rollout teacher-style supervision):** A teacher, expert, discriminator, reference model, previous checkpoint, or privileged-context model supervises that exact student trajectory.
- **C3 (Objective consumption):** The training objective consumes the teacher-style signal, not only a scalar reward.

---

## Memo

### VLA-OPD

Clean strict-OPD case. The paper describes a three-phase on-policy loop (S3.1): (1) the student VLA generates trajectories in the environment, (2) the frozen teacher is queried for action logits at every state the student visits, (3) the student is optimized via a Reverse-KL objective computed from per-token logit ratios.

The supervision is dense (every timestep), distributional (full action distributions, not scalars), and consumed directly as a KL divergence rather than as a scalar reward fed to policy gradient. The paper is explicit about the coupling: "By distilling knowledge strictly on trajectories the student naturally visits, our approach achieves a 'gentle' alignment" (S3.3). All three criteria are met without ambiguity.

### PRISM

PRISM satisfies C1 clearly: the student samples N responses per prompt from its current policy (S3.2.4). It also has a form of C2: the MoE discriminator evaluates those same student rollouts, and both discriminator experts are "optimized jointly with the policy...so that they function as on-policy discriminators that continuously adapt to the evolving rollout distribution" (S3.2.4).

The critical gap is C3. The discriminator produces a **single scalar reward per response** (`r(x,y) = alpha * Dv(x,c) + (1-alpha) * Dr(x,t)`, Eq.1), which is consumed via GRPO policy gradient with group-normalized advantages (`Ai = [r(x,yi) - mean] / std`, Eq.3). There is no token-level KL, no logit-level distillation loss, and the authors explicitly remove KL regularization: "we remove the KL regularization term commonly used to anchor the policy near its SFT initialization, since such a constraint would directly oppose the goal of correcting SFT-induced distributional drift" (S3.2.4). The paper frames this as a "response-level adversarial game" (S3.2.1) with minimax objective `min_theta max_phi E[r_phi(x,y+) - r_phi(x,y-)]` (Eq.4).

The adversarial co-adaptation of discriminator and policy distinguishes PRISM from fixed-reward RL and justifies `borderline_strict` over plain `partial_opd`. The discriminator continuously re-trains on the student's evolving rollouts, making it a genuine same-distribution supervisor. But the supervision channel is a scalar reward, not a distributional signal, failing strict C3.

---

## Decision Rules Applied

1. **VLA action-token supervision satisfies strict OPD** because the student generates the action trajectory and the expert teacher supervises those same action tokens with a Reverse-KL distillation objective (per task specification).

2. **PRISM remains below strict** because the paper shows only response-level scalar discriminator rewards consumed via GRPO policy gradient, not non-scalar same-rollout distillation supervision. A response-level discriminator reward is `borderline_strict`, not `strict_opd` (per task specification).

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | supervision_granularity | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|
| VLA-OPD | https://arxiv.org/abs/2603.26666 | S3.1 Framework Overview; S3.2 On-Policy Trajectory Sampling; S3.3 Dense Teacher Supervision; S3.4 Optimization Objective and Analysis | "Phase 1: On-Policy Sampling (Exploration). The student pi_theta generates trajectories T_student in the environment." (S3.1) | "For every timestep t in a student-generated trajectory tau in D_k, we query the teacher to obtain the target action distribution: q_t(a) = pi_tea(a\|s_t)." (S3.3); "For every state s_t visited by the student, the teacher provides its action logits." (S3.1) | "max_theta J(theta) = E_{s~pi_theta}[-D_KL(pi_theta(.\|s) \|\| pi_tea(.\|s))]" (S3.4 Eq.5); "r_t^{OPD}(s_t,a_t) = -(log pi_theta(a_t\|s_t) - log pi_tea(a_t\|s_t))" (S3.4 Eq.6) | Token-level (per-timestep action distributions from teacher; Reverse-KL computed at every action token) | strict_opd | none | high |
| PRISM | https://arxiv.org/abs/2604.28123 | S3.2 Distribution Alignment via On-Policy Distillation; S3.2.2 Mixture-of-Experts Discriminator; S3.2.4 Adversarial On-Policy Distillation | "For each input x, we sample a group of N responses {y_i^-}_{i=1}^N from the current policy G(.\|x)" (S3.2.4) | "r(x,y) = alpha * D_v(x,c) + (1-alpha) * D_r(x,t)" (S3.2.2 Eq.1); "both experts are optimized jointly with the policy throughout alignment, so that they function as on-policy discriminators that continuously adapt to the evolving rollout distribution" (S3.2.4) | "A_i = [r(x,y_i^-) - mean({r(x,y_j^-)})] / std({r(x,y_j^-)})" (S3.2.4 Eq.3) -- scalar rewards consumed via GRPO policy gradient; "we remove the KL regularization term" (S3.2.4); minimax game: "min_theta max_phi E[r_phi(x,y+) - r_phi(x,y-)]" (Eq.4) | Response-level (single MoE scalar per response; no token-level KL or logit distillation) | borderline_strict | C1 and C2 satisfied; C3 fails strict: discriminator provides response-level scalar rewards, not token-level distributional supervision; policy trained via GRPO policy gradient with group-normalized advantages, not KL/distillation objective; KL regularization explicitly removed; paper self-describes as "response-level adversarial game" | high |

---

## Comparison with ChatGPT Audit

Cross-referencing `wp9-line-audit-vla-prism-chatgpt.md`:

| Field | ChatGPT verdict | Claude verdict | Agreement |
|---|---|---|---|
| VLA-OPD label | strict_opd | strict_opd | agree |
| VLA-OPD confidence | high | high | agree |
| PRISM label | borderline_strict | borderline_strict | agree |
| PRISM C3 quote | "not found" | Found: Eq.3 (advantage), Eq.4 (minimax), KL removal quote | Claude found explicit scalar-reward evidence supporting downgrade |
| PRISM confidence | medium-high | high | Claude slightly more confident due to additional C3 evidence |

Both auditors converge on the same labels. The key difference is that this audit located explicit C3 quotes from PRISM (the advantage computation equation, the minimax objective, and the KL-removal statement) that confirm the scalar-reward mechanism, whereas the ChatGPT audit marked C3 as "not found."

---

## Verification Methodology

1. **Abstract extraction**: WebFetch of each `arxiv.org/abs/` page to identify title, authors, and high-level method.

2. **HTML full-text extraction**: Two rounds of WebFetch against `arxiv.org/html/` endpoints per paper:
   - Round 1: Section map + method summary with key equations.
   - Round 2: Targeted verbatim extraction for C1/C2/C3 quotes, specific equation numbers, and self-characterization statements.

3. **Conservative rules applied**:
   - Reward-only scalar signals default to borderline_strict or below.
   - Self-characterization ("response-level adversarial game") is treated as supporting evidence for downgrade.
   - Adversarial co-adaptation of discriminator justifies borderline_strict over partial_opd.
   - No inferences from title alone; all claims grounded in section-level quotes.
