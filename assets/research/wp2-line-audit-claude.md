# WP2 Line-Level Primary-Source Audit: Black-Box / API OPD

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-12
**Sources:** arXiv HTML full text for all 5 primary + 3 secondary methods
**Methodology:** Three independent research agents fetched primary-source HTML, extracted exact quotes, and assessed C1/C2/C3. Results reconciled against `taxonomies/strict-opd-definition.md` and the WP2 decision rules for black-box supervision.

---

## Scope

Primary queue (5 methods):

> GAD, PRISM, OVD, SODA, ORPO-Distill

Secondary spot-checks (3 methods):

> ADPA, PAD, daDPO

## OPD Test

Strict OPD requires all three conditions:

- **C1 (Student rollout):** The current student/policy generates the training trajectory.
- **C2 (Same-rollout supervision):** A black-box teacher, discriminator, preference judge, verbal-feedback model, or teacher-derived process supervises that exact student-generated state.
- **C3 (Distillation-style consumption):** The training objective consumes that supervision as a distillation signal, not merely as reward-only RLVR, scalar judge reward, or static preference optimization.

---

## Memo

### Central finding

**Strict black-box OPD remains empty after line-level audit.** None of the 5 primary candidates satisfy all three criteria. The structural barrier is confirmed: without token logits, black-box teacher supervision collapses to response-level signals (scalar scores, preferences, verbal feedback) that are consumed either as:

1. **Scalar rewards via GRPO/PPO** (GAD, PRISM) -- this is reward-based RL, C3 fails
2. **Rejection sampling filters** (OVD) -- verbal scores never enter the loss, C3 fails
3. **Static preference pairs via DPO** (SODA) -- frozen snapshot, not current-policy, C1 fails
4. **Independent teacher-vs-student preference pairs** (ORPO-Distill) -- teacher never supervises student rollout, C2 fails

### GAD and PRISM: honest about their mechanism

Both papers explicitly acknowledge the scalar-reward nature of their student objective:
- GAD: "we treat D(G(x)) as a reward" and "our discriminator can be interpreted as an on-policy reward model" (Section 1)
- PRISM: "divergence-based distillation inapplicable" (Section 3.2.1); KL regularization "deliberately disabled" (coeff=0.0)

The adversarial co-evolution of the discriminator with the student is a genuine structural advantage over static reward RL (discriminator adapts to student distribution, preventing reward hacking). This justifies `borderline_strict` over `partial_opd`. But from the student's perspective, the update is standard GRPO policy gradient on a scalar.

### OVD: verbal scores are a filter, not a loss signal

The most critical finding: OVD's verbal scores (0-9 from teacher) **never appear in any loss equation**. They are used exclusively for rejection sampling (Eq. 5: acceptance = 1[S(y) >= theta]). The actual GRPO training loss (Eq. 4) uses outcome-based rewards: R(y) = F1 for web Q&A, exact match for math (Eq. 2). The teacher's contribution to training is: (a) filtering which student trajectories enter training, and (b) providing replacement demonstrations for rejected trajectories. This is "RL with teacher-curated data," not distillation.

### SODA: the paper self-identifies as semi on-policy

SODA explicitly places itself between off-policy and on-policy: "the negative signal comes from the student's own distribution (on-policy), but is captured once and held fixed (off-policy in the temporal sense)" (Section 2.4). Student outputs come from frozen q0, not the evolving theta. The DPO loss trains on (teacher-chosen, frozen-student-rejected) pairs. The teacher never examines student output.

### ORPO-Distill: teacher never supervises student rollouts

The teacher generates its own CoT traces independently. Positive/negative labels come from ground-truth answer matching, not from teacher evaluation of student outputs. The teacher never sees, scores, or provides feedback on any student trace. C2 fails.

### Secondary spot-checks: all require white-box access

- **ADPA** (ICLR 2025 Spotlight): "precompute log pi_dpo(.|s_t) - log pi_ref(.|s_t) for the top 50 probabilities" -- requires teacher logits
- **PAD** (ACL 2025): "our method requires token-level probabilities, which are unavailable in some black-box models" -- explicitly white-box
- **daDPO**: "we consider the white-box setting, where the teacher model's distribution pi_te is fully accessible" -- explicitly white-box

PAD is structurally the closest to true OPD (student generates, teacher scores student outputs via log-likelihood, JSD distillation objective) but is explicitly white-box and belongs in WP1, not WP2.

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAD | https://arxiv.org/abs/2511.10643 | S2.2, Eq.2,5-7, S1, App.A.1 | "For each input prompt x, we sample a group of N student responses {y_s^i} from q_G(.\|x)" (A.1 Eq.7) | "the discriminator predicts a sequence-level scalar score D([x,y]) given prompt x and response y" (S2.1) | "we treat D(G(x)) as a reward and optimize it using policy gradient" (S2.2); r_s^i = D(y_s^i) -> GRPO advantage A^i (Eq.5-7); "We employ GRPO to train the student" | discriminator | response | grpo_reward | current_policy | borderline_strict | C3 fails strict: discriminator produces scalar score consumed as GRPO reward; paper says "discriminator can be interpreted as an on-policy reward model" (S1); no KL, no distributional target, no per-token likelihood matching; adversarial co-evolution justifies borderline over partial | high |
| PRISM | https://arxiv.org/abs/2604.28123 | S3.2.4, Eq.1-4, S3.2.1, App.A.1 | "we sample a group of N responses {y_i^-} from the current policy G(.\|x)" (S3.2.4 Eq.3) | "the policy is optimized to improve the quality of its own rollouts under the reward provided by the MoE discriminator" (S3.2.4) | "We alternate between updating the policy via GRPO and updating the two discriminator experts" (S3.2.4); r(x,y) = alpha*Dv + (1-alpha)*Dr -> GRPO advantage (Eq.1,3); "divergence-based distillation inapplicable" (S3.2.1) | discriminator | response | grpo_reward | current_policy | borderline_strict | C3 fails strict: MoE discriminator produces scalar reward consumed via GRPO; KL regularization "deliberately disabled" (coeff=0.0, App.A.1); paper says "divergence-based distillation inapplicable"; adversarial co-evolution justifies borderline | high |
| OVD | https://arxiv.org/abs/2601.21968 | S3.2-3.4, Eq.2-6 | "the student generates N trajectories per problem: y_i^(j) ~ pi_S(.\|x_i) for j=1,...,N" (S3.3) | "The teacher outputs a distribution over a discrete verbal vocabulary of size v=10 (scores {0,1,...,9})" (S3.2) | C3 FAILS: verbal scores used only for filtering via "a(y) = 1[S(y) >= theta]" (Eq.5); GRPO loss (Eq.4) uses outcome rewards "R(y) = F1 for web Q&A; exact match for math" (Eq.2); verbal score S(y) never appears in any loss equation | verbal | response | grpo_reward | current_policy | partial_opd | C3 fails: verbal scores are a data-curation filter, not a distillation signal consumed by the loss; actual gradient driven entirely by F1/exact-match outcome rewards; rejected trajectories replaced by teacher demonstrations; weaker than GAD/PRISM because scores don't even enter the loss as rewards | high |
| SODA | https://arxiv.org/abs/2604.03873 | S2.3-2.4, Eq.3-5 | C1 FAILS: "we sample responses from the base student q_0 before any fine-tuning" (Eq.3); paper says "captured once and held fixed (off-policy in the temporal sense)" (S2.4) | C2 PARTIAL: "teacher responses as preferred and the base small model's own responses as dispreferred" (Eq.4) -- teacher generates independently, never evaluates student output | "L_DPO(theta) = -E[log sigma(beta log q_theta(y+\|x)/q_w(y+\|x) - beta log q_theta(y-\|x)/q_w(y-\|x))]" (Eq.5) -- DPO on static teacher-chosen / frozen-student-rejected pairs | black_box | sequence | dpo_preference | static_snapshot | partial_opd | C1 fails: rollouts from frozen q0, not current theta; paper self-identifies as "semi on-policy" and acknowledges "utility diminishes as q_theta diverges"; C2 partial: teacher generates independently; C3: DPO on static pairs, not same-rollout supervision | high |
| ORPO-Distill | https://arxiv.org/abs/2509.25100 | S2.1-2.3 | "Rejected is a student CoT trace leading to a negative or incorrect answer" (S2.2) -- mixed/on-policy student traces | C2 FAILS: "Chosen is a teacher CoT trace leading to the positive or correct answer" (S2.2) -- teacher generates independently; positive/negative determined by ground-truth labels, NOT by teacher evaluation of student output | "L_OR = -log sigma(log odds_theta(y_P\|x) / odds_theta(y_N\|x))" (S2.1) -- ORPO contrastive preference between independent teacher and student traces | black_box | sequence | orpo_preference | mixed_policy | adjacent | C2 fails: teacher never examines, scores, or provides feedback on any student output; generates own traces independently; labels assigned by ground-truth answer matching; this is contrastive preference optimization between independently generated sets, not same-rollout supervision | high |

### Secondary spot-check table

| method | source_url | exact_section | black_box? | C1 | C2 | C3 | final_label | key_quote | confidence |
|---|---|---|---|---|---|---|---|---|---|
| ADPA | https://arxiv.org/abs/2502.17927 | S3.3, S4.1 | NO -- "precompute log pi_dpo(.\|s_t) - log pi_ref(.\|s_t) for the top 50 probabilities" (S4.1) | pass | partial (precomputed advantages) | pass (advantage-weighted distillation) | not_wp2 (white-box) | Requires teacher logit distributions; structurally a WP1 white-box method | high |
| PAD | https://arxiv.org/abs/2502.14272 | S4.1-4.3 | NO -- "our method requires token-level probabilities, which are unavailable in some black-box models" (Limitations) | pass | pass (teacher log-likelihood of student text) | pass (JSD preference distribution alignment) | not_wp2 (white-box) | Structurally closest to true OPD (student generates, teacher scores, JSD loss) but explicitly white-box; belongs in WP1 | high |
| daDPO | https://arxiv.org/abs/2506.15717 | S3.2, S4.2 | NO -- "we consider the white-box setting, where the teacher model's distribution, pi_te, is fully accessible" (S3) | pass | partial (role-based preference + pi_te(y^s)) | hybrid (DPO + distributional) | not_wp2 (white-box, off-policy) | Explicitly white-box; off-policy (pre-sampled responses); role-based preference assignment | high |

---

## Cross-validation with existing repo labels

| Method | Current repo label | This audit | Change? |
|---|---|---|---|
| GAD | partial_opd | borderline_strict | **UPGRADE** -- adversarial co-evolution of discriminator on student rollouts is stronger than static reward RL; discriminator encodes teacher distribution information adaptively |
| PRISM | borderline_strict | borderline_strict | confirmed |
| OVD | partial_opd | partial_opd | confirmed -- verbal scores never enter loss; weaker than GAD/PRISM |
| SODA | partial_opd | partial_opd | confirmed -- frozen snapshot q0; paper self-identifies as semi on-policy |
| ORPO-Distill | adjacent | adjacent | confirmed -- teacher never supervises student rollouts |

**One proposed upgrade:** GAD from partial_opd to borderline_strict. Rationale: GAD's adversarial discriminator co-evolves with the student, providing an adaptive, on-policy teacher-derived supervision signal. This is structurally equivalent to PRISM (already borderline_strict) -- both consume discriminator scalar rewards via GRPO. The key distinction from reward-only RL is that the discriminator continuously updates to distinguish current student outputs from teacher outputs, encoding teacher distribution information into the reward signal. Both papers are borderline for the same reason (C3 fails strict), so they should carry the same label.

---

## Reconciliation notes

### Why borderline_strict and not strict_opd for GAD/PRISM

Both methods satisfy C1 (on-policy student rollouts) and C2 (discriminator trained on teacher data supervises student rollouts). The discriminator co-evolves adversarially, which is structurally more than a static reward model. However:

1. The student's loss is standard GRPO policy gradient on a scalar reward
2. There is no distributional target, no KL divergence to teacher, no per-token likelihood matching
3. Both papers explicitly acknowledge this: GAD says "we treat D(G(x)) as a reward"; PRISM says "divergence-based distillation inapplicable"
4. PRISM deliberately disables KL regularization (coeff=0.0)

The adversarial structure resides in discriminator training, not in the form of the student's loss. The "distillation" occurs indirectly: teacher knowledge is compressed into the discriminator's parameters, which then shapes the student's reward landscape. This is a real form of knowledge transfer, but the student's objective is reward maximization, not distribution matching.

### Why partial_opd and not borderline_strict for OVD

OVD's verbal scores from the teacher never enter any loss equation. They serve exclusively as a rejection sampling filter. The actual GRPO loss uses F1/exact-match outcome rewards. This is strictly weaker than GAD/PRISM, where at least the discriminator score IS the reward in the loss. In OVD, the teacher contributes to training only through data curation (which trajectories enter training) and demonstration mixing (rejected trajectories replaced by teacher demos). The training objective itself is blind to the teacher.

### Why partial_opd and not adjacent for SODA

SODA's DPO loss does consume student-specific information (frozen q0 outputs as the rejected side), providing a student-distribution-aware contrastive signal. This is more than purely off-policy methods where the student never generates any training data. However, the frozen snapshot means the signal degrades as training progresses ("most informative when q_theta is still close to q_0"). The paper's own "semi on-policy" label is accurate.

### Why adjacent for ORPO-Distill

ORPO-Distill's teacher never examines student output. The teacher generates its own traces independently. Positive/negative labeling comes from ground-truth answer matching, not teacher evaluation. The teacher and student traces are paired contrastively, but the teacher provides no supervision of the student's specific rollout. This is contrastive preference optimization between independently generated sets -- a valid training strategy, but not same-rollout OPD.

---

## Verification methodology

1. **Primary-source fetching:** Three agents fetched arXiv HTML for GAD/PRISM, OVD/SODA, and ORPO-Distill/ADPA/PAD/daDPO respectively.

2. **Loss equation tracing:** For each method, the exact equation where the teacher signal enters the student's gradient was identified. The critical test: does the teacher signal appear as (a) a scalar reward in a policy gradient, (b) a filter/selector for data, or (c) a distributional/KL target in a distillation loss?

3. **Paper self-identification:** Both GAD and PRISM honestly describe their mechanisms as reward-based RL. OVD's verbal scores provably never enter any loss. SODA self-identifies as "semi on-policy."

4. **Conservative rules applied:**
   - Response-level discriminator scalar via GRPO = borderline_strict, not strict_opd
   - Verbal scores used only for filtering = partial_opd (C3 fails)
   - Frozen-snapshot DPO = partial_opd (C1 fails)
   - Teacher generating independently without examining student output = adjacent (C2 fails)
   - White-box teacher requirement = not WP2 scope (redirect to WP1)
