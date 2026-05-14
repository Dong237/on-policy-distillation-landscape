# Speculative Decoding and OPD

Speculative decoding methods are included when they have an OPD-like training loop
(draft/student generates, target/teacher supervises those exact generations).

## Inclusion criteria

- Drafter or student generates candidate sequences during training.
- Target or teacher model supervises those generated sequences.

## Exclusion criteria

- Target-model generated data used offline to train a drafter.
- Inference-only speculative decoding.

---

## 1. DistillSpec (Zhou et al., ICLR 2024)

**Paper:** DistillSpec: Improving Speculative Decoding via Knowledge Distillation
**arXiv:** [2310.08461](https://arxiv.org/abs/2310.08461)
**OPD classification:** `strict_opd`

### Core idea

DistillSpec applies knowledge distillation to improve speculative decoding acceptance
rates. The draft model generates on-policy candidate tokens and the target model
provides distributional supervision via a tailored f-divergence.

### On-policy training loop

The general KD objective is (Equation 2):

```
theta* := argmin E_{(x,y)~G} [ D(M_p || M_{q_theta})(y|x) ]
```

where G can be constructed from draft-model generations (on-policy), target-model
generations, or mixed. The per-sequence divergence is:

```
D(M_p || M_{q_theta})(y|x) = (1/|y|) sum_{t=1}^{|y|} D(p(.|x,y_{<t}) || q_theta(.|x,y_{<t}))
```

**Key finding:** Using the draft model M_{q_theta} for on-policy data generation
achieves "similar or superior performance compared to the target model M_p, but at
much lower cost."

### F-divergence family (tailored divergence)

DistillSpec explores multiple divergences and finds the best choice is task-dependent:

| Divergence | Formula | Property |
|---|---|---|
| Forward KL | D_KL(P \|\| Q) | Mode-covering |
| Reverse KL | D_RKL(P \|\| Q) = D_KL(Q \|\| P) | Mode-seeking |
| Jensen-Shannon | D_JS(P \|\| Q) = (1/2)(D_KL(P \|\| M) + D_KL(Q \|\| M)), M = (P+Q)/2 | Symmetric |
| Total Variation | D_TVD(P, Q) | Direct acceptance rate proxy |
| Generalized JSD | D_JSD[beta](P \|\| Q) = beta D_KL(P \|\| beta P + (1-beta)Q) + (1-beta) D_KL(Q \|\| beta P + (1-beta)Q) | Interpolation |

### TVD-acceptance rate connection (Equation 3)

```
alpha(x) = 1 - E_{y~p<=T(y|x)} [ sum_{t=1}^{|y|} D_TVD(p(y_t), q(y_t)) ] / L_p(x)
```

This directly connects minimizing TVD to maximizing acceptance rate. However,
empirically TVD does not consistently yield best results -- other divergences perform
better on specific tasks.

### Theorem 4.1: On-policy justification

Bounds acceptance rate given on-policy KD loss epsilon:

```
E_{x~X}[alpha(x)] >= 1 - T * E_{x~X}[T / L_p(x)] * epsilon
```

Simplified (constant length T):

```
E_{x~X}[alpha(x)] >= 1 - T * epsilon
```

This proves on-policy distillation ensures improved acceptance rate despite optimizing
over student (draft) distribution rather than target distribution.

### Lossy speculative decoding extension

Lenience function f modifies acceptance:

```
min(1, f(p(y_t), epsilon) / q(y_t))
```

with evaluated functions: linear f_lin(p,e) = p/e, quadratic f_sq(p,e) = p/e^2,
exponential f_exp(p,e) = p^e.

### C1/C2/C3 evidence

- **C1:** Draft model M_{q_theta} generates candidate tokens on-policy.
- **C2:** Target model M_p distribution supervises those draft-generated states.
- **C3:** Tailored f-divergence objective directly trains the draft model on those
  states.

**Rollout freshness:** current_policy (draft generates during training).
**Teacher kind:** target_language_model (larger target for speculative decoding).

---

## 2. Speculative Knowledge Distillation (SKD) (2024)

**Paper:** Speculative Knowledge Distillation
**arXiv:** [2410.11325](https://arxiv.org/abs/2410.11325)
**OPD classification:** `partial_opd`

### Core idea

SKD uses a speculative-decoding-inspired interleaved generation protocol during
training: the student proposes tokens and the teacher conditionally accepts or rejects
them, creating mixed trajectories.

### Interleaved generation protocol (Algorithm 1)

For each training step and each decoding position i:

1. Sample token from student: `y_i ~ M_s(.|y_{<i}, x_j)`
2. Check if y_i falls in teacher's top-K: `if y_i not in topK(M_t(.|y_{<i}, x_j))`
3. If rejected, resample from teacher: `y_i ~ M_t(.|y_{<i}, x_j)`
4. Apply gradient descent minimizing the divergence

Default K = 25.

### Loss function (Equation 1)

```
D(M_t || M_s)(y|x) = (1/L_y) sum_{i=1}^{L_y} D(M_T(.|y_{<i}, x) || M_s(.|y_{<i}, x))
```

### Why this creates mixed-policy trajectories

The generated trajectory y contains a mix of student-proposed tokens (accepted by
top-K check) and teacher-resampled tokens (replacing rejected student tokens). This
means the prefix y_{<i} at each position t may contain tokens from either model.

**Dynamic behavior:**
- Early in training: high rejection rate, behaves like supervised KD.
- Late in training: high acceptance rate, behaves like on-policy KD.
- The method "dynamically adjusts the balance between supervised and on-policy KD
  based on the distribution gap between teacher and student models."

### How SKD differs from DistillSpec

| Aspect | DistillSpec | SKD |
|---|---|---|
| Trajectory source | Pure student on-policy | Mixed student+teacher tokens |
| Acceptance criterion | N/A during training | Top-K teacher membership |
| Divergence | Tailored f-divergence | Standard KL |
| Rollout freshness | current_policy | mixed_freshness |
| OPD strictness | strict_opd | partial_opd |

### C1/C2/C3 evidence

- **C1:** Weakened. Student proposes tokens but rejected tokens are replaced by teacher
  tokens, so the trajectory is mixed-policy rather than pure student on-policy.
- **C2:** Teacher logits supervise the interleaved trajectory.
- **C3:** Token-KL objective consumes the teacher supervision signal.

**OPD strictness:** `partial_opd` because C1 is weakened by the interleaved
teacher-corrected trajectories. The student does not generate pure on-policy rollouts.

---

## 3. Lightning OPD (2026)

**Paper:** Lightning OPD: Offline Approximation to On-Policy Distillation
**arXiv:** [2604.13010](https://arxiv.org/abs/2604.13010)
**OPD classification:** `partial_opd`

### Core idea

Lightning OPD precomputes teacher log-probabilities once over SFT rollouts and reuses
them during training, eliminating the need for a live teacher server. It achieves 4.0x
higher training efficiency versus standard OPD.

### Teacher consistency condition

Requires that the SFT-stage teacher pi_T^SFT and OPD-stage teacher pi_T^OPD be the
same model. Violating this introduces an irreducible gradient bias of magnitude
G * sigma_Delta, where sigma_Delta measures teacher divergence under the reference
policy distribution.

### Offline OPD dataset

Rollouts are sampled once from pi_ref (the SFT-initialized student):

```
D_OPD = {(q^j, x^j, {log pi_T(a_t^j | s_t^j)}_t) | x^j ~ pi_ref(.|q^j)}
```

### Loss functions

**Standard online OPD:**

```
J_on(theta) = E_{q~p, x~pi_theta} [ sum_t A_t(theta) ]
```

**Offline OPD approximation (Lightning OPD):**

```
J_off(theta) = E_{q~p, x~pi_ref} [ sum_t A_t(theta) ]
```

**Per-token advantage:**

```
A_t(theta) = log pi_T(a_t | s_t) - log pi_theta(a_t | s_t)
```

### Why C1 fails (rollouts are not current-policy)

The offline objective samples from pi_ref (fixed SFT-initialized policy), not from
pi_theta (current student policy). This means training data is not generated by the
current student policy, violating strict C1.

However, the paper proves this works under teacher consistency because:

1. At initialization, pi_theta = pi_ref, so both objectives coincide exactly
   (chi^2(pi_ref || pi_ref) = 0).
2. As the student diverges, the offline objective introduces implicit regularization.

### Gradient discrepancy bound (Theorem 3.5)

```
||grad J_on(theta) - grad J_off(theta)||_2 <= G * sigma_A * sqrt(chi^2(pi_theta || pi_ref))
```

### Teacher consistency impact (Theorem 3.11)

With mismatched teachers:

```
||grad J_on(theta) - grad J_off(theta)||_2 <= G * (sigma_A * sqrt(chi^2) + sigma_Delta)
```

The sigma_Delta term is "irreducible and persists even at initialization."

### Implicit regularization

The offline gradient decomposes as:

```
grad J_off(theta) = grad J_on(theta) - Cov_{pi_ref}[w(x;theta), f(x;theta)]
```

The covariance term vanishes when pi_theta = pi_ref and grows as the student diverges,
acting as an implicit trust-region penalty without explicit KL regularization.

### Algorithm

**Stage 1: SFT** -- Collect D_SFT from teacher-generated trajectories, fine-tune base model.

**Stage 2a: Preprocessing** -- For each prompt q^j, sample x^j from pi_ref, store
log pi_T(a_t^j | s_t^j) for all tokens t. Form D_OPD.

**Stage 2b: Training** -- Initialize pi_theta <- pi_ref. For each step: sample
mini-batch from D_OPD, compute A_t(theta), update theta.

### Performance

- Qwen3-8B-Base reaches 69.9% on AIME 2024 in 30 GPU hours (vs 120 for standard OPD).
- Qwen3-30B-A3B achieves 71.0% on AIME 2024 on a single 8xH100 node.

### C1/C2/C3 evidence

- **C1:** Fails current-policy freshness. Teacher log-probs are precomputed on SFT/cached
  rollouts from pi_ref, not from the evolving pi_theta.
- **C2:** Teacher log-probs exist and supervise the rollouts.
- **C3:** KL-like advantage objective consumes the teacher supervision.

**OPD strictness:** `partial_opd` because C1 fails strict current-policy requirement.
Under teacher consistency, Lightning OPD provably shares the same optimum as standard
OPD with bounded gradient discrepancy, but the training distribution is fixed.

---

## 4. DistiLLM / DistiLLM-2 (Ko et al., ICML 2024/2025)

**Paper:** DistiLLM / DistiLLM-2: Streamlined/Contrastive Distillation for LLMs
**arXiv:** [2402.03898](https://arxiv.org/abs/2402.03898) / [2503.07067](https://arxiv.org/abs/2503.07067)
**OPD classification:** `partial_opd`

### Core idea (DistiLLM-2)

Uses contrastive loss with different divergences for teacher-generated vs
student-generated outputs. Student-generated outputs (SGOs) are collected in batches
per epoch rather than fresh every iteration.

### Loss function (DistiLLM-2, Equation 2)

```
L_DistiLLM-2 = (1/2|D|) sum [ (1-beta) D^{alpha_t}_{SKL}(x, y_t) + beta D^{alpha_s}_{SRKL}(x, y_s) ]
```

where SKL = skew KL (for teacher responses), SRKL = skew reverse KL (for student
responses), and beta is gradually increased during training.

### On-policy status

Student generates its own outputs per epoch (batch approach), but these are
previous-epoch or batched -- not fresh current-policy per iteration.

### C1/C2/C3 evidence

- **C1:** Weakened. Student-generated outputs are batched/per-epoch, not fresh.
- **C2:** Teacher logits are used.
- **C3:** Contrastive skew-KL consumes the signal.

**OPD strictness:** `partial_opd` (rollout freshness is replay_buffer_stale).

---

## Summary table

| Method | Year | arXiv | OPD Strictness | C1 | C2 | C3 | Key innovation |
|---|---|---|---|---|---|---|---|
| DistillSpec | 2024 | 2310.08461 | strict_opd | Draft on-policy | Target distributional | Tailored f-divergence | Task-dependent divergence choice |
| SKD | 2024 | 2410.11325 | partial_opd | Mixed (teacher replaces) | Teacher top-K + logits | Token KL | Interleaved generation |
| Lightning OPD | 2026 | 2604.13010 | partial_opd | Precomputed SFT rollouts | Teacher log-probs cached | Advantage objective | 4x speedup via offline approx |
| DistiLLM-2 | 2025 | 2503.07067 | partial_opd | Batch per-epoch SGO | Teacher logits | Contrastive skew-KL | Different losses for TGO/SGO |
