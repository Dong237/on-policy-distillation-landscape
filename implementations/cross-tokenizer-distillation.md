# Cross-Tokenizer Distillation

Cross-tokenizer OPD is difficult because teacher and student probabilities may not
align token-by-token when vocabularies differ.

## Track

- Whether teacher and student share tokenizer.
- Whether token-level probabilities are available.
- Alignment method: sequence-level loss, span alignment, top-k transfer, optimal
  transport, byte-level interface, or auxiliary projector.
- Whether the method remains on-policy after alignment.

## Classification caution

Cross-tokenizer distillation can be strict OPD only when the aligned supervision is
applied to student-generated states during training.

---

## Landscape of Cross-Tokenizer KD Methods

### 1. ULD -- Universal Logit Distillation (Boizard et al., 2024)

**arXiv:** [2402.12030](https://arxiv.org/abs/2402.12030)
**Alignment method:** Optimal transport (Wasserstein distance)
**On-policy:** No (offline distillation)
**OPD classification:** `not_opd`

**Mechanism:** Uses the Wasserstein-1 distance between sorted probability vectors.
Both vocabularies are padded to equal size with zeros, sorted independently by
probability magnitude, then compared element-wise.

**Loss function (Equation 4):**

```
L_ULD = sum_{t=1}^{|x|} CE(t) + lambda * W_1[p_{theta_S}(.|x_{<t}^S), q_{theta_T}(.|x_{<t}^T)]
```

where lambda = 1.5 and W_1 is computed via sorting-based approximation (O(n log n)):

```
W_1 = sum_{t=1}^{|x|} sum_{i=1}^{|Omega|} |p(x_{sigma_S(i)}^S | x_{<t}^S) - q(x_{sigma_T(i)}^T | x_{<t}^T)|
```

**C1/C2/C3:** Missing C1 entirely. Student never generates; it trains on teacher
outputs via teacher forcing. Offline KD with cross-tokenizer innovation.

---

### 2. DSKD -- Dual-Space Framework (Zhang et al., 2025)

**arXiv:** [2504.11426](https://arxiv.org/abs/2504.11426)
**Alignment method:** Dual projectors + exact token alignment (ETA) algorithm
**On-policy:** Yes is claimed for an on-policy mode, while the method also has off-policy settings.
**OPD classification:** `partial_opd` at method level after candidate line audit. A strict on-policy-mode subrow could be split later if experiment-level evidence is needed.

**Mechanism:** Linear projectors map hidden states between teacher and student
representation spaces. The ETA algorithm aligns tokens between differently-tokenized
sequences by matching character-level boundaries.

**ETA alignment set (Equation 14):**

```
A = {(i,j) | y^t_{<j} = y^s_{<i}, y^t_j = y^s_i, hat{y}^t_j in V_{stu}}
```

**Cross-tokenizer projector initialization (Equation 16):**

```
W^{t->s} = tilde{W}^t * (tilde{W}^s)^+
```

using overlapped vocabulary tilde{V} = V_tea intersection V_stu.

**Off-policy loss (Equation 13):**

```
L_dskd = L^{kd}_{stu} + L^{ce}_{t->s} + L^{kd}_{tea}
```

**On-policy loss (Equation 18):**

```
L^{kd}_{op} = (1/|hat{y}|) sum D(p(y_i | hat{y}_{<i}, x) || q_theta(y_i | hat{y}_{<i}, x))
```

where hat{y} is sampled from q_theta(.|x). For cross-tokenizer on-policy mode, the
ETA algorithm uses student-generated outputs hat{y}^s and hat{y}^t.

**C1/C2/C3 (on-policy mode):**
- C1: Student generates with its own tokenizer.
- C2: Teacher supervises via projected distributions on aligned tokens.
- C3: Dual-space KL objective consumes the signal.

---

### 3. SimCT -- Recovering Lost Supervision (Sun et al., 2026)

**arXiv:** [2605.07711](https://arxiv.org/abs/2605.07711)
**Alignment method:** Minimal aligned units + multi-token continuation scoring
**On-policy:** Yes, source-verified by candidate line audit.
**OPD classification:** `strict_opd`.

The candidate line audit promotes SimCT as the cleanest strict cross-tokenizer OPD seed currently in the repo.

**Mechanism:** Constructs a common supervision space containing shared tokens and
"minimal aligned units" -- text continuations that both tokenizers can express.

**Supervision space (Equation 6):**

```
U_SimCT = (V_T intersection V_S) union A
```

where A is the set of minimal aligned units.

**Multi-token continuation score (Equation 7):**

```
s_M(u | x_{<t}) = (1/k) log p_M(u | x_{<t}) = (1/k) sum_{j=1}^k log p_M(v_j | x_{<t}, v_{<j})
```

**Normalized supervision distribution (Equation 8):**

```
q_M^SimCT(u | x_{<t}) = exp(s_M(u | x_{<t})) / sum_{u' in U_SimCT} exp(s_M(u' | x_{<t}))
```

**Loss function (Equation 9):**

```
L_SimCT(x_{<t}) = D(q_S^SimCT(. | x_{<t}), q_T^SimCT(. | x_{<t}))
```

where D is reverse KL in practice.

**Theorem 1 (Minimal aligned units):** These units form the "finest
boundary-consistent" partition jointly expressible by both tokenizers.

**Proposition 1 (Signal loss under coarsening):**

```
KL(q_S^min || q_T^min) >= KL(q_S^C || q_T^C)
```

proving coarser units erase within-unit distinctions.

**C1/C2/C3:**
- C1: Student generates prefixes using its own tokenizer and policy.
- C2: Teacher evaluates student-generated text via common supervision space.
- C3: Reverse-KL over SimCT supervision distributions consumes the signal.

---

### 4. Byte-Level Interface (Singh et al., 2026)

**arXiv:** [2604.07466](https://arxiv.org/abs/2604.07466)
**Alignment method:** Byte-level probability conversion via BPE covering
**On-policy:** No (offline, teacher probabilities precomputed)
**OPD classification:** `not_opd`

**Mechanism:** Converts teacher token probabilities to byte-level using the covering
approach. The student receives an additional lightweight byte-level decoder module
that is discarded after distillation.

**Byte probability:**

```
P_T(b) = sum_{y_i in cover_T(b)} prod_{t_j(i) in s_i} f_T(t_j(i) | t_{<j(i)})
```

**Loss:**

```
L = sum [ CE(delta(t_l), f_s(t_{<l})) + (1/n_l) sum CE(delta(b_j(l)), f_S^b(t_{<l}, j)) + KL(P_T(b_j(l) | b_{<j(l)}, t_{<l}), f_S^b(t_{<l}, j)) ]
```

with lambda_KL = 0.1, lambda_b = 1.0.

**C1/C2/C3:** Missing C1. Teacher byte probabilities are precomputed offline.

---

### 5. Multi-Level OT (Cui et al., 2024)

**arXiv:** [2412.14528](https://arxiv.org/abs/2412.14528)
**Alignment method:** Multi-level optimal transport (token + sequence)
**On-policy:** Not explicitly on-policy
**OPD classification:** `not_opd` or `adjacent`

**Loss:**

```
L = sum_t L_CE(y(t), s(t)) + alpha * (L_HAD + beta * L_SL + gamma * L_SD)
```

where L_HAD = holistic absolute difference, L_SL = sequential logarithmic,
L_SD = Sinkhorn distance. Default: alpha=0.15, beta=0.1, gamma=0.1.

---

### 6. Cross-Tokenizer Likelihood Scoring (Phan et al., 2025)

**arXiv:** [2512.14954](https://arxiv.org/abs/2512.14954)
**Alignment method:** BPE recursive structure for likelihood ratio computation
**On-policy:** Not specified
**OPD classification:** `adjacent` (scoring algorithm, not a full training method)

**Mechanism:** Leverages implicit recursive structure in BPE to compute exact
likelihoods across tokenizers. Subset case achieves O(1) model evaluations per token.

---

### 7. CTPD -- Cross-Tokenizer Preference Distillation (Nguyen et al., 2026)

**arXiv:** [2601.11865](https://arxiv.org/abs/2601.11865)
**Alignment method:** Aligned Span Projection to character-level spans
**On-policy:** Not explicitly specified
**OPD classification:** `adjacent` (DPO-style preference, not dense token OPD)

**Mechanism:** Maps teacher and student tokens to shared character-level spans.
Uses Token-level Importance Sampling DPO (TIS-DPO) and Teacher-Anchored Reference.

**Loss:**

```
L_CTPD = -E_{(x,y_w,y_l)~D} [ log sigma(beta (r(x,y_w) - r(x,y_l))) ]
```

where reward r(x,y) = sum_i w_i log(pi_theta(p_i|x,p_{<i}) / pi_ref(p_i|x,p_{<i})).

---

### 8. DWA-KD (Vu et al., 2026)

**arXiv:** [2602.21669](https://arxiv.org/abs/2602.21669)
**Alignment method:** Entropy-weighted token + Soft-DTW sequence alignment
**On-policy:** Not specified
**OPD classification:** `adjacent` or `partial_opd` (pending source audit)

**Mechanism:** Dual-space KL with entropy-based token weighting (up-weight where
student is uncertain and teacher is confident). Soft Dynamic Time Warping aligns
embedding and hidden-state layers.

---

### 9. CDM -- Contextual Dynamic Mapping (Chen et al., 2025)

**arXiv:** [2502.11104](https://arxiv.org/abs/2502.11104)
**Alignment method:** Contextual information for sequence alignment + dynamic
vocabulary mapping
**On-policy:** Not specified
**OPD classification:** `adjacent` (pending source audit)

---

### 10. SRA -- Span Representation Alignment (Dao et al., 2026)

**arXiv:** [2605.01205](https://arxiv.org/abs/2605.01205)
**Alignment method:** Center-of-mass attention-weighted span representations
**On-policy:** Not specified
**OPD classification:** `adjacent` (pending source audit)

**Mechanism:** Shifts alignment unit from tokens to tokenizer-agnostic spans, using
attention-weighted averaging for center-of-mass representation plus a geometric
regularizer.

---

### 11. TIDE -- Cross-Architecture Diffusion LLM Distillation (Zhang et al., 2026)

**arXiv:** [2604.26951](https://arxiv.org/abs/2604.26951)
**Alignment method:** Reverse CALM (inverted chunk-level likelihood matching)
**On-policy:** Not specified
**OPD classification:** `adjacent` (diffusion LLM distillation, not autoregressive OPD)

---

### 12. KDFlow Framework (Zhang et al., 2026)

**arXiv:** [2603.01875](https://arxiv.org/abs/2603.01875)
**Alignment method:** Extensible APIs for cross-tokenizer KD
**On-policy:** Supports both on-policy and off-policy
**OPD classification:** Framework, not a method. Strictness depends on configuration.

**Systems innovation:** Uses FSDP2 for student training and SGLang for teacher
inference. Transfers only teacher hidden states (not full logits) via zero-copy,
recomputing logits on the student side.

---

## Summary: Which cross-tokenizer methods qualify as strict OPD?

| Method | Year | Cross-tokenizer | On-policy | OPD strictness | Alignment |
|---|---|---|---|---|---|
| **SimCT** | 2026 | Yes | Yes | strict_opd | Minimal aligned units |
| **DSKD** | 2025 | Yes | Optional on-policy mode | partial_opd at method level | Projectors + ETA |
| ULD | 2024 | Yes | No | not_opd | Optimal transport sort |
| Byte-Level | 2026 | Yes | No | not_opd | Byte covering |
| Multi-Level OT | 2024 | Yes | No | not_opd / adjacent | Multi-level OT |
| CTPD | 2026 | Yes | Unclear | adjacent | Span projection |
| CDM | 2025 | Yes | Unclear | adjacent | Contextual mapping |
| SRA | 2026 | Yes | Unclear | adjacent | CoM span |
| DWA-KD | 2026 | Yes | Unclear | adjacent | DTW + entropy |
| TIDE | 2026 | Yes | Unclear | adjacent | Reverse CALM |
| KDFlow | 2026 | Yes | Configurable | framework | API-based |

**Key finding:** The candidate line audit promotes **SimCT** to strict cross-tokenizer
OPD. **DSKD v2** remains partial at method level because the on-policy mode is
optional and the main framework evidence is mixed.
