# WP1 Expanded Audit: White-Box OPD and Divergence/Objective Methods

Checked: 2026-05-12  
Scope: LLM white-box on-policy distillation. Primary sources only.  
Extends: `wp1-white-box-divergence-audit.md` (the baseline audit covering 14 methods).  
This document adds 20+ newly discovered methods and deepens the objective taxonomy.

---

## 1. Research Memo

### What was searched

- arXiv API and Semantic Scholar for papers citing GKD (2306.13649) and MiniLLM (2306.08543), 2024--2026.
- Keyword queries: "on-policy distillation", "online knowledge distillation LLM", "token-level distillation", "cross-tokenizer distillation", "adaptive divergence distillation", "f-divergence distillation LLM".
- Forward citation chains from DistiLLM, REOPOLD, DistillSpec, Veto, and Entropy-Aware OPD.
- Targeted searches for: top-K logit matching, sparse logit KD, variance reduction for KD gradients, position-aware token weighting, Renyi/alpha/Bregman divergence KD, progressive divergence scheduling.

### What was found (new additions)

10 new strict_opd candidates, 2 new borderline/partial candidates with strict substages, and 8+ adjacent methods with novel objectives combinable with OPD. The field has grown rapidly since early 2026, with most innovation in:

1. **Token-level importance weighting** (TIP, Rock Tokens, SOD, SelecTKD): which tokens to train on during OPD.
2. **Variance reduction** (vOPD, AOPD): stabilizing the policy-gradient view of reverse-KL OPD.
3. **Failure mode fixes** (StableOPD for length inflation, CaOPD for miscalibration, Veto for gradient explosion).
4. **Cross-tokenizer OPD** (SimCT, DSKD): the first truly on-policy cross-tokenizer methods.

### What was ruled out

- **No f-divergence generalizations** (alpha-divergence, Renyi divergence) have been published specifically for on-policy LLM distillation as of May 2026. JSD is the most exotic divergence used in practice.
- **EGAD** (2605.01732) is off-policy (supervised distillation, not on student rollouts).
- **OKD** (2409.12512) adapts the teacher online instead of generating student rollouts -- adjacent.
- **DynSDPB** (2411.16991) uses previous mini-batch logits as self-teacher -- adjacent self-distillation.
- **LUFFY** (2504.14945) uses off-policy trajectories with GRPO, not white-box teacher logit supervision -- adjacent RL.

### What changed the landscape

The OPD-as-RL equivalence (independently proven by G-OPD, REOPOLD, KDRL, vOPD) is now the dominant theoretical framework. Most new 2026 methods build on this view and address its instabilities rather than proposing entirely new divergences. The practical frontier has shifted from "which divergence?" to "which tokens to distill on?" and "how to stabilize the policy-gradient surrogate?"

---

## 2. Expanded Method Family Taxonomy

### Family A: Canonical Distribution Matching (Foundational)

| Method | Divergence | Key property |
|--------|-----------|-------------|
| GKD | Generalized JSD(beta), FKL, RKL | Lambda-mixing on/off-policy; stop-gradient on sampling |
| MiniLLM | Sequence reverse KL | Policy gradient with teacher log-ratio rewards; teacher-mixed sampling |

### Family B: Skewed/Smoothed KL (Partial OPD)

| Method | Divergence | Key property |
|--------|-----------|-------------|
| DistiLLM | Skew KL: `KL(p || alpha*p + (1-alpha)*q)` | Replay buffer + adaptive off-policy; alpha=0.1 |
| DistiLLM-2 | Contrastive SKL/SRKL with adaptive per-sample alpha | Pre-generated SGO; asymmetric loss-data pairing |
| AMiD | Alpha-mixture assistant distribution family | Continuous interpolation generalizing DistiLLM; off-policy |

### Family C: Adaptive/Entropy-Gated Divergence

| Method | Divergence | Key property |
|--------|-----------|-------------|
| Entropy-Aware OPD | RKL + binary-gated top-k FKL (`I[H_T > tau]`) | Preserves teacher uncertainty at high-entropy positions |
| Veto | KL to geometric bridge `Q ~ P_T * P_S^beta` | Polynomial damping suppresses pathological gradients |
| SCOPE | KL on incorrect + MLE on correct (correctness-routed) | Perplexity-weighted adaptive teacher confidence gating |
| ToDi | Per-token sigmoid-weighted FKL/RKL blend | Log-ratio-based automatic divergence selection per token |
| CaOPD | OPD with calibration-corrected targets | Fixes systematic overconfidence in standard OPD |
| DPSW | Dual-perspective safety weighting (teacher conf. x student risk) | Safety alignment; product-of-experts token weights |

### Family D: Dense Log-Ratio / OPD-as-RL

| Method | Divergence | Key property |
|--------|-----------|-------------|
| G-OPD | `lambda * log(pi_T/pi_ref) - KL(pi_theta || pi_ref)` | Reward extrapolation (lambda > 1) surpasses teacher |
| REOPOLD | Clipped RKL with entropy-based token masking | Mixture-based reward clipping; exploration-to-refinement phases |
| KDRL | `J_GRPO - beta * D_KL^{k2}(pi_theta || pi_T)` | Joint loss; k2 is unbiased gradient estimator |
| vOPD | RKL with closed-form control variate baseline | `V(c_t) = -D_KL(pi_theta(.|c_t) || pi_T(.|c_t))`; no critic |
| AOPD | Asymmetric advantage: divergence minimization replaces negative reinforcement | Addresses high variance and vanishing gradients |

### Family E: Token Importance / Selective Training

| Method | Selection mechanism | Key property |
|--------|-------------------|-------------|
| TIP | 2D soft-OR: student entropy x teacher-student divergence | Top-rho fraction; 50% tokens matches full training |
| Rock Tokens | Persistent high-loss token identification + gradient freeze | Context-aware rock score; causal intervention verification |
| SOD | Step-level divergence ratio reweighting | Cascading error dampening for agentic/tool-use |
| SelecTKD | Propose-and-verify: student proposes, teacher accepts/rejects | Token Acceptance Rate as implicit curriculum |
| RSKD | Importance-sampling-based random sparse logit caching | Unbiased gradient estimate with <10% logit overhead |
| BiLD | Top-k bi-directional logit differences | Filters long-tail noise; preserves internal ranking |

### Family F: Stabilization and Failure Mode Fixes

| Method | Failure mode addressed | Key property |
|--------|----------------------|-------------|
| StableOPD | Truncation collapse / length inflation | Reference divergence constraint + rollout mixture |
| CaOPD | Systematic miscalibration (overconfidence) | Empirical confidence from rollouts replaces teacher targets |
| Veto | Forward KL gradient explosion on low-confidence tokens | Polynomial damping via geometric bridge |
| REOPOLD | Heavy-tailed negative rewards; entropy collapse | Mixture-based clipping + entropy-gated masking |

### Family G: Efficiency and Systems OPD

| Method | Optimization | Key property |
|--------|-------------|-------------|
| Fast OPD | Prefix truncation of student rollouts | 2x--47x FLOP reduction; prefix scheduling |
| Lightning OPD | Offline precomputation of teacher log-probs | 4x speedup; fixes rollouts to pi_ref (not strict) |
| DistillSpec | Draft-model alignment for speculative decoding | Task-dependent f-divergence selection |
| Speculative KD | Interleaved student-propose/teacher-replace | Dynamic on/off-policy balance (partial) |
| AdaSPEC | Selective token filtering for speculative decoders | Reference model identifies hard tokens to skip |

### Family H: Cross-Tokenizer OPD (New)

| Method | Alignment mechanism | Key property |
|--------|-------------------|-------------|
| SimCT | Minimal aligned units via multi-token continuation scoring | First strict cross-tokenizer OPD; Theorem 1: finest boundary-consistent partition |
| DSKD | Dual projectors with Exact Token Alignment (ETA) | Strict in on-policy mode; projector init from overlapped vocab |
| ULD | Wasserstein distance on sorted probability vectors | Off-policy only; adjacent |
| MultiLevelOT | Token-level OT with Sinkhorn distance | Off-policy; novel Wasserstein objective |
| DWA-KD | Dual-space entropy weighting + soft DTW | Off-policy; novel token weighting |

### Family I: Self-Distillation OPD (Privileged Context)

| Method | Teacher substitute | Key property |
|--------|-------------------|-------------|
| OPSDL | Short-context self-teacher for long-context generation | Point-wise RKL; long-context generalization |
| OPSD | Privileged-context self (verified reasoning traces) | Token-level KL; already tracked in WP3 |
| OPSDC/CRISP | Concise self-view for reasoning compression | RKL for reasoning compression; already tracked in WP3 |

### Family J: Privacy and Safety Variants

| Method | Modification | Key property |
|--------|-------------|-------------|
| DP-OPD | DP-SGD on student only; frozen teacher | Standard OPD loop with differential privacy noise |
| DPSW | Dual-perspective safety token weighting | Combined teacher confidence x student risk |

---

## 3. Evidence Ledger (New Methods)

### Strict OPD Candidates

| id | Title | arXiv | Year | C1 | C2 | C3 | Classification | Exact objective | Rollout freshness | Teacher access / kind | Evidence strength |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vopd-2026 | KL for a KL: On-Policy Distillation with Control Variate Baseline | 2605.07865 | 2026 | Yes: student generates trajectories | Yes: teacher logits provide KL signal | Reverse KL with variance-reduced PG surrogate; `V(c_t) = -D_KL(pi_theta \|\| pi_T)` as closed-form baseline | `strict_opd` | Reverse KL with control variate: `a_t = r_t + D_KL(pi_theta(.\|c_t) \|\| pi_T(.\|c_t))` | current_policy | White-box teacher logits; larger_llm | Medium: preprint, needs line audit |
| tip-2026 | Token Importance in On-Policy Distillation | 2604.14084 | 2026 | Yes: student on-policy rollouts | Yes: teacher distributions on student states | Standard OPD with selective token masking; top-rho by soft-OR score | `strict_opd` | `L_TIP = (1/\|T\|) sum_{t in T} D_KL(P_S \|\| P_T)` where `T = TopK(s_t, rho*m)` | current_policy | White-box teacher logits; larger_llm | Medium: preprint, novel selection mechanism |
| aopd-2026 | Asymmetric On-Policy Distillation | 2605.06387 | 2026 | Yes: student on-policy rollouts | Yes: teacher log-probs provide advantage | Asymmetric advantage: divergence minimization replaces negative reinforcement in non-positive-advantage regions | `strict_opd` | Asymmetric PG with localized divergence minimization for A_t <= 0 | current_policy | White-box teacher logits; larger_llm | Medium: preprint, May 2026 |
| stableopd-2026 | Demystifying OPD: Length Inflation and Stabilization | 2604.08527 | 2026 | Yes: student generates trajectories (which exhibit length inflation) | Yes: teacher supervises those trajectories | Modified OPD with reference divergence constraint + rollout mixture | `strict_opd` | OPD + KL(pi_theta \|\| pi_ref) reference constraint + rollout mixture distillation | current_policy | White-box teacher logits; larger_llm | Medium: preprint, identifies truncation collapse |
| caopd-2026 | Decoupling Capability and Calibration in OPD | 2604.16830 | 2026 | Yes: student rollouts for training and confidence estimation | Yes: teacher distribution used then corrected | OPD with calibration-corrected targets from student rollout statistics | `strict_opd` | Standard OPD loss with empirical confidence replacing teacher-conditioned targets | current_policy | White-box teacher logits; larger_llm | Medium: preprint, Salesforce AI Research |
| sod-2026 | Step-wise On-policy Distillation | 2605.07725 | 2026 | Yes: student generates agentic trajectories | Yes: teacher logits supervise those trajectories | Step-level divergence-reweighted KL + GRPO: `L = L_GRPO + L_OPD^step` | `strict_opd` | `d_k = mean \|log pi_theta - log pi_T\|`; `w_k = min(prod(d_u/d_{u+1}), 1+delta)` | current_policy | White-box teacher logits; larger_llm | Medium: preprint, agentic focus |
| simct-2026 | Recovering Lost Supervision for Cross-Tokenizer OPD | 2605.07711 | 2026 | Yes: student on-policy with own tokenizer | Yes: teacher multi-token continuation scoring on aligned units | Reverse KL over SimCT supervision distributions with minimal aligned units | `strict_opd` | RKL over `U_SimCT = (V_T intersect V_S) union A`; Theorem 1: finest boundary-consistent partition | current_policy | White-box teacher logits; larger_llm (cross-tokenizer) | Medium: preprint, first strict cross-tokenizer OPD |
| opsdl-2026 | On-Policy Self-Distillation for Long-Context LMs | 2604.17535 | 2026 | Yes: student generates in long-context mode | Yes: short-context self-teacher supervises on those outputs | Point-wise reverse KL under relevant short-context | `strict_opd` | Point-wise RKL: privileged short-context self-teacher | current_policy | Privileged short-context self; privileged_self | Medium: preprint, self-distillation variant |
| dp-opd-2026 | Differentially Private On-Policy Distillation | 2604.04461 | 2026 | Yes: student generates trajectories | Yes: frozen teacher provides dense supervision | Standard OPD objective with DP-SGD noise on student gradients | `strict_opd` | OPD + DP-SGD (privacy constraint only) | current_policy | White-box frozen teacher logits; larger_llm | Medium: preprint, privacy variant |
| rock-tokens-2026 | Cornerstones or Stumbling Blocks? | 2605.09253 | 2026 | Yes: student on-policy rollouts | Yes: teacher KL supervision on student states | KL with gradient freeze on persistent high-loss tokens; causal intervention verification | `strict_opd` | `L_weighted = E sum w(x_t,t) * l_t`; `w=lambda` if rock token, else 1; lambda=0 is gradient freeze | current_policy | White-box teacher logits; larger_llm | Medium: preprint, diagnostic + fix |

### Borderline/Partial Candidates

| id | Title | arXiv | Year | C1 | C2 | C3 | Classification | Exact objective | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dskd-2025 | Dual-Space KD via Projectors and ETA | 2504.11426 | 2025 | Yes in on-policy mode | Yes via dual projectors | Cross-vocabulary KL with ETA alignment | `partial_opd` to `strict_opd` (mode-dependent) | Projector-aligned KL across vocabularies | Needs line audit for on-policy loop confirmation |
| hpd-2026 | Hybrid Policy Distillation | 2604.20244 | 2026 | Mixed on/off-policy sampling | Yes: teacher logits | Hybrid FKL+RKL with mixed-policy reweighting | `partial_opd` | Token-level reweighted log-likelihood | Off-policy component weakens C1 |
| selectkd-2025 | Selective Token-Weighted KD | 2510.24021 | 2025 | Conditional: supports on-policy mode | Yes: teacher top-k verification | Acceptance-masked KL with TAR curriculum | `partial_opd` to `strict_opd` | Token Acceptance Rate implicit curriculum | Depends on data mode; needs on-policy mode audit |
| tvd-plus-2024 | TVD++ for Speculative Decoding Draft Alignment | 2403.00858 | 2024 | Partial: mixed teacher/draft data | Yes: target logits | TVD with PG variance reduction | `partial_opd` | Total Variation Distance ++ | Speculative decoding context |
| adaswitch-2025 | AdaSwitch: Adaptive Switching for LLM Distillation | 2510.07842 | 2025 | Partial: student until threshold, then teacher takes over | Yes: teacher logits on hybrid trace | KL on switched sequences | `partial_opd` | `tau = K * d_bar; switch to teacher if d_i > tau` | Hybrid student-teacher generation |
| todi-2025 | Token-wise Distillation via Fine-Grained Divergence Control | 2505.16297 | 2025 | Needs verification (likely off-policy base) | Yes: teacher logits per token | Per-token sigmoid-weighted FKL/RKL | `partial_opd` or `adjacent` | Sigmoid blend of FKL/RKL per token based on log-ratio | Novel objective; needs C1 confirmation |

### Adjacent Methods (Novel Objectives, Not On-Policy by Default)

| id | Title | arXiv | Year | Why adjacent | Novel contribution |
|---|---|---|---|---|---|
| csd-2025 | Concrete Score Matching for LLM Distillation | 2509.25837 | 2025 | Not inherently on-policy | Discrete score matching bypassing softmax; non-KL/non-f-divergence |
| amid-2025 | Alpha-Mixture Assistant Distribution KD | 2510.15982 | 2025 | Off-policy objective formulation | Continuous interpolation generalizing DistiLLM skew KL family |
| bild-2024 | Bi-directional Logits Difference | 2406.13555 | 2024 | Likely off-policy | Top-k logit difference preserving ranking; filters long-tail noise |
| rskd-2025 | Sparse Logit Sampling for KD | 2503.16870 | 2025 | Efficiency enabler, not on-policy by itself | Importance-sampling-based unbiased sparse logit caching |
| okd-2024 | Online Teacher Adaptation KD | 2409.12512 | 2024 | Teacher adapts, not student on-policy | Online modules in teacher adapt to student distribution |
| egad-2026 | Entropy-Guided Adaptive Distillation | 2605.01732 | 2026 | Off-policy supervised distillation | Entropy curriculum + adaptive temperature per token |
| multilevelot-2024 | Multi-Level Optimal Transport for Cross-Tokenizer KD | 2412.14528 | 2024 | Off-policy; novel Wasserstein objective | Sinkhorn distance at token and sequence level |
| uld-2024 | Universal Logit Distillation | 2402.12030 | 2024 | Off-policy only | Wasserstein on sorted probability vectors; cross-tokenizer |
| dwa-kd-2026 | Dual-Space Weighting + Time-Warped Alignment | 2602.21669 | 2026 | Likely off-policy | Dual-space entropy weighting + soft DTW |
| dynsdpb-2024 | Dynamic Self-Distillation from Previous Mini-Batches | 2411.16991 | 2024 | Previous mini-batch self-teacher; no external teacher | Dynamic temperature self-KL from prior iteration |

---

## 4. Objective Taxonomy (Expanded)

### 4a. KL-Family Divergences (on student-generated states)

| Divergence | Formula (per token position t) | Methods |
|-----------|-------------------------------|---------|
| Forward KL | `D_KL(pi_T(.\|c_t) \|\| pi_theta(.\|c_t))` | GKD (FKL variant), PACED (forward track), DistillSpec (FKL variant) |
| Reverse KL | `D_KL(pi_theta(.\|c_t) \|\| pi_T(.\|c_t))` | MiniLLM, Fast OPD, REOPOLD, vOPD, SimCT |
| Generalized JSD(beta) | `beta*KL(P\|\|beta*P+(1-beta)*Q) + (1-beta)*KL(Q\|\|beta*P+(1-beta)*Q)` | GKD |
| Skew Forward KL | `KL(p \|\| alpha*p + (1-alpha)*q)` | DistiLLM |
| Skew Reverse KL | `KL(q \|\| (1-alpha)*p + alpha*q)` | DistiLLM, DistiLLM-2 |
| Contrastive SKL/SRKL | SKL on teacher data + SRKL on student data (asymmetric) | DistiLLM-2 |
| Entropy-gated hybrid | `L_RKL + I[H_T > tau] * L_FKL^{top-k}` | Entropy-Aware OPD |
| Geometric bridge KL | `KL(Q \|\| pi_theta)` where `Q ~ pi_T * pi_theta^beta` | Veto |
| Calibration-corrected KL | Standard KL with empirical confidence targets | CaOPD |
| DP-noised KL | Standard KL + DP-SGD gradient noise | DP-OPD |

### 4b. Dense Log-Ratio / Policy-Gradient Surrogates

| Variant | Token reward/advantage | Methods |
|---------|----------------------|---------|
| Standard OPD advantage | `A_t = log pi_T(a_t\|s_t) - log pi_theta(a_t\|s_t)` | REOPOLD, KDRL (k2), Lightning OPD |
| Lambda-scaled reward | `r = lambda * log(pi_T/pi_ref)` | G-OPD / ExOPD |
| Control variate baseline | `a_t = r_t + D_KL(pi_theta(.\|c_t) \|\| pi_T(.\|c_t))` | vOPD |
| Asymmetric advantage | Replace negative reinforcement with divergence minimization | AOPD |
| Clipped/relaxed reward | `max(R_t, log(lambda)/(1-lambda))` | REOPOLD |

### 4c. Token Selection / Weighting Mechanisms

| Mechanism | Selection criterion | Methods |
|-----------|-------------------|---------|
| All tokens (baseline) | None | GKD, MiniLLM, standard OPD |
| Entropy gate (binary) | `I[H_T > tau]` | Entropy-Aware OPD |
| 2D soft-OR score | `s_t = h_t + delta_t - h_t*delta_t` (entropy x divergence) | TIP |
| Persistent loss (rock score) | `R(v) = mean_loss(v) * freq(v)` | Rock Tokens |
| Step-level divergence ratio | `w_k = min(prod(d_u/d_{u+1}), 1+delta)` | SOD |
| Propose-and-verify (TAR) | Student proposes, teacher accepts/rejects via top-k | SelecTKD |
| Correctness routing | Binary verifier splits rollouts | SCOPE |
| Perplexity weighting | Teacher PPL (incorrect), Student PPL (correct) | SCOPE |
| Entropy-based masking | `M_t = I[H_t >= tau_beta]` (refinement phase) | REOPOLD |
| Reference-model filtering | Reference model identifies hard tokens to skip | AdaSPEC |
| Gradient freeze | `w=0` for rock tokens; causal intervention verification | Rock Tokens |

### 4d. Stabilization Techniques

| Technique | Instability addressed | Methods |
|-----------|---------------------|---------|
| Geometric bridge target | FKL gradient explosion on low-P_S tokens | Veto |
| Mixture-based reward clipping | Heavy-tailed negative log-ratio rewards | REOPOLD |
| Reference divergence constraint | Truncation collapse / length inflation | StableOPD |
| Teacher-mixed sampling | Reward hacking in pure student sampling | MiniLLM |
| Rollout mixture distillation | Distribution shift after policy updates | StableOPD |
| Exploration-to-refinement phasing | Entropy collapse killing exploration | REOPOLD |
| Prefix truncation + scheduling | Compute cost of full-length OPD | Fast OPD |
| Offline OPD approximation | Online teacher serving cost | Lightning OPD |
| Calibration correction | Systematic overconfidence | CaOPD |
| DP-SGD noise | Privacy leakage from teacher | DP-OPD |

### 4e. Cross-Tokenizer Alignment

| Technique | Alignment approach | On-policy? | Methods |
|-----------|-------------------|-----------|---------|
| Minimal aligned units | Multi-token continuation scoring; boundary-consistent partition | Yes (strict) | SimCT |
| Dual projectors + ETA | Learned projectors with overlapped vocab initialization | Yes (in on-policy mode) | DSKD |
| Wasserstein on sorted vectors | Optimal transport without token alignment | No | ULD |
| Multi-level OT + Sinkhorn | Token and sequence OT | No | MultiLevelOT |
| Dual-space DTW | Soft dynamic time warping at embedding/hidden layers | No | DWA-KD |
| Byte-level interface | Convert both to byte sequences | No | BLD |

---

## 5. C1/C2/C3 Classification Summary

### Strict OPD (all three criteria met)

**Previously verified (14 methods):**
GKD, MiniLLM, DistillSpec, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD, SCOPE (borderline_strict), KDRL (borderline_strict), SDPO, OPCD, OPSD, OPSDC/CRISP

**New strict candidates (10 methods, pending line audit):**

| Method | C1 evidence | C2 evidence | C3 evidence |
|--------|------------|------------|------------|
| vOPD | Student generates trajectories | Teacher logits provide KL signal | RKL with closed-form control variate baseline |
| TIP | Student on-policy rollouts | Teacher distributions on student states | Standard OPD with top-rho token masking by soft-OR score |
| AOPD | Student on-policy rollouts | Teacher log-probs provide advantage | Asymmetric PG: divergence minimization for negative advantage |
| StableOPD | Student generates (with length inflation) | Teacher supervises those trajectories | OPD + reference divergence constraint + rollout mixture |
| CaOPD | Student rollouts for training and confidence | Teacher distribution corrected for calibration | OPD with calibration-corrected targets |
| SOD | Student generates agentic trajectories | Teacher logits supervise | Step-level divergence-reweighted KL + GRPO |
| SimCT | Student on-policy with own tokenizer | Teacher multi-token continuation scoring | RKL over SimCT minimal aligned units |
| OPSDL | Student generates in long-context mode | Short-context self-teacher supervises | Point-wise reverse KL |
| DP-OPD | Student generates trajectories | Frozen teacher dense supervision | Standard OPD + DP-SGD noise |
| Rock Tokens | Student on-policy rollouts | Teacher KL on student states | KL with gradient freeze on persistent outliers |

### Partial OPD (C1 weakened or mixed)

**Previously classified:**
DistiLLM, DistiLLM-2, Speculative KD, PACED, Lightning OPD, AdaSwitch

**New partial candidates:**
DSKD (mode-dependent), HPD (mixed policy), SelecTKD (mode-dependent), TVD++ (mixed data), ToDi (needs C1 audit)

### Adjacent (novel objective, not on-policy by default)

CSD, AMiD, BiLD, RSKD, OKD, EGAD, MultiLevelOT, ULD, DWA-KD, DynSDPB

---

## 6. Rollout Freshness Summary

| Freshness level | Methods |
|----------------|---------|
| `current_policy` (strict) | GKD (lambda=1), MiniLLM, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD, DistillSpec (draft variant), vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens |
| `per_iteration` | REOPOLD (near-policy refreshed), KDRL |
| `mixed_freshness` | PACED (pass-rate on-policy, FKL track off-policy), AdaSwitch, HPD, SelecTKD |
| `replay_buffer_stale` | DistiLLM |
| `precomputed_sft_rollouts` | Lightning OPD, DistiLLM-2 (pre-generated SGO) |
| `fresh_hybrid_trace` | Speculative KD (interleaved student/teacher tokens) |

---

## 7. Teacher Access and Teacher Kind Summary

| Teacher kind | Methods |
|-------------|---------|
| `larger_llm` (white-box logits) | GKD, MiniLLM, DistiLLM, DistiLLM-2, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD, PACED, vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, DP-OPD, Rock Tokens |
| `reference_model` (target LM) | DistillSpec, Speculative KD, TVD++, AdaSPEC |
| `privileged_self` | OPSDL (short-context self), OPSD, OPSDC/CRISP, SDPO |
| `larger_llm + reference` | G-OPD (teacher + reference model) |
| `multi_teacher` | Already tracked in WP5/other WPs (MAD-OPD, Nemotron, MiMo) |

---

## 8. False Positives and Downgrade Notes

### Confirmed downgrades (from baseline audit)

1. **DistiLLM** -> `partial_opd`: Paper explicitly says "adaptive off-policy"; code has `ReplayBuffer` with `deque(maxlen=1000)` and `--init-threshold 0.0`. Authors' own terminology confirms non-fresh rollouts.
2. **DistiLLM-2** -> `partial_opd`: Paper states "batch approach...rather than on-policy." Code uses separate `generate_vllm.py` script; no in-loop generation. `num_train_epochs: 1` on static dataset.
3. **Speculative KD** -> `partial_opd`: Interleaved student-propose/teacher-replace creates mixed-policy trajectories. C1 weakened by teacher token injection.
4. **PACED** -> `partial_opd`: Forward-KL track trains on teacher sequences (off-policy). Only reverse-KL self-distillation track is on-policy.
5. **Lightning OPD** -> `partial_opd`: Deliberately offline. Rollouts from pi_ref, teacher log-probs cached. C1 fails current-policy freshness by design.
6. **AdaSwitch** -> `partial_opd`: Student generates until divergence threshold, then teacher takes over mid-sequence. Not clean current-student trajectory.

### New downgrade warnings

7. **ToDi** (2505.16297): Per-token FKL/RKL blend is a novel objective innovation but the base setting is likely off-policy. Do NOT classify as strict without explicit C1 evidence from the paper. Place as `adjacent` until line audit confirms student rollout generation.
8. **EGAD** (2605.01732): Despite "entropy-guided adaptive" name, this is supervised distillation on fixed data. Off-policy. Not OPD.
9. **OKD** (2409.12512): Adapts teacher to student, not student to its own rollouts. Adjacent -- novel approach but inverts the OPD loop.
10. **DynSDPB** (2411.16991): Self-distillation from previous mini-batch logits. No external teacher, no current-policy rollout generation in the OPD sense. Adjacent self-distillation.
11. **AMiD** (2510.15982): Generalizes DistiLLM's skew family theoretically but is itself an objective formulation paper, not an on-policy training method. Adjacent until combined with OPD loop.
12. **HPD** (2604.20244): "Hybrid Policy Distillation" mixes off-policy and on-policy sampling. Do not classify as strict; the off-policy component weakens C1. `partial_opd`.

### Do NOT invent strict rows for

- Any cross-tokenizer method (ULD, MultiLevelOT, DWA-KD, BLD) unless explicit student-rollout loop is documented. SimCT and DSKD are the only confirmed candidates.
- Any method that only proposes a new divergence without demonstrating on-policy training. The objective is novel and worth tracking as adjacent, but strict_opd requires C1+C2+C3.
- RSKD or BiLD: these are efficiency enablers for any KD loop, not on-policy methods themselves.

---

## 9. Open Gaps Requiring Line-Level Audit

### High priority (new strict candidates)

1. **vOPD** (2605.07865): Verify exact control variate formulation; confirm no approximation errors in the closed-form baseline. Check if teacher logits are full-vocabulary or top-k.
2. **TIP** (2604.14084): Verify soft-OR score implementation; confirm rho parameter sensitivity. Check if 50% token selection degrades on specific task types.
3. **AOPD** (2605.06387): Verify what "localized divergence minimization" means concretely in code. Is it a projection step? A clipped loss? An auxiliary objective?
4. **StableOPD** (2604.08527): Verify truncation collapse is a real failure mode (not just artifact of specific hyperparameters). Check reference divergence constraint formulation.
5. **CaOPD** (2604.16830): Verify how empirical confidence is estimated and at what frequency. Does recalibration require additional rollout batches beyond training?
6. **SOD** (2605.07725): Verify step-level divergence computation; is it truly on-policy or does it cache step-level statistics? Check GRPO integration.
7. **SimCT** (2605.07711): Verify multi-token continuation scoring is tractable for large vocabularies. Check if minimal aligned units are computed once or dynamically.
8. **OPSDL** (2604.17535): Verify the short-context extraction mechanism. Is the relevant short context selected automatically or manually?
9. **Rock Tokens** (2605.09253): Verify causal intervention (knockout policy) methodology. Is the rock token set computed once or updated during training?
10. **DP-OPD** (2604.04461): Verify DP-SGD noise scale does not destroy the OPD signal. Check privacy-utility tradeoff curves.

### Medium priority (partial candidates needing mode confirmation)

11. **DSKD** (2504.11426): Line-audit the on-policy mode. Does the released code actually implement student-rollout generation with cross-tokenizer alignment?
12. **SelecTKD** (2510.24021): Confirm the on-policy data mode exists in released code, not just described in the paper.
13. **ToDi** (2505.16297): Check if any experiment uses student-generated sequences. The per-token divergence selection is novel regardless.
14. **HPD** (2604.20244): Quantify the off-policy vs on-policy mixture ratio. If primarily on-policy, could upgrade to borderline.

### Low priority (adjacent methods -- objective innovations)

15. **CSD** (2509.25837): Monitor for follow-up work combining concrete score matching with on-policy loop.
16. **AMiD** (2510.15982): Monitor for empirical validation of alpha-mixture family in on-policy setting.
17. **RSKD** (2503.16870): Evaluate whether unbiased sparse logit caching could reduce OPD teacher-serving cost.
18. **MultiLevelOT** (2412.14528): Check if OT alignment can be made compatible with on-policy student rollouts.

### Structural gaps in the landscape

19. **No Renyi/alpha-divergence OPD**: No published work studies alpha-divergence or Renyi divergence specifically for on-policy LLM distillation. This is an open research direction.
20. **No Bregman divergence OPD**: Bregman divergences over logit space have not been explored for OPD.
21. **No progressive divergence scheduling**: While individual methods switch between FKL and RKL (PACED, Entropy-Aware OPD), no work systematically studies optimal divergence scheduling during OPD training.
22. **No online teacher updating in OPD**: OKD adapts the teacher but is not on-policy for the student. The combination (student on-policy + teacher adaptation) is unexplored.
23. **Cross-tokenizer OPD at scale**: SimCT and DSKD are the first strict candidates, but neither has been validated at industrial scale (>30B parameters).
24. **OPD for code generation**: While some methods report code benchmarks, no method specifically addresses code-generation-specific OPD challenges (syntax constraints, execution feedback integration).
