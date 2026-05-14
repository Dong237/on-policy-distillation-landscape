# WP1: White-Box OPD and Divergence/Objectives Audit (Claude)

Checked: 2026-05-12

Scope: LLM white-box on-policy distillation (OPD) and objective/divergence methods. This audit uses primary papers, code repositories, and arXiv sources. VLM, RLVR-only, offline KD, DPO, and industrial reports are excluded except where a method is a useful downgrade or contrast case.

Classification follows `taxonomies/strict-opd-definition.md`:

- C1: current student/policy rollout generates the supervised state.
- C2: teacher-style supervision is evaluated on that exact state.
- C3: the objective directly consumes the teacher-style signal.

Search methodology: arXiv API and Semantic Scholar citation chains from GKD (2306.13649) and MiniLLM (2306.08543), 2024--2026. Keyword queries for "on-policy distillation", "online knowledge distillation LLM", "token-level distillation", "cross-tokenizer distillation", "adaptive divergence distillation LLM", "f-divergence distillation LLM". Forward citation chains from DistiLLM, REOPOLD, DistillSpec, Veto, Entropy-Aware OPD.

---

## Memo: Families and Objective Taxonomy

White-box OPD is distribution matching on student-generated text states. The key divide is not "KL vs reward wording"; it is whether the supervised prefix is fresh student policy data and whether the teacher distribution/log-prob is consumed by the update.

The field has converged on the **OPD-as-RL equivalence**: reverse KL on student rollouts equals policy gradient with token reward `r_t = log(pi_T(a_t|s_t) / pi_theta(a_t|s_t))`. This is independently proven by G-OPD, REOPOLD, KDRL, and vOPD. Innovation since early 2026 has shifted from "which divergence" to three questions: (1) which tokens to train on, (2) how to stabilize the policy-gradient surrogate, and (3) how to fix failure modes (length inflation, miscalibration, entropy collapse, gradient explosion).

No published work explores Renyi, alpha-divergence, or Bregman divergence specifically for on-policy LLM distillation as of May 2026. JSD (via GKD, MAD-OPD) is the most exotic divergence used in practice. This is an open gap.

| Family | Methods | Objective pattern | WP1 classification tendency |
|---|---|---|---|
| Canonical on-policy distribution matching | GKD; MiniLLM | KL-family divergence on student-generated completions | Strict when pure student rollout is used |
| Stabilized/adaptive divergences | DistiLLM; DistiLLM-2; Entropy-Aware OPD; Veto; CaOPD | Skew KL, reverse KL, entropy-gated FKL/RKL, adaptive bridge targets, calibration-corrected targets | Strict only if rollout freshness holds; otherwise partial |
| RL-equivalent dense log-ratio OPD | G-OPD; REOPOLD; KDRL; vOPD; AOPD; Lightning OPD | Dense teacher log-ratio advantage, often equivalent to sequence reverse KL; control variate or asymmetric treatment | Strict for fresh student rollouts; Lightning is partial/offline |
| Token importance / selective training | TIP; Rock Tokens; SOD; SelecTKD; SCOPE | 2D importance scoring, persistent-loss identification, step-level reweighting, propose-and-verify | Strict when selection is applied to student-generated rollouts |
| Stabilization and failure mode fixes | StableOPD; CaOPD; Veto; REOPOLD | Reference divergence constraint, calibration correction, polynomial damping, mixture-based clipping | Strict; these address specific OPD instabilities |
| Systems/efficiency OPD | DistillSpec; Fast OPD; Lightning OPD; Speculative KD; AdaSPEC | Draft-model or prefix-limited teacher scoring, cached/offline approximations | Split by rollout source; systems goal alone does not change C1/C2/C3 |
| Hybrid/interleaved rollout methods | Speculative KD; PACED; DistiLLM variants; AdaSwitch; HPD | Teacher-student mixed traces, replay buffers, pass-rate curricula, mid-sequence switching | Whole-method label is usually partial; subcomponents may be strict |
| Cross-tokenizer OPD | SimCT; DSKD; ULD; MultiLevelOT; DWA-KD | Minimal aligned units, dual projectors, optimal transport | SimCT and DSKD (on-policy mode) are the only strict candidates; others are offline KD |
| Self-distillation OPD | OPSDL; OPSD; OPSDC/CRISP | Privileged-context self-teacher on student rollouts | Strict when privileged self-teacher supervises current-policy states |
| Privacy / safety OPD | DP-OPD; DPSW | Standard OPD with DP-SGD noise or dual-perspective safety weighting | Strict; constraint does not change C1/C2/C3 |

Objective taxonomy:

- Forward KL: `KL(pi_T(.|s_t) || pi_theta(.|s_t))`; mode-covering, used in GKD variants, DistillSpec variants, and PACED forward track.
- Reverse KL / sequence RKL: `KL(pi_theta(.|q) || pi_T(.|q))`, or sampled-token policy-gradient surrogate with advantage `log pi_T(a_t|s_t) - log pi_theta(a_t|s_t)`; core for MiniLLM, Fast OPD, REOPOLD, Lightning baseline, vOPD, SimCT, and several OPD reasoning papers.
- Generalized JSD: GKD uses beta-JSD as a tunable alternative to KL.
- Skew KL / skew reverse KL: DistiLLM and DistiLLM-2 smooth the teacher/student target distribution with a mixture. DistiLLM uses fixed alpha=0.1; DistiLLM-2 uses adaptive per-sample alpha via curriculum.
- Adaptive target KL: Veto defines a target `Q(.|s_t) proportional to pi_T(.|s_t) * pi_theta(.|s_t)^beta`, then minimizes FKL or RKL to `Q`. Theorem 1 proves polynomial damping `P_S(y)^beta` suppresses FKL gradient explosion.
- Entropy-aware hybrid KL: Entropy-Aware OPD adds teacher-top-k forward KL only for high-teacher-entropy tokens on top of an OPD/RKL surrogate: `L_EOPD = L_OPD + I[H_T > tau] * L_FKL^{top-k}` with tau=0.8, k=16.
- Dense log-ratio reward: G-OPD and REOPOLD express OPD as policy optimization with token reward `r_t = log(pi_T/pi_theta)`. G-OPD extends with reward extrapolation (lambda > 1 surpasses teacher). REOPOLD adds mixture-based clipping `max(R_t, log(lambda)/(1-lambda))` and entropy-based token masking. This remains OPD when the signal is teacher distributional supervision, not reward-only RLVR.
- Control variate baseline: vOPD derives `V(c_t) = -D_KL(pi_theta(.|c_t) || pi_T(.|c_t))` as a closed-form baseline requiring no additional critic, reducing variance proportional to squared KL.
- Asymmetric advantage: AOPD replaces ineffective negative reinforcement (negative advantage tokens) with localized divergence minimization, addressing high variance, vanishing gradients, and exploration bottlenecks.
- Token selection mechanisms: TIP uses 2D soft-OR score `s_t = h_t + delta_t - h_t*delta_t` over student entropy and teacher-student divergence; Rock Tokens uses gradient freeze on persistent outliers verified by causal intervention; SOD uses step-level divergence ratio reweighting; SCOPE routes by correctness with perplexity-weighted KL.
- Calibration-corrected KL: CaOPD identifies systematic overconfidence in standard OPD and replaces teacher-conditioned targets with student-grounded empirical confidence estimates.
- Stabilized OPD: StableOPD adds `KL(pi_theta || pi_ref)` reference divergence constraint and rollout mixture distillation to fix truncation collapse / length inflation.
- Cross-tokenizer OPD: SimCT constructs minimal aligned units `U = (V_T intersect V_S) union A` via multi-token continuation scoring. Theorem 1 proves these are the finest boundary-consistent partition. DSKD uses dual projectors with Exact Token Alignment (ETA).
- Cached/offline OPD approximation: Lightning OPD uses the same dense advantage as OPD but fixes the rollout distribution to an SFT/reference model and caches teacher log-probs, so C1 freshness fails. Theorems 3.5 and 3.11 bound gradient discrepancy under teacher consistency.
- DP-noised OPD: DP-OPD applies DP-SGD gradient noise only to the student; frozen teacher provides dense supervision without needing DP teacher training.

---

## Method Classification Matrix

### Verified strict seeds (already in `tables/opd_papers.md`)

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Rollout freshness | Teacher access and kind |
|---|---|---|---|---|---|---|---|
| GKD | Yes for lambda=1; mixed for lambda<1 | Yes | Yes | `strict_opd` for pure on-policy; mixed variants `partial_opd` | `L_GKD=(1-lambda) E_data[D(T||S)] + lambda E_{y~S}[D(T||S)]`; `D` can be FKL/RKL/JSD(beta); stop-gradient on sampling | Current student samples for on-policy term | White-box teacher full logits over vocabulary; larger LM |
| MiniLLM | Yes, with teacher-mixed sampling caveat | Yes | Yes | `strict_opd` | Sequence reverse KL `KL(q_theta || p_T)`; policy-gradient surrogate with PPO clipping; rewards `R_t = sum_{t'>=t} log(p_T/q_theta)`; length normalization | Fresh student or teacher-mixed behavior samples (alpha=0.2 teacher mix-in) | White-box teacher logits for rewards AND mixed generation; larger LM |
| Entropy-Aware OPD | Yes | Yes | Yes | `strict_opd` | OPD clipped/RKL surrogate plus high-entropy teacher-top-k FKL: `L_EOPD = L_OPD + I[H_T>tau] L_FKL`; tau=0.8, k=16 | Current/previous-policy PPO-style behavior rollout per iteration | White-box teacher log-probs, entropy, top-k logits; larger reasoning LM |
| G-OPD | Yes | Yes | Yes | `strict_opd` | `max E_{y~pi_theta}[lambda log(pi_T/pi_ref) - KL(pi_theta||pi_ref)]`; lambda>1 extrapolates beyond teacher; lambda=1 recovers standard OPD | Current student rollout | White-box teacher and reference log-probs; larger teacher + reference LM |
| REOPOLD | Yes | Yes | Yes | `strict_opd` | Relaxed/clipped RKL: `R_hat_t = max(R_t, log(lambda)/(1-lambda))`; entropy-based token masking `M_t = I[H_t >= tau_beta]`; exploration-to-refinement multi-stage | Current/near-current behavior rollout | White-box teacher logits/log-probs; larger reasoning LM |
| Veto | Yes | Yes | Yes | `strict_opd` | Adaptive target `Q(.|s_t) proportional to exp(z_T + beta*z_S)`; optimize `KL(Q||pi_theta)` or `KL(pi_theta||Q)`; Theorem 1 proves gradient veto suppresses pathological FKL gradients; linear beta decay | Current student trajectories | White-box teacher logits; larger LM |
| Fast OPD | Yes for retained prefixes | Yes | Yes | `strict_opd` | Prefix-limited reverse KL: `E_{x~pi_s} sum_{t=1}^{L_train} log(pi_s(a_t|s_t)/pi_T(a_t|s_t))`; prefix scheduling `L_train += Delta_L` per step; Delta_L=256 | Current student prefix rollouts, truncated/scheduled length | White-box teacher log-probs; larger reasoning LM |
| DistillSpec | Yes for draft-generated-data variant | Yes | Yes | `strict_opd` for draft/on-policy variant | Token divergence on draft states; studied FKL, RKL, JSD, TVD; Theorem 4.1 bounds acceptance rate given on-policy loss epsilon | Draft-current for strict variant | White-box target/teacher LM full distribution; target model for speculative decoding |

### New strict candidates (pending WP9 line audit)

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Rollout freshness | Teacher access and kind |
|---|---|---|---|---|---|---|---|
| vOPD | Yes: student generates trajectories | Yes: teacher logits provide per-token KL signal | Reverse KL with closed-form control variate `V(c_t) = -D_KL(pi_theta(.\|c_t) \|\| pi_T(.\|c_t))` | `strict_opd` | Reverse KL; per-token advantage `a_t = r_t + D_KL(pi_theta \|\| pi_T)` eliminates need for critic; variance reduction proportional to squared KL | current_policy | White-box teacher logits; larger LM |
| TIP | Yes: student on-policy rollouts | Yes: teacher distributions on student states | Standard OPD with top-rho token masking by 2D soft-OR score | `strict_opd` | `L_TIP = (1/\|T\|) sum_{t in T} D_KL(P_S \|\| P_T)` where `T = TopK(s_t, rho*m)`; `s_t = h_t + delta_t - h_t*delta_t`; rho=0.5 matches full OPD | current_policy | White-box teacher logits; larger LM |
| AOPD | Yes: student on-policy rollouts | Yes: teacher log-probs provide advantage | Asymmetric PG: divergence minimization replaces negative reinforcement for `A_t <= 0` | `strict_opd` | Asymmetric advantage-weighted policy gradient; localized divergence minimization in non-positive advantage regions | current_policy | White-box teacher logits; larger LM |
| StableOPD | Yes: student generates (with length inflation) | Yes: teacher supervises those trajectories | OPD + `KL(pi_theta \|\| pi_ref)` reference constraint + rollout mixture distillation | `strict_opd` | Standard OPD objective with reference divergence constraint preventing truncation collapse; 7.2% avg improvement | current_policy | White-box teacher logits; larger LM |
| CaOPD | Yes: student rollouts for training and confidence | Yes: teacher distribution, corrected | OPD with calibration-corrected targets from student rollout statistics | `strict_opd` | Standard OPD loss with empirical confidence replacing raw teacher-conditioned targets | current_policy | White-box teacher logits; larger LM |
| SOD | Yes: student generates agentic trajectories | Yes: teacher logits supervise | Step-level divergence-reweighted KL + GRPO | `strict_opd` | `d_k = mean \|log pi_theta - log pi_T\|` per step; `w_k = min(prod(d_u/d_{u+1}), 1+delta)`; `L = L_GRPO + L_OPD^step` | current_policy | White-box teacher logits; larger LM |
| SimCT | Yes: student on-policy with own tokenizer | Yes: teacher multi-token continuation scoring on aligned units | Reverse KL over SimCT supervision distributions with minimal aligned units | `strict_opd` | RKL over `U_SimCT = (V_T intersect V_S) union A`; Theorem 1: finest boundary-consistent partition; first strict cross-tokenizer OPD | current_policy | White-box teacher logits (cross-tokenizer); larger LM |
| OPSDL | Yes: student generates in long-context mode | Yes: short-context self-teacher supervises on those outputs | Point-wise reverse KL under relevant short context | `strict_opd` | Point-wise RKL: `D_KL(pi_theta^{long}(.\|c_t) \|\| pi_theta^{short}(.\|c_t^{relevant}))` | current_policy | Privileged short-context self; privileged_self |
| DP-OPD | Yes: student generates trajectories | Yes: frozen teacher provides dense supervision | Standard OPD + DP-SGD gradient noise on student | `strict_opd` | Standard OPD objective; DP-SGD adds per-sample gradient clipping and Gaussian noise | current_policy | White-box frozen teacher logits; larger LM |
| Rock Tokens | Yes: student on-policy rollouts | Yes: teacher KL on student states | KL with gradient freeze on persistent high-loss tokens | `strict_opd` | `L_weighted = E sum w(x_t,t) * l_t`; `w=lambda if x_t in R, else 1`; lambda=0 is gradient freeze; rock score `R(v) = mean_loss(v) * freq(v)` with context-consistency rate | current_policy | White-box teacher logits; larger LM |

### Borderline strict (already in table)

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Rollout freshness | Teacher access and kind |
|---|---|---|---|---|---|---|---|
| SCOPE | Yes: current-policy rollouts routed by correctness | Yes: teacher supervises incorrect trajectories | Teacher-perplexity-weighted KL on incorrect branch; student-perplexity-weighted MLE on correct branch | `borderline_strict` | `J = sum w_i^stu * L_MLE(correct) + sum w_i^tea * L_OPD(incorrect)`; `w_i^tea = PPL_T^{-1/tau}` | current_policy | White-box teacher logits; larger LM |
| KDRL | Yes: on-policy rollouts for both GRPO and KD | Yes: teacher token log-probs on rollouts | Joint loss `J_KDRL = J_GRPO - beta * D_KL^{k2}(pi_theta \|\| pi_T)`; k2 is unbiased gradient estimator | `borderline_strict` | Reverse KL (k2 approximation) + GRPO; annealing schedule `beta = max(beta_init - delta*step, beta_min)` | current_policy | White-box teacher logits; larger LM |

### Partial OPD (already in table)

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Rollout freshness | Teacher access and kind |
|---|---|---|---|---|---|---|---|
| DistiLLM | Weak/mixed: stochastic generation with replay buffer, `--init-threshold 0.0` | Yes | Yes | `partial_opd` | Skew KL `KL(p || alpha*p + (1-alpha)*q)` and skew RKL `KL(q || (1-alpha)*p + alpha*q)`; alpha=0.1 fixed; code has `ReplayBuffer(deque(maxlen=1000))` | Replay-buffer stale; `samp_threshold = threshold * (1 - step/total)` decreasing generation probability | White-box teacher logits; larger LM |
| DistiLLM-2 | Weak/mixed: pre-generated offline by `generate_vllm.py` | Yes | Yes | `partial_opd` | Contrastive SKL/SRKL: `(1/2)(2-beta)*SKL(teacher_data) + beta*SRKL(student_data)`; adaptive per-sample alpha via curriculum; beta ramps from ~1.0 to ~1.5 | Per-epoch batched SGO; `num_train_epochs: 1` on static dataset; paper says "batch approach...rather than on-policy" | White-box teacher logits; larger LM |
| Speculative KD | Mixed/interleaved: student proposes, teacher top-k accepts/rejects/replaces | Yes | Yes | `partial_opd` | Token-level KL on interleaved sequences; student-proposed tokens accepted if in teacher top-K (K=25), otherwise resampled from teacher | Fresh but hybrid student/teacher token trace; early training has high rejection (like supervised KD) | White-box teacher logits/top-k; target/reference LM |
| PACED | Mixed: forward-KL track teacher-forced, reverse-KL track student rollout | Yes | Yes | `partial_opd` | Beta-kernel weighting `w(p) = p^alpha * (1-p)^beta` (default alpha=beta=1); forward KL on teacher sequences; reverse KL on student sequences; two-stage schedule | Pass-rate estimates from student rollouts (on-policy), but forward track is off-policy | White-box teacher logits; larger LM or GRPO-finetuned self-teacher |
| Lightning OPD | No fresh current rollout: `J_off = E_{x~pi_ref} sum A_t` | Yes (cached) | Yes | `partial_opd` / offline approximation | Cached reverse-KL log-ratio surrogate; Theorems 3.5 and 3.11 bound `\|\|grad J_on - grad J_off\|\| <= G*sigma_A*sqrt(chi^2(pi_theta \|\| pi_ref))`; implicit trust-region regularization | Precomputed SFT/reference rollouts; cached teacher log-probs; 4x speedup | White-box teacher log-probs cached once; larger reasoning LM |
| AdaSwitch | Partial: student generates until `d_i > tau`, then teacher takes over permanently | Yes | Yes | `partial_opd` | Token-level KL on switched sequences; `tau = K * d_bar_{i-1}` (K=3, sliding window average of KL divergences) | Mixed: student prefix + teacher suffix within each sequence | White-box teacher logits; larger LM |

### New partial/borderline candidates (pending line audit)

| Method | C1 | C2 | C3 | Classification | Exact objective/divergence | Notes |
|---|---|---|---|---|---|---|
| DSKD | Yes in on-policy mode | Yes via dual projectors with ETA | Cross-vocabulary KL with Exact Token Alignment | `partial_opd` to `strict_opd` (mode-dependent) | Dual projectors: `W^{t->s} = tilde{W}^t * (tilde{W}^s)^+`; needs on-policy mode confirmation in code | arXiv 2504.11426 |
| HPD | Mixed on/off-policy sampling | Yes: teacher logits | Hybrid FKL+RKL with mixed-policy reweighting | `partial_opd` | Token-level reweighted log-likelihood; off-policy component weakens C1 | arXiv 2604.20244 |
| SelecTKD | Conditional: supports on-policy mode | Yes: teacher top-k verification | Acceptance-masked KL with Token Acceptance Rate curriculum | `partial_opd` to `strict_opd` | Propose-and-verify: accepted tokens get full loss, rejected masked | arXiv 2510.24021 |
| TVD++ | Partial: mixed teacher/draft data | Yes: target logits | Total Variation Distance with PG variance reduction | `partial_opd` | Speculative decoding context; ICLR 2024 Workshop | arXiv 2403.00858 |
| ToDi | Needs verification (likely off-policy base) | Yes: teacher logits per token | Per-token sigmoid-weighted FKL/RKL blend | `adjacent` until C1 confirmed | EMNLP 2025 Oral; novel divergence selection per token | arXiv 2505.16297 |

---

## Evidence Ledger

### Verified entries (primary source confirmed)

| Method | Primary source | Evidence used | Conservative note |
|---|---|---|---|
| GKD | [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) (ICLR 2024) | Defines `D(T\|\|S)(y\|x)` as token-averaged divergence; `L_GKD` mixes fixed data and student samples by lambda; supports FKL/RKL/JSD(beta). Stop-gradient on sampling. TRL `GKDTrainer` implements it. | Mark strict only for lambda=1 setting. |
| MiniLLM | [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) (ICLR 2024) | Minimizes sequence RKL via policy-gradient/PPO with dense log-ratio rewards; teacher-mixed sampling `alpha*p + (1-alpha)*q` (alpha=0.2) with importance weighting; length normalization; cross-entropy + entropy regularization. Code at `microsoft/LMOps/tree/main/minillm`. | Teacher-mixed sampling is a stabilizer; not purely current-student at every token. |
| DistiLLM | [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) (ICML 2024) | Skew KL and skew RKL with `alpha=0.1`; code confirms `ReplayBuffer` with `deque(maxlen=1000)`, `--init-threshold 0.0`, `--replay-ratio decreasing`; paper says "adaptive off-policy". Code at `github.com/jongwooko/distillm`. | Authors themselves use "off-policy"; rollout freshness is definitively not current-policy. |
| DistiLLM-2 | [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://arxiv.org/abs/2503.07067) (ICML 2025 Oral) | Contrastive SKL/SRKL with adaptive per-sample alpha; `generate_vllm.py` runs offline, `num_train_epochs: 1` on static dataset; paper states "batch approach...rather than on-policy approach, which samples at every training iteration". Code at `github.com/jongwooko/distillm-2`. | Whole method partial; sequences never refreshed during training. |
| DistillSpec | [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) (ICLR 2024) | Defines target-vs-draft token divergence; studies FKL, RKL, JSD, TVD on draft-generated data; Theorem 4.1: `E[alpha(x)] >= 1 - T*epsilon`. | Strict only for draft-generated/on-policy variant. |
| Entropy-Aware OPD | [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079) | Entropy-gated objective: `L_EOPD = L_OPD + I[H_T > tau] * L_FKL^{top-k}`; tau=0.8, k=16; PPO-style clipped RKL with `eps=0.2`. | Top-k FKL is approximate support matching. |
| G-OPD | [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) | OPD reformulated as KL-constrained RL: `J = max E[lambda*log(pi_T/pi_ref) - KL(pi_theta\|\|pi_ref)]`; token-level implicit reward `r_t = log(pi_T(y_t)/pi_ref(y_t))`; lambda>1 for extrapolation beyond teacher. Multi-teacher ExOPD is the only method producing students surpassing all domain teachers. | Dense teacher logit supervision, not RLVR-only. |
| REOPOLD | [Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) | OPD as policy optimization: `R_t = log pi_T - log pi_theta`; mixture-based reward clipping `max(R_t, log(lambda)/(1-lambda))`; entropy-based token masking; exploration-to-refinement phasing. 6.7-12x sample efficiency. | By same first author as DistiLLM; this is the strict-OPD version of the same line of work. |
| Veto | [Stable On-Policy Distillation through Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) (Findings of ACL 2026) | Bridge target `Q ~ P_T * P_S^beta`; Theorem 1 (gradient veto): `lim_{p->0+} p^beta * log(p) = 0`; Theorem 2 (sharpening): student converges to `P_T^{1/(1-beta)}`; Theorem 3 (REINFORCE bridge). | Preprint; author/code normalization needed. |
| Fast OPD | [Fast and Effective On-Policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) | Prefix-limited reverse KL; empirical finding: training signal concentrated in first 2000-4000 tokens; prefix scheduling `L_train += Delta_L` per step; 2x-47x FLOP reduction. On-policy + reverse KL is the strongest setting under truncation. | Strict for prefix states only. |
| Speculative KD | [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) | Interleaved protocol: student proposes, teacher accepts if in top-K (K=25), else resamples from teacher; 71-85% higher acceptance rate. | Mixed-policy trace: rejected tokens are teacher-generated, not student-generated. |
| PACED | [PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) | Beta-kernel `w(p) = p^alpha*(1-p)^beta` (inverse Bernoulli Fisher information); Proposition 2: leading-order maximum-parsimony family; Theorem 6: minimax-robust; two-stage FKL-then-RKL schedule. | Forward-KL track is off-policy; only reverse-KL self-distill track is on-policy. |
| Lightning OPD | [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010) | Replaces `E_{x~pi_theta}` with `E_{x~pi_ref}`; Theorems 3.5 and 3.11: bounded gradient discrepancy under teacher consistency; 4x speedup; 69.9% AIME 2024 in 30 GPU hours. | C1 fails by design. |
| SCOPE | [SCOPE: Signal-Calibrated On-Policy Distillation Enhancement](https://arxiv.org/abs/2604.10688) | Correctness-routed dual-path: teacher-PPL-weighted KL on incorrect, student-PPL-weighted MLE on correct; `w_i^tea = PPL_T^{-1/tau}` downweights where teacher is uncertain; importance sampling ratio `rho_t`. | Borderline because correct branch uses MLE not teacher KL. |
| KDRL | [KDRL: Post-Training Reasoning LLMs via Unified KD and RL](https://arxiv.org/abs/2506.02208) | Joint loss `J_KDRL = J_GRPO - beta * D_KL^{k2}`; k2 is unbiased estimator (+1.4% vs k3); reward-guided KD masking suppresses distillation on correct responses; annealing `beta = max(beta_init - delta*step, beta_min)`. | Full method borderline because of GRPO hybridization; KD-RKL subcomponent is strict. |
| AdaSwitch | [AdaSwitch: Adaptive Switching for Efficient LLM Distillation](https://arxiv.org/abs/2510.07842) | `tau = K * d_bar_{i-1}` (K=3); student generates until divergence exceeds threshold, then teacher takes over for remaining tokens. | Hybrid student-teacher generation is not clean student-on-policy. |

### New method evidence (pending WP9)

| Method | Primary source | Evidence used | Conservative note |
|---|---|---|---|
| vOPD | [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865) | Casts OPD as policy-gradient RL; derives `V(c_t) = -D_KL(pi_theta(.\|c_t) \|\| pi_T(.\|c_t))` as closed-form value function for control variate; no additional critic or inference. Variance reduction scales with squared KL. | Preprint May 2026; verify derivation in line audit. |
| TIP | [Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) | 2D taxonomy: Q1 (high entropy, high divergence) = most valuable; Q2/Q3/Q4 less so. Soft-OR scoring `s_t = h_t + delta_t - h_t*delta_t`; top-rho selection. 50% tokens matches full training; 47% memory reduction. | Preprint Apr 2026; novel token selection; verify rho sensitivity. |
| AOPD | [Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level](https://arxiv.org/abs/2605.06387) | Replaces negative reinforcement with localized divergence minimization for non-positive advantage regions; addresses high variance, vanishing gradients, and exploration bottlenecks. | Preprint May 2026; verify concrete implementation of divergence minimization. |
| StableOPD | [Demystifying OPD: Length Inflation and Stabilization Strategies](https://arxiv.org/abs/2604.08527) | Identifies truncation collapse failure mode (abrupt length inflation); proposes reference divergence constraint `KL(pi_theta \|\| pi_ref)` and rollout mixture distillation; 7.2% average improvement. | Preprint Apr 2026; verify truncation collapse is robust finding, not hyperparameter artifact. |
| CaOPD | [The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation](https://arxiv.org/abs/2604.16830) | Identifies OPD causes severe overconfidence despite accuracy gains; CaOPD estimates empirical confidence from model rollouts and replaces teacher-conditioned targets. | Preprint Apr 2026; Salesforce AI Research. Verify confidence estimation frequency and cost. |
| SOD | [Step-wise On-policy Distillation for Small Language Model Agents](https://arxiv.org/abs/2605.07725) | Step-level divergence `d_k = mean \|log pi_theta - log pi_T\|` per step; `w_k = min(prod(d_u/d_{u+1}), 1+delta)`; combined with GRPO. 0.6B student achieves 26.13% on AIME 2025. | Preprint May 2026; agentic/tool-use focus overlaps WP6. |
| SimCT | [Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2605.07711) | Minimal aligned units via multi-token continuation scoring; `s_M(u\|x) = (1/k) sum log p_M(v_j \| x, v_{<j})`; Theorem 1: finest boundary-consistent partition. First strict cross-tokenizer OPD. | Preprint May 2026; verify tractability for large vocabularies and at scale. |
| OPSDL | [On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535) | Short-context self-teacher supervises long-context generation; point-wise reverse KL per token under relevant extracted short context. | Preprint Apr 2026; verify short-context extraction mechanism (automatic vs manual). |
| DP-OPD | [Differentially Private On-Policy Distillation for Language Models](https://arxiv.org/abs/2604.04461) | DP-SGD applied only to student; frozen teacher provides dense supervision; eliminates need for DP teacher training. | Preprint Apr 2026; verify DP noise does not destroy OPD signal. |
| Rock Tokens | [Cornerstones or Stumbling Blocks? Understanding and Handling Rock Tokens in OPD](https://arxiv.org/abs/2605.09253) | Persistent high-loss tokens (~18%); rock score `R(v) = mean_loss * freq`; context-aware rock score `R_ctx = R * CCR`; gradient freeze `w=0`; causal intervention (knockout policy) verifies necessity. | Preprint May 2026; verify knockout methodology and rock token set stability during training. |

### Adjacent methods (novel objectives, not on-policy by default)

| Method | Primary source | Why adjacent | Novel contribution |
|---|---|---|---|
| CSD | [Distillation via Concrete Score Matching](https://arxiv.org/abs/2509.25837) (ICLR 2026) | Not inherently on-policy; objective paper | Discrete score matching bypassing softmax; non-KL non-f-divergence; mode-seeking and mode-covering variants |
| AMiD | [alpha-mixture Assistant Distribution KD](https://arxiv.org/abs/2510.15982) | Off-policy objective formulation | Continuous interpolation generalizing DistiLLM skew family |
| BiLD | [Bi-directional Logits Difference Loss](https://arxiv.org/abs/2406.13555) (COLING 2025) | Likely off-policy | Top-k (k=8) logit difference preserving internal ranking; filters long-tail noise |
| RSKD | [Sparse Logit Sampling for KD](https://arxiv.org/abs/2503.16870) (ACL 2025 Oral) | Efficiency enabler, not on-policy | Importance-sampling-based unbiased gradient estimate with <10% logit overhead |
| OKD | [Online Teacher Adaptation KD](https://arxiv.org/abs/2409.12512) | Teacher adapts instead of student on-policy | Online modules in teacher adapt to student distribution; inverts OPD loop |
| EGAD | [Entropy-Guided Adaptive Distillation](https://arxiv.org/abs/2605.01732) | Off-policy supervised KD | Entropy curriculum + adaptive temperature per token; despite name, not on-policy |
| MultiLevelOT | [Multi-Level OT for Cross-Tokenizer KD](https://arxiv.org/abs/2412.14528) (AAAI 2025 Oral) | Off-policy cross-tokenizer | Token-level OT with Sinkhorn distance; Wasserstein-based objective |
| ULD | [Universal Logit Distillation](https://arxiv.org/abs/2402.12030) | Off-policy only | Wasserstein on sorted probability vectors; cross-tokenizer |
| ToDi | [Token-wise Distillation via Fine-Grained Divergence Control](https://arxiv.org/abs/2505.16297) (EMNLP 2025 Oral) | Likely off-policy; needs C1 audit | Per-token sigmoid-weighted FKL/RKL blend based on log-ratio |
| HPD | [Hybrid Policy Distillation](https://arxiv.org/abs/2604.20244) | Mixes off/on-policy sampling | Token-level reweighted log-likelihood; partial at best |
| DWA-KD | [Dual-Space Weighting + Time-Warped Alignment](https://arxiv.org/abs/2602.21669) (EACL Findings) | Likely off-policy | Dual-space entropy weighting + soft DTW |
| DynSDPB | [Dynamic Self-Distillation from Previous Mini-Batches](https://arxiv.org/abs/2411.16991) | Previous mini-batch self-teacher; no external teacher | Dynamic temperature self-KL from prior iteration |
| AdaSPEC | [Selective KD for Efficient Speculative Decoders](https://arxiv.org/abs/2510.19779) | Mixed; reference model filters | Selective token filtering using reference model; 15% improvement over DistillSpec |

---

## Suggested Markdown Row Fragments

These fragments validate or extend the relevant fields in `tables/opd_papers.md`. The 10 strict candidates are provided as full 44-column rows in `assets/research/wp1/wp1-candidate-rows.md`. Below are compact fragments for cross-reference.

### Verified entries (confirm existing table rows)

| id | method_family | opd_strictness | divergence_or_objective | rollout_source | rollout_freshness | teacher_access | teacher_kind | strictness_evidence | notes |
|---|---|---|---|---|---|---|---|---|---|
| gkd-2024 | white_box_opd | strict_opd | generalized_kl_fkl_rkl_jsd | student_on_policy | current_policy | teacher_logprobs | larger_llm | C1 pure lambda=1 samples student outputs; C2 teacher scores those outputs; C3 generalized KL consumes teacher distribution. | Mixed lambda variants are partial. TRL GKDTrainer is the primary public implementation. |
| minillm-2024 | reverse_kl_opd | strict_opd | sequence_reverse_kl | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student sequence distribution is optimized; C2 teacher distribution supplies log-ratio rewards; C3 reverse-KL/PG objective consumes it. | Teacher-mixed sampling (alpha=0.2) is a stabilizer. PPO clipping + length normalization. Code at microsoft/LMOps. |
| distillm-2024 | adaptive_off_policy_kd | partial_opd | skew_kl_and_skew_reverse_kl | mixed_policy | replay_buffer_stale | teacher_logits | larger_llm | C1 weakened by replay/static mixture; code has ReplayBuffer(deque(maxlen=1000)); paper says "adaptive off-policy". | Authors' own terminology confirms non-fresh rollouts. alpha=0.1 fixed. |
| distillm2-2025 | contrastive_kd | partial_opd | contrastive_skl_srkl | mixed_policy | precomputed_sft_rollouts | teacher_logits | larger_llm | C1 weakened: paper says "batch approach...rather than on-policy"; code uses separate generate_vllm.py; num_train_epochs: 1 on static dataset. | Adaptive per-sample alpha; ICML 2025 Oral. |
| distillspec-2024 | systems_opd | strict_opd | task_specific_f_divergence | student_on_policy | current_policy | teacher_logits | reference_model | C1 draft-generated sequences; C2 target model scores draft states; C3 tailored divergence trains draft. Theorem 4.1 bounds acceptance rate. | Strict only for draft-generated/on-policy variant. |
| entropy-aware-opd-2026 | adaptive_divergence_opd | strict_opd | entropy_gated_forward_and_reverse_kl | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student trajectories; C2 teacher log-probs/entropy/top-k; C3 EOPD consumes both. tau=0.8, k=16. | Top-k FKL is approximate support matching. |
| g-opd-2026 | opd_rl_hybrid | strict_opd | kl_constrained_reward_extrapolation | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student trajectories; C2 teacher/reference dense log-ratio; C3 G-OPD objective consumes signal. lambda>1 extrapolates beyond teacher. | Not reward-only RLVR. Dense teacher supervision. |
| reopold-2026 | relaxed_opd | strict_opd | relaxed_reverse_kl | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student/old-policy rollouts; C2 teacher log-ratio on tokens; C3 relaxed RKL surrogate with clipping and masking. | 6.7-12x sample efficiency. Exploration-to-refinement phasing. |
| veto-2026 | target_reformulation_opd | strict_opd | adaptive_target_kl | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student trajectories; C2 adaptive target combines teacher and student logits; C3 KL to target updates student. Theorem 1: gradient veto. | Findings of ACL 2026. Normalize authors/code. |
| fast-opd-2026 | prefix_opd | strict_opd | prefix_reverse_kl | student_on_policy | current_policy | teacher_logits | larger_llm | C1 student prefix rollout; C2 teacher scores retained prefix; C3 reverse KL consumes it. Prefix scheduling Delta_L=256. | 2x-47x FLOP reduction. On-policy + RKL is strongest under truncation. |
| speculative-kd-2024 | hybrid_speculative_kd | partial_opd | interleaved_token_kl | mixed_policy | mixed_freshness | teacher_logits | reference_model | C1 weakened by teacher replacement at rejected positions (top-K=25); C2 teacher logits; C3 token KL. | Mixed trace; 71-85% higher acceptance rate. |
| paced-2026 | curriculum_opd | partial_opd | beta_weighted_kl | mixed_policy | mixed_freshness | teacher_logits | larger_llm | C1 mixed by track; C2 teacher/self-teacher logits; C3 weighted KL. Beta-kernel w(p)=p(1-p) is inverse Bernoulli Fisher info. | FKL-then-RKL two-stage schedule. Reverse-KL subtrack is strict. |
| lightning-opd-2026 | offline_opd_approximation | partial_opd | cached_teacher_kl | off_policy | precomputed_sft_rollouts | teacher_logits | larger_llm | Missing C1: rollouts from pi_ref; C2 cached teacher log-probs; C3 offline objective. Theorems 3.5/3.11 bound gradient discrepancy. | 4x speedup. 69.9% AIME 2024 in 30 GPU hours. |

### New strict candidates (pending WP9, full rows in wp1-candidate-rows.md)

| id | method_family | opd_strictness | divergence_or_objective | rollout_source | rollout_freshness | teacher_kind | strictness_evidence |
|---|---|---|---|---|---|---|---|
| vopd-2026 | variance_reduced_opd | strict_opd | reverse_kl_with_control_variate | student_on_policy | current_policy | larger_llm | C1 student generates; C2 teacher logits; C3 RKL with V(c_t)=-D_KL as baseline. |
| tip-2026 | selective_token_opd | strict_opd | selective_token_kl | student_on_policy | current_policy | larger_llm | C1 student rollouts; C2 teacher on student states; C3 OPD on top-rho tokens by soft-OR. |
| aopd-2026 | asymmetric_advantage_opd | strict_opd | asymmetric_advantage_kl | student_on_policy | current_policy | larger_llm | C1 student rollouts; C2 teacher log-probs; C3 asymmetric PG with divergence minimization for A_t<=0. |
| stableopd-2026 | stabilized_opd | strict_opd | opd_with_reference_constraint | student_on_policy | current_policy | larger_llm | C1 student generates; C2 teacher supervises; C3 OPD + KL(pi_theta\|\|pi_ref) constraint. |
| caopd-2026 | calibration_corrected_opd | strict_opd | calibration_corrected_kl | student_on_policy | current_policy | larger_llm | C1 student rollouts; C2 teacher corrected; C3 OPD with empirical confidence targets. |
| sod-2026 | step_weighted_opd | strict_opd | step_divergence_weighted_kl | student_on_policy | current_policy | larger_llm | C1 student generates agentic trajectories; C2 teacher supervises; C3 step-weighted KL + GRPO. |
| simct-2026 | cross_tokenizer_opd | strict_opd | reverse_kl_over_aligned_units | student_on_policy | current_policy | larger_llm | C1 student on-policy own tokenizer; C2 teacher multi-token continuation scoring; C3 RKL on SimCT space. |
| opsdl-2026 | self_distillation_opd | strict_opd | pointwise_reverse_kl | student_on_policy | current_policy | privileged_self | C1 student in long-context; C2 short-context self-teacher; C3 point-wise RKL. |
| dp-opd-2026 | privacy_preserving_opd | strict_opd | opd_with_dp_sgd | student_on_policy | current_policy | larger_llm | C1 student generates; C2 frozen teacher dense supervision; C3 standard OPD + DP-SGD noise. |
| rock-tokens-2026 | diagnostic_opd | strict_opd | kl_with_gradient_freeze | student_on_policy | current_policy | larger_llm | C1 student rollouts; C2 teacher KL; C3 KL with w=0 on rock tokens verified by knockout. |

---

## False Positives and Downgrade Notes

### Confirmed downgrades

1. **DistiLLM** -> `partial_opd`: Paper explicitly says "adaptive off-policy"; code has `ReplayBuffer(deque(maxlen=1000))` with `--init-threshold 0.0` (initially NO student generation). `samp_threshold = threshold * (1 - step/total)` decreasing generation probability. Authors' own terminology is unambiguous.
2. **DistiLLM-2** -> `partial_opd`: Paper states "batch approach...rather than on-policy approach, which samples at every training iteration." Code uses separate `generate_vllm.py` offline script; `num_train_epochs: 1` on static dataset. No in-loop generation. No importance weighting for staleness.
3. **Speculative KD** -> `partial_opd`: Interleaved student-propose/teacher-replace creates mixed-policy trajectories. Rejected student tokens (anything not in teacher top-K=25) are replaced by teacher tokens. The supervised prefix is a hybrid, not clean student rollout.
4. **PACED** -> `partial_opd`: Forward-KL track trains on teacher sequences (off-policy for the distillation loss). Only the reverse-KL self-distillation track trains on student sequences (on-policy for that track). Pass-rate estimation is on-policy. Whole method must stay partial unless rows split.
5. **Lightning OPD** -> `partial_opd`: Deliberately offline by design. `J_off = E_{x~pi_ref} sum A_t` explicitly replaces current-policy rollouts with reference/SFT rollouts. Teacher log-probs precomputed once. C1 fails current-policy freshness. Useful efficiency contrast but not strict OPD.
6. **AdaSwitch** -> `partial_opd`: Student generates until `d_i > tau = K * d_bar`, then teacher takes over permanently for remaining tokens. The resulting sequence has a student prefix and teacher suffix; not clean current-student trajectory.

### New downgrade warnings

7. **ToDi** (arXiv 2505.16297): Per-token FKL/RKL blend is a novel objective innovation, but the base experimental setting is likely off-policy (supervised KD on fixed dataset). Do NOT classify as strict without explicit C1 evidence. Place as `adjacent` until line audit confirms student-rollout generation. EMNLP 2025 Oral.
8. **EGAD** (arXiv 2605.01732): Despite "entropy-guided adaptive" name, this is supervised distillation on fixed data with entropy-based curriculum. Student does not generate rollouts. Off-policy. Not OPD.
9. **OKD** (arXiv 2409.12512): Adapts teacher to student distribution via online modules in the teacher network, instead of having the student generate on-policy rollouts. Inverts the OPD loop. Adjacent.
10. **DynSDPB** (arXiv 2411.16991): Self-distillation from previous mini-batch logits. No external teacher, no current-policy rollout generation in the OPD sense. Adjacent self-distillation.
11. **AMiD** (arXiv 2510.15982): Generalizes DistiLLM's skew KL family with continuous alpha-mixture interpolation. Theoretical contribution; not an on-policy training method by itself. Adjacent until combined with OPD loop.
12. **HPD** (arXiv 2604.20244): "Hybrid Policy Distillation" explicitly mixes off-policy and on-policy sampling. The off-policy component weakens C1. `partial_opd` at best.
13. **Cross-tokenizer ULD/MultiLevelOT/DWA-KD/BLD**: White-box KD enablers solving tokenizer mismatch, but all are offline/off-policy. Do NOT add strict OPD row without current-rollout proof. SimCT and DSKD (on-policy mode) are the only strict cross-tokenizer candidates found.
14. **RSKD/BiLD**: Efficiency enablers (sparse logit caching, top-k logit differences) for any KD loop. Not on-policy methods themselves. Adjacent.
15. **G-OPD and REOPOLD** use RL-style notation but should NOT be downgraded for policy-gradient wording. The supervision is dense teacher log-prob/logit guidance on student-generated states, satisfying C1/C2/C3. The OPD-as-RL equivalence is a mathematical reformulation, not a change in supervision type.

---

## Open Gaps Requiring Line-Level Audit

### High priority (new strict candidates)

1. **vOPD** (2605.07865): Verify control variate derivation is exact (not approximate). Confirm teacher logits are full-vocabulary. Check if variance reduction claim holds across model sizes.
2. **TIP** (2604.14084): Verify soft-OR score implementation and rho parameter sensitivity. Confirm 50% token selection does not degrade on specific task types. Check if selection criterion adapts during training.
3. **AOPD** (2605.06387): Verify what "localized divergence minimization" means concretely in code. Is it a projection step? A clipped loss? An auxiliary KL objective? Implementation details matter for reproducibility.
4. **StableOPD** (2604.08527): Verify truncation collapse is a robust failure mode (not just artifact of specific hyperparameters/models). Check reference divergence constraint coefficient.
5. **CaOPD** (2604.16830): Verify how empirical confidence is estimated and at what frequency. Does recalibration require additional rollout batches beyond training?
6. **SOD** (2605.07725): Verify step-level divergence computation is truly on-policy. Check whether step-level statistics are cached. Confirm GRPO integration does not change C1/C2/C3 for the OPD component.
7. **SimCT** (2605.07711): Verify multi-token continuation scoring is tractable for large vocabularies. Check if minimal aligned units are computed once or dynamically. Test at >30B scale.
8. **OPSDL** (2604.17535): Verify short-context extraction mechanism (automatic or manual). Check if the privileged short-context is always available or sometimes degenerate.
9. **Rock Tokens** (2605.09253): Verify causal intervention (knockout policy) methodology. Is the rock token set computed once or updated? What happens to previously-frozen tokens after they unfreeze?
10. **DP-OPD** (2604.04461): Verify DP-SGD noise scale does not destroy the OPD signal. Check privacy-utility tradeoff curves. Is the privacy guarantee meaningful for realistic epsilon values?

### Medium priority (existing entries)

11. **Veto**: Normalize author list, code status, and final equation numbering; source evidence for the adaptive target is sufficient but metadata is rough. ACL 2026 Findings venue now confirmed.
12. **Entropy-Aware OPD**: Audit exact top-k renormalization and whether teacher entropy is computed over full vocabulary or top-k support in all experiments.
13. **PACED**: Decide whether to split into two rows: forward-KL teacher-sequence PACED (`partial_opd`) and reverse-KL self-distillation PACED (`strict_opd` substage).
14. **Lightning OPD**: Verify whether any experimental ablation uses current student rollouts; the main algorithm is offline/reference-rollout only.
15. **DistiLLM/DistiLLM-2**: Inspect whether any reported experimental setting disables replay/batching enough to warrant a separate strict row. Current whole-method labels should stay partial.
16. **Fast OPD**: Audit tokenizer/vocabulary assumptions and special-token handling; same-vocab limitation matters for cross-tokenizer claims.
17. **DistillSpec**: Split rows by data source if the table needs precision: draft-generated strict, target/generated offline variants not strict.
18. **G-OPD**: Record exact reference model choices by experiment; lambda/reference access affects teacher-access cost and reproducibility.

### Low priority (partial candidates needing mode confirmation)

19. **DSKD** (2504.11426): Line-audit the on-policy mode. Does the released code actually implement student-rollout generation with cross-tokenizer alignment?
20. **SelecTKD** (2510.24021): Confirm on-policy data mode exists in released code, not just described in the paper.
21. **ToDi** (2505.16297): Check if any experiment uses student-generated sequences. The per-token divergence selection is novel regardless.
22. **HPD** (2604.20244): Quantify off-policy vs on-policy mixture ratio. If primarily on-policy, could upgrade to borderline.

### Structural gaps in the landscape

23. **No Renyi/alpha-divergence OPD**: No published work studies alpha-divergence or Renyi divergence specifically for on-policy LLM distillation. This is an open research direction.
24. **No Bregman divergence OPD**: Bregman divergences over logit space have not been explored for OPD.
25. **No systematic divergence scheduling**: While individual methods switch between FKL and RKL (PACED, Entropy-Aware OPD), no work systematically studies optimal divergence scheduling during OPD training (when to use which divergence, at which training phase, for which token types).
26. **No online teacher updating combined with on-policy student**: OKD adapts the teacher but is not on-policy for the student. The combination (student on-policy + teacher adaptation) is unexplored.
27. **Cross-tokenizer OPD at scale**: SimCT and DSKD are the first strict candidates, but neither has been validated at industrial scale (>30B parameters).
28. **OPD for code generation**: While some methods report code benchmarks, no method specifically addresses code-generation-specific OPD challenges (syntax constraints, execution feedback integration, cross-language distillation).
29. **OPD for safety/alignment**: DP-OPD addresses privacy; DPSW addresses safety weighting. But systematic study of OPD for alignment (harmlessness, helpfulness, honesty) beyond standard benchmarks is missing.
30. **Theoretical convergence guarantees**: Only Lightning OPD provides convergence analysis. No finite-sample convergence rate for standard OPD is published.
