# White-Box Distributional OPD

White-box OPD uses teacher logits, probabilities, or log-probs to supervise student-generated trajectories.

## Include

- GKD-style generalized KL methods.
- MiniLLM-style reverse-KL methods.
- DistiLLM-style skewed/adaptive KL methods.
- Entropy-aware or token-position-aware divergence selection.
- Top-K/local-support matching and sampled-token failure analyses.
- OPD-as-RL equivalence methods (G-OPD, REOPOLD, vOPD, AOPD).
- Token importance / selective training (TIP, Rock Tokens, SOD).
- Stabilization methods (StableOPD, CaOPD, Veto).
- Cross-tokenizer OPD (SimCT, DSKD).
- Privacy and safety variants (DP-OPD, DPSW).

## Key questions

- Does the student generate the supervised prefix or trajectory?
- Are teacher logits/log-probs evaluated on that student state?
- Is the loss token-level, sequence-level, or hybrid?
- What compute cost comes from teacher scoring?
- Which tokens are trained on? (all, entropy-gated, importance-selected, gradient-frozen)
- How is the policy-gradient surrogate stabilized? (clipping, control variate, asymmetric advantage)

## Method families

### A. Canonical distribution matching (foundational)

GKD (generalized JSD/FKL/RKL with lambda-mixing) and MiniLLM (sequence reverse KL via policy gradient with teacher-mixed sampling). These are the two foundational references for white-box OPD.

### B. Skewed/smoothed KL (partial OPD)

DistiLLM (skew KL with replay buffer), DistiLLM-2 (contrastive SKL/SRKL with pre-generated SGO), AMiD (alpha-mixture assistant distribution, adjacent). These smooth the divergence denominator but weaken rollout freshness.

### C. Adaptive/entropy-gated divergence

Entropy-Aware OPD (binary entropy gate: RKL + FKL at high-entropy positions), Veto (geometric bridge target suppressing gradient explosion), SCOPE (correctness-routed perplexity-weighted KL/MLE), CaOPD (calibration-corrected targets). These adaptively select which divergence or target to use per token or per trajectory.

### D. Dense log-ratio / OPD-as-RL equivalence

G-OPD (dense teacher log-ratio as reward; reward extrapolation with lambda > 1), REOPOLD (clipped reverse KL with entropy-based token masking), KDRL (joint GRPO + KD-RKL with k2 unbiased estimator), vOPD (control variate baseline from closed-form KL value function), AOPD (asymmetric advantage: divergence minimization replaces negative reinforcement). These recast OPD as policy optimization with teacher log-ratio as token reward.

### E. Token importance / selective training

TIP (2D soft-OR score from entropy x divergence; top-rho selection), Rock Tokens (gradient freeze on persistent outliers via causal intervention), SOD (step-level divergence ratio reweighting for agentic trajectories), SelecTKD (propose-and-verify with Token Acceptance Rate), RSKD (unbiased sparse logit caching, adjacent enabler).

### F. Stabilization and failure mode fixes

StableOPD (reference divergence constraint for truncation collapse / length inflation), CaOPD (calibration correction for systematic overconfidence), Veto (polynomial damping for FKL gradient explosion), REOPOLD (mixture-based reward clipping for heavy-tailed negatives).

### G. Efficiency and systems OPD

Fast OPD (prefix truncation with scheduling; 2x-47x FLOP reduction), Lightning OPD (offline OPD approximation; 4x speedup but partial), DistillSpec (draft-model alignment for speculative decoding), Speculative KD (interleaved student-propose/teacher-replace; partial).

### H. Cross-tokenizer OPD

SimCT (minimal aligned units via multi-token continuation scoring; source-verified strict), DSKD v2 (dual projectors with ETA; partial at method level), ULD/MultiLevelOT/DWA-KD (off-policy cross-tokenizer KD; adjacent).

### I. Privacy and safety variants

DP-OPD (standard OPD with DP-SGD on student; frozen teacher), DPSW (dual-perspective safety token weighting).

## Audit Results

Detailed audit provenance is archived under [assets/research](../assets/research/), including white-box syntheses, line audits, baseline audits, and candidate rows.

### Verified strict seeds (in table)

- GKD pure on-policy variants.
- MiniLLM reverse-KL OPD, with teacher-mixed sampling noted as an implementation caveat.
- Entropy-Aware OPD.
- G-OPD.
- REOPOLD.
- Veto.
- Fast OPD for retained current-policy prefixes.
- DistillSpec only for the draft-generated/on-policy drafter-training variant.

### Promoted strict candidates

These were promoted into the main paper table after candidate line audit.

- vOPD: variance reduction via closed-form control variate baseline.
- TIP: 2D token importance selection; 50% tokens matches full OPD.
- AOPD: asymmetric advantage treatment; localized divergence minimization.
- CaOPD: calibration-corrected OPD targets.
- SOD: step-level divergence reweighting for agentic OPD.
- SimCT: strict cross-tokenizer OPD via minimal aligned units.
- OPSDL: short-context self-teacher for long-context distillation.
- DP-OPD: differentially private OPD with frozen teacher.
- Rock Tokens: gradient freeze on persistent high-loss tokens.

### Partial or downgrade cases

- DistiLLM and DistiLLM-2 use skew-KL variants but weaken rollout freshness through replay, batching, or teacher/student response mixtures.
- StableOPD has a strict OPD core but a mixed full recipe, so it is partial at method level.
- Speculative KD uses fresh hybrid traces, but teacher replacement means the supervised sequence is not cleanly the student's exact rollout.
- PACED is a mixed method: the reverse-KL self-distillation subtrack can be strict, but the full method should remain partial unless split.
- Lightning OPD is an offline approximation: it caches teacher log-probs over reference/SFT rollouts and therefore fails current-policy freshness.
- AdaSwitch switches from student to teacher mid-sequence; hybrid trajectory.
- HPD mixes off-policy and approximate on-policy sampling; partial.
- DSKD v2 is strict only in an optional on-policy mode; method-level label is partial.
- SelecTKD and ToDi remain adjacent after candidate audit.

### Adjacent methods (novel objectives, not on-policy by default)

- CSD (concrete score matching; non-KL/non-f-divergence).
- AMiD (alpha-mixture assistant distribution; generalizes DistiLLM).
- BiLD (top-k bi-directional logit differences).
- RSKD (sparse logit caching enabler).
- OKD (online teacher adaptation; inverts the OPD loop).
- EGAD (entropy-guided adaptive KD; off-policy despite name).
- MultiLevelOT, ULD, DWA-KD (cross-tokenizer KD; off-policy).

### Open gaps

- No Renyi/alpha-divergence or Bregman divergence OPD published as of May 2026.
- No systematic divergence scheduling study (progressive FKL-to-RKL).
- Cross-tokenizer OPD at scale (>30B) is unvalidated.
- SimCT is the only source-verified strict cross-tokenizer OPD row; DSKD v2 is partial at method level; all others are off-policy or adjacent until new evidence appears.
