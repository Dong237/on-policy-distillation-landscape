# WP5 — Industrial and Systems OPD Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML for all technical reports; official blogs, GitHub repos, and docs for frameworks
**Methodology:** Five independent research agents fetched primary-source HTML and verified C1/C2/C3 evidence against exact section quotes. Framework support verified by checking actual docs/READMEs for the student-generates → teacher-scores → KL-loss loop.

---

## 1. Memo

### 1.1 Central finding

**Three industrial methods have confirmed strict OPD stages with exact loss equations; one has suggestive but mechanistically unconfirmed evidence; two are borderline or unconfirmed; and seven are confirmed false positives.** Among frameworks, only TRL GKDTrainer and NeMo RL implement strict OPD out of the box; OpenRLHF and verl are reward-RL-only.

### 1.2 Industrial OPD landscape

| Industrial method | OPD evidence strength | Key gap |
|---|---|---|
| **MiMo-V2-Flash MOPD** | **Strong** — exact loss equation (Eq.7-9), domain teacher selection, systems details | none |
| **Nemotron-Cascade2 MOPD** | **Strong** — exact loss equation (Eq.4), truncated importance weighting, cascade ordering | none |
| **DistillSpec** | **Strong** — f-divergence with on-policy justification (Theorem 4.1), multiple divergence options | none |
| **Qwen3 OPD stage** | **Suggestive** — "on-policy knowledge transfer" + "logit distillation" but NO equation, NO algorithm, NO mechanistic detail | No KL equation; "on-policy" is descriptive, not mechanistic |
| **Gemma 2 post-training** | **Moderate** — "distillation on the student's distribution" + GKD/MiniLLM citation | No loss equation; relies on GKD paper citation |
| **Gemma 3** | **Weak** — "all models use knowledge distillation" but no on-policy evidence | KD and RL are separate; no student-generated rollout confirmation for KD |

### 1.3 Confirmed false positives

| Claimed distillation | Actual mechanism | Why not OPD |
|---|---|---|
| **DeepSeek-R1 distilled** | SFT on 800k teacher-generated reasoning traces | "We directly fine-tuned...using the 800k samples curated with DeepSeek-R1" (S2.4); "only SFT and do not include an RL stage" |
| **Minitron** | Offline logit KD on fixed 8T pretraining corpus | Student trains on a fixed dataset with teacher logit supervision; never generates own rollouts |
| **InternVL3.5** | Online GSPO/MPO (reward-RL) | "Online RL is disclosed, but no teacher/discriminator/reference distillation signal on student trajectories" |
| **MiniCPM-V 4.5** | Hybrid RL + DPO | "Strict OPD evidence is not disclosed" |
| **Qwen3-VL** | No OPD evidence in public report | "Public evidence does not establish strict OPD" |
| **Gemini 2.5** | "RL-based post-training refinement" | Closed model; no teacher-on-rollout evidence |
| **VLM-R1** | Reward-only GRPO | "Missing C2 and C3: on-policy reward optimization is disclosed, but no teacher-style distillation signal" |

### 1.4 Framework OPD support

| Framework | Full OPD loop? | Explicit recipe? | Evidence |
|---|---|---|---|
| **TRL GKDTrainer** | **Yes** | Yes | `lmbda=1.0` gives pure on-policy JSD; student generates → teacher logits → generalized JSD loss |
| **NeMo RL** | **Yes** | Yes (`run_distillation.py`) | "Student generates on-policy sequences and aligns logits to a larger teacher via KL" |
| **OpenRLHF** | **No** | No | PPO, REINFORCE++, GRPO, RLOO, DAPO only; no KD/distillation implementation found |
| **verl** | **No** | No | PPO, GRPO, GSPO, DAPO only; no teacher-student distillation loop |

### 1.5 Systems tricks for reducing teacher cost

| Technique | Method | How it works | Speedup |
|---|---|---|---|
| **Cached teacher log-probs** | Lightning OPD | Teacher queried once on SFT rollouts; stored per-token log-probs reused during training | **4x** (120→30 GPU-hrs at 8B) |
| **Prefix-only scoring** | Fast OPD | Teacher scores only informative reasoning prefixes, not full traces | Reported "reducing full-trajectory OPD cost" |
| **Truncated importance weighting** | Nemotron-Cascade2 | Masks tokens where train/inference policy ratio is outside [0.5, 2.0] | Enables train-inference mismatch handling |
| **Rollout routing replay (R3)** | MiMo-V2-Flash | Reuses routed expert assignments from rollout in training | Reduces expert re-computation |
| **Staleness-aware sampling** | MiMo-V2-Flash | Truncated importance sampling for partially stale rollouts | Improves sample efficiency |
| **Sampled-logit KD** | Gemma 3 | "Sample 256 logits per token, weighted by teacher probabilities" | Reduces logit transfer bandwidth |
| **Token selection** | TIP, Rock Tokens | Teacher scores all positions but loss applied only to selected tokens | Reduces effective training tokens |
| **Entropy masking** | REOPOLD | Filters out uninformative tokens with high predictive uncertainty | Reduces wasted gradient steps |
| **Cross-tokenizer aligned units** | SimCT | Multi-token continuation scoring aligns teacher/student tokenizations | Enables cross-architecture OPD |
| **Draft-target f-divergence** | DistillSpec | Draft model generates on-policy; target model provides tailored f-divergence supervision | Acceptance-rate-optimal divergence choice |
| **Control variate baseline** | vOPD | Detached closed-form KL baseline reduces variance of reverse-KL estimator | Same teacher cost, lower variance |

---

## 2. Evidence Ledger

### 2.1 Industrial methods — confirmed strict OPD stages

| method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2-Flash MOPD | https://arxiv.org/abs/2601.02780 | S4.4 Eq.7-9, S4.1, S4.6 | "student samples from its evolving distribution" (S4.4) | "Domain-specialized teachers provide dense and token-level reward" (S4.1); `sg[log pi_domain(y_t)/pi_theta(y_t)]` (Eq.9) | `A^MOPD,t = sg[log pi_domain(y_t)/pi_theta(y_t)] + alpha * A^ORM` (Eq.9); surrogate loss (Eq.7) | multi_teacher white_box logprobs | R3 rollout routing replay; staleness-aware truncated importance sampling (S4.6) | token | current_policy | strict_opd (stage-scoped) | Strict for MOPD stage only; full pipeline is SFT→RL→MOPD | high (source_verified) |
| Nemotron-Cascade2 MOPD | https://arxiv.org/abs/2603.19220 | S4.4 Eq.3-4, S4.1, Fig.2 | "strict on-policy student training" (S4.4); GRPO single-gradient-update rollouts | `a_t^MOPD = log pi^domain_i(y_t\|s_t) - log pi^train(y_t\|s_t)` (Eq.4) -- "positive when the domain teacher assigns higher probability" | `L_MOPD = -E[1/\|V(y)\| sum w_t * sg[a_t^MOPD] * log pi_train(y_t\|s_t)]` (Eq.4); truncated importance: `w_t = sg[r_t] * 1[0.5 <= r_t <= 2.0]` (Eq.3) | multi_teacher (3 domain teachers from cascade checkpoints) | NeMo RL; Megatron parallelism; "rollout size 4, 128 prompts per update, effective batch 512, converges within 40-50 steps" | token | current_policy | strict_opd (stage-scoped) | Strict for MOPD stage; cascade is IF-RL→Multi-domain RL→MOPD→RLHF→Long-context RL→Code RL→SWE RL | high (source_verified) |
| DistillSpec | https://arxiv.org/abs/2310.08461 | S3-4, Eq.2, Theorem 4.1 | "draft model generates on-policy candidate tokens" (Eq.2) | "target model distributional supervision via FKL/RKL/JS/TVD/GenJSD" (Table 1) | "tailored f-divergence objective trains draft; Theorem 4.1 bounds acceptance rate given on-policy loss epsilon" | white_box target-model logits | speculative decoding verification; f-divergence selection per task | token | current_policy | strict_opd | none | high (source_verified) |

### 2.2 Industrial methods — suggestive or borderline

| method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 OPD stage | https://arxiv.org/abs/2505.09388 | S1, S4 | NOT FOUND — no mechanistic description of student generating sequences | "directly distilling the output logits from teacher models into lightweight student models" (S4) | NOT FOUND — no KL equation, no loss function for distillation | white_box logits (stated) | "1/10 of the GPU hours" vs full 4-stage pipeline; small models only (0.6B-8B) | token (implied from "logits") | unclear ("on-policy knowledge transfer" in S1, but no mechanism) | strict_opd (borderline confidence) | Report says "leveraging both off-policy and on-policy knowledge transfer" (S1) but provides NO equation, NO algorithm, NO description of student generating. The "on-policy" phrase is descriptive, not mechanistic. Existing repo label kept due to wp0_consensus, but confidence should remain medium. | medium |
| Gemma 2 post-training | https://arxiv.org/abs/2408.00118 | S4 | Implied by "on the student's distribution" | "We also run distillation from the teacher on the student's distribution [Agarwal et al. 2024, Gu et al. 2024]" (S4) -- GKD and MiniLLM cited | NOT FOUND — no loss equation; relies on GKD paper citation for C3 | white_box logits (implied) | not described | token (implied from GKD citation) | unclear (implied current_policy from GKD) | borderline_strict | "on the student's distribution" + GKD/MiniLLM citation is strong indirect evidence, but no loss equation or explicit "on-policy" confirmation | medium (source_verified) |
| Gemma 3 | https://arxiv.org/abs/2503.19786 | S2.2, S3 | NOT FOUND for KD phase | "sample 256 logits per token, weighted by teacher probabilities...via cross-entropy loss" (S2.2) | "cross-entropy loss" for pretraining KD (S2.2); post-training: "improved version of knowledge distillation from a large IT teacher" (S3) with no equation | white_box logits (sampled 256 per token) | sampled-logit KD reduces bandwidth; RL phase uses BOND/WARM/WARP separately | token (sampled 256) | unclear | adjacent | KD is confirmed but not connected to student on-policy generation; "All Gemma 3 models are trained with knowledge distillation" but no on-policy evidence; RL phase (BOND/WARP) is independent of KD | medium |

### 2.3 Speculative and systems OPD

| method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Speculative KD | https://arxiv.org/abs/2410.11325 | Alg.1, Eq.1 | "student proposes y_i~M_s but rejected tokens replaced by teacher" (Alg.1) | "teacher logits D(M_t\|\|M_s) supervise" (Eq.1) | "token-KL on interleaved trajectory" | white_box target logits | speculative propose-and-verify; top-K acceptance | token | mixed (student+teacher tokens in trajectory) | partial_opd | C1 weakened: mixed-policy trajectory (student proposes, teacher replaces); early training ≈ offline KD, late training ≈ on-policy KD | medium (source_verified) |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | S3.2, Thm 3.5-3.7 | C1 FAILS: "teacher is queried once to precompute and store per-token log-probabilities" on SFT rollouts (S3.2) | "teacher log-probs exist on SFT distribution" | "A_t = log pi_T - log pi_theta consumed as advantage; Thm 3.5 bounds gradient gap" | white_box logits (precomputed) | **4x speedup**: zero live teacher serving; precomputed on pi_ref; equivalence under teacher consistency (sigma_Delta=0) | token | off_policy (precomputed on pi_ref) | partial_opd | Provably equivalent optimum under teacher consistency; 4x speedup trades strict C1 freshness for efficiency | high (source_verified) |
| SimCT (cross-tokenizer) | https://arxiv.org/abs/2605.07711 | S3-4 | "student generates text states with its own tokenizer" | "teacher distribution evaluated on those states through aligned continuation units" | "reverse-KL OPD loss consumes the aligned supervision" | white_box logits (cross-tokenizer projected) | minimal aligned units + multi-token continuation scoring enable different tokenizers | token | current_policy | strict_opd | none | high (source_verified) |

### 2.4 Frameworks

| method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRL GKDTrainer | https://huggingface.co/docs/trl/gkd_trainer | Docs | "When lmbda=1.0, the loss reduces to on-policy JSD, where the student generates output sequences" | "token-specific feedback on these sequences from the teacher" | "generalized JSD loss interpolating forward/reverse KL via beta" | white_box teacher logits | Accelerate/DeepSpeed backend; PEFT/LoRA support | token | current_policy (lmbda=1.0) | strict_opd_support | lmbda<1.0 mixes offline data, reducing on-policy strength | high |
| NeMo RL OPD | https://github.com/NVIDIA-NeMo/RL | README | "Student generates on-policy sequences" | "aligns logits to a larger teacher via KL" | Dedicated recipe: `run_distillation.py` with Qwen3-1.7B student, Qwen3-4B teacher | white_box teacher logits | Megatron parallelism; Ray-based teacher serving alongside student rollouts | token | current_policy | strict_opd_support | VLM OPD recipe not yet confirmed | high |
| OpenRLHF | https://github.com/OpenRLHF/OpenRLHF | README | student generates for RL | NO teacher logit supervision found | NO KD/distillation loss | N/A for OPD | Ray + DeepSpeed distributed RL | N/A | current_policy (for RL) | reward_rl_only | No distillation/KD implementation; PPO/REINFORCE++/GRPO/RLOO/DAPO only | high |
| verl | https://github.com/verl-project/verl | README | student generates for RL | NO teacher logit supervision found | NO KD/distillation loss | N/A for OPD | Ray distributed engine; async rollouts | N/A | current_policy (for RL) | reward_rl_only | No distillation support; PPO/GRPO/GSPO/DAPO/PRIME only | high |

---

## 3. Candidate Table Rows / Updates

### 3.1 Updates to `tables/industrial_reports.md`

**Qwen3 (row 7):** No label change, but update notes:
- Current: `confidence: high`
- Proposed: `confidence: medium`
- Reason: Report says "on-policy knowledge transfer" + "logit distillation" but provides NO equation, NO algorithm pseudocode, NO mechanistic description of student generating sequences. Evidence is suggestive but not confirmatory. Keep `strict_opd` label due to wp0_consensus, but downgrade confidence.

**Gemma 3 (row 13):** No label change needed. Already `adjacent` with `medium` confidence. Confirmed by this audit: KD is present ("all models use knowledge distillation") but no on-policy evidence for the KD phase.

### 3.2 Updates to `tables/frameworks.md`

**OpenRLHF (row 10):** Update `strict_opd_support` from `unclear` to `no`. No KD/distillation implementation found.

**verl (row 9):** Update `strict_opd_support` from `unclear` to `no`. No KD/distillation implementation found.

### 3.3 No new rows needed

All industrial methods and frameworks are already tracked.

---

## 4. Adjacent / False-Positive Ledger

### 4.1 Offline KD / SFT on traces (not OPD)

| method | link | mechanism | missing_condition | key_quote | confidence |
|---|---|---|---|---|---|
| **DeepSeek-R1 Distilled** | https://arxiv.org/abs/2501.12948 | SFT on 800k teacher-generated reasoning traces | C1: student never generates own rollouts; C3: SFT, not KL distillation | "We directly fine-tuned...using the 800k samples curated with DeepSeek-R1" (S2.4); "only SFT and do not include an RL stage" | high |
| **Minitron** | https://arxiv.org/abs/2407.14679 | Offline logit KD on fixed 8T pretraining corpus after pruning | C1: student trains on fixed dataset, never generates own data | "pruned student retrained on Nemotron-4 curated 8T base pretraining dataset" by minimizing CLM + logit KD + hidden-state loss | high |
| **Gemma 3 KD** | https://arxiv.org/abs/2503.19786 | KD at pretraining and post-training; no on-policy evidence | C1: no student-generated rollout confirmation for KD phase | "All Gemma 3 models are trained with knowledge distillation" but mechanism unspecified; RL (BOND/WARP) is separate | medium |

### 4.2 Reward-only RLVR (on-policy rollouts but no teacher distillation)

| method | link | mechanism | missing_condition | confidence |
|---|---|---|---|---|
| **InternVL3.5** | https://internvl.github.io/blog/2025-08-26-InternVL-3.5/ | Online GSPO/MPO cascade RL | C2/C3: no teacher distillation signal on student rollouts | medium |
| **VLM-R1** | https://arxiv.org/abs/2504.07615 | GRPO with outcome reward only | C2/C3: no teacher at all | high |
| **Qwen3-VL** | https://arxiv.org/abs/2511.21631 | Multimodal training report | No OPD evidence in public materials | medium |
| **MiniCPM-V 4.5** | https://github.com/OpenBMB/MiniCPM-V | Hybrid RL + DPO | C2/C3: no teacher distillation signal disclosed | medium |
| **Gemini 2.5** | https://huggingface.co/papers/2507.06261 | "RL-based post-training refinement" | Closed model; no teacher-on-rollout evidence | medium |

### 4.3 Frameworks (reward-RL only, not OPD)

| framework | link | what_it_supports | why_not_opd | confidence |
|---|---|---|---|---|
| **OpenRLHF** | https://github.com/OpenRLHF/OpenRLHF | PPO, REINFORCE++, GRPO, RLOO, DAPO | No KD/distillation implementation; no teacher-student loop | high |
| **verl** | https://github.com/verl-project/verl | PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, DAPO, PRIME | No KD/distillation support; optimized for verifiable-reward RL | high |

---

## 5. Gap Map

### 5.1 Evidence gaps requiring resolution

| Method | Current label | Gap | Priority | Resolution path |
|---|---|---|---|---|
| **Qwen3 OPD stage** | strict_opd (medium confidence) | No KL equation, no algorithm, no mechanistic description of student generating sequences; "on-policy knowledge transfer" is the only evidence | **high** | Wait for Qwen3 training code release or detailed blog post; current evidence is suggestive but not confirmatory |
| **Gemma 2 post-training** | borderline_strict (medium) | No loss equation; relies entirely on GKD/MiniLLM citation + "on the student's distribution" phrase | medium | Google unlikely to publish more detail; accept as borderline based on indirect evidence |
| **Gemma 3 post-training KD** | adjacent | Post-training KD uses "improved version of knowledge distillation from a large IT teacher" but no on-policy evidence | low | If Google publishes Gemma 3 training recipe, check for student-generated rollout evidence |
| **Qwen3-VL** | unclear | "Public evidence does not establish strict OPD" | medium | Check for updated arXiv version or detailed training blog |

### 5.2 Systems gaps

| System question | Status | What's needed |
|---|---|---|
| **Multi-teacher routing cost** | MiMo and Nemotron both use domain teachers; neither quantifies per-method switching overhead | Benchmark: multi-teacher serving latency vs single-teacher |
| **Cross-tokenizer OPD serving** | SimCT proves feasibility; no industrial deployment reported | Check if any Qwen→Llama or Gemma→Mistral cross-tokenizer OPD has been deployed |
| **Teacher logit compression** | Gemma 3 uses "256 sampled logits per token"; no other method reports logit bandwidth optimization | Benchmark: full vocab logits vs sampled-k logits vs top-k logits for OPD accuracy |
| **VLM OPD framework support** | NeMo RL has VLM GRPO but VLM OPD is "planned"; no framework has a turnkey VLM OPD recipe | Track NeMo RL and TRL for VLM-GKD or VLM-OPD recipes |

### 5.3 Industrial coverage gaps

| Industrial report | Status | What's needed |
|---|---|---|
| **Llama 4** | Not tracked | Check Meta's Llama 4 technical report (if released) for any distillation stages |
| **Yi-Lightning** | Not tracked | Check 01.AI reports for distillation methodology |
| **Mistral Large/Medium** | Not tracked | Check Mistral technical reports for any KD pipeline disclosures |
| **Claude training** | Not applicable | Anthropic does not publish training methodology details |

### 5.4 Verification actions

| Action | Priority | Target | What to do |
|---|---|---|---|
| Monitor | high | Qwen3 | Watch for training code release or detailed blog confirming on-policy distillation mechanism |
| Table update | medium | `tables/frameworks.md` | Set `strict_opd_support: no` for OpenRLHF and verl |
| Table update | medium | `tables/industrial_reports.md` | Downgrade Qwen3 confidence from `high` to `medium` |
| Monitor | low | NeMo RL | Track VLM OPD recipe development |
| Monitor | low | Gemma 3 | Watch for training recipe publication |

---

## 6. Cross-validation with existing repo labels

| Method | Prior repo label | This audit | Change? |
|---|---|---|---|
| MiMo-V2-Flash MOPD | strict_opd (source_verified) | strict_opd | confirmed |
| Nemotron-Cascade2 MOPD | strict_opd (wp0_consensus) | strict_opd | confirmed; exact equations extracted |
| DistillSpec | strict_opd (source_verified) | strict_opd | confirmed |
| Qwen3 OPD stage | strict_opd (wp0_consensus, high) | strict_opd (medium confidence) | **downgrade confidence from high to medium** |
| Gemma 2 post-training | borderline_strict (source_verified, medium) | borderline_strict | confirmed |
| Gemma 3 | adjacent (medium) | adjacent | confirmed |
| Speculative KD | partial_opd (source_verified) | partial_opd | confirmed |
| Lightning OPD | partial_opd (source_verified) | partial_opd | confirmed |
| SimCT | strict_opd (source_verified) | strict_opd | confirmed |
| DeepSeek-R1 Distilled | not_opd (adjacent_work) | not_opd | confirmed |
| Minitron | adjacent (legacy) | not_opd | confirmed offline KD |
| InternVL3.5 | adjacent | adjacent | confirmed |
| MiniCPM-V 4.5 | adjacent | adjacent | confirmed |
| Qwen3-VL | unclear | unclear | confirmed |
| TRL GKDTrainer | strict_opd_support (high) | strict_opd_support | confirmed |
| NeMo RL | strict_opd_support (medium) | strict_opd_support | confirmed; explicit recipe found |
| OpenRLHF | unclear | **reward_rl_only** | **RESOLVED: no OPD support** |
| verl | unclear | **reward_rl_only** | **RESOLVED: no OPD support** |

### Summary of changes

| Change | Type | Reason |
|---|---|---|
| Qwen3 confidence high→medium | downgrade confidence | No mechanistic OPD evidence in report; "on-policy knowledge transfer" is descriptive only |
| OpenRLHF strict_opd_support unclear→no | resolved | No KD/distillation implementation found; reward-RL only |
| verl strict_opd_support unclear→no | resolved | No KD/distillation implementation found; reward-RL only |
| NeMo RL confidence medium→high | upgrade confidence | Explicit OPD recipe confirmed (`run_distillation.py`) |

---

## 7. Architectural patterns

### 7.1 MiMo vs Nemotron: Two approaches to industrial MOPD

| Dimension | MiMo-V2-Flash | Nemotron-Cascade2 |
|---|---|---|
| **KD+RL fusion** | `A_KD + alpha * A_ORM` (KL + outcome reward) | Pure `A_KD` (KL only, no outcome reward) |
| **Domain teachers** | "Flexible: RL-derived, SFT, or self" | 3 specific checkpoints from cascade stages |
| **Stability mechanism** | R3 rollout routing replay + staleness-aware sampling | Truncated importance weighting `w_t = 1[0.5 <= r_t <= 2.0]` |
| **Pipeline position** | Stage 3 (after SFT→RL) | Middle of 7-stage cascade (IF-RL→Multi-RL→**MOPD**→RLHF→...) |
| **Convergence** | Not reported | "40-50 steps" with batch 512 |
| **Framework** | Custom | NeMo RL |

### 7.2 Qwen3 vs Gemma vs DeepSeek: Three distillation philosophies

| Dimension | Qwen3 | Gemma 2/3 | DeepSeek-R1 |
|---|---|---|---|
| **Distillation type** | Claimed "on-policy + off-policy knowledge transfer" | Gemma 2: GKD-style on student distribution; Gemma 3: KD at pretraining + post-training | Pure SFT on teacher traces |
| **Mechanistic evidence** | None (no equation, no algorithm) | Moderate (GKD citation + "student's distribution") | Explicit ("800k samples curated with DeepSeek-R1") |
| **Which models** | Small models only (0.6B-8B) | All models | Specific distilled checkpoints |
| **RL involvement** | RL is separate (4-stage pipeline for large models) | RL (BOND/WARP) is separate from KD | "Do not include an RL stage" for distilled models |
| **OPD label** | strict_opd (medium confidence) | Gemma 2: borderline_strict; Gemma 3: adjacent | not_opd |

---

## Sources

- [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [MiMo-V2-Flash Technical Report (arXiv 2601.02780)](https://arxiv.org/abs/2601.02780)
- [Nemotron-Cascade 2 (arXiv 2603.19220)](https://arxiv.org/abs/2603.19220)
- [Gemma 2 Technical Report (arXiv 2408.00118)](https://arxiv.org/abs/2408.00118)
- [Gemma 3 Technical Report (arXiv 2503.19786)](https://arxiv.org/abs/2503.19786)
- [DistillSpec (arXiv 2310.08461)](https://arxiv.org/abs/2310.08461)
- [Speculative KD (arXiv 2410.11325)](https://arxiv.org/abs/2410.11325)
- [Lightning OPD (arXiv 2604.13010)](https://arxiv.org/abs/2604.13010)
- [SimCT (arXiv 2605.07711)](https://arxiv.org/abs/2605.07711)
- [DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)
- [Minitron (arXiv 2407.14679)](https://arxiv.org/abs/2407.14679)
- [TRL GKDTrainer Docs](https://huggingface.co/docs/trl/gkd_trainer)
- [NeMo RL GitHub](https://github.com/NVIDIA-NeMo/RL)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [verl GitHub](https://github.com/verl-project/verl)
- [InternVL3.5 Blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [MiniCPM-V GitHub](https://github.com/OpenBMB/MiniCPM-V)
- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
