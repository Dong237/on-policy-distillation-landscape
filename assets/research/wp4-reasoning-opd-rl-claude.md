# WP4 — Reasoning and OPD+RL Hybrids for Math, Code, Long-CoT, and Reasoning-Heavy LLM Training

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML/PDF for all primary methods; repo `tables/opd_papers.md` and `tables/adjacent_work.md` cross-referenced against `taxonomies/strict-opd-definition.md`
**Methodology:** Five independent research agents fetched primary-source HTML for RLKD, G-OPD, REOPOLD, industrial methods (MiMo-V2-Flash, Nemotron-Cascade2, Qwen3), and stability-focused papers (StableOPD, KDRL, SCOPE, Lightning OPD). Results reconciled below. Repo sweep confirmed no unresolved alternating KD/RL methods remain.

---

## 1. Memo

### 1.1 Central finding

**The reasoning OPD landscape divides into five structural patterns along the KD-RL axis.** The boundary between strict OPD and adjacent reward-RL is cleanly defined by whether the teacher's signal remains **distributional and per-token in the loss** (strict) or is **compressed to a scalar reward consumed by GRPO/PPO** (adjacent). Intermediate patterns produce borderline or partial labels.

### 1.2 Five structural patterns

| Pattern | Teacher signal form | Loss structure | Typical label | Examples |
|---|---|---|---|---|
| **A. Pure OPD** | Dense teacher logits per token | Token-level KL/JSD only | strict_opd | GKD, MiniLLM, vOPD, TIP, AOPD, Rock Tokens, Entropy-Aware OPD, Fast OPD, OPSD, CRISP, SDPO, OPCD, SDFT |
| **B. OPD reframed as RL** | Teacher log-ratio treated as token-level reward | Policy-gradient with log pi_T/pi_ref as reward | strict_opd | G-OPD (ExOPD), REOPOLD |
| **C. Joint KD+RL (dense teacher arm preserved)** | Dense teacher KL advantage + scalar outcome reward | Token-level KL advantage added to GRPO advantage | strict_opd to borderline | SOD, VOLD, MiMo-V2-Flash MOPD, Nemotron-Cascade2, KDRL, SCOPE |
| **D. Mixed on/off-policy distillation** | Teacher logits on mixed or partial rollouts | KL on mixed-provenance data | partial_opd | PACED, StableOPD, HDPO, Lightning OPD |
| **E. RL with teacher-informed reward** | Teacher structure/critique → scalar reward | Standard GRPO/PPO on scalar rewards | adjacent | RLKD, GAD, OVD, PRISM |

### 1.3 Which methods combine dense teacher supervision with GRPO/PPO/RLVR?

**Pattern C (joint KD+RL) methods** are the most important WP4 subfamily. They share a common formula:

```
A_total = A_KD + alpha * A_reward
```

where `A_KD = log pi_T(y_t|s_t) - log pi_theta(y_t|s_t)` (token-level KL advantage) and `A_reward` is outcome/verifier reward, consumed jointly in a policy-gradient update.

| Method | KD signal | RL signal | How combined | Label |
|---|---|---|---|---|
| **SOD** | Step-wise teacher KL per token | GRPO outcome reward | "step-divergence-weighted KL plus GRPO" — dense teacher term preserved alongside GRPO | strict_opd |
| **VOLD** | Text-teacher KL on VLM rollouts | GRPO on reasoning correctness | Unified Stage 2: KL + GRPO co-optimized | strict_opd |
| **MiMo-V2-Flash MOPD** | `sg[log pi_domain(y_t) / pi_theta(y_t)]` | `alpha * A^ORM` | Token-level KL advantage + outcome reward advantage in single GRPO step (Eq.9) | strict_opd |
| **Nemotron-Cascade2** | `a_t^MOPD = log pi^domain(y_t|s_t) - log pi^train(y_t|s_t)` | Cascade RL rewards | Dense distillation advantage + truncated importance weighting; MOPD is one stage in cascade | strict_opd |
| **KDRL** | Reverse KL `D_KL(pi_theta \|\| pi_T)` | GRPO outcome reward | `J_KDRL = J_GRPO - beta * D_KL`; beta anneals from 5e-3 to 1e-3 (KD-heavy early → reward-heavy late) | borderline_strict |
| **SCOPE** | Teacher-perplexity-weighted KL on incorrect rollouts | Student-perplexity-weighted MLE on correct rollouts | Routing by correctness: incorrect → teacher KL, correct → student MLE | borderline_strict |

**Key distinction from reward-only RL:** In Pattern C, the teacher's log-probability distribution appears in the loss function and its gradient flows through per-token terms. In Pattern E (RLKD, GAD), the teacher signal is compressed to a scalar before entering the loss.

### 1.4 Which methods are reward-only RLVR despite on-policy rollouts?

| Method | Why adjacent | Teacher signal compression point |
|---|---|---|
| **RLKD** | GSRM converts teacher reasoning traces → step-match scalar reward via LLM judge; `R_total = 3*R_acc + 3*R_gsrm + 2*R_format + 2*R_tag` consumed by standard GRPO (Eq.2) | Teacher traces → structured comparison → scalar reward → GRPO advantage |
| **GAD** | Discriminator compares student/teacher outputs → scalar score → GRPO reward | Teacher text → discriminator → scalar → GRPO |
| **OVD** | Verbal scores (0-9) from API teacher → rejection sampling + GRPO; scores never enter loss | Teacher API → verbal score → filter + GRPO |
| **VLM-R1** | Outcome reward only; no teacher at all | No teacher signal |

### 1.5 Which methods have strict subcomponents but partial method-level recipes?

| Method | Strict subcomponent | Why method-level is partial |
|---|---|---|
| **PACED** | Reverse-KL self-distillation track samples student sequences and uses teacher supervision | Full method mixes forward-KL track (teacher-generated sequences) with reverse-KL track; curriculum weighting changes data provenance |
| **StableOPD** | Core OPD loop (student rollout + teacher logits + KL) is strict | Proposed stabilization recipe mixes student rollouts with golden/auxiliary data and reference KL regularization |
| **HDPO** | GRPO arm generates current student rollouts (but reward-only) | Distillation arm generates privileged-teacher rollouts and teacher-forces student; C1 fails for distillation |
| **Lightning OPD** | The optimized objective is identical to OPD's gradient in expectation | C1 fails: teacher log-probs precomputed on SFT rollouts, not current student; equivalence holds only under teacher consistency (sigma_Delta=0) |
| **OEL** | OPCD-style consolidation substage | Full method includes deployment collection and knowledge extraction stages |
| **GATES** | On-policy arm (lambda_on=0.1): student generates, tutor scores | Off-policy arm (lambda_off=1.0, dominant): SFT on tutor-generated traces |

### 1.6 Long-CoT drift, mode collapse, length inflation, and teacher over-imitation

| Problem | Which methods address it | Mechanism | Key quote |
|---|---|---|---|
| **Length inflation** | StableOPD | Identifies self-reinforcing feedback loop where repetitive tokens get disproportionately large RKL advantages; proposes golden-data anchoring + reference KL | "once [repetitive tokens] become sufficiently common, their combination of frequency and disproportionately large token-level advantages allows them to steer subsequent OPD updates toward repetitive continuations" (S3.5) |
| **Mode collapse / diversity loss** | SCOPE | Routes correct rollouts to student-weighted MLE (reinforces low-confidence successes) and incorrect to teacher KL; prevents "Pass@k Paradox" where Pass@32 degrades while Pass@1 rises | "high teacher perplexity indicates severe context degradation" (S2.2) |
| **Teacher over-imitation** | G-OPD, KDRL | G-OPD: reward extrapolation with lambda>1 overshoots teacher distribution; KDRL: annealing beta from KD-heavy to reward-heavy | G-OPD: "ExOPD with appropriate reward extrapolation (lambda=1.25) consistently outperforms OPD and domain teacher" (S4.1.2) |
| **Gradient instability** | REOPOLD, vOPD | REOPOLD: clipped reverse-KL with principled floor `log(lambda/(1-lambda))` + entropy masking; vOPD: control variate baseline reduces variance | REOPOLD: "We employ this asymptotic limit as a principled floor to truncate the heavy-tailed negative rewards" (S4.1) |
| **Noisy teacher on flawed prefixes** | SCOPE, TIP | SCOPE: teacher-perplexity gating filters noisy teacher guidance; TIP: token importance selection skips uninformative positions | SCOPE: correct → student MLE, incorrect → teacher KL with perplexity weighting |

### 1.7 Compute and serving costs

| Method | Teacher serving cost | Claimed speedup | Key trade-off |
|---|---|---|---|
| **Lightning OPD** | **Zero** (precomputed once on SFT rollouts) | **4.0x** (120→30 GPU-hrs at 8B, S4.3) | Requires teacher consistency (same teacher for SFT and OPD stages) |
| **Fast OPD** | Reduced (teacher scores reasoning prefixes only, not full traces) | Reported as "reducing full-trajectory OPD cost" | Prefix selection strategy critical |
| **REOPOLD** | Same as OPD (one teacher forward per token) | **2-6.7x efficiency** (600 steps vs 2000 for ProRL, S5.1) | Clipping and entropy masking reduce wasted gradient steps |
| **TIP** | Same teacher cost but fewer effective training tokens | Not quantified as wall-clock | Token selection reduces gradient updates |
| **G-OPD (ExOPD)** | OPD + one reference model forward (for lambda!=1) | No wall-clock numbers | Extra pi_ref forward pass per token |
| **MiMo-V2-Flash MOPD** | Multi-teacher domain scoring per token | Not isolated from full pipeline | Domain-teacher selection amortizes teacher pool |
| **Nemotron-Cascade2** | Multi-teacher scoring + truncated importance weighting | Not isolated | MOPD positioned as regression-recovery in cascade; high cost but short stage |
| **Qwen3** | Offline distillation (no live teacher scoring during RL) | N/A (separate stages) | KD and RL never run simultaneously |
| **KDRL** | Teacher forward pass + GRPO rollout | Not quantified | Beta annealing reduces KD influence over time |

### 1.8 Industrial integration patterns

Three distinct industrial approaches have emerged:

1. **MiMo/Nemotron: Fused KD+RL** — Token-level teacher KL advantage is computed alongside outcome reward advantage in a single GRPO step. The teacher signal is stop-gradiented but remains dense. This is the strongest industrial OPD+RL pattern.

2. **Qwen3: Sequential KD then RL** — Distillation and RL are separate pipeline stages. Small models receive offline strong-to-weak distillation; all models then undergo GRPO-based reasoning RL. No joint KD+RL objective exists.

3. **Gemma 2: Implicit OPD** — Post-training distillation references GKD/MiniLLM but the exact rollout procedure is not public. Borderline_strict by default.

---

## 2. Evidence Ledger

### 2.1 Strict OPD — Pure OPD for reasoning (Pattern A)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Entropy-Aware OPD | https://arxiv.org/abs/2603.07079 | S3-4 | "student follows its own trajectories" | "teacher token distributions and entropy supervise those trajectories" | "entropy-aware KL objective consumes the signal" | white_box logits | token | entropy-gated FKL/RKL | current_policy | strict_opd | none | medium (wp0_consensus) |
| Fast OPD | https://arxiv.org/abs/2602.15260 | S3-4 | "student generates reasoning rollouts" | "teacher token feedback is focused on prefixes" | "reverse-KL prefix loss trains the student" | white_box logits | token (prefix only) | prefix reverse KL | current_policy | strict_opd | none (prefix-only is a cost optimization, not a structural downgrade) | medium (wp0_consensus) |
| vOPD | https://arxiv.org/abs/2605.07865 | S3-4 | "student samples from the current policy" | "teacher logits provide per-token reverse-KL supervision" | "variance-reduced reverse-KL objective" with detached KL control variate | white_box logits | token | reverse KL + control variate | current_policy | strict_opd | none | high (source_verified) |
| TIP | https://arxiv.org/abs/2604.14084 | S3-4 | "student generates on-policy rollouts" | "teacher distributions score student-generated positions" | "reverse-KL OPD loss applied to selected tokens" | white_box logits | token (selected) | selective token reverse KL | current_policy | strict_opd | none | high (source_verified) |
| AOPD | https://arxiv.org/abs/2605.06387 | S3-4 | "student trains on its own trajectories" | "teacher token probabilities supervise student-generated states" | "asymmetric KL objective" — advantage and localized divergence derived from teacher logits | white_box logits | token | asymmetric advantage KL | current_policy | strict_opd | none | high (source_verified) |
| Rock Tokens | https://arxiv.org/abs/2605.09253 | S3-4 | "student rollouts are sampled" | "teacher token distributions supervise those rollouts" | "reverse-KL OPD loss reweighted or frozen for rock tokens" | white_box logits | token | reverse KL + rock-token masking | current_policy | strict_opd | none | high (source_verified) |

### 2.2 Strict OPD — OPD reframed as RL (Pattern B)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G-OPD | https://arxiv.org/abs/2602.12125 | S3.2 Eq.9, S4.1.2 Eq.12 | "student trajectories define the training distribution" | "teacher logits converted into dense reward-like supervision": `r_t^OPD = log pi*(y_t) / pi_ref(y_t)` (Eq.9) | "KL-constrained objective consumes that supervision"; ExOPD lambda>1 overshoots teacher (Eq.12) | white_box logits + reference | token | KL-constrained reward extrapolation | current_policy | strict_opd | none — RL notation wraps a dense teacher log-ratio per token; no scalar compression | high (source_verified) |
| REOPOLD | https://arxiv.org/abs/2603.11137 | S3.1 Remark 3.1, S4.1 Eq.7, S4.2 Eq.8 | "student rollouts are sampled" | "teacher-student log-likelihood ratio acts as a token reward" (Remark 3.1); clipped by `log(lambda/(1-lambda))` (Eq.7) | "relaxed reverse-KL objective" + entropy-based token masking (Eq.8) | white_box logits | token | clipped reverse KL + entropy masking | current_policy (near-policy refreshed) | strict_opd | none — clipping is a stabilizer, not a structural downgrade | high (source_verified) |

### 2.3 Strict OPD — Joint KD+RL with dense teacher arm (Pattern C, strict)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SOD | https://arxiv.org/abs/2605.07725 | S3-4 | "student generates trajectories" | "teacher logits supervise student-visited states" | "step-wise OPD KL term consumes teacher signal alongside GRPO" — dense KL preserved | white_box logits | token | step-weighted KL + GRPO | current_policy | strict_opd | GRPO is auxiliary; dense KL term is the primary distillation signal | high (source_verified) |
| VOLD | https://arxiv.org/abs/2510.23497 | Stage 2 | "VLM student generates on-policy reasoning traces" | "text teacher provides distillation supervision" | "KL plus GRPO objective consumes teacher and reward signals" | white_box logprobs (text teacher) | token | KL + GRPO | current_policy | strict_opd | Strict only for Stage 2 unified RL+OPD | high (source_verified) |
| MiMo-V2-Flash MOPD | https://arxiv.org/abs/2601.02780 | S4.4 Eq.9 | "student samples from its evolving distribution" | "domain teachers provide token-level KL rewards" | `A^MOPD,t = sg[log pi_domain(y_t)/pi_theta(y_t)] + alpha * A^ORM` (Eq.9) — dense teacher KL advantage + outcome reward | multi_teacher logprobs | token | dense KL advantage + outcome reward via GRPO | current_policy | strict_opd | Dense teacher term preserved in the advantage; not compressed to scalar | high (source_verified) |
| Nemotron-Cascade2 | https://arxiv.org/abs/2603.19220 | S4.4 | "strict on-policy student training" | "best-performing domain teachers supervise sampled tokens": `a_t^MOPD = log pi^domain(y_t|s_t) - log pi^train(y_t|s_t)` | "dense distillation advantage objective" + truncated importance weighting | multi_teacher logprobs | token | dense token distillation advantage | current_policy | strict_opd | MOPD is one stage in cascade; strict for the MOPD stage only | high (wp0_consensus) |

### 2.4 Borderline strict — Joint KD+RL with structural caveats (Pattern C, borderline)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KDRL | https://arxiv.org/abs/2506.02208 | S3.1, S3.3-3.4 | "on-policy rollouts are used" via GRPO | "teacher token log-probs supervise the rollout" via RKL term | `J_KDRL = J_GRPO - beta * D_KL(pi_theta \|\| pi_T)` (S3.1); beta anneals 5e-3→1e-3; reward-guided KL masking suppresses distillation on positive-reward responses (S3.4) | white_box logits | token (RKL) + sequence (GRPO reward) | reverse KL + GRPO | current_policy | borderline_strict | Full method hybridizes: KD-RKL subcomponent is strict, but annealing shifts to reward-dominant late training; masking selectively disables distillation on high-reward rollouts | high (source_verified) |
| SCOPE | https://arxiv.org/abs/2604.10688 | S2.1-2.2 | "current-policy rollouts are routed by correctness" | "teacher policy supervises incorrect trajectories" with teacher-perplexity weighting | "teacher-perplexity-weighted KL consumes that signal on the incorrect branch" (S2.2) | white_box logits | token (incorrect branch) + sequence (correct branch MLE) | selective teacher KL + student MLE | current_policy | borderline_strict | Correct-branch uses student-weighted MLE (no teacher), not full-method teacher supervision; "high teacher perplexity indicates severe context degradation" filters noisy teacher | high (source_verified) |

### 2.5 Partial OPD — Mixed on/off-policy or partial rollout coverage (Pattern D)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PACED | https://arxiv.org/abs/2603.11178 | S3-4 | "forward-KL track uses teacher sequences while reverse-KL self-distillation samples student sequences" | "teacher or self-teacher supervises selected track" | "KL objective consumes the signal" — curriculum-weighted | white_box logits | token | curriculum-weighted KL | mixed (FKL teacher-gen + RKL student-gen) | partial_opd | Full method mixes teacher-generated and student-generated tracks; RKL substage alone would be strict | high (source_verified) |
| StableOPD | https://arxiv.org/abs/2604.08527 | S3.5, S4.1 | "student rollouts mixed with golden or auxiliary data" | "teacher logits supervise the mixture" | "OPD plus reference constraint and mixture" | white_box logits | token | OPD + reference KL + golden-data anchor | mixed (student + golden) | partial_opd | Core OPD is strict but stabilization recipe mixes provenance; "golden data serves as an anchor" (S4.1) | high (source_verified) |
| HDPO | https://arxiv.org/abs/2603.23871 | S3.2 | GRPO arm: student generates K rollouts; JSD arm: "generate privileged rollouts y-bar ~ pi_theta(.\|x+y*)" | "pi_T and pi_theta share the same weights" | `L_HDPO = L_GRPO + lambda * L_JSD`; JSD on privileged-teacher rollouts | privileged_context | token (JSD) + sequence (GRPO) | JSD + GRPO | mixed (student for GRPO, privileged-teacher for JSD) | partial_opd | C1 fails for distillation arm: privileged teacher generates rollouts; structurally inverse to OPCD | high (source_verified) |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | S3.2, Thm 3.5-3.7 | C1 FAILS: "teacher is queried once to precompute and store per-token log-probabilities" on SFT rollouts (S3.2) | "teacher log-probs exist" on SFT distribution | `A_t = log pi_T - log pi_theta` consumed as advantage; Thm 3.5 bounds gradient gap by G*sigma_A*sqrt(chi^2) | white_box logits (precomputed) | token | KL on cached rollouts | off_policy (precomputed on pi_ref) | partial_opd | C1 fails strict freshness; provably equivalent optimum under teacher consistency (sigma_Delta=0); **4x speedup** | high (source_verified) |

### 2.6 Industrial — Sequential or pipeline OPD

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 OPD | https://arxiv.org/abs/2505.09388 | S4 (post-training) | "lightweight model generates on-policy sequences" | "larger Qwen teacher provides logits" | "KL minimization fine-tunes the student" — **separate stage from RL** | white_box logits | token | KL to teacher logits | current_policy (for distillation stage) | strict_opd | Strict only for the documented OPD phase; Qwen3 keeps KD and RL as independent sequential stages; "directly distilling output logits from teacher models into lightweight student models" | high (wp0_consensus) |
| Gemma 2 post-training | https://arxiv.org/abs/2408.00118 | Post-training section | C1 implied but "exact rollout procedure not found" | "teacher distillation is stated"; GKD/MiniLLM cited | "exact post-training objective is not found" | white_box logits (implied) | token (implied) | KL (implied) | unclear | borderline_strict | Public report too terse to certify strict; rollout procedure undisclosed | medium (source_verified) |

---

## 3. Candidate Markdown Rows

### 3.1 No new rows needed

All WP4 priority methods are already tracked in `tables/opd_papers.md` or `tables/adjacent_work.md`. The existing labels are confirmed by this audit:

| Method | Existing label | WP4 verdict | Change needed? |
|---|---|---|---|
| G-OPD | strict_opd | strict_opd | no |
| REOPOLD | strict_opd | strict_opd | no |
| Fast OPD | strict_opd | strict_opd | no (upgrade confidence from medium when line audit completes) |
| PACED | partial_opd | partial_opd | no |
| KDRL | borderline_strict | borderline_strict | no |
| RLKD | adjacent | adjacent | no |
| SCOPE | borderline_strict | borderline_strict | no |
| Lightning OPD | partial_opd | partial_opd | no |
| HDPO | partial_opd | partial_opd | no |
| SOD | strict_opd | strict_opd | no |
| MiMo-V2-Flash MOPD | strict_opd | strict_opd | no |
| Nemotron-Cascade2 | strict_opd | strict_opd | no |
| Qwen3 OPD | strict_opd | strict_opd | no |
| StableOPD | partial_opd | partial_opd | no |
| vOPD | strict_opd | strict_opd | no |
| TIP | strict_opd | strict_opd | no |
| AOPD | strict_opd | strict_opd | no |
| Rock Tokens | strict_opd | strict_opd | no |
| Entropy-Aware OPD | strict_opd | strict_opd | no |

---

## 4. Adjacent / False-Positive Ledger

### 4.1 Reward-only RLVR (on-policy rollouts but scalar reward, not distillation)

| method | link | why_adjacent | missing_condition | confidence |
|---|---|---|---|---|
| **RLKD** | https://arxiv.org/abs/2505.16142 | GSRM converts teacher reasoning traces → structured step-match via LLM judge → scalar reward `R_gsrm`; `R_total = 3*R_acc + 3*R_gsrm + 2*R_format + 2*R_tag` consumed by standard GRPO (Eq.2); zero teacher logits computed on student tokens | C2 fails (no token-level teacher signal); C3 fails (scalar reward, not distillation) | high (source-verified this audit) |
| **VLM-R1** | https://arxiv.org/abs/2504.07615 | On-policy GRPO with outcome reward only; no teacher model | C2 fails (no teacher); C3 fails (reward-only) | high |
| **GAD** | https://arxiv.org/abs/2511.10643 | Discriminator score → GRPO reward; "we treat D(G(x)) as a reward" | C3 fails (scalar discriminator reward via GRPO) | high |
| **OVD** | https://arxiv.org/abs/2601.21968 | Verbal scores (0-9) → rejection sampling + GRPO; scores never enter loss | C3 fails (verbal score is filter, not loss signal) | high |

### 4.2 Static DPO / off-policy preference distillation

| method | link | why_adjacent | missing_condition | confidence |
|---|---|---|---|---|
| **LUFFY** | https://arxiv.org/abs/2504.14945 | Off-policy demonstrations + on-policy RLVR; teacher traces and reward shaping are off-policy | C1/C2 fail (mixed/off-policy guidance) | high |
| **DeepSeek-R1-Distill** | https://arxiv.org/abs/2501.12948 | SFT on teacher reasoning traces; no RL or on-policy component | C1 fails (no student rollouts); C3 fails (SFT) | high |

### 4.3 SFT on teacher reasoning traces

| method | link | why_adjacent | missing_condition | confidence |
|---|---|---|---|---|
| **DASD** | https://arxiv.org/abs/2601.09088 | Teacher-generated completions; student does not receive supervision on current-policy rollouts | C1 fails (teacher-generated data) | medium |
| **SuperCorrect** | https://arxiv.org/abs/2410.09008 | Cross-model DPO on teacher correction traces; static pairs | C1 fails (teacher-generated); C3 fails (static DPO) | high |

---

## 5. Gap Map — Open Questions

### 5.1 Confidence upgrades needed

| Method | Current confidence | What's needed | Priority |
|---|---|---|---|
| **Fast OPD** | medium (wp0_consensus) | Line-level audit: extract exact prefix-selection mechanism and reverse-KL equation; confirm prefix-only teacher scoring doesn't weaken C2 | high |
| **Entropy-Aware OPD** | medium (wp0_consensus) | Line-level audit: extract entropy gating mechanism and FKL/RKL switching rule | medium |
| **Nemotron-Cascade2** | high (wp0_consensus) | Upgrade to source_verified: extract exact MOPD loss equation and truncated importance weighting formula | medium |
| **Qwen3 OPD** | high (wp0_consensus) | Upgrade to source_verified: find exact OPD stage description (currently inferred from "directly distilling output logits" quote) | medium |
| **CaOPD** | medium | Verify calibration correction preserves teacher distributional signal | low |

### 5.2 Structural questions

1. **KDRL annealing endpoint:** When beta→1e-3, the KD term becomes negligible relative to GRPO. Does late-training KDRL degenerate to pure reward RL? If so, should the label be "borderline_strict (early training) → adjacent (late training)"? The current single label is conservative.

2. **SCOPE diversity claim:** The "Pass@k Paradox" (Pass@32 degrades while Pass@1 improves) is a strong diagnostic. Is this specific to OPD or also present in reward-only RL? If OPD-specific, it strengthens the case for SCOPE-style routing as a necessary OPD component.

3. **Lightning OPD teacher consistency requirement:** Theorem 3.7 proves equivalence under sigma_Delta=0 (same teacher for SFT and OPD). What happens empirically when different teachers are used? This determines whether Lightning OPD can be applied in multi-teacher settings (MiMo, Nemotron).

4. **StableOPD golden-data fraction:** How much golden/auxiliary data is needed? If the fraction is high (>50%), the method becomes closer to offline KD with OPD augmentation rather than OPD with stabilization.

### 5.3 Missing methods

5. **DAPO / GSPO standalone papers:** DAPO and GSPO appear as framework algorithm variants in PRISM and InternVL3.5 but no standalone published OPD+KD papers were found. If standalone papers emerge, they should be added to WP4 queue.

6. **Alternating KD/RL schedules beyond KDRL:** Methods that alternate KD epochs with RL epochs (rather than combining in a single loss) may exist but are not yet tracked. MiMo-V2-Flash and Nemotron-Cascade2 use stage-level alternation; a per-epoch alternation pattern could produce different dynamics.

### 5.4 Verification actions

| Action | Priority | Method | What to do |
|---|---|---|---|
| Line audit | high | Fast OPD | Fetch HTML, extract prefix RKL equation, confirm C1/C2/C3 |
| Line audit | medium | Entropy-Aware OPD | Fetch HTML, extract entropy-gating mechanism |
| Quote extraction | medium | Nemotron-Cascade2 | Extract Eq. for MOPD loss and importance weighting |
| Quote extraction | medium | Qwen3 OPD | Find exact OPD stage description in technical report |
| Targeted search | low | DAPO, GSPO | Check arXiv for standalone KD+RL papers using these names |

---

## 6. Cross-validation with existing repo labels

All 19 priority methods confirmed. No label changes proposed.

| Label | Count | Methods |
|---|---|---|
| **strict_opd** | 13 | G-OPD, REOPOLD, Fast OPD, vOPD, TIP, AOPD, Rock Tokens, Entropy-Aware OPD, SOD, VOLD, MiMo-V2-Flash MOPD, Nemotron-Cascade2, Qwen3 OPD |
| **borderline_strict** | 2 | KDRL, SCOPE |
| **partial_opd** | 4 | PACED, StableOPD, HDPO, Lightning OPD |
| **adjacent** | 1 | RLKD |

---

## 7. Verification methodology

1. **Primary-source fetching:** Five agents fetched arXiv HTML for RLKD (primary gap), G-OPD + REOPOLD (RL-boundary details), industrial methods (MiMo, Nemotron, Qwen3), and stability papers (StableOPD, KDRL, SCOPE, Lightning OPD).

2. **KD/RL boundary test:** For each method, the critical test was: does the teacher's probability distribution appear **in the loss function's gradient** per token, or is it compressed to a scalar before entering the loss? Methods where the teacher log-ratio appears directly (G-OPD, REOPOLD, KDRL, SOD, MiMo, Nemotron) are strict or borderline. Methods where the teacher signal becomes a reward score (RLKD, GAD, OVD) are adjacent.

3. **Pattern classification:** Methods were assigned to one of five structural patterns (A-E) based on (a) whether a separate RL reward exists alongside the teacher signal, (b) whether the teacher signal remains dense or is compressed, and (c) whether rollouts are purely on-policy or mixed.

4. **Conservative rules applied:**
   - Dense teacher log-ratio per token in the loss = strict (even if written in RL notation)
   - Dense teacher KL advantage + scalar reward in a single GRPO step = strict (Pattern C)
   - Teacher KL + GRPO with annealing or selective masking = borderline (KDRL, SCOPE)
   - Mixed on/off-policy rollouts with teacher logits = partial (PACED, StableOPD)
   - Teacher structure → scalar reward → GRPO = adjacent (RLKD)
   - Reward-only GRPO/PPO = adjacent (VLM-R1)

---

## Sources

- [G-OPD — Generalized OPD with Reward Extrapolation (arXiv 2602.12125)](https://arxiv.org/abs/2602.12125)
- [REOPOLD — Relaxed On-Policy Distillation (arXiv 2603.11137)](https://arxiv.org/abs/2603.11137)
- [Fast OPD — OPD from Reasoning Prefixes (arXiv 2602.15260)](https://arxiv.org/abs/2602.15260)
- [PACED — Progress-Aware Curriculum for Efficient Distillation (arXiv 2603.11178)](https://arxiv.org/abs/2603.11178)
- [KDRL — Knowledge Distillation Meets RL (arXiv 2506.02208)](https://arxiv.org/abs/2506.02208)
- [RLKD — RL from Knowledge Distillation (arXiv 2505.16142)](https://arxiv.org/abs/2505.16142)
- [SCOPE — Signal-Calibrated On-Policy Distillation Enhancement (arXiv 2604.10688)](https://arxiv.org/abs/2604.10688)
- [Lightning OPD — Offline Approximation to OPD (arXiv 2604.13010)](https://arxiv.org/abs/2604.13010)
- [HDPO — Hybrid Distillation Policy Optimization (arXiv 2603.23871)](https://arxiv.org/abs/2603.23871)
- [SOD — Step-wise On-policy Distillation (arXiv 2605.07725)](https://arxiv.org/abs/2605.07725)
- [MiMo-V2-Flash — Multi-domain OPD (arXiv 2601.02780)](https://arxiv.org/abs/2601.02780)
- [Nemotron-Cascade 2 — Multi-Domain OPD (arXiv 2603.19220)](https://arxiv.org/abs/2603.19220)
- [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388)
- [StableOPD — Length Inflation and Stabilization (arXiv 2604.08527)](https://arxiv.org/abs/2604.08527)
- [vOPD — Variance-Reduced OPD (arXiv 2605.07865)](https://arxiv.org/abs/2605.07865)
- [TIP — Token Importance in OPD (arXiv 2604.14084)](https://arxiv.org/abs/2604.14084)
- [AOPD — Asymmetric OPD (arXiv 2605.06387)](https://arxiv.org/abs/2605.06387)
- [Rock Tokens — Understanding Rock Tokens in OPD (arXiv 2605.09253)](https://arxiv.org/abs/2605.09253)
- [Entropy-Aware OPD (arXiv 2603.07079)](https://arxiv.org/abs/2603.07079)
- [VOLD — Reasoning Transfer via OPD (arXiv 2510.23497)](https://arxiv.org/abs/2510.23497)
- [Gemma 2 Technical Report (arXiv 2408.00118)](https://arxiv.org/abs/2408.00118)
