# Red-Team Verification Audit — On-Policy Distillation Landscape

**Date:** 2026-05-11  
**Auditor:** Automated red-team (5 parallel research agents, primary-source verified)  
**Scope:** 49 candidate methods across 4 audit groups + 4 hallucination checks

---

## 1. Memo of Major Findings

### Overall Statistics
- **49 methods audited** (19 core/frontier + 26 conflict + 4 hallucination checks)
- **28 confirmed strict_opd** (high confidence)
- **1 borderline_strict upgraded to strict_opd** (Uni-OPD — previously borderline, now primary-verified as strict)
- **5 borderline_strict** (require human judgment on edge cases)
- **2 partial_opd** (on-policy generation but scalar supervision signal)
- **5 adjacent / not_opd** (missing C1, C2, or C3)
- **4 hallucination/name-collision flags** confirmed
- **2 dual-identity methods** (SDFT has two distinct papers; Gemma 2 has two phases)

### Key Surprises
1. **All 10 core OPD candidates verified as strict_opd with high confidence.** No downgrades needed in the seed set.
2. **Uni-OPD should be upgraded from borderline_strict to strict_opd** — primary source (arXiv:2605.03677) confirms calibrated token-level reverse KL on student rollouts.
3. **VLA-OPD should be upgraded from borderline_strict to strict_opd** — reverse KL ablation on action tokens is fully documented.
4. **MAD-OPD should be upgraded from borderline_strict to strict_opd** — multi-agent debate produces token-level distributions consumed by JSD/reverse KL.
5. **PRISM remains borderline_strict** — adversarial MoE discriminator is richer than scalar reward but not token-level logit-KL.
6. **SDFT is two distinct papers**: 2024 version (Yang et al.) is not_opd; 2026 version (Shenfeld et al.) is strict_opd.
7. **Gemma 2 has a strict OPD post-training phase** despite the pre-training KD being off-policy.
8. **RLKD is NOT strict OPD** — teacher signal is mediated through a scalar reward model (GSRM), violating C3.
9. **LUFFY is explicitly off-policy guidance** — the paper's own title says "Off-Policy Guidance."
10. **All 4 hallucination candidates confirmed as problematic**: fabricated expansions, name collisions, or misclassified methods.

### Critical Corrections to Repository
- OPSDC/CRISP author attribution: **Sang et al.**, not Zhao et al. (Zhao et al. is OPSD)
- SCOPE expansion is "Signal-Calibrated On-Policy Distillation Enhancement", NOT "Self-play Contrastive On-Policy Evaluation"
- DBKD expansion is "Decision-Based KD", NOT "Distribution-Based KD"
- TED is "Task-aware layEr-wise Distillation" (encoder KD, ICML 2023), NOT OPD

---

## 2. Full Classification Table

### Legend
| Label | Meaning |
|-------|---------|
| `strict_opd` | All C1+C2+C3 verified from primary source |
| `borderline_strict` | OPD principle present but edge case in rollout purity, signal density, or teacher type |
| `partial_opd` | Some but not all conditions met; on-policy generation with scalar supervision |
| `adjacent` | Missing OPD supervision (reward-only, offline KD, curriculum-only) |
| `not_opd` | No student on-policy training states or no teacher |
| `name_collision` | Real paper exists but name/expansion is wrong or method is not OPD |

---

### A. Core Strict OPD Seeds (10 methods)

| # | Method | arXiv | Venue | C1 | C2 | C3 | Teacher Kind | Loss | Verdict | Conf. |
|---|--------|-------|-------|----|----|----|----|------|---------|-------|
| 1 | MiniLLM | [2306.08543](https://arxiv.org/abs/2306.08543) | ICLR 2024 | Student MC sampling | Larger LLM logits on student seqs | Reverse KL via policy gradient | White-box larger LLM | Reverse KL (policy gradient) | **strict_opd** | high |
| 2 | GKD | [2306.13649](https://arxiv.org/abs/2306.13649) | ICLR 2024 | Student self-generated seqs | Teacher token-level logits on student seqs | Configurable divergence (FKL/RKL/JSD) | White-box larger LLM | FKL / RKL / JSD | **strict_opd** | high |
| 3 | DistillSpec | [2310.08461](https://arxiv.org/abs/2310.08461) | ICLR 2024 | Draft model on-policy generation | Target model logits on draft seqs | KL divergence (task-dependent) | White-box target model | KL divergence | **strict_opd** | high |
| 4 | Entropy-Aware OPD | [2603.07079](https://arxiv.org/abs/2603.07079) | Preprint 2026 | Student trajectories | Teacher token-level logits + entropy | Entropy-gated RKL/FKL switching | White-box larger LLM | Entropy-gated hybrid KL | **strict_opd** | high |
| 5 | G-OPD | [2602.12125](https://arxiv.org/abs/2602.12125) | Preprint 2026 | Student rollouts | Teacher logits + flexible reference | Dense KL-constrained RL with reward scaling | White-box LLM + reference | Dense KL-constrained RL | **strict_opd** | high |
| 6 | REOPOLD | [2603.11137](https://arxiv.org/abs/2603.11137) | Preprint 2026 | Student on-policy rollouts | Teacher logits with reward clipping | Relaxed reverse KL + token-level sampling | White-box larger LLM | Relaxed RKL | **strict_opd** | high |
| 7 | Veto | [2601.07155](https://arxiv.org/abs/2601.07155) | Preprint 2026 | Student on-policy rollouts | Teacher logits via geometric PoE bridge | Divergence to intermediate target Q | White-box larger LLM | KL to PoE target | **strict_opd** | high |
| 8 | Fast OPD | [2602.15260](https://arxiv.org/abs/2602.15260) | Preprint 2026 | Student prefix rollouts | Teacher token-level logits on prefixes | Standard OPD divergence on prefix | White-box larger LLM | Prefix-truncated KL | **strict_opd** | high |
| 9 | OPSD | [2601.18734](https://arxiv.org/abs/2601.18734) | Preprint 2026 | Student on-policy rollouts | Privileged self (same model + ground-truth) | Per-token FKL/RKL/JSD | Privileged self | Token-level divergence | **strict_opd** | high |
| 10 | CRISP/OPSDC | [2603.05433](https://arxiv.org/abs/2603.05433) | Preprint 2026 | Student on-policy rollouts | Privileged self (conciseness instruction) | Per-token reverse KL | Privileged self | RKL | **strict_opd** | high |

---

### B. Industrial & Frontier OPD (9 methods)

| # | Method | arXiv | Venue | C1 | C2 | C3 | Teacher Kind | Loss | Verdict | Conf. |
|---|--------|-------|-------|----|----|----|----|------|---------|-------|
| 11 | Qwen3 OPD stage | [2505.09388](https://arxiv.org/abs/2505.09388) | Tech report 2025 | Student on-policy (think/no_think) | Qwen3-32B/235B logits | Token-level KL divergence | White-box larger Qwen3 | KL divergence | **strict_opd** | high |
| 12 | Nemotron-Cascade 2 | [2603.19220](https://arxiv.org/abs/2603.19220) | Preprint 2026 | Student on-policy sampling | Domain-best checkpoint teachers | Token-level log-ratio advantage | Domain-specific checkpoints | MOPD advantage (GRPO) | **strict_opd** | high |
| 13 | X-OPD | [2603.24596](https://arxiv.org/abs/2603.24596) | Preprint 2026 | Speech LLM rollouts | Text backbone LLM (cross-modal) | Policy gradient + token-level RKL | Cross-modal text LLM | PG + RKL | **strict_opd** | high |
| 14 | VOLD | [2510.23497](https://arxiv.org/abs/2510.23497) | ICLR 2026 | Student VLM on-policy | Text-only LLM (Qwen3-8B) | GRPO + masked token-level RKL | Cross-modal text LLM | GRPO + masked RKL | **strict_opd** | high |
| 15 | Video-OPD | [2602.02994](https://arxiv.org/abs/2602.02994) | Preprint 2026 | Student MLLM on-policy | Frozen frontier multimodal teacher | Token-level reverse KL | Frozen frontier MLLM | RKL | **strict_opd** | high |
| 16 | Uni-OPD | [2605.03677](https://arxiv.org/abs/2605.03677) | Preprint 2026 | Student rollouts | Calibrated teacher(s) logits | Calibrated token-level RKL | Single/multi expert teachers | Calibrated RKL | **strict_opd** | high |
| 17 | PRISM | [2604.28123](https://arxiv.org/abs/2604.28123) | Preprint 2026 | Student on-policy VLM rollouts | MoE discriminator (perception + reasoning) | Adversarial minimax | Black-box MoE discriminator | GAN-style adversarial | **borderline_strict** | medium |
| 18 | VLA-OPD | [2603.26666](https://arxiv.org/abs/2603.26666) | Preprint 2026 | Student VLA action trajectories | Frozen expert VLA teacher logits | Token-level reverse KL on actions | Frozen expert VLA | RKL | **strict_opd** | high |
| 19 | MAD-OPD | [2605.01347](https://arxiv.org/abs/2605.01347) | Preprint 2026 | Student on-policy rollouts | Multi-teacher debate (weighted token distributions) | Task-adaptive JSD/RKL | Multi-agent debate collective | JSD or RKL | **strict_opd** | high |

---

### C. Conflict Candidates — Batch 1 (10 methods)

| # | Method | arXiv | Venue | C1 | C2 | C3 | Teacher Kind | Loss | Verdict | Conf. |
|---|--------|-------|-------|----|----|----|----|------|---------|-------|
| 20 | DistiLLM | [2402.03898](https://arxiv.org/abs/2402.03898) | ICML 2024 | Partial: adaptive off-policy replay buffer | Yes: white-box logits on SGOs | Yes: Skew KL | White-box LLM | Skew FKL + Skew RKL | **borderline_strict** | high |
| 21 | DistiLLM-2 | [2503.07067](https://arxiv.org/abs/2503.07067) | ICML 2025 | Yes: SGOs are first-class | Yes: white-box logits on SGOs | Yes: contrastive SKL/SRKL | White-box LLM | CALD (contrastive) | **strict_opd** | high |
| 22 | GAD | [2511.10643](https://arxiv.org/abs/2511.10643) | Preprint 2025 | Yes: student as generator | Yes: co-trained discriminator | Partial: scalar reward via GRPO | Black-box + discriminator | Adversarial + GRPO | **partial_opd** | medium |
| 23 | Lion | [2305.12870](https://arxiv.org/abs/2305.12870) | Preprint 2023 | Partial: for curriculum only | Partial: instruction-level referee | No: SFT on teacher outputs | Black-box API teacher | SFT cross-entropy | **adjacent** | high |
| 24 | OVD | [2601.21968](https://arxiv.org/abs/2601.21968) | Preprint 2026 | Yes: student rollouts | Yes: verbal scores 0-9 | Partial: scalar scores via GRPO | Black-box (verbal output only) | GRPO + verbal rejection | **partial_opd** | high |
| 25 | Speculative KD | [2410.11325](https://arxiv.org/abs/2410.11325) | Preprint 2024 | Partial: interleaved student-teacher | Yes: teacher corrects tokens | Yes: token-level KL | White-box LLM | Token-level KL | **borderline_strict** | medium |
| 26 | PACED | [2603.11178](https://arxiv.org/abs/2603.11178) | Preprint 2026 | Yes: K=8 rollouts for pass-rate | Yes: token-level logits | Yes: Beta-weighted KL | White-box LLM | Weighted FKL/RKL | **strict_opd** | high |
| 27 | AdaSwitch | [2510.07842](https://arxiv.org/abs/2510.07842) | Preprint 2025 | Partial: student starts, teacher finishes | Yes: logit-level divergence | Yes: token-level KL | White-box LLM | Token-level KL | **borderline_strict** | high |
| 28 | BOND | [2407.14622](https://arxiv.org/abs/2407.14622) | ICLR 2025 | Yes: policy generates N samples | Partial: reward model, not teacher LLM | Partial: scalar reward-based | Reward model (not teacher LLM) | Jeffreys divergence | **adjacent** | high |
| 29 | Lightning OPD | [2604.13010](https://arxiv.org/abs/2604.13010) | Preprint 2026 | Yes: on-policy rollouts | Partial: teacher logits precomputed on SFT dist | Yes: token-level KL | White-box LLM (precomputed) | Standard OPD KL | **borderline_strict** | medium |

---

### D. Conflict Candidates — Batch 2 (16 methods)

| # | Method | arXiv | Venue | C1 | C2 | C3 | Teacher Kind | Loss | Verdict | Conf. |
|---|--------|-------|-------|----|----|----|----|------|---------|-------|
| 30 | DASD | [2601.09088](https://arxiv.org/abs/2601.09088) | Preprint 2026 | Partial: student prefixes only | Partial: teacher completes, not scores | No: SFT loss | External LLM | SFT cross-entropy | **adjacent** | high |
| 31 | DDT | [2602.12222](https://arxiv.org/abs/2602.12222) | Preprint 2026 | Partial: distribution analysis | No: no teacher model | No: re-weighted SFT | None (theory framework) | Re-weighted SFT | **adjacent** | high |
| 32 | GATES | [2602.20574](https://arxiv.org/abs/2602.20574) | Preprint 2026 | No: tutor generates traces | Yes: self-teacher (privileged) | Yes: trajectory-level distillation | Privileged self-teacher | Consensus-gated distill. | **partial_opd** | medium |
| 33 | SDPO | [2601.20802](https://arxiv.org/abs/2601.20802) | Preprint 2026 | Yes: on-policy rollouts | Yes: self-teacher (feedback-conditioned) | Yes: token-level KL/JSD | Self-teacher + feedback | Token-level KL/JSD | **strict_opd** | high |
| 34 | OPCD | [2602.12275](https://arxiv.org/abs/2602.12275) | Preprint 2026 | Yes: student samples responses | Yes: context-conditioned teacher | Yes: token-level reverse KL | Context-conditioned teacher | RKL | **strict_opd** | high |
| 35 | OEL | [2603.16856](https://arxiv.org/abs/2603.16856) | Preprint 2026 | Yes: on-policy rollouts | Yes: self-teacher (experiential context) | Yes: token-level RKL, top-k | Self-teacher + experiential knowledge | RKL (top-k) | **strict_opd** | high |
| 36 | HDPO | [2603.23871](https://arxiv.org/abs/2603.23871) | Preprint 2026 | Yes (GRPO part) | Yes: self-teacher (privileged) | Yes (JSD) — but on teacher rollouts | Self-teacher + ground-truth | GRPO + JSD (cliff prompts) | **partial_opd** | medium |
| 37 | SDFT (2024) | [2402.13669](https://arxiv.org/abs/2402.13669) | ACL 2024 | No: pre-generated rewrite data | No: no teacher on rollouts | No: standard SFT | None (data preprocessing) | SFT | **not_opd** | high |
| 38 | SDFT (2026) | [2601.19897](https://arxiv.org/abs/2601.19897) | Preprint 2026 | Yes: on-policy trajectories | Yes: self-teacher (demo-conditioned) | Yes: token-level reverse KL | Self-teacher + demonstration | RKL | **strict_opd** | high |
| 39 | Priv. Info. Distill. | [2602.04942](https://arxiv.org/abs/2602.04942) | Preprint 2026 | Yes (OPSD variant) | Yes: PI-conditioned teacher | Yes: RL + reverse KL penalty | Self-teacher + privileged info | RL + RKL penalty | **strict_opd** | high |
| 40 | SCOPE | [2604.10688](https://arxiv.org/abs/2604.10688) | Preprint 2026 | Yes: on-policy rollouts | Yes: teacher KL on incorrect trajectories | Yes: dual-path KL + MLE | External teacher | Selective KL + weighted MLE | **strict_opd** | high |
| 41 | RLKD | [2505.16142](https://arxiv.org/abs/2505.16142) | AAAI 2026 | Yes: on-policy via GRPO | Indirect: GSRM trained on teacher structure | No: scalar reward via GSRM | Teacher-informed reward model | GRPO + GSRM reward | **adjacent** | high |
| 42 | KDRL | [2506.02208](https://arxiv.org/abs/2506.02208) | Preprint 2025 | Yes: on-policy rollouts | Yes: teacher token-level log-probs | Yes: unified RKL + GRPO | External teacher | RKL + GRPO (annealing) | **strict_opd** | high |
| 43 | LUFFY | [2504.14945](https://arxiv.org/abs/2504.14945) | Preprint 2025 | Partial: mixed on/off-policy | No: teacher provides off-policy traces | No: mixed-policy GRPO | Off-policy teacher traces | Mixed-policy GRPO | **adjacent** | high |
| 44 | Gemma 2 KD | [2408.00118](https://arxiv.org/abs/2408.00118) | Tech report 2024 | Yes (post-training phase) | Yes: teacher logits on student completions | Yes: cross-entropy KD | White-box larger model | Cross-entropy KD | **strict_opd** (post-train only) | medium |
| 45 | MiMo-V2-Flash | [2601.02780](https://arxiv.org/abs/2601.02780) | Preprint 2026 | Yes: student on-policy | Yes: multi-domain teacher logits | Yes: token-level RKL + outcome reward | Multiple domain-specialized teachers | MOPD (RKL + reward) | **strict_opd** | high |
| 46 | TML Blog | [blog post](https://thinkingmachines.ai/blog/on-policy-distillation/) | Blog 2025 | Yes: student sampling | Yes: teacher token-level grading | Yes: reverse KL | White-box larger model | RKL | **strict_opd** | medium |

---

### E. Hallucination / Name-Collision Check (4 items)

| # | Name Checked | Real Paper? | Actual Identity | OPD? | Verdict |
|---|-------------|-------------|-----------------|------|---------|
| 47 | DistillDirect | Yes (minor, arXiv:2405.00715) | Sub-method in clinical NLP paper; on-policy DPO variant | Technically on-policy but not a recognized OPD method | **name_collision** — not a canonical OPD method |
| 48 | DBKD | Yes (arXiv:2306.08909, ACL 2023) | "Decision-Based KD" (NOT "Distribution-Based KD") | No — offline encoder KD on NLU tasks | **name_collision** — wrong expansion, not OPD |
| 49 | TED as OPD | Yes (arXiv:2210.01351, ICML 2023) | "Task-aware layEr-wise Distillation" (NOT "Task-Embedded") | No — layer-wise encoder model KD | **not_opd** — misclassified |
| 50 | SCOPE as "Self-play Contrastive On-Policy Evaluation" | Yes (arXiv:2604.10688) | Real name: "Signal-Calibrated On-Policy Distillation Enhancement" | Yes — the real SCOPE IS strict OPD | **name_collision** — expansion completely fabricated |

---

## 3. Downgrade List

These methods should be **downgraded from strict_opd** if currently classified as such:

| Method | Current | Should Be | Reason |
|--------|---------|-----------|--------|
| DistiLLM | — | borderline_strict | Self-describes as "adaptive off-policy"; replay buffer means stale student rollouts (C1 weakened) |
| GAD | — | partial_opd | Discriminator provides scalar reward via GRPO, not dense token-level distribution (C3 partial) |
| Lion | — | adjacent | Student outputs used only for curriculum; training is SFT on teacher outputs (C1/C3 fail) |
| OVD | — | partial_opd | Verbal scores are scalar (0-9) consumed by GRPO, not distributional supervision (C3 partial) |
| Speculative KD | — | borderline_strict | Interleaved sampling means training data is student-teacher mixture, not purely student (C1 partial) |
| AdaSwitch | — | borderline_strict | Student starts, teacher finishes each sequence; hybrid on/off-policy (C1 partial) |
| BOND | — | adjacent | Reward model (not teacher LLM) provides scalar scores; this is RLHF, not KD (C2/C3 fail) |
| Lightning OPD | — | borderline_strict | Teacher logits precomputed on SFT distribution, not student's current distribution (C2 approximate) |
| DASD | — | adjacent | Student provides prefixes only; teacher completes; loss is SFT (all three conditions partial/fail) |
| DDT | — | adjacent | No teacher model; theoretical framework for improving SFT (C2 fails) |
| GATES | — | partial_opd | Tutor generates traces, student learns from curated tutor outputs (C1 fails — off-policy for student) |
| HDPO | — | partial_opd | JSD distillation occurs on teacher-generated privileged rollouts, not student rollouts (C1 fails for distill component) |
| SDFT (2024) | — | not_opd | Data preprocessing, not on-policy training; no teacher on rollouts |
| RLKD | — | adjacent | Teacher structure -> GSRM -> scalar reward -> GRPO; supervision is reward-mediated (C3 fails) |
| LUFFY | — | adjacent | Explicitly "Off-Policy Guidance"; teacher provides pre-generated traces, not rollout supervision |
| PRISM | — | borderline_strict | MoE discriminator is richer than scalar reward but not token-level logit-KL |

---

## 4. Safe to Keep as strict_opd (28 methods)

These methods are **confirmed strict_opd with high confidence** from primary sources:

| # | Method | arXiv | Key Evidence |
|---|--------|-------|--------------|
| 1 | MiniLLM | 2306.08543 | Student MC sampling + teacher logits + reverse KL policy gradient |
| 2 | GKD | 2306.13649 | Student self-generated seqs + teacher token logits + configurable divergence |
| 3 | DistillSpec | 2310.08461 | Draft on-policy + target logits + KL divergence |
| 4 | Entropy-Aware OPD | 2603.07079 | Student rollouts + entropy-gated teacher logits + hybrid KL |
| 5 | G-OPD | 2602.12125 | Student rollouts + teacher + reference logits + dense KL-constrained RL |
| 6 | REOPOLD | 2603.11137 | Student rollouts + clipped teacher logits + relaxed RKL |
| 7 | Veto | 2601.07155 | Student rollouts + PoE bridge target + KL to intermediate |
| 8 | Fast OPD | 2602.15260 | Student prefix rollouts + teacher logits + prefix-truncated KL |
| 9 | OPSD | 2601.18734 | Student rollouts + privileged self + per-token divergence |
| 10 | CRISP/OPSDC | 2603.05433 | Student rollouts + conciseness-conditioned self + RKL |
| 11 | Qwen3 OPD stage | 2505.09388 | Student on-policy + Qwen3-32B/235B logits + token-level KL |
| 12 | Nemotron-Cascade 2 | 2603.19220 | Student on-policy + domain-best checkpoints + MOPD advantage |
| 13 | X-OPD | 2603.24596 | Speech LLM rollouts + text teacher + PG + token-level RKL |
| 14 | VOLD | 2510.23497 | VLM on-policy + text LLM teacher + GRPO + masked RKL |
| 15 | Video-OPD | 2602.02994 | MLLM on-policy + frontier teacher + token-level RKL |
| 16 | Uni-OPD | 2605.03677 | Student rollouts + calibrated teacher(s) + calibrated RKL |
| 17 | VLA-OPD | 2603.26666 | VLA action trajectories + frozen expert + RKL on actions |
| 18 | MAD-OPD | 2605.01347 | Student rollouts + multi-teacher debate + JSD/RKL |
| 19 | DistiLLM-2 | 2503.07067 | SGOs first-class + teacher logits + contrastive SKL/SRKL |
| 20 | PACED | 2603.11178 | K=8 rollouts + teacher logits + Beta-weighted KL |
| 21 | SDPO | 2601.20802 | On-policy rollouts + feedback-conditioned self-teacher + token KL |
| 22 | OPCD | 2602.12275 | Student responses + context-conditioned teacher + RKL |
| 23 | OEL | 2603.16856 | On-policy rollouts + experiential self-teacher + RKL top-k |
| 24 | SDFT (2026) | 2601.19897 | On-policy trajectories + demo-conditioned self-teacher + RKL |
| 25 | Priv. Info. Distill. | 2602.04942 | On-policy rollouts + PI-conditioned teacher + RL + RKL |
| 26 | SCOPE | 2604.10688 | On-policy rollouts + teacher KL on incorrect + dual-path loss |
| 27 | KDRL | 2506.02208 | On-policy rollouts + teacher token log-probs + unified RKL + GRPO |
| 28 | MiMo-V2-Flash | 2601.02780 | On-policy + multi-domain teachers + MOPD (RKL + reward) |

**Medium-confidence strict (require minor caveats):**

| # | Method | arXiv | Caveat |
|---|--------|-------|--------|
| 29 | Gemma 2 KD (post-train) | 2408.00118 | Only post-training phase qualifies; pre-training KD is off-policy |
| 30 | TML Blog | blog post | Secondary source only; no peer-reviewed paper. Implementation available on GitHub. |

---

## 5. Unresolved Items Requiring Human Audit

| Method | Issue | Recommended Action |
|--------|-------|--------------------|
| **PRISM** | Adversarial MoE discriminator: richer than scalar reward but not token-level logit-KL. Is adversarial distribution matching sufficient for C3? | Human judgment needed on whether discriminator-based adversarial signal counts as "directly consuming supervision" |
| **DistiLLM** | Self-describes as "adaptive off-policy" but does use student outputs with teacher logits. The replay buffer makes rollouts stale. | Human judgment on whether cached student rollouts (not current-policy) satisfy C1 |
| **Speculative KD** | Interleaved student-teacher token generation. Training data is a mixture, not purely student-generated. | Human judgment on C1: are interleaved sequences "student rollouts"? |
| **AdaSwitch** | Student generates prefix, teacher completes. Token-level hybrid. | Same C1 question as Speculative KD |
| **Lightning OPD** | Teacher logits precomputed on SFT distribution with theoretical "teacher consistency" guarantee. Approximation is bounded. | Human judgment on whether approximate C2 (bounded gradient discrepancy) is sufficient |
| **GATES** | Tutor (privileged) generates traces; student learns from consensus-gated tutor outputs. Is the student or tutor generating the training data? | Human judgment on C1: who generates the rollouts? |
| **HDPO** | JSD distillation on teacher-generated privileged rollouts for cliff prompts. On-policy GRPO for non-cliff prompts. | Is the distillation component on student or teacher rollouts? |
| **GAD** | Discriminator co-evolves with student and provides scalar GRPO reward. Richer than static RM but still scalar. | Same question as PRISM but with even weaker signal (pure scalar vs. MoE-disentangled) |
| **OVD** | Verbal scores 0-9 are richer than binary but still discrete scalars per trajectory. Paper explicitly avoids dense supervision by design. | Is structured verbal feedback sufficient for C3? |
| **Gemma 2 KD** | Post-training phase is strict OPD, but pre-training is the main contribution. Should this be listed as strict_opd with a phase caveat? | Classification granularity: phase-level vs. paper-level? |
| **SDFT name collision** | Two completely different papers use this acronym (2024 vs. 2026). Which should the repo track? | Both, with clear disambiguation. Only 2026 version is OPD. |
| **TML Blog** | No peer-reviewed paper. Implementation exists. | Should secondary-source-only items be tracked as strict_opd? |
| **DistillDirect** | Real paper (arXiv:2405.00715) but minor clinical NLP contribution, not a recognized OPD method. | Exclude from OPD landscape or track as adjacent curiosity |

---

## Appendix: Author Attribution Corrections

| Method | Currently Attributed To | Correct Authors |
|--------|------------------------|-----------------|
| CRISP/OPSDC | "Zhao et al." | **Sang et al.** (Hejian Sang, Yuanda Xu, et al.) |

*Note: Zhao et al. (arXiv:2601.18734) is OPSD, a related but distinct paper.*

## Appendix: Name/Expansion Corrections

| Name | Claimed Expansion | Correct Expansion | Source |
|------|-------------------|-------------------|--------|
| SCOPE | "Self-play Contrastive On-Policy Evaluation" | "Signal-Calibrated On-Policy Distillation Enhancement" | arXiv:2604.10688 |
| DBKD | "Distribution-Based Knowledge Distillation" | "Decision-Based Knowledge Distillation" | arXiv:2306.08909 |
| TED | "Task-Embedded Distillation" | "Task-aware layEr-wise Distillation" | arXiv:2210.01351 |
