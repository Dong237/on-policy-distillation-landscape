# On Policy Distillation Landscape for VLMs & MLLMs

> A taxonomy-first, implementation-aware research map of On-Policy Distillation for Vision-Language Models and Multimodal Large Language Models.
>
> Last updated: 2026-05-13

---

## Table of Contents

- [Definition](#definition)
- [Taxonomy of VLM/MLLM OPD Methods](#taxonomy-of-vlmmllm-opd-methods)
- [Historical Timeline](#historical-timeline)
- [Must-Read First 10 Papers](#must-read-first-10-papers-for-vlm-opd)
- [Detailed Paper Catalog](#detailed-paper-catalog)
  - [Core VLM/MLLM OPD Methods](#1-core-vlmmllm-opd-methods)
  - [Text-Only LLM Teacher → VLM Student (Cross-Modal OPD)](#2-text-only-llm-teacher--vlm-student-cross-modal-opd)
  - [VLM RL / RLVR (OPD-Adjacent)](#3-vlm-rl--rlvr-opd-adjacent)
  - [Multimodal Self-Distillation & Self-Improvement](#4-multimodal-self-distillation--self-improvement)
  - [Black-Box Multimodal OPD](#5-black-box-multimodal-opd)
  - [VLM Compression via KD (Offline Baselines)](#6-vlm-compression-via-kd-offline-baselines)
  - [Multimodal Agent / Tool-Use Distillation](#7-multimodal-agent--tool-use-distillation)
  - [Multi-Teacher & Debate-Based OPD](#8-multi-teacher--debate-based-opd)
- [Video-Language OPD](#video-language-opd)
- [Industrial VLM Systems & Technical Reports](#industrial-vlm-systems--technical-reports)
- [Frameworks & Tools Supporting VLM OPD](#frameworks--tools-supporting-vlm-opd)
- [Summary Table (GitHub README)](#summary-table)
- [Open Problems in VLM OPD](#open-problems-in-vlm-opd)

---

## Definition

### Strict On-Policy Distillation (OPD) for VLMs

A method qualifies as **Strict OPD** if and only if:

1. **The VLM/MLLM student generates its own multimodal trajectories**, completions, rationales, or intermediate states during training (on-policy rollouts with visual input).
2. **A teacher, verifier, discriminator, privileged-context model, reward model, or reference model provides supervision** on those student-generated trajectories.
3. **The training objective consumes that supervision as distillation-style signal**, not merely scalar reward, static preference, or SFT on teacher traces.

### Classification Criteria

| Category | Rollout Source | Supervision Source | Example |
|---|---|---|---|
| **Strict OPD** | VLM student generates | Teacher/verifier scores student outputs | VOLD, Video-OPD, Uni-OPD, VLA-OPD, GUI-SD, LiteGUI guided stage |
| **Partial OPD** | Mixed student+teacher or student generates but supervision is sparse | Teacher provides partial signal | LLaVA-MoD (preference stage), VL-Rethinker |
| **Adjacent** | Student generates for RL | Verifier/reward only (no teacher distribution matching) | VLM-R1, Visionary-R1, Perception-R1, GRPO-based VLM RL |
| **Not OPD** | Teacher generates all data | Student trains on teacher data only | LLaVA (GPT-4 data), LLaVA-KD (offline), standard visual instruction tuning |

### Distinguishing VLM OPD from Related Paradigms

| Paradigm | Student Rollouts? | Teacher Signal? | Difference from VLM OPD |
|---|---|---|---|
| **Visual Instruction Tuning (SFT)** | No — trains on fixed dataset | No teacher logits | Standard multimodal SFT on curated data |
| **Offline Multimodal KD** | No — trains on teacher outputs | Yes (logits/text) | No student rollouts; train-test mismatch |
| **Multimodal DPO/Preference** | Pre-collected pairs | Implicit preference | Offline preference optimization |
| **VLM RLVR (GRPO)** | Yes — VLM generates | Verifiable reward (correctness) | No teacher distribution; reward from answer correctness |
| **VLM OPD** | Yes — VLM generates | Teacher provides logits/scores on student rollouts | Teacher feedback on student's multimodal outputs |

---

## Taxonomy of VLM/MLLM OPD Methods

```
On-Policy Distillation for VLMs/MLLMs
├── By Teacher Type
│   ├── Text-Only LLM Teacher → VLM Student (Cross-Modal OPD)
│   │   ├── On-policy KL + GRPO (VOLD)
│   │   ├── Text-only SFT+RL transfer (X-Reasoner)
│   │   └── Cross-modal formalization (R1-Onevision)
│   ├── Stronger VLM Teacher → Smaller VLM Student
│   │   ├── White-box logit distillation (LLaVA-KD, EM-KD, LLAVADI)
│   │   ├── MoE + progressive KD (LLaVA-MoD)
│   │   └── Unified OPD framework (Uni-OPD for MLLMs)
│   ├── Multimodal Verifier / Reward Model
│   │   ├── Visual perception reward (Perception-R1)
│   │   ├── Caption + answer reward (Visionary-R1, Vision-SR1)
│   │   └── Perception-centric PRM (Perceval)
│   ├── Discriminator-Based
│   │   └── GAN-style black-box OPD (GAD extended to multimodal)
│   └── Self-Distillation (Teacher-Free)
│       ├── Privileged visual context (self as teacher)
│       ├── Self-critique + revision (SelfReVision)
│       └── Self-rewarding VLM (Vision-SR1)
│
├── By Modality
│   ├── Image-Text (majority of current work)
│   ├── Video-Text (emerging, temporal reasoning)
│   ├── Document/Chart/Table (OCR-heavy, DocVQA, ChartQA)
│   ├── GUI / Web (agent-level OPD: UITRON, T3-Agent)
│   └── Embodied / Agentic (MAD-OPD agentic tasks)
│
├── By Training Stage
│   ├── Visual alignment / pre-training KD (Gemma 2/3 vision, Mini-InternVL)
│   ├── Post-SFT distillation (LLaVA-KD, LLAVADI, EM-KD)
│   ├── Inside RL/RLVR (VOLD, VLM-R1, Perception-R1)
│   ├── Preference alignment (LLaVA-MoD DPO stage, InternVL3 MPO)
│   └── Compression (Mini-InternVL, NanoVLMs)
│
├── By Method Family
│   ├── White-Box OPD (teacher logits accessible)
│   │   ├── Token-level KL on student rollouts (VOLD, Uni-OPD)
│   │   ├── Vision-language affinity KD (EM-KD)
│   │   └── Multimodal logit + relation KD (LLaVA-KD)
│   ├── Black-Box OPD (text-only teacher access)
│   │   ├── Teacher as data generator (LLaVA paradigm)
│   │   ├── Cross-modal black-box KD (ARMADA)
│   │   └── Proxy-based approaches
│   ├── OPD-RL Hybrid
│   │   ├── Unified GRPO + on-policy KL (VOLD)
│   │   ├── SFT → GRPO for VLMs (X-Reasoner, VLAA-Thinker)
│   │   └── Mixed perception + cognition reward (VLAA-Thinking)
│   └── Multi-Teacher OPD
│       ├── Debate-driven collective supervision (MAD-OPD)
│       └── Mixture of visual encoders (MoVE-KD)
│
└── By Application Domain
    ├── Math / STEM visual reasoning (VOLD, R1-Onevision, ThinkLite-VL)
    ├── General VQA / instruction following (LLaVA-KD, Mini-InternVL)
    ├── Document / chart understanding (EM-KD, Qwen2.5-VL)
    ├── GUI agent navigation (UITRON, T3-Agent, GUI-R1)
    ├── Tool-use reasoning (VTool-R1, VPD)
    └── Hallucination mitigation (LLaVA-MoD, Perception-R1)
```

---

## Historical Timeline

| Year | Milestone | Paper/System |
|---|---|---|
| **2023 Apr** | LLaVA: visual instruction tuning from GPT-4 data | Liu et al. — LLaVA |
| **2023 Jun** | GKD: foundational OPD paper (text LLMs) | Agarwal et al. — GKD (notes future multimodal extension) |
| **2023 Oct** | LLaVA-1.5: improved visual instruction baselines | Liu et al. — LLaVA-1.5 |
| **2023 Dec** | Visual Program Distillation (VPD) | Hu et al. — VPD (CVPR 2024 Oral) |
| **2024 Jul** | LLAVADI: what matters for MLLM distillation | LLAVADI study |
| **2024 Aug** | LLaVA-MoD: MoE + progressive KD + DPO | Shu et al. — LLaVA-MoD (ICLR 2025) |
| **2024 Aug** | VLM-KD: distilling from VLM for long-tail recognition | VLM-KD |
| **2024 Oct** | Mini-InternVL 2.0: 5% params, 90% performance | Mini-InternVL 2.0 |
| **2024 Oct** | LLaVA-KD: multi-stage MLLM distillation | Cai et al. — LLaVA-KD (ICCV 2025) |
| **2024 Nov** | EM-KD: distillation with unbalanced vision tokens | Feng et al. — EM-KD |
| **2025 Feb** | Qwen2.5-VL: dynamic resolution VLM | Qwen Team — Qwen2.5-VL |
| **2025 Mar** | R1-Onevision: cross-modal formalization for reasoning | Yang et al. — R1-Onevision (ICCV 2025) |
| **2025 Apr** | InternVL3: native multimodal pre-training + MPO | OpenGVLab — InternVL3 |
| **2025 Apr** | SFT or RL? Early investigation for reasoning VLMs | Chen et al. — VLAA-Thinking (TMLR) |
| **2025 May** | X-Reasoner: text-only SFT+RL generalizes to vision | Liu et al. — X-Reasoner (Microsoft) |
| **2025 May** | MASSV: self-data distillation for VLM speculative decoding | MASSV |
| **2025 Jun** | Perception-R1: visual perception reward for GRPO | Xiao et al. — Perception-R1 |
| **2025 Jul** | SelfReVision: self-critical distillation for VLM planning | SelfReVision |
| **2025 Oct** | **VOLD: first text→VLM on-policy distillation + GRPO** | **Bousselham et al. — VOLD** |
| **2025 Nov** | GAD: black-box on-policy distillation (text LLMs) | Ye et al. — GAD (Microsoft) |
| **2026 Feb** | ARMADA: cross-modal black-box KD (VLM→LLM) | ARMADA |
| **2026 May** | **Uni-OPD: unified OPD for LLMs + MLLMs** | **Hou et al. — Uni-OPD** |
| **2026 May** | MAD-OPD: multi-agent debate OPD (incl. agentic) | Wang et al. — MAD-OPD |

---

## Must-Read First 10 Papers for VLM OPD

| # | Paper | Year | Why Read First |
|---|---|---|---|
| 1 | [VOLD: Reasoning Transfer from LLMs to VLMs via On-Policy Distillation](https://arxiv.org/abs/2510.23497) | 2025 | **First explicit text→VLM on-policy distillation.** Combines GRPO with on-policy KL from text-only teacher. |
| 2 | [Uni-OPD: Unifying On-Policy Distillation](https://arxiv.org/abs/2605.03677) | 2026 | **Unified OPD for LLMs + MLLMs.** Dual-perspective recipe, cross-modal distillation experiments. |
| 3 | [X-Reasoner: Generalizable Reasoning Across Modalities](https://arxiv.org/abs/2505.03981) | 2025 | Proves text-only post-training transfers reasoning to vision. SFT+RL with text generalizes cross-modally. |
| 4 | [SFT or RL? Training R1-Like Reasoning VLMs](https://arxiv.org/abs/2504.11468) | 2025 | Critical analysis: SFT induces pseudo-reasoning that harms RL. RL is more effective for VLM reasoning. |
| 5 | [LLaVA-KD: Distilling Multimodal LLMs](https://arxiv.org/abs/2410.16236) | 2024 | Key MLLM KD framework: multimodal + relation distillation. Three-stage training scheme. (ICCV 2025) |
| 6 | [LLaVA-MoD: MoE Knowledge Distillation](https://arxiv.org/abs/2408.15881) | 2024 | Progressive KD: mimic → preference (DPO). MoE for efficiency. (ICLR 2025) |
| 7 | [MAD-OPD: Multi-Agent Debate On-Policy Distillation](https://arxiv.org/abs/2605.01347) | 2026 | Breaks single-teacher ceiling via multi-teacher debate. First OPD for agentic tasks. |
| 8 | [Perception-R1: Visual Perception Reward](https://arxiv.org/abs/2506.07218) | 2025 | Addresses visual perception deficit in VLM RL. Novel perception reward for GRPO. |
| 9 | [R1-Onevision: Cross-Modal Formalization](https://arxiv.org/abs/2503.10615) | 2025 | Images → formal text → language reasoning. SFT+RL on cross-modal pipeline. (ICCV 2025) |
| 10 | [Visual Program Distillation (VPD)](https://arxiv.org/abs/2312.03052) | 2024 | Distills programmatic tool-use reasoning into VLMs. (CVPR 2024 Oral) |

---

## Detailed Paper Catalog

### 1. Core VLM/MLLM OPD Methods

---

#### VOLD: Reasoning Transfer from LLMs to VLMs via On-Policy Distillation
- **Year:** 2025 (arXiv Oct 2025)
- **Authors:** Walid Bousselham, Hilde Kuehne, Cordelia Schmid
- **Link:** [arXiv:2510.23497](https://arxiv.org/abs/2510.23497) | [OpenReview](https://openreview.net/forum?id=lkv7sOGtfk)
- **Type:** Paper
- **Model Type:** VLM (Qwen2.5-VL-3B student)
- **Modality:** Image-text (math/visual reasoning)
- **Method Family:** Cross-modal OPD-RL hybrid (text LLM teacher → VLM student)
- **Teacher Type:** Text-only LLM (Qwen3-8B)
- **Teacher Access:** White-box (logits, shared tokenizer for KL computation)
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Token-level KL divergence + GRPO outcome reward (unified objective)
- **Training Stage:** Post-SFT, inside RL
- **Benchmarks:** MMMU-Pro, MathVision, MathVista, LogicVista
- **Code:** Not released
- **Strictness:** **Strict OPD** — VLM student generates on-policy; text-only teacher provides token-level KL supervision on student rollouts
- **Summary:** The first paper to explicitly apply on-policy distillation for cross-modal reasoning transfer from a text-only LLM teacher to a VLM student. Uses a two-stage pipeline: (1) cold-start SFT alignment with teacher-generated reasoning traces, (2) unified GRPO + on-policy KL objective on text-only reasoning data. The text-only reasoning skills transfer to visual tasks at inference time.
- **Why It Matters:** Demonstrates that OPD can bridge the modality gap — text-only reasoning transfers to vision without vision-specific reasoning data during OPD training. First explicit cross-modal OPD for VLMs.
- **Limitations:** Withdrawn from ICLR 2026; cold-start alignment is critical and finicky; text-only training may not capture all visual reasoning patterns.

---

#### Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe
- **Year:** 2026 (arXiv May 2026)
- **Authors:** Wenjin Hou et al. (15 authors)
- **Link:** [arXiv:2605.03677](https://arxiv.org/abs/2605.03677)
- **Type:** Paper
- **Model Type:** LLM + MLLM (unified framework)
- **Modality:** Text + image-text (multi-domain)
- **Method Family:** Unified OPD framework
- **Teacher Type:** Stronger LLM/MLLM (single or multi-teacher)
- **Teacher Access:** White-box (logits)
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Token-level KL with outcome-guided margin calibration
- **Training Stage:** Post-training
- **Benchmarks:** 5 domains, 16 benchmarks (LLM + MLLM)
- **Code:** Not specified
- **Strictness:** **Strict OPD**
- **Summary:** Identifies two core OPD bottlenecks: insufficient exploration of informative states and unreliable teacher supervision on student rollouts. Proposes dual-perspective optimization: (1) student-side data balancing for better exploration, (2) teacher-side outcome-guided margin calibration to restore order-consistency between correct/incorrect trajectories. Covers both LLM and MLLM settings, including cross-modal distillation.
- **Why It Matters:** First unified OPD framework explicitly validated on both LLMs and MLLMs. Establishes that OPD for MLLMs is "largely underexplored" and provides the first systematic treatment.
- **Limitations:** Early-stage for MLLM experiments; vanilla OPD used as baseline since no prior MLLM OPD methods exist.

---

#### EM-KD: Distilling Efficient MLLMs with Unbalanced Vision Tokens
- **Year:** 2025 (arXiv Nov 2025)
- **Authors:** Ze Feng, Sen Yang, Boqiang Duan, Wankou Yang, Jingdong Wang
- **Link:** [arXiv:2511.21106](https://arxiv.org/abs/2511.21106)
- **Type:** Paper
- **Model Type:** MLLM (efficient variants with compressed vision tokens)
- **Modality:** Image-text
- **Method Family:** White-box MLLM KD with spatial alignment
- **Teacher Type:** Vanilla MLLM (full vision tokens)
- **Teacher Access:** White-box (logits + vision token alignment)
- **Rollout Source:** Off-policy (dataset-based)
- **Supervision Signal:** Vision-Language Affinity Distillation (VLAD) + Vision Semantic Distillation (VSD, reverse KL)
- **Training Stage:** Post-SFT distillation
- **Benchmarks:** Diverse VQA, document understanding, OCR
- **Code:** Not specified
- **Strictness:** **Not OPD** — offline distillation, no student rollouts. Included as a key MLLM-specific KD technique.
- **Summary:** Addresses the fundamental challenge of unbalanced vision tokens between teacher (full tokens) and student (compressed tokens). Uses Hungarian matching on Manhattan distance to align vision logits spatially, then applies VLAD for cross-modal affinity and VSD for semantic alignment.
- **Why It Matters:** Solves a multimodal-specific KD problem that doesn't exist in text-only LLM distillation — different numbers of vision tokens between teacher and student.
- **Limitations:** Off-policy; doesn't address distribution mismatch from student generation.

---

### 2. Text-Only LLM Teacher → VLM Student (Cross-Modal OPD)

---

#### X-Reasoner: Generalizable Reasoning Across Modalities and Domains
- **Year:** 2025 (arXiv May 2025)
- **Authors:** Qianchu Liu et al. (Microsoft)
- **Link:** [arXiv:2505.03981](https://arxiv.org/abs/2505.03981) | [GitHub](https://github.com/microsoft/x-reasoner)
- **Type:** Paper
- **Model Type:** VLM (Qwen2.5-VL-7B base)
- **Modality:** Image-text, medical imaging
- **Method Family:** Text-only SFT+RL transferring to vision
- **Teacher Type:** Text-only reasoning data (distilled long-CoT + RL rewards)
- **Teacher Access:** Offline data (SFT stage) + verifiable rewards (RL stage)
- **Rollout Source:** Student on-policy (during GRPO RL stage)
- **Supervision Signal:** SFT on distilled CoT traces + GRPO with math verifier
- **Training Stage:** Post-training (SFT → RL)
- **Benchmarks:** MathVista, MedQA, multimodal medical reasoning
- **Code:** [GitHub](https://github.com/microsoft/x-reasoner)
- **Strictness:** **Partial OPD** — SFT stage is offline distillation; RL stage is on-policy with verifiable rewards but no teacher logit supervision. Demonstrates cross-modal reasoning transfer.
- **Summary:** Answers the question "Is reasoning generalizable across modalities?" affirmatively. Post-trains a VLM using only general-domain text via SFT + GRPO RL. The text-trained reasoning skills transfer to vision tasks. Includes forced-exiting mechanism and domain specialization recipe (X-Reasoner-Med).
- **Why It Matters:** Proves that pure text-based post-training can enable strong visual reasoning — more effective than in-domain multimodal training. Establishes the "cross-modal reasoning transfer" paradigm.
- **Limitations:** No explicit teacher on student rollouts during RL; vision encoder is frozen.

---

#### R1-Onevision: Cross-Modal Formalization for Multimodal Reasoning
- **Year:** 2025 (arXiv Mar 2025, ICCV 2025)
- **Authors:** Yi Yang et al. (Zhejiang University, Tencent)
- **Link:** [arXiv:2503.10615](https://arxiv.org/abs/2503.10615) | [GitHub](https://github.com/Fancy-MLLM/R1-Onevision)
- **Type:** Paper (ICCV 2025)
- **Model Type:** VLM (Qwen2.5-VL 3B/7B)
- **Modality:** Image-text (math, physics, chemistry, biology)
- **Method Family:** Cross-modal formalization + SFT + RL
- **Teacher Type:** Cross-modal pipeline (image → formal text → reasoning)
- **Teacher Access:** Offline data generation (cross-modal pipeline)
- **Rollout Source:** Student on-policy (during RL on CLEVR)
- **Supervision Signal:** SFT on R1-Onevision dataset + RL on CLEVR
- **Training Stage:** SFT → RL
- **Benchmarks:** MathVision, MathVerse, MathVista, R1-Onevision-Bench
- **Code:** [GitHub](https://github.com/Fancy-MLLM/R1-Onevision)
- **Strictness:** **Partial OPD** — SFT stage uses offline cross-modal data; RL stage is on-policy but with verifiable rewards rather than teacher logits
- **Summary:** Transforms images into formal textual representations for language-based reasoning. Constructs a dataset with step-by-step multimodal reasoning annotations. SFT then RL fine-tuning enables deep CoT reasoning, outperforming GPT-4o on multiple benchmarks.
- **Why It Matters:** Demonstrates that converting visual problems to formal text representations enables effective cross-modal reasoning with standard LLM techniques.
- **Limitations:** Cross-modal formalization may lose visual information; RL is limited to CLEVR domain.

---

### 3. VLM RL / RLVR (OPD-Adjacent)

---

#### SFT or RL? An Early Investigation into Training R1-Like Reasoning VLMs
- **Year:** 2025 (arXiv Apr 2025, TMLR)
- **Authors:** Hardy Chen, Haoqin Tu et al. (UC Santa Cruz)
- **Link:** [arXiv:2504.11468](https://arxiv.org/abs/2504.11468) | [GitHub](https://github.com/UCSC-VLAA/VLAA-Thinking)
- **Type:** Paper (TMLR)
- **Model Type:** VLM (Qwen2.5-VL 3B)
- **Modality:** Image-text (math, VQA)
- **Method Family:** GRPO RL with mixed perception+cognition rewards
- **Teacher Type:** No teacher (self-generated + verifiable rewards)
- **Teacher Access:** N/A (reward-based)
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Mixed reward: perception (caption accuracy) + cognition (answer correctness)
- **Training Stage:** Post-training (SFT comparison, RL, SFT+RL)
- **Benchmarks:** VLAA-Thinking benchmark, Open LMM Reasoning Leaderboard
- **Code:** [GitHub](https://github.com/UCSC-VLAA/VLAA-Thinking)
- **Strictness:** **Adjacent** — student generates on-policy with GRPO, but supervision is reward-based, no teacher distribution matching
- **Summary:** Critical finding: SFT induces "pseudo reasoning paths" from expert models that lock models into rigid, imitative reasoning modes. RL alone is significantly more effective. Combining SFT and GRPO also hurts performance. Introduces VLAA-Thinking dataset and mixed perception+cognition reward.
- **Why It Matters:** Foundational empirical study for VLM reasoning training. Challenges the standard SFT→RL pipeline and provides evidence that direct RL is better for VLMs.
- **Limitations:** Small-scale (3B); limited evaluation scope.

---

#### Visionary-R1: Mitigating Shortcuts in Visual Reasoning with RL
- **Year:** 2025 (ICLR 2026 submission)
- **Authors:** Xia et al.
- **Link:** [OpenReview](https://openreview.net/forum?id=bya3KOdLeS)
- **Type:** Paper
- **Model Type:** VLM
- **Modality:** Image-text
- **Method Family:** GRPO with caption-reason-answer format
- **Teacher Type:** No teacher (LLM-based caption reward)
- **Teacher Access:** LLM judge for caption evaluation
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Caption reward + answer correctness reward
- **Training Stage:** RL post-training
- **Benchmarks:** Visual reasoning benchmarks
- **Code:** Not specified
- **Strictness:** **Adjacent** — on-policy GRPO with visual grounding rewards, but no teacher KL matching
- **Summary:** Addresses the "visual shortcut" problem where VLMs bypass visual reasoning and rely on language priors. Enforces a caption→reason→answer format. Training on 273K CoT-free visual QA pairs, outperforms GPT-4o, Claude 3.5-Sonnet, and Gemini-1.5-Pro.
- **Why It Matters:** Explicitly addresses the visual neglect problem in VLM RL training — a core multimodal-specific challenge.
- **Limitations:** Caption reward evaluation adds overhead; relies on LLM judge accuracy.

---

#### Perception-R1: Visual Perception Reward for MLLMs
- **Year:** 2025 (arXiv Jun 2025)
- **Authors:** Tong Xiao et al.
- **Link:** [arXiv:2506.07218](https://arxiv.org/abs/2506.07218) | [GitHub](https://github.com/tongxiao2002/Perception-R1)
- **Type:** Paper
- **Model Type:** MLLM (Qwen2.5-VL-7B)
- **Modality:** Image-text
- **Method Family:** GRPO with visual perception reward
- **Teacher Type:** No teacher (verifiable visual perception reward)
- **Teacher Access:** N/A
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Visual perception reward + answer correctness
- **Training Stage:** RL post-training
- **Benchmarks:** Multimodal perception and reasoning benchmarks
- **Code:** [GitHub](https://github.com/tongxiao2002/Perception-R1)
- **Strictness:** **Adjacent** — on-policy RL with perception-specific reward, no teacher distribution
- **Summary:** Shows that existing RLVR methods fail to enhance multimodal perception (verified via McNemar's test). Introduces a visual perception reward that explicitly encourages accurate visual content perception, improving both perception and reasoning.
- **Why It Matters:** First to demonstrate empirically that standard VLM RLVR does not improve perception capabilities. Perception-specific reward is a novel contribution.
- **Limitations:** Perception reward design is task-specific; may not generalize to all visual domains.

---

#### Vision-SR1: Self-Rewarding VLM via Reasoning Decomposition
- **Year:** 2025 (arXiv Aug 2025)
- **Authors:** Not specified
- **Link:** [arXiv:2508.19652](https://arxiv.org/abs/2508.19652)
- **Type:** Paper
- **Model Type:** VLM
- **Modality:** Image-text
- **Method Family:** Self-rewarding with two-stage visual+answer rollout
- **Teacher Type:** Self (self-rewarding)
- **Teacher Access:** Self-generated reward scores
- **Rollout Source:** Student on-policy (two-stage: visual + answer)
- **Supervision Signal:** Separate visual grounding reward + answer correctness reward
- **Training Stage:** RL post-training
- **Benchmarks:** Visual reasoning
- **Code:** Not specified
- **Strictness:** **Partial OPD** — self-rewarding with decomposed visual and reasoning rewards; no external teacher but structured self-supervision
- **Summary:** Addresses the question of whether RL training truly improves visual reasoning or just language reasoning. Uses a two-stage rollout with separate visual and answer rewards. Only 10-15% more expensive than standard GRPO.
- **Why It Matters:** Introduces the self-rewarding paradigm to VLMs with explicit visual grounding accountability.
- **Limitations:** Self-rewarding may accumulate biases; visual reward calibration is challenging.

---

#### VL-Rethinker: Self-Reflection with RL for VLMs
- **Year:** 2025 (arXiv Apr 2025)
- **Authors:** Not specified
- **Link:** [arXiv:2504.08837](https://arxiv.org/abs/2504.08837)
- **Type:** Paper
- **Model Type:** VLM
- **Modality:** Image-text
- **Method Family:** GRPO with Selective Sample Replay
- **Teacher Type:** No teacher (self-reflection via RL)
- **Teacher Access:** N/A
- **Rollout Source:** Student on-policy
- **Supervision Signal:** GRPO rewards with SSR for stability
- **Training Stage:** RL post-training
- **Benchmarks:** Multimodal reasoning
- **Code:** Not specified
- **Strictness:** **Adjacent** — on-policy RL with self-reflection, no teacher
- **Summary:** Explores direct RL training for multimodal reasoning without distillation from stronger teachers. Identifies the "vanishing advantages problem" in GRPO where identical group rewards eliminate gradient signal. Proposes Selective Sample Replay (SSR) for stable training.
- **Why It Matters:** Addresses a fundamental VLM-specific GRPO instability issue.
- **Limitations:** Pure RL without teacher may have lower sample efficiency.

---

#### VTool-R1: VLMs Learn to Think with Images via RL on Tool Use
- **Year:** 2025/2026
- **Authors:** Not specified
- **Link:** [OpenReview](https://openreview.net/forum?id=Idst6X6gmy)
- **Type:** Paper
- **Model Type:** VLM with tool use
- **Modality:** Image-text + tool outputs
- **Method Family:** RFT with tool-augmented multimodal CoT
- **Teacher Type:** No teacher (outcome-based rewards)
- **Teacher Access:** N/A
- **Rollout Source:** Student on-policy (generates tool calls + text)
- **Supervision Signal:** Outcome-based rewards for tool use correctness
- **Training Stage:** RL post-training
- **Benchmarks:** Multimodal tool-use reasoning
- **Code:** Not specified
- **Strictness:** **Adjacent** — on-policy RL with tool use, reward-based
- **Summary:** First RFT framework training VLMs to generate multimodal chains of thought interleaving text and visual tool operations. Integrates Python-based visual editing tools into the RL process. Elicits strategic visual tool use without process-based supervision.
- **Why It Matters:** Extends VLM RL to tool-augmented reasoning, combining visual understanding with programmatic tool execution.
- **Limitations:** Tool integration adds complexity; limited to Python-based visual tools.

---

### 4. Multimodal Self-Distillation & Self-Improvement

---

#### MASSV: Self-Data Distillation for VLM Speculative Decoding
- **Year:** 2025 (arXiv May 2025, EMNLP 2025 Findings)
- **Authors:** Not specified
- **Link:** [arXiv:2505.10526](https://arxiv.org/abs/2505.10526)
- **Type:** Paper
- **Model Type:** VLM (Qwen2.5-VL, Gemma3 families)
- **Modality:** Image-text
- **Method Family:** Self-data distillation for speculative decoding
- **Teacher Type:** Self (target VLM generates training data for drafter)
- **Teacher Access:** Full model outputs
- **Rollout Source:** Target model generates (off-policy from drafter's perspective)
- **Supervision Signal:** Self-distilled visual instruction tuning
- **Training Stage:** Inference optimization
- **Benchmarks:** Visually-grounded tasks, inference speedup
- **Code:** Not specified
- **Strictness:** **Not OPD** — target model generates data for drafter; no drafter on-policy rollouts during training
- **Summary:** Transforms small language models into multimodal drafters through (1) connecting the target VLM's vision encoder via a trainable projector, (2) self-distilled visual instruction tuning. Achieves up to 1.46x inference speedup.
- **Why It Matters:** Extends speculative decoding to VLMs with cross-architecture alignment.
- **Limitations:** Off-policy data generation; drafter doesn't train on its own outputs.

---

#### SelfReVision: Self-Critical Distillation for VLM Planning
- **Year:** 2025 (arXiv Jul 2025)
- **Authors:** Not specified
- **Link:** [arXiv:2507.08224](https://arxiv.org/abs/2507.08224)
- **Type:** Paper
- **Model Type:** VLM (3B to 72B)
- **Modality:** Image-text (procedural planning, robotics)
- **Method Family:** Iterative self-critique + self-distillation
- **Teacher Type:** Self (iterative critique-revise-verify loop)
- **Teacher Access:** Self-generated critiques
- **Rollout Source:** Student on-policy (generates plans, critiques, revisions)
- **Supervision Signal:** Self-generated quality assessments
- **Training Stage:** Post-training (iterative)
- **Benchmarks:** Procedural planning, robotics
- **Code:** Not specified
- **Strictness:** **Partial OPD** — student generates on-policy and self-critiques, but no external teacher. Self-distillation loop.
- **Summary:** VLMs iteratively critique, revise, and verify their own plans without external supervision. Through this self-distillation loop, models generate higher-quality plans usable at inference and for continued fine-tuning. Outperforms models 100x larger.
- **Why It Matters:** Teacher-free self-improvement for VLM planning — doesn't require any external teacher or supervision signal.
- **Limitations:** Self-critique quality limited by model capability; may not work for very small models.

---

#### ThinkLite-VL: MCTS-Guided Sample Selection for Visual Reasoning
- **Year:** 2025 (arXiv Apr/May 2025)
- **Authors:** Not specified
- **Link:** [arXiv:2504.07934](https://arxiv.org/abs/2504.07934)
- **Type:** Paper
- **Model Type:** VLM (7B)
- **Modality:** Image-text (math)
- **Method Family:** MCTS-guided data curation for RL fine-tuning
- **Teacher Type:** Self (MCTS for sample difficulty assessment)
- **Teacher Access:** N/A
- **Rollout Source:** Student on-policy (RL fine-tuning)
- **Supervision Signal:** Curated high-impact training data + RL rewards
- **Training Stage:** Post-training (RFT)
- **Benchmarks:** MathVista (SoTA 75.1), OlympiadBench, MathVision
- **Code:** Not specified
- **Strictness:** **Adjacent** — MCTS used for training data selection, then RL on selected data
- **Summary:** Uses MCTS during training (not inference) to assess sample difficulty and curate a high-impact training subset for reinforcement fine-tuning. Achieves 75.1 on MathVista, surpassing GPT-4o, O1, and Qwen2.5-VL-72B.
- **Why It Matters:** Data-centric approach to VLM reasoning — careful sample selection dramatically improves RL fine-tuning effectiveness.
- **Limitations:** MCTS during training is computationally expensive.

---

### 5. Black-Box Multimodal OPD

---

#### ARMADA: Cross-Modal Black-Box KD from VLMs to Language Models
- **Year:** 2026 (arXiv Mar 2026)
- **Authors:** Not specified
- **Link:** [arXiv:2603.10877](https://arxiv.org/abs/2603.10877)
- **Type:** Paper
- **Model Type:** Language model student (VLM teacher)
- **Modality:** Cross-modal (VLM → text-only LLM)
- **Method Family:** Architecture-agnostic cross-modal black-box KD
- **Teacher Type:** Black-box VLM teacher
- **Teacher Access:** Text outputs only (black-box)
- **Rollout Source:** Off-policy
- **Supervision Signal:** Novel alignment techniques for cross-modal transfer
- **Training Stage:** Post-training
- **Benchmarks:** 12 NLU tasks + 8 generative reasoning tasks
- **Code:** Not specified
- **Strictness:** **Not OPD** — off-policy black-box distillation (VLM→LLM direction)
- **Summary:** First architecture-agnostic cross-modal KD from black-box VLM teachers to language-only students. Achieves up to 3.4% improvement on language understanding without multimodal pre-training.
- **Why It Matters:** Demonstrates feasibility of extracting visual knowledge from VLMs into text-only models via black-box access. Reverse direction from typical VLM distillation.
- **Limitations:** Black-box access limits knowledge transfer depth; reverse direction (VLM→LLM) rather than the more typical LLM→VLM.

---

### 6. VLM Compression via KD (Offline Baselines)

---

#### LLaVA-KD: A Framework of Distilling Multimodal LLMs
- **Year:** 2024 (arXiv Oct 2024, ICCV 2025)
- **Authors:** Cai et al.
- **Link:** [arXiv:2410.16236](https://arxiv.org/abs/2410.16236) | [GitHub](https://github.com/Fantasyele/LLaVA-KD)
- **Type:** Paper (ICCV 2025)
- **Model Type:** MLLM
- **Modality:** Image-text
- **Method Family:** Multi-stage white-box MLLM distillation
- **Teacher Type:** Large MLLM (LLaVA-7B/13B → 1B student)
- **Teacher Access:** White-box (logits + internal representations)
- **Rollout Source:** Off-policy (dataset-based)
- **Supervision Signal:** Multimodal Distillation (MDist) + Relation Distillation (RDist)
- **Training Stage:** Three stages: Distilled Pre-Training → SFT → Distilled Fine-Tuning
- **Benchmarks:** 5 VQA benchmarks
- **Code:** [GitHub](https://github.com/Fantasyele/LLaVA-KD)
- **Strictness:** **Not OPD** — offline distillation with no student rollouts
- **Summary:** Comprehensive MLLM KD framework with MDist (visual+textual distribution alignment) and RDist (visual token relationship transfer). Three-stage training. LLaVA-KD-1B outperforms BLIP2-13B and InstructBLIP-7B.
- **Why It Matters:** Reference framework for MLLM-to-MLLM distillation. Establishes important baselines for future on-policy multimodal KD work.
- **Limitations:** Fully offline; suffers from distribution mismatch at inference.

---

#### LLAVADI: What Matters For Multimodal LLM Distillation
- **Year:** 2024 (arXiv Jul 2024)
- **Authors:** Not specified
- **Link:** [arXiv:2407.19409](https://arxiv.org/abs/2407.19409)
- **Type:** Paper
- **Model Type:** MLLM (MobileLLaMA 2.7B student, LLaVA-v1.5-13B teacher)
- **Modality:** Image-text
- **Method Family:** Comprehensive KD study (feature, logit, affinity, data-driven)
- **Teacher Type:** Larger MLLM
- **Teacher Access:** White-box
- **Rollout Source:** Off-policy
- **Supervision Signal:** Multiple (feature, logit, affinity distillation)
- **Training Stage:** Post-training
- **Benchmarks:** Multiple VQA benchmarks
- **Code:** Not specified
- **Strictness:** **Not OPD** — systematic study of offline KD approaches
- **Summary:** Extensive empirical study of what matters in MLLM distillation: training strategies, model choices, and algorithms. Key finding: joint alignment for both tokens and logits is critical. A 2.7B student can match 7B-13B teacher performance.
- **Why It Matters:** Essential reference for understanding MLLM KD design space.
- **Limitations:** All methods are offline; doesn't explore on-policy approaches.

---

#### LLaVA-MoD: MoE Knowledge Distillation
- **Year:** 2024 (arXiv Aug 2024, ICLR 2025)
- **Authors:** Fangxun Shu et al.
- **Link:** [arXiv:2408.15881](https://arxiv.org/abs/2408.15881) | [GitHub](https://github.com/shufangxun/LLaVA-MoD)
- **Type:** Paper (ICLR 2025)
- **Model Type:** MLLM (sparse MoE, 2B activated)
- **Modality:** Image-text
- **Method Family:** Progressive KD (mimic → preference/DPO)
- **Teacher Type:** Larger MLLM
- **Teacher Access:** White-box (logits for mimic distillation + outputs for DPO)
- **Rollout Source:** Mixed — mimic stage is off-policy; DPO preference stage uses student outputs
- **Supervision Signal:** KL divergence (mimic stage) + DPO loss (preference stage)
- **Training Stage:** Progressive (two stages)
- **Benchmarks:** MMB, MME, hallucination benchmarks
- **Code:** [GitHub](https://github.com/shufangxun/LLaVA-MoD)
- **Strictness:** **Partial OPD** — preference distillation stage uses student outputs with DPO (on-policy element); mimic stage is off-policy
- **Summary:** Integrates sparse MoE architecture for efficiency. Two-stage progressive KD: (1) mimic distillation with KL divergence, (2) preference distillation with DPO where student learns to outperform teacher, especially on hallucination. 2B activated params surpass 7B+ baselines.
- **Why It Matters:** Combines architectural efficiency (MoE) with progressive knowledge transfer. DPO stage introduces on-policy elements for hallucination mitigation.
- **Limitations:** DPO stage is only partially on-policy; MoE routing adds complexity.

---

#### Mini-InternVL 2.0: 5% Parameters, 90% Performance
- **Year:** 2024 (Oct 2024)
- **Authors:** OpenGVLab
- **Link:** [Blog](https://internvl.github.io/blog/2024-10-21-Mini-InternVL-2.0/)
- **Type:** Technical report / blog
- **Model Type:** MLLM (1B-4B)
- **Modality:** Image-text
- **Method Family:** Vision encoder KD + visual instruction tuning
- **Teacher Type:** InternViT-6B → InternViT-300M distillation
- **Teacher Access:** White-box
- **Rollout Source:** Off-policy
- **Supervision Signal:** Feature-level distillation for vision encoder
- **Training Stage:** Visual alignment / compression
- **Benchmarks:** MMBench, ChartQA, DocVQA, MathVista
- **Code:** [GitHub](https://github.com/opengvlab/internvl)
- **Strictness:** **Not OPD** — offline vision encoder distillation
- **Summary:** Achieves 90% of InternVL2-76B performance with only 5% of parameters (1B-4B). Key: distilling InternViT-6B to InternViT-300M, then building efficient MLLMs on top.
- **Why It Matters:** Demonstrates that aggressive vision encoder compression via KD retains most multimodal capability.
- **Limitations:** All offline; vision encoder distillation only.

---

### 7. Multimodal Agent / Tool-Use Distillation

---

#### Visual Program Distillation (VPD)
- **Year:** 2024 (CVPR 2024 Oral)
- **Authors:** Yushi Hu et al. (Google)
- **Link:** [arXiv:2312.03052](https://arxiv.org/abs/2312.03052) | [Project](https://visual-program-distillation.github.io/)
- **Type:** Paper (CVPR 2024 Oral)
- **Model Type:** VLM (PaLI-X)
- **Modality:** Image-text (compositional reasoning)
- **Method Family:** Tool-use program distillation into VLM
- **Teacher Type:** LLM (program generator) + specialized vision tools
- **Teacher Access:** Programs + tool execution results
- **Rollout Source:** Off-policy (LLM generates programs, tools execute)
- **Supervision Signal:** Verified programs converted to language reasoning chains
- **Training Stage:** Instruction tuning
- **Benchmarks:** MMBench, OK-VQA, A-OKVQA, TallyQA, POPE
- **Code:** Not specified
- **Strictness:** **Not OPD** — teacher (LLM + tools) generates all data; student does not generate during training
- **Summary:** Uses LLM-generated programs + specialized vision tools to create verified reasoning chains, which are converted to natural language and used to train VLMs. Student VLM learns compositional reasoning (counting, spatial relations) via instruction tuning.
- **Why It Matters:** Pioneered distilling programmatic/tool-use reasoning into VLMs, achieving CVPR 2024 Oral.
- **Limitations:** Fully offline; program quality depends on LLM and tool accuracy.

---

#### T3-Agent / Multi-modal Agent Tuning
- **Year:** 2024/2025 (ICLR 2025 Spotlight)
- **Authors:** Not specified
- **Link:** [arXiv:2412.15606](https://arxiv.org/abs/2412.15606) | [Project](https://mat-agent.github.io/)
- **Type:** Paper (ICLR 2025 Spotlight)
- **Model Type:** VLM agent (MiniCPM-V-8.5B, Qwen2-VL-7B)
- **Modality:** GUI / tool-use
- **Method Family:** Trajectory tuning for tool usage
- **Teacher Type:** Automated data synthesis pipeline
- **Teacher Access:** Trajectory data
- **Rollout Source:** Off-policy (synthetic trajectories)
- **Supervision Signal:** Trajectory tuning on MM-Traj dataset (20K tasks)
- **Training Stage:** Post-training (agent tuning)
- **Benchmarks:** GTA, GAIA
- **Code:** Not specified
- **Strictness:** **Not OPD** — trains on synthetic trajectory data, no on-policy student rollouts
- **Summary:** Automatically generates multimodal tool-usage data and tunes VLMs as controllers for tool-use reasoning. 20% improvement over untrained VLMs.
- **Why It Matters:** Establishes VLM-as-agent paradigm with tool-use distillation.
- **Limitations:** Offline trajectory data; no RL or on-policy refinement.

---

#### MAD-OPD: Multi-Agent Debate On-Policy Distillation (see Section 8)

---

### 8. Multi-Teacher & Debate-Based OPD

---

#### MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate
- **Year:** 2026 (arXiv May 2026)
- **Authors:** Jianze Wang et al.
- **Link:** [arXiv:2605.01347](https://arxiv.org/abs/2605.01347)
- **Type:** Paper
- **Model Type:** LLM (with agentic task extension)
- **Modality:** Text (code + agentic tasks)
- **Method Family:** Multi-teacher debate-driven OPD + agentic distillation (OPAD)
- **Teacher Type:** Multiple teachers forming deliberative collective
- **Teacher Access:** White-box (post-debate confidence-weighted logits)
- **Rollout Source:** Student on-policy
- **Supervision Signal:** Emergent collective token-level supervision from teacher debate
- **Training Stage:** Post-training (agentic + code)
- **Benchmarks:** 5 agentic and code benchmarks
- **Code:** Not specified
- **Strictness:** **Strict OPD** — student generates on-policy; multi-teacher debate provides supervision
- **Summary:** Multiple teachers debate over student's on-policy states, producing emergent collective intelligence for token-level supervision. On-Policy Agentic Distillation (OPAD) adds step-level sampling for multi-step error compounding. Task-adaptive divergence: JSD for agentic, reverse KL for code.
- **Why It Matters:** Breaks the single-teacher capability ceiling. First OPD method designed for agentic tasks where per-step errors compound. Notes OPD is adopted in DeepSeek-V4 post-training.
- **Limitations:** Debate adds inference cost for teacher; primarily text/code (not vision), but agentic framework applies to multimodal agents.

---

## Video-Language OPD

Video-language OPD remains largely **unexplored** as of May 2026. Current approaches include:

### Related Works

| Paper | Year | Approach | OPD Status |
|---|---|---|---|
| [Distilling VLMs on Millions of Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Distilling_Vision-Language_Models_on_Millions_of_Videos_CVPR_2024_paper.pdf) | 2024 | Offline KD: distill CLIP-based VLMs to video understanding models | **Not OPD** |
| Qwen2.5-VL Dynamic FPS | 2025 | Native dynamic FPS sampling for video understanding | **N/A** (architecture, not distillation) |
| InternVL3 video capabilities | 2025 | Video understanding via native multimodal pre-training + MPO | **Partial OPD** (MPO uses preference data) |

### Open Gaps

- **No dedicated video-language OPD paper** exists. GKD (2023) explicitly noted "extending GKD to auto-regressive models for video" as future work.
- **Temporal reasoning distillation** (understanding temporal order, causality in video) has not been addressed with on-policy approaches.
- **Video reward design** for VLM RL is nascent — verifiable rewards for video understanding (e.g., temporal grounding IoU) are underexplored.
- **Long-video OPD** faces challenges from sequence length, requiring efficient on-policy generation strategies.

---

## Industrial VLM Systems & Technical Reports

| System | Org | Year | OPD Use | Distillation Details | Link |
|---|---|---|---|---|---|
| **Qwen2.5-VL** | Alibaba | 2025 | **N/A** (architecture paper) | Native dynamic-resolution ViT, window attention. No explicit OPD in report. | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| **Qwen3 (VL component)** | Alibaba | 2025 | **Partial OPD** | Strong-to-weak distillation for small models. Uses Qwen2.5-VL for data augmentation (OCR on PDFs). On-policy KL for small model distillation. | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| **InternVL3** | OpenGVLab | 2025 | **Partial OPD** (MPO) | Native multimodal pre-training + Mixed Preference Optimization (MPO) with positive/negative supervision. MPO introduces offline RL-style preference alignment. | [arXiv:2504.10479](https://arxiv.org/abs/2504.10479) |
| **InternVL3.5** | OpenGVLab | 2025 | **Partial OPD** (Cascade RL) | MPO (offline RL) → GSPO (online RL). Cascade RL training with online component. | [Blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/) |
| **Mini-InternVL 2.0** | OpenGVLab | 2024 | **Not OPD** (offline) | Vision encoder KD (InternViT-6B → 300M). 5% params, 90% perf. | [Blog](https://internvl.github.io/blog/2024-10-21-Mini-InternVL-2.0/) |
| **Gemma 2/3** | Google | 2024-25 | **Strict OPD** (text) | All models use on-policy KD (GKD-style). Applies to text component. | [arXiv:2408.00118](https://arxiv.org/html/2408.00118v1) |
| **DeepSeek-R1 Distilled** | DeepSeek | 2025 | **Offline KD** | 800K CoT traces distilled via SFT. Primarily text reasoning. | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **MiniCPM-V** | OpenBMB | 2025 | **Offline KD** | Efficient edge models. 8B outperforms GPT-4V on 11 benchmarks. | [Nature Comms](https://www.nature.com/articles/s41467-025-61040-5) |
| **NanoVLMs** | Community | 2025 | **Offline KD** | Ultra-small VLMs trained on GPT-4o-generated descriptions. | [arXiv:2502.07838](https://arxiv.org/abs/2502.07838) |

### Key Observation

**Strict on-policy distillation for VLMs in industry is extremely rare.** The dominant industrial pattern remains:
1. **Offline KD** for bootstrapping (teacher generates data, student SFTs)
2. **RL/RLVR** for reasoning refinement (on-policy generation with rewards, but no teacher logit supervision)
3. **Preference optimization** (MPO, DPO) for alignment

VOLD, Video-OPD, X-OPD, Uni-OPD, VLA-OPD, GUI-SD, and LiteGUI now form the small verified or caveated strict multimodal core. The dominant industrial pattern still remains offline KD, reward-only RL/RLVR, and preference optimization.

The multimodal candidate line audit adds KEPO, D-OPSD, and Flow-OPD as borderline multimodal/image-generation OPD rows. GTR-Turbo remains adjacent because its KL variant is consumed as PPO reward shaping.

---

## Frameworks & Tools Supporting VLM OPD

| Framework | Org | VLM OPD Support | Key Features | Link |
|---|---|---|---|---|
| **OpenRLHF-M** | OpenRLHF | **Native** — VLM RLHF/RL + KD (MiniLLM) | PPO/REINFORCE++/GRPO for VLMs, LMM-R1 merged, vLLM generation, Ray distributed | [GitHub](https://github.com/OpenRLHF/OpenRLHF-M) |
| **OpenRLHF** | OpenRLHF | **Partial** — VLM RLHF added in v0.10 | VLM support (Qwen3.5-VL), KD (MiniLLM), PPO/GRPO/DAPO | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| **LMM-R1** | TideDra | **Native** — multimodal RL for reasoning | Two-stage framework (FRE + MGT), PPO/REINFORCE++/RLOO for VLMs, 4.7x speedup over R1-V | [GitHub](https://github.com/TideDra/lmm-r1) |
| **TRL (HuggingFace)** | HuggingFace | **Partial** — text OPD trainers (GKD, GOLD, MiniLLM), VLM support growing | GKDTrainer, GOLDTrainer, DistillationTrainer; VLM support via SFTTrainer | [GitHub](https://github.com/huggingface/trl) |
| **EasyDistill** | Alibaba | **Partial** — data synthesis + KD for LLMs, VLM extension possible | Black-box + white-box KD, CogPO, AgentKD | [GitHub](https://github.com/modelscope/easydistill) |
| **InternVL codebase** | OpenGVLab | **Partial** — MPO + SFT + RL pipelines | MPO implementation, mini-model training, multi-stage distillation | [GitHub](https://github.com/opengvlab/internvl) |
| **verl** | verl-project | **Partial** — VLM RL via extensions | GRPO/PPO for VLMs (via SDPO, rLLM extensions), 671B scale | [GitHub](https://github.com/verl-project/verl) |
| **NeMo RL** | NVIDIA | **Planned** — multimodal RL support in roadmap | On-policy distillation, GRPO/GSPO/DAPO, Megatron parallelism | [GitHub](https://github.com/NVIDIA-NeMo/RL) |

---

## Summary Table

| Paper | Year | Model Type | Modality | Method | Teacher | Rollout | OPD Type | Key Innovation |
|---|---|---|---|---|---|---|---|---|
| [VOLD](https://arxiv.org/abs/2510.23497) | 2025 | VLM | Image-text | OPD-RL hybrid | Text LLM | Student | **Strict OPD** | First text→VLM on-policy distillation + GRPO |
| [Video-OPD](https://arxiv.org/abs/2602.02994) | 2026 | MLLM | Video-text | Reverse-KL OPD | Frontier MLLM | Student | **Strict OPD** | Temporal video grounding via on-policy distillation |
| [X-OPD](https://arxiv.org/abs/2603.24596) | 2026 | Speech / LLM | Speech-text | Cross-modal OPD | Text teacher | Student | **Strict OPD** | Speech-model rollout supervision by text teacher |
| [Uni-OPD](https://arxiv.org/abs/2605.03677) | 2026 | LLM+MLLM | Multi | Unified OPD | Stronger LLM/MLLM | Student | **Strict OPD** | First unified OPD for LLMs + MLLMs |
| [VLA-OPD](https://arxiv.org/abs/2603.26666) | 2026 | VLA | Vision-language-action | Action-token reverse KL | Expert VLA | Student | **Strict OPD** | Expert labels student-visited robot/action states |
| [GUI-SD](https://arxiv.org/abs/2605.00642) | 2026 | VLM agent | GUI grounding | Privileged visual self-distillation | Privileged self-teacher | Student | **Strict OPD** | Grounding-only coordinate-token self-distillation |
| [LiteGUI](https://arxiv.org/abs/2605.07505) | 2026 | VLM agent | GUI | Guided OPD/GKD stage | Larger GUI VLM teacher | Student | **Strict OPD stage** | Guided GUI distillation before separate GRPO |
| [KEPO](https://arxiv.org/abs/2602.00400) | 2026 | VLM | Medical VQA | Quality-gated teacher divergence + GRPO | Qwen3-VL teacher | Student | Borderline | Medical VLM OPD candidate with gated teacher divergence |
| [D-OPSD](https://arxiv.org/abs/2605.05204) | 2026 | Diffusion | Text-to-image | Privileged self-distillation | EMA privileged self-teacher | Student | Borderline | Velocity-field MSE on student denoising rollouts |
| [Flow-OPD](https://arxiv.org/abs/2605.08063) | 2026 | Flow matching | Text-to-image | Multi-teacher velocity reward | Flow teacher ensemble | Student | Borderline | KL-derived velocity reward consumed through PPO-style update |
| [MAD-OPD](https://arxiv.org/abs/2605.01347) | 2026 | LLM (agentic) | Text+agentic | Multi-teacher debate OPD | Multiple teachers | Student | **Strict OPD** | Multi-teacher debate; first agentic OPD |
| [X-Reasoner](https://arxiv.org/abs/2505.03981) | 2025 | VLM | Image-text | Text SFT+RL transfer | Text data+verifier | Student (RL) | Partial OPD | Text-only training transfers to vision |
| [R1-Onevision](https://arxiv.org/abs/2503.10615) | 2025 | VLM | Image-text | Cross-modal formalization | Cross-modal pipeline | Student (RL) | Partial OPD | Image→formal text→reasoning |
| [VLAA-Thinking](https://arxiv.org/abs/2504.11468) | 2025 | VLM | Image-text | GRPO mixed reward | Self (reward) | Student | Adjacent | SFT harms RL; mixed perception+cognition reward |
| [Visionary-R1](https://openreview.net/forum?id=bya3KOdLeS) | 2025 | VLM | Image-text | GRPO caption-reason | Self (caption reward) | Student | Adjacent | Caption reward mitigates visual shortcuts |
| [Perception-R1](https://arxiv.org/abs/2506.07218) | 2025 | MLLM | Image-text | GRPO perception reward | Self (perception reward) | Student | Adjacent | Visual perception reward for GRPO |
| [Vision-SR1](https://arxiv.org/abs/2508.19652) | 2025 | VLM | Image-text | Self-rewarding | Self | Student | Partial OPD | Two-stage visual+answer rollout |
| [VL-Rethinker](https://arxiv.org/abs/2504.08837) | 2025 | VLM | Image-text | GRPO + SSR | Self | Student | Adjacent | Vanishing advantages fix for VLM GRPO |
| [VTool-R1](https://openreview.net/forum?id=Idst6X6gmy) | 2025 | VLM | Image-text+tools | RFT tool-use | Self (outcome) | Student | Adjacent | First VLM tool-use RFT |
| [ThinkLite-VL](https://arxiv.org/abs/2504.07934) | 2025 | VLM | Image-text | MCTS+RFT | Self (MCTS) | Student | Adjacent | MCTS data curation for VLM RFT |
| [SelfReVision](https://arxiv.org/abs/2507.08224) | 2025 | VLM | Image-text | Self-critique loop | Self | Student | Partial OPD | Teacher-free iterative self-improvement |
| [LLaVA-KD](https://arxiv.org/abs/2410.16236) | 2024 | MLLM | Image-text | Multi-stage KD | Larger MLLM | Off-policy | Not OPD | MDist + RDist for MLLM compression |
| [LLAVADI](https://arxiv.org/abs/2407.19409) | 2024 | MLLM | Image-text | KD study | Larger MLLM | Off-policy | Not OPD | What matters for MLLM KD |
| [LLaVA-MoD](https://arxiv.org/abs/2408.15881) | 2024 | MLLM | Image-text | MoE + mimic→DPO | Larger MLLM | Mixed | Partial OPD | Progressive KD with MoE |
| [EM-KD](https://arxiv.org/abs/2511.21106) | 2025 | MLLM | Image-text | Vision token alignment KD | Vanilla MLLM | Off-policy | Not OPD | Hungarian matching for unbalanced tokens |
| [VPD](https://arxiv.org/abs/2312.03052) | 2024 | VLM | Image-text | Tool-use distillation | LLM+tools | Off-policy | Not OPD | Programmatic reasoning distillation |
| [Mini-InternVL](https://internvl.github.io/blog/2024-10-21-Mini-InternVL-2.0/) | 2024 | MLLM | Image-text | Vision encoder KD | InternViT-6B | Off-policy | Not OPD | 5% params, 90% perf |
| [MASSV](https://arxiv.org/abs/2505.10526) | 2025 | VLM | Image-text | Self-data distillation | Self (target) | Off-policy | Not OPD | VLM speculative decoding |
| [T3-Agent](https://arxiv.org/abs/2412.15606) | 2025 | VLM agent | GUI/tools | Trajectory tuning | Synthetic pipeline | Off-policy | Not OPD | Agent tool-use distillation |
| [ARMADA](https://arxiv.org/abs/2603.10877) | 2026 | LLM (from VLM) | Cross-modal | Black-box cross-modal KD | Black-box VLM | Off-policy | Not OPD | VLM→LLM black-box KD |
| [HKD4VLM](https://arxiv.org/abs/2506.13038) | 2025 | VLM | Image-text | Progressive hybrid KD | Stronger VLM | Mixed | Partial OPD | Online + refinement distillation |
| [InternVL3](https://arxiv.org/abs/2504.10479) | 2025 | MLLM | Image-text+video | MPO (preference) | N/A (preference data) | Mixed | Partial OPD | Native pre-training + MPO |

---

## Open Problems in VLM OPD

### 1. The Visual Neglect Problem
- VLMs trained with text-heavy RL (GRPO) often bypass visual reasoning, relying on language priors.
- Existing solutions (Visionary-R1 caption reward, Perception-R1 perception reward) are partial fixes.
- **Open:** Principled methods to ensure VLM RL training genuinely improves visual perception, not just language reasoning.

### 2. Cross-Modal OPD: Text Teacher → VLM Student
- VOLD is the first work, but it was withdrawn from ICLR 2026.
- Core challenge: how to provide meaningful token-level KL supervision from a text-only teacher to a VLM that processes both images and text.
- **Open:** Principled cross-modal teacher supervision that accounts for modality-specific tokens.

### 3. Video-Language OPD Is Sparse
- Video-OPD provides a strict seed for temporal video grounding.
- Video understanding still requires broader temporal reasoning, temporal grounding, and long-sequence on-policy generation coverage.
- **Open:** OPD frameworks for long video-language models beyond temporal grounding, with clear teacher supervision on student-generated temporal traces.

### 4. Vision Token Mismatch in OPD
- Different VLMs produce different numbers of vision tokens (EM-KD addresses this for offline KD).
- On-policy OPD with different teacher/student vision architectures is unsolved.
- **Open:** Cross-architecture on-policy distillation for VLMs with different vision encoders.

### 5. Multimodal Agent OPD
- MAD-OPD introduces agentic OPD for text agents.
- GUI-SD and LiteGUI provide GUI grounding or GUI-stage seeds, while many GUI agents still use offline trajectory training or reward-only RL.
- **Open:** Long-horizon multimodal GUI/tool-use OPD where the student interacts with environments and receives teacher action-token supervision on visited states.

### 6. Hallucination-Aware OPD
- LLaVA-MoD's DPO stage targets hallucination, but with offline data.
- On-policy hallucination detection + correction during OPD training is unexplored.
- **Open:** OPD methods that specifically address visual hallucination via on-policy feedback.

### 7. OPD for Document/Chart/OCR-Heavy Reasoning
- EM-KD addresses efficient distillation for document understanding.
- No on-policy OPD work specifically targets OCR-heavy or chart reasoning tasks.
- **Open:** Domain-specific OPD for document understanding VLMs.

### 8. Scaling Laws for VLM OPD
- No study on how VLM OPD scales with student size, teacher size, vision encoder capacity, or data.
- **Open:** Compute-optimal allocation between visual and textual on-policy generation.

### 9. Multi-Teacher VLM OPD
- MAD-OPD demonstrates multi-teacher debate for text, but not for multimodal.
- Using multiple VLM teachers with different visual strengths (e.g., one strong at charts, another at natural images) is unexplored.
- **Open:** Multi-teacher OPD where teachers have complementary visual expertise.

### 10. Reproducibility and Benchmarks
- No standardized benchmark exists for evaluating VLM OPD.
- OPD for MLLMs is "largely underexplored" (Uni-OPD, 2026).
- **Open:** Community-standard benchmarks and protocols for VLM OPD research.

---

## Citation

```bibtex
@misc{on-policy-distillation-landscape-vlm-2026,
  title={On Policy Distillation Landscape for VLMs & MLLMs},
  year={2026},
  url={https://github.com/YOUR_USERNAME/on-policy-distillation-landscape}
}
```

## Key Survey References

- Song et al. (2026). [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626)
- KD Survey (2025). [Knowledge Distillation and Dataset Distillation of Large Models](https://arxiv.org/abs/2504.14772)
- RL in Vision Survey (2025). [Reinforcement Learning in Vision: A Survey](https://arxiv.org/html/2508.08189v1)
- Awesome-RLVR. [GitHub](https://github.com/opendilab/awesome-RLVR)

---

*This document was compiled through systematic search of arXiv, OpenReview, CVPR/ICCV/ICLR proceedings, Google Scholar, GitHub, and major lab technical reports. Every entry has been verified against its source. Last verified: May 2026.*
