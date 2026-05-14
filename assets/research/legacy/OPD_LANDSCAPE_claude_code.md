# Awesome On-Policy Distillation (OPD) for LLMs — Research Landscape

> A taxonomy-first, implementation-aware research map of On-Policy Distillation for Large Language Models.
>
> Last updated: 2026-05-11

---

## Table of Contents

- [Definition](#definition)
- [Taxonomy of LLM OPD Methods](#taxonomy-of-llm-opd-methods)
- [Historical Timeline](#historical-timeline)
- [Must-Read First 10 Papers](#must-read-first-10-papers)
- [Detailed Paper Catalog](#detailed-paper-catalog)
  - [Foundational Works](#1-foundational-works)
  - [Core OPD Methods](#2-core-opd-methods-2023-2026)
  - [OPD + RL/RLVR Hybrids](#3-opd--rlrlvr-hybrids)
  - [Iterative Self-Distillation & Self-Play](#4-iterative-self-distillation--self-play)
  - [Black-Box OPD](#5-black-box-opd)
  - [Speculative Decoding + Distillation](#6-speculative-decoding--distillation)
  - [Cross-Tokenizer OPD](#7-cross-tokenizer-opd)
  - [Preference-Aligned Distillation](#8-preference-aligned-distillation)
- [Industrial Systems & Technical Reports](#industrial-systems--technical-reports)
- [Frameworks & Tools Supporting OPD](#frameworks--tools-supporting-opd)
- [Summary Table (GitHub README)](#summary-table)
- [Markdown-Table Comprehensive Table](#markdown-table-comprehensive-table)
- [Open Problems in LLM OPD](#open-problems-in-llm-opd)

---

## Definition

### Strict On-Policy Distillation (OPD)

A method qualifies as **Strict OPD** if and only if:

1. **The student/policy generates its own trajectories**, completions, or intermediate states during training (on-policy rollouts).
2. **A teacher, verifier, discriminator, privileged-context model, reward model, or reference model provides supervision** on those student-generated trajectories.

### Classification Criteria

| Category | Rollout Source | Supervision Source | Example |
|---|---|---|---|
| **Strict OPD** | Student generates | Teacher/verifier scores student outputs | GKD, MiniLLM, OPSD |
| **Partial OPD** | Mixed student+teacher | Teacher provides partial signal | DistiLLM (adaptive off-policy), RAFT |
| **Adjacent** | Student generates | Self-reward or verifier only (no teacher) | STaR, SPIN, GRPO |
| **Not OPD** | Teacher generates all data | Student trains on teacher data only | Offline KD, SeqKD, standard SFT |

### Distinguishing OPD from Related Paradigms

| Paradigm | Student Rollouts? | Teacher Signal? | Difference from OPD |
|---|---|---|---|
| **Offline KD** | No — trains on teacher outputs | Yes (logits/text) | No student rollouts; train-test mismatch |
| **Standard SFT** | No — trains on fixed dataset | No teacher | No teacher, no student rollouts |
| **DPO** | No — uses pre-collected pairs | Implicit reward | Offline preference optimization |
| **RLHF (PPO)** | Yes — student generates | Reward model scores | No teacher distribution matching; reward only |
| **RLVR (GRPO)** | Yes — student generates | Verifiable reward | No teacher; reward from answer correctness |
| **OPD** | Yes — student generates | Teacher provides logits/scores/text on student rollouts | Teacher feedback on student distribution |

---

## Taxonomy of LLM OPD Methods

```
On-Policy Distillation for LLMs
├── By Teacher Access
│   ├── White-Box OPD (teacher logits accessible)
│   │   ├── Token-level KL minimization (GKD, MiniLLM, DistiLLM, ToDi, EOPD)
│   │   ├── f-divergence framework (f-DISTILL, GKD generalized JSD)
│   │   └── Cross-tokenizer OPD (DSKD, ULD, GOLD, ALM, BLD)
│   ├── Black-Box OPD (text-only teacher access)
│   │   ├── Adversarial distillation (GAD)
│   │   ├── Rejection sampling + teacher scoring (RAFT, RFT, ReST)
│   │   └── LLM-as-Judge feedback (Self-Rewarding, RLAIF)
│   └── Teacher-Free / Self-Distillation
│       ├── Self-play (SPIN)
│       ├── Privileged-context self-distillation (OPSD, OPSDL)
│       └── Iterative self-improvement (STaR, Self-Rewarding)
│
├── By Feedback Signal
│   ├── Logit-based (dense, token-level)
│   │   ├── Forward KL / Reverse KL / JSD (GKD, MiniLLM)
│   │   ├── Skewed KL (DistiLLM)
│   │   ├── Adaptive per-token divergence (ToDi, EOPD)
│   │   └── Hybrid token+sequence (Uni-OPD)
│   ├── Outcome-based (sparse, sequence-level)
│   │   ├── Binary correctness reward (ReST, RFT, RAFT)
│   │   ├── Reward model score (RLHF-style)
│   │   └── Discriminator reward (GAD)
│   ├── Process-based (step-level)
│   │   ├── Process Reward Models (PRIME, R-PRM, ThinkPRM)
│   │   └── Structure-aware rewards (RLKD with GSRM)
│   └── Self-play / Contrastive
│       ├── Distinguish self vs. reference (SPIN)
│       └── Self-critique + revision (Constitutional AI)
│
├── By Training Stage
│   ├── Pre-training distillation (Gemma 2/3, Ministral Cascade)
│   ├── Post-training / SFT-stage (GKD, MiniLLM, most methods)
│   ├── RL-stage hybrid (KDRL, RLAD, RLKD)
│   └── Inference-time (speculative decoding: DistillSpec, DVI)
│
├── By Iteration Strategy
│   ├── Single-round OPD (GKD, MiniLLM)
│   ├── Iterative OPD (ReST^EM, STaR, Self-Rewarding)
│   └── Continual/progressive (Cascade Distillation, Born-Again Networks)
│
└── By Application Domain
    ├── General instruction following (GKD, Zephyr, Qwen3)
    ├── Math reasoning (OPSD, RLKD, KDRL, PRIME)
    ├── Code reasoning (ReST^EM on APPS, DeepSeek-R1 distillation)
    ├── Long-context extension (OPSDL)
    ├── Alignment / safety (Constitutional AI, DPKD, AlignDistil)
    └── Inference acceleration (DistillSpec, OSD, DVI)
```

---

## Historical Timeline

| Year | Milestone | Paper/System |
|---|---|---|
| **2006** | Model compression concept | Bucila et al. — Model Compression |
| **2011** | DAgger: on-policy imitation learning | Ross, Gordon & Bagnell — DAgger (AISTATS 2011) |
| **2015** | Knowledge Distillation formalized | Hinton, Vinyals & Dean — Distilling the Knowledge in a Neural Network |
| **2016** | Sequence-level KD for NLP | Kim & Rush — Sequence-Level Knowledge Distillation (EMNLP 2016) |
| **2018** | Self-distillation concept | Furlanello et al. — Born Again Neural Networks (ICML 2018) |
| **2022** | Bootstrapping reasoning with self-training | Zelikman et al. — STaR: Self-Taught Reasoner (NeurIPS 2022) |
| **2022** | AI feedback for alignment | Bai et al. — Constitutional AI (Anthropic, Dec 2022) |
| **2023 Jun** | **On-policy KD for LLMs formalized** | **Agarwal et al. — GKD (arXiv Jun 2023, ICLR 2024)** |
| **2023 Jun** | Reverse KL for LLM distillation | **Gu et al. — MiniLLM (arXiv Jun 2023, ICLR 2024)** |
| **2023 Jul** | f-divergence framework for SeqKD | Wen et al. — f-DISTILL (ACL 2023) |
| **2023 Aug** | Rejection sampling fine-tuning | Yuan et al. — RFT for math reasoning |
| **2023 Aug** | Reinforced Self-Training | Gulcehre et al. — ReST (Aug 2023) |
| **2023 Oct** | Zephyr: distilled alignment | Tunstall et al. — Zephyr (HuggingFace, Oct 2023) |
| **2023 Dec** | Weak-to-strong generalization | Burns et al. — Weak-to-Strong (OpenAI, Dec 2023) |
| **2023 Dec** | Scaling self-training beyond human data | Singh et al. — ReST^EM (Google, Dec 2023) |
| **2024 Jan** | Self-play fine-tuning | Chen et al. — SPIN (UCLA, Jan 2024) |
| **2024 Jan** | Self-Rewarding Language Models | Yuan et al. — Self-Rewarding (Meta, Jan 2024) |
| **2024 Feb** | Efficient on-policy KD with skewed KL | Ko et al. — DistiLLM (ICML 2024) |
| **2024 Apr** | Reward-ranked fine-tuning | Dong et al. — RAFT (TMLR 2023) |
| **2024 Jun** | Gemma 2: on-policy distillation in pre-training | Google — Gemma 2 Technical Report |
| **2024 Jun** | Dual-space cross-tokenizer KD | Zhang et al. — DSKD (EMNLP 2024) |
| **2024 Jun** | Preference knowledge distillation | Li et al. — DPKD (Microsoft, Jun 2024) |
| **2024 Oct** | Speculative decoding + KD | Zhou et al. — DistillSpec (ICLR 2024) |
| **2024 Dec** | DeepSeek-V3 with R1 reasoning distillation | DeepSeek — V3 Technical Report |
| **2025 Jan** | **DeepSeek-R1: reasoning model + distillation** | **DeepSeek — R1 Technical Report** |
| **2025 Feb** | Implicit process rewards for on-policy RL | Sun et al. — PRIME |
| **2025 Mar** | Cross-tokenizer distillation via ALM | Boizard et al. — ALM |
| **2025 Mar** | Gemma 3: all models use KD | Google — Gemma 3 Technical Report |
| **2025 Mar** | AlignDistil: token-level alignment as distillation | Zhang et al. — AlignDistil (ACL 2025) |
| **2025 Apr** | DistilQwen2.5: industrial distillation practices | Alibaba — DistilQwen2.5 (ACL 2025) |
| **2025 May** | ToDi: token-wise adaptive divergence | Jung et al. — ToDi (EMNLP 2025) |
| **2025 May** | **Qwen3: on-policy strong-to-weak distillation** | **Qwen Team — Qwen3 Technical Report** |
| **2025 May** | RLKD: RL-based reasoning distillation | Xu et al. — RLKD (AAAI 2025) |
| **2025 Jun** | KDRL: unified KD+RL post-training | Xu et al. — KDRL |
| **2025 Nov** | Black-box adversarial OPD | Ye et al. — GAD (Microsoft) |
| **2026 Jan** | Ministral 3: cascade distillation | Mistral — Ministral 3 Technical Report |
| **2026 Jan** | On-policy self-distillation for reasoning | Zhao et al. — OPSD |
| **2026 Feb** | RL-aware distillation | Zhang et al. — RLAD |
| **2026 Mar** | Entropy-aware on-policy distillation | Jin et al. — EOPD |
| **2026 Apr** | **First comprehensive OPD survey** | **Song et al. — A Survey of OPD for LLMs** |
| **2026 Apr** | Rethinking OPD: failure modes & recipes | Li et al. — Rethinking OPD (THUNLP) |
| **2026 Apr** | Self-distillation for long context | Zhang et al. — OPSDL (Baidu) |

---

## Must-Read First 10 Papers

| # | Paper | Year | Why Read First |
|---|---|---|---|
| 1 | [GKD: On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) (Agarwal et al.) | 2023 | **The foundational OPD paper.** Formalizes on-policy KD with mixture sampling, flexible divergences, and RL integration. |
| 2 | [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) (Gu et al.) | 2023 | Introduces reverse KL for LLM distillation with RLHF-like on-policy training pipeline. |
| 3 | [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) (Song et al.) | 2026 | **The first comprehensive OPD survey.** Unified f-divergence framework, 3-axis taxonomy, industrial analysis. |
| 4 | [DeepSeek-R1: Incentivizing Reasoning Capability](https://arxiv.org/abs/2501.12948) (DeepSeek) | 2025 | Landmark reasoning model. Demonstrates that distilling reasoning traces from R1 to small models outperforms direct RL. |
| 5 | [Rethinking OPD: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016) (Li et al.) | 2026 | Deep mechanistic analysis of when/why OPD succeeds or fails. Identifies compatibility conditions. |
| 6 | [Self-Distilled Reasoner: OPSD](https://arxiv.org/abs/2601.18734) (Zhao et al.) | 2026 | Teacher-free OPD using privileged context — single model as both teacher and student. |
| 7 | [DistiLLM: Streamlined Distillation](https://arxiv.org/abs/2402.03898) (Ko et al.) | 2024 | Skewed KL + adaptive off-policy for 4.3x speedup. Practical efficiency improvements. |
| 8 | [KDRL: Unified KD + RL Post-Training](https://arxiv.org/abs/2506.02208) (Xu et al.) | 2025 | Unified framework combining GRPO with teacher KL supervision. Key OPD+RL hybrid. |
| 9 | [GAD: Black-Box On-Policy Distillation](https://arxiv.org/abs/2511.10643) (Ye et al.) | 2025 | Adversarial OPD for black-box teachers. Student-as-generator, discriminator-as-reward. |
| 10 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al.) | 2015 | **The foundational KD paper.** Soft targets, temperature scaling — everything builds on this. |

---

## Detailed Paper Catalog

### 1. Foundational Works

---

#### Distilling the Knowledge in a Neural Network
- **Year:** 2015
- **Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)
- **Link:** [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **Type:** Paper
- **Method Family:** Knowledge Distillation (original formulation)
- **Teacher Access:** White-box (soft targets / logits)
- **Supervision Signal:** KL divergence between teacher and student softmax outputs with temperature scaling
- **Rollout Source:** Teacher-generated (offline)
- **Loss/Objective:** Weighted sum of cross-entropy with hard labels and KL divergence with soft targets
- **Training Stage:** General (classification)
- **Model Family:** Various neural networks
- **Tasks/Benchmarks:** MNIST, speech recognition
- **Code:** N/A (widely reimplemented)
- **Strictness:** **Not OPD** — uses teacher outputs, no student rollouts
- **Summary:** Introduces the concept of "distillation" where a smaller student model is trained to match the softened output distribution of a larger teacher model. The soft targets provide richer gradient information than hard labels, enabling the student to capture inter-class relationships learned by the teacher.
- **Why It Matters:** The foundational paper that established the KD paradigm. All subsequent OPD work builds on its core insight that soft targets transfer more knowledge than hard labels.
- **Limitations:** Offline only; does not address distribution mismatch in autoregressive generation.

---

#### Sequence-Level Knowledge Distillation
- **Year:** 2016
- **Authors:** Yoon Kim, Alexander M. Rush (Harvard)
- **Link:** [arXiv:1606.07947](https://arxiv.org/abs/1606.07947) | [ACL Anthology](https://aclanthology.org/D16-1139/)
- **Type:** Paper (EMNLP 2016)
- **Method Family:** Sequence-level KD for NMT
- **Teacher Access:** White-box/Black-box (teacher beam search outputs)
- **Supervision Signal:** Sequence-level distribution matching
- **Rollout Source:** Teacher-generated sequences
- **Loss/Objective:** Word-level KD + sequence-level KD (training on teacher beam search outputs)
- **Training Stage:** NMT training
- **Model Family:** Seq2seq (NMT models)
- **Tasks/Benchmarks:** WMT translation
- **Code:** N/A
- **Strictness:** **Not OPD** — student trains on teacher-generated sequences, not its own rollouts
- **Summary:** Extends KD from classification to sequence generation. Proposes training students on full sequences generated by the teacher's beam search (SeqKD), which somewhat surprisingly eliminates the need for beam search at inference time.
- **Why It Matters:** First to address KD for autoregressive sequence models, directly inspiring later work on distribution mismatch in LLM distillation.
- **Limitations:** Teacher-generated data creates distribution mismatch with student's own generation behavior.

---

#### Born Again Neural Networks
- **Year:** 2018
- **Authors:** Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar
- **Link:** [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- **Type:** Paper (ICML 2018)
- **Method Family:** Self-distillation
- **Teacher Access:** White-box (same-architecture teacher)
- **Supervision Signal:** Soft targets from teacher of identical architecture
- **Rollout Source:** N/A (classification, not generation)
- **Loss/Objective:** KD loss with identical student-teacher architectures
- **Training Stage:** Training from scratch
- **Model Family:** DenseNets, ResNets
- **Tasks/Benchmarks:** CIFAR-10, CIFAR-100, language modeling
- **Code:** [GitHub (unofficial)](https://github.com/nocotan/born_again_neuralnet)
- **Strictness:** **Not OPD** — classification setting, no autoregressive generation
- **Summary:** Demonstrates that distilling a model into a student of identical architecture (Born-Again Networks) surprisingly improves performance. Multiple generations of self-distillation yield cumulative gains.
- **Why It Matters:** Established that self-distillation can improve models even without compression, directly inspiring iterative self-distillation approaches in LLMs.
- **Limitations:** Classification-focused; language modeling results were secondary.

---

#### DAgger: A Reduction of Imitation Learning to No-Regret Online Learning
- **Year:** 2011
- **Authors:** Stephane Ross, Geoffrey Gordon, Drew Bagnell (CMU)
- **Link:** [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) | [PMLR](https://proceedings.mlr.press/v15/ross11a.html)
- **Type:** Paper (AISTATS 2011)
- **Method Family:** On-policy imitation learning
- **Teacher Access:** Oracle expert providing corrective actions
- **Supervision Signal:** Expert labels on learner-visited states
- **Rollout Source:** Student/learner policy rollouts
- **Loss/Objective:** Supervised learning on aggregated dataset
- **Training Stage:** Imitation learning
- **Model Family:** General policies
- **Tasks/Benchmarks:** SuperTux, Mario Bros
- **Code:** Multiple reimplementations
- **Strictness:** **Adjacent** — on-policy imitation in RL, not LLM distillation, but the key conceptual ancestor
- **Summary:** Proposes Dataset Aggregation (DAgger), where the learner rolls out its own policy, the expert provides corrections, and the dataset is aggregated over iterations. Provably addresses covariate shift in imitation learning.
- **Why It Matters:** The conceptual foundation of on-policy distillation — the student learns from its own mistakes corrected by an expert. GKD and MiniLLM are direct descendants of this idea.
- **Limitations:** Requires an interactive expert; designed for RL, not language generation.

---

#### STaR: Self-Taught Reasoner
- **Year:** 2022
- **Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah Goodman (Stanford, Google)
- **Link:** [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- **Type:** Paper (NeurIPS 2022)
- **Method Family:** Iterative self-training for reasoning
- **Teacher Access:** Self (with rationalization using correct answers)
- **Supervision Signal:** Binary correctness filter + rationalization with correct answer
- **Rollout Source:** Student-generated rationales
- **Loss/Objective:** SFT on rationales that yield correct answers
- **Training Stage:** Post-training / fine-tuning
- **Model Family:** GPT-J
- **Tasks/Benchmarks:** Arithmetic, math word problems, CommonsenseQA
- **Code:** [GitHub](https://github.com/ezelikman/STaR)
- **Strictness:** **Partial OPD** — student generates rollouts filtered by correctness, but supervision is self-generated + answer-conditioned rationalization rather than an external teacher
- **Summary:** Iteratively generates chain-of-thought rationales, filters by correctness, and fine-tunes on correct ones. Introduces "rationalization" where the model re-generates rationales conditioned on the correct answer for problems it initially failed.
- **Why It Matters:** Pioneered iterative self-improvement through self-generated reasoning traces. Directly inspired ReST, ReST^EM, and subsequent reasoning model training approaches.
- **Limitations:** Requires initial above-chance reasoning ability; smaller models (GPT-2) cannot bootstrap.

---

#### Constitutional AI: Harmlessness from AI Feedback
- **Year:** 2022
- **Authors:** Yuntao Bai et al. (Anthropic)
- **Link:** [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **Type:** Paper
- **Method Family:** RLAIF (RL from AI Feedback)
- **Teacher Access:** Self (model critiques/revises its own outputs) + constitutional principles
- **Supervision Signal:** AI-generated preference judgments based on constitutional principles
- **Rollout Source:** Student-generated responses + self-critiques
- **Loss/Objective:** SFT on revised responses + RLHF with AI preference model
- **Training Stage:** Post-training (SFT + RL)
- **Model Family:** Anthropic models (~52B)
- **Tasks/Benchmarks:** Helpfulness, harmlessness evaluation
- **Code:** N/A (proprietary)
- **Strictness:** **Partial OPD** — student generates outputs, but the "teacher" is itself (self-critique), and preference model is trained from self-judgments rather than an external teacher
- **Summary:** Two-phase training: (1) SL phase where the model self-critiques and revises responses according to constitutional principles, then (2) RLAIF where the model evaluates pairs of its own outputs to train a preference model used for RL. Eliminates need for human harmlessness labels.
- **Why It Matters:** Introduced RLAIF concept — the model providing its own supervision — which is a form of self-distillation. Foundational for teacher-free OPD approaches.
- **Limitations:** Relies on model being capable enough for meaningful self-critique; may amplify biases.

---

### 2. Core OPD Methods (2023-2026)

---

#### GKD: On-Policy Distillation of Language Models
- **Year:** 2023 (arXiv Jun 2023, ICLR 2024)
- **Authors:** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem (Google DeepMind)
- **Link:** [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) | [OpenReview](https://openreview.net/forum?id=3zKtaqxLhW)
- **Type:** Paper (ICLR 2024)
- **Method Family:** On-policy KD with flexible divergences
- **Teacher Access:** White-box (teacher logits)
- **Supervision Signal:** KL divergence / Reverse KL / JSD between teacher and student on student-generated outputs
- **Rollout Source:** Student (with mixture parameter λ controlling student vs. dataset ratio)
- **Loss/Objective:** Generalized f-divergence on on-policy samples: `L = (1-λ)·L_offline + λ·L_on-policy`
- **Training Stage:** Post-training (SFT / distillation)
- **Model Family:** T5 (small to XL), PaLM
- **Tasks/Benchmarks:** Summarization, translation, arithmetic reasoning, instruction tuning
- **Code:** [HuggingFace TRL GKDTrainer](https://huggingface.co/docs/trl/gkd_trainer)
- **Reproducibility:** High (implemented in TRL)
- **Strictness:** **Strict OPD**
- **Summary:** The canonical OPD paper. Formalizes on-policy distillation through a mixture sampling policy π_mix that interpolates between the dataset and the student's own generations. Demonstrates that training on student-generated outputs with teacher feedback addresses the train-test distribution mismatch in autoregressive KD. Shows reverse KL and JSD substantially outperform forward KL.
- **Why It Matters:** Established the theoretical and practical framework for all subsequent OPD work. Nearly matched PaLM-540B performance with a 7000x smaller model (T5-small). First to show seamless integration of distillation with RLHF.
- **Limitations:** Requires teacher to fit on GPU for on-policy teacher scoring; increased training cost from student generation.

---

#### MiniLLM: Knowledge Distillation of Large Language Models
- **Year:** 2023 (arXiv Jun 2023, ICLR 2024)
- **Authors:** Yuxian Gu et al. (Tsinghua, Microsoft)
- **Link:** [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | [OpenReview](https://openreview.net/forum?id=5h0qf7IBZZ)
- **Type:** Paper (ICLR 2024)
- **Method Family:** Reverse KL minimization with on-policy optimization
- **Teacher Access:** White-box (teacher logits)
- **Supervision Signal:** Reverse KL divergence (student → teacher)
- **Rollout Source:** Student-generated completions
- **Loss/Objective:** Minimizes reverse KLD with an RLHF-style on-policy optimization approach
- **Training Stage:** Post-training (instruction following)
- **Model Family:** GPT-2 (120M-1.5B), GPT-J (6B), OPT (1.3B-13B)
- **Tasks/Benchmarks:** Instruction following (Dolly, Self-Instruct, Vicuna, S-NI)
- **Code:** [GitHub (microsoft/LMOps)](https://github.com/microsoft/LMOps/tree/main/minillm) | [TRL MiniLLMTrainer](https://huggingface.co/docs/trl/main/minillm)
- **Reproducibility:** High
- **Strictness:** **Strict OPD**
- **Summary:** Argues that forward KL forces the student to overestimate low-probability regions (mode-covering), making it suboptimal for generation. Proposes minimizing reverse KL with an RLHF-like on-policy pipeline where the student generates, the teacher scores, and the student updates. Produces more precise responses with less exposure bias.
- **Why It Matters:** Demonstrated that reverse KL is superior to forward KL for LLM distillation, a finding confirmed by most subsequent work. Showed OPD scales from 120M to 13B parameters.
- **Limitations:** Higher computational cost than offline KD; reverse KL can be mode-seeking (missing some valid outputs).

---

#### DistiLLM: Towards Streamlined Distillation for Large Language Models
- **Year:** 2024 (arXiv Feb 2024, ICML 2024)
- **Authors:** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun (KAIST, Microsoft)
- **Link:** [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)
- **Type:** Paper (ICML 2024)
- **Method Family:** Skewed KL with adaptive off-policy
- **Teacher Access:** White-box (teacher logits)
- **Supervision Signal:** Skewed KL divergence
- **Rollout Source:** Adaptive mixture of student and dataset (reduces student sampling cost)
- **Loss/Objective:** Skewed KL divergence with theoretical properties for stability
- **Training Stage:** Post-training
- **Model Family:** GPT-2, OPT, OpenLLaMA
- **Tasks/Benchmarks:** Instruction following, downstream NLP
- **Code:** [GitHub](https://github.com/jongwooko/distillm)
- **Reproducibility:** High
- **Strictness:** **Partial OPD** — uses adaptive off-policy approach to reduce cost, not purely on-policy
- **Summary:** Addresses the computational overhead of on-policy KD by introducing a skewed KL divergence loss with favorable theoretical properties and an adaptive off-policy approach that achieves 4.3x speedup over MiniLLM while maintaining comparable quality.
- **Why It Matters:** Made on-policy distillation practical at scale by dramatically reducing computational costs. The skewed KL provides a principled middle ground between forward and reverse KL.
- **Limitations:** Adaptive off-policy is not truly on-policy; some distribution mismatch remains.

---

#### f-DISTILL: f-Divergence Minimization for Sequence-Level KD
- **Year:** 2023 (ACL 2023)
- **Authors:** Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou
- **Link:** [arXiv:2307.15190](https://arxiv.org/abs/2307.15190) | [ACL Anthology](https://aclanthology.org/2023.acl-long.605/)
- **Type:** Paper (ACL 2023)
- **Method Family:** f-divergence framework for sequence-level KD
- **Teacher Access:** White-box
- **Supervision Signal:** General f-divergence (total variation, Jensen-Shannon, etc.)
- **Rollout Source:** Mixed (step-wise decomposition)
- **Loss/Objective:** f-divergence minimization with tractable step-wise decomposition
- **Training Stage:** Post-training
- **Model Family:** Transformer (NMT, summarization)
- **Tasks/Benchmarks:** WMT translation, summarization, paraphrase, dialogue
- **Code:** N/A
- **Strictness:** **Partial OPD** — step-wise decomposition approximates on-policy behavior
- **Summary:** Formulates sequence-level KD as minimizing a generalized f-divergence. Derives tractable word-level losses from intractable sequence-level divergences. Shows symmetric divergences (JSD, TVD) outperform asymmetric ones (KL, RKL).
- **Why It Matters:** Provided the theoretical foundation for the unified f-divergence framework later adopted by the OPD survey.
- **Limitations:** Pre-LLM scale experiments; tractable decomposition involves approximations.

---

#### ToDi: Token-wise Distillation via Fine-Grained Divergence Control
- **Year:** 2025 (EMNLP 2025)
- **Authors:** Seongryong Jung, Suwan Yoon, DongGeon Kim, Hwanhee Lee
- **Link:** [arXiv:2505.16297](https://arxiv.org/abs/2505.16297) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.409/)
- **Type:** Paper (EMNLP 2025)
- **Method Family:** Token-level adaptive divergence
- **Teacher Access:** White-box
- **Supervision Signal:** Per-token adaptive combination of Forward KL and Reverse KL
- **Rollout Source:** Offline (dataset), but technique is applicable to on-policy
- **Loss/Objective:** Sigmoid-weighted blend of FKL and RKL based on log-ratio of teacher/student probabilities
- **Training Stage:** Post-training
- **Model Family:** GPT-2, TinyLLaMA, LLaMA-2
- **Tasks/Benchmarks:** Instruction following, downstream NLP
- **Code:** N/A
- **Strictness:** **Adjacent** — primarily offline but the per-token adaptive mechanism directly inspired EOPD and other on-policy variants
- **Summary:** Reveals through gradient analysis that FKL boosts underestimated tokens while RKL suppresses overestimated ones, showing their complementary roles. Proposes per-token sigmoid-weighted combination with O(V) complexity.
- **Why It Matters:** Established that uniform divergence across all tokens is suboptimal, inspiring per-token adaptive methods in OPD.
- **Limitations:** Not itself on-policy; improvements are modest on some benchmarks.

---

#### EOPD: Entropy-Aware On-Policy Distillation
- **Year:** 2026 (arXiv Mar 2026)
- **Authors:** Jin et al.
- **Link:** [arXiv:2603.07079](https://arxiv.org/abs/2603.07079)
- **Type:** Paper
- **Method Family:** Entropy-adaptive on-policy KD
- **Teacher Access:** White-box
- **Supervision Signal:** Entropy-adaptive blend of reverse KL + forward KL
- **Rollout Source:** Student on-policy
- **Loss/Objective:** Reverse KL by default, augmented with forward KL on high-entropy tokens
- **Training Stage:** Post-training
- **Model Family:** Qwen2.5, LLaMA-3
- **Tasks/Benchmarks:** GSM8K, MATH, AIME, out-of-domain transfer
- **Code:** N/A
- **Strictness:** **Strict OPD**
- **Summary:** Argues that pure reverse-KL OPD becomes brittle when the teacher distribution has high entropy. Augments reverse KL with forward KL on high-entropy tokens to preserve diversity. Achieves significantly higher Pass@k scores, especially on hard benchmarks.
- **Why It Matters:** Addresses a fundamental failure mode of reverse-KL OPD and shows strong out-of-domain transfer.
- **Limitations:** Requires entropy estimation; additional hyperparameter for threshold.

---

#### Rethinking On-Policy Distillation: Phenomenology, Mechanism, and Recipe
- **Year:** 2026 (arXiv Apr 2026)
- **Authors:** Yaxuan Li et al. (THUNLP, Tsinghua)
- **Link:** [arXiv:2604.13016](https://arxiv.org/abs/2604.13016) | [GitHub](https://github.com/thunlp/OPD)
- **Type:** Paper
- **Method Family:** Mechanistic analysis of OPD
- **Teacher Access:** White-box
- **Supervision Signal:** Token-level KL divergence
- **Rollout Source:** Student on-policy
- **Loss/Objective:** Standard on-policy KD objectives with proposed fixes
- **Training Stage:** Post-training
- **Model Family:** Various
- **Tasks/Benchmarks:** Math reasoning, instruction following
- **Code:** [GitHub](https://github.com/thunlp/OPD)
- **Strictness:** **Strict OPD** (analysis paper)
- **Summary:** Provides the first systematic investigation of when and why OPD fails. Identifies two necessary conditions: (1) compatible thinking patterns between student and teacher, (2) teacher must offer genuinely new capabilities. Shows successful OPD is characterized by progressive alignment on high-probability tokens. Proposes off-policy cold start and teacher-aligned prompt selection as practical fixes.
- **Why It Matters:** Critical for practitioners — explains failure modes that were previously mysterious and provides actionable remedies.
- **Limitations:** Analysis is primarily on math reasoning; may not fully generalize.

---

#### A Survey of On-Policy Distillation for Large Language Models
- **Year:** 2026 (arXiv Apr 2026)
- **Authors:** Mingyang Song et al.
- **Link:** [arXiv:2604.00626](https://arxiv.org/abs/2604.00626)
- **Type:** Survey paper
- **Method Family:** Survey / meta-analysis
- **Strictness:** N/A (survey)
- **Summary:** The first comprehensive OPD survey. Introduces a unified f-divergence framework and organizes the landscape along three dimensions: feedback signal (logit-based, outcome-based, self-play), teacher access (white-box, black-box, teacher-free), and loss granularity (token-level, sequence-level, hybrid). Systematically covers industrial deployments and identifies open problems.
- **Why It Matters:** Essential reference for anyone entering the field. Provides the definitive taxonomy and identifies key open research directions.

---

#### OPSD: Self-Distilled Reasoner
- **Year:** 2026 (arXiv Jan 2026)
- **Authors:** Siyan Zhao et al.
- **Link:** [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
- **Type:** Paper
- **Method Family:** On-policy self-distillation (teacher-free)
- **Teacher Access:** Self (privileged context — ground-truth solution)
- **Supervision Signal:** KL divergence between privileged teacher policy and standard student policy
- **Rollout Source:** Student on-policy (standard context, no solution)
- **Loss/Objective:** Reverse KL between teacher (with privileged info) and student (without)
- **Training Stage:** Post-training (math reasoning)
- **Model Family:** Various
- **Tasks/Benchmarks:** Math reasoning benchmarks
- **Code:** [Project page](https://siyan-zhao.github.io/blog/2026/opsd/)
- **Strictness:** **Strict OPD** — student generates rollouts; teacher is the same model with privileged context
- **Summary:** Proposes that a single LLM can serve as both teacher and student: the teacher sees the ground-truth solution (privileged context) while the student sees only the problem. The model's in-context learning ability lets it rationalize solutions when given privileged info, creating a natural teacher-student gap without separate models.
- **Why It Matters:** Eliminates the need for a separate larger teacher, making OPD accessible without multi-model infrastructure. Demonstrates superior token efficiency compared to RL methods.
- **Limitations:** Requires the model to be sufficiently capable for privileged context to help; may not work for very small models.

---

#### OPSDL: On-Policy Self-Distillation for Long-Context LLMs
- **Year:** 2026 (arXiv Apr 2026)
- **Authors:** Xinsen Zhang et al. (Baidu)
- **Link:** [arXiv:2604.17535](https://arxiv.org/abs/2604.17535)
- **Type:** Paper
- **Method Family:** Self-distillation for long-context extension
- **Teacher Access:** Self (short-context capability as teacher)
- **Supervision Signal:** Token-level reverse KL between short-context "teacher" and long-context "student"
- **Rollout Source:** Student on-policy (long-context generation)
- **Loss/Objective:** Reverse KL divergence on-policy
- **Training Stage:** Post-training (long-context)
- **Model Family:** Qwen2.5-7B to 32B
- **Tasks/Benchmarks:** Long-context benchmarks (varying lengths)
- **Code:** N/A
- **Strictness:** **Strict OPD** — student generates in long-context; short-context self provides teacher signal
- **Summary:** Leverages the asymmetry between a model's strong short-context and weaker long-context abilities. Distills the model's own short-context behavior into its long-context behavior via on-policy training, without external teachers or reward models.
- **Why It Matters:** Novel application of self-distillation to long-context extension. Achieves performance comparable to models explicitly pre-trained for million-token contexts.
- **Limitations:** Assumes short-context performance is reliable; may not help if short-context is also weak.

---

### 3. OPD + RL/RLVR Hybrids

---

#### ReST: Reinforced Self-Training for Language Modeling
- **Year:** 2023 (arXiv Aug 2023)
- **Authors:** Caglar Gulcehre et al. (Google DeepMind)
- **Link:** [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- **Type:** Paper
- **Method Family:** Iterative on-policy generation + reward filtering
- **Teacher Access:** Reward model (trained on human preferences)
- **Supervision Signal:** Reward model scores for filtering
- **Rollout Source:** Student/policy generates, then filtered by reward
- **Loss/Objective:** Offline RL on reward-filtered student-generated data
- **Training Stage:** Post-training
- **Model Family:** Various (demonstrated on translation)
- **Tasks/Benchmarks:** Machine translation
- **Code:** N/A
- **Strictness:** **Strict OPD** — student generates, reward model provides supervision signal
- **Summary:** Two-loop algorithm: "Grow" (generate from policy, filter by reward) and "Improve" (fine-tune on filtered data). More efficient than online RLHF because data is generated offline and reused.
- **Why It Matters:** Bridged self-training and RLHF. Directly inspired ReST^EM and became a template for iterative OPD.
- **Limitations:** Fixed reward model may become stale; diminishing returns after few iterations.

---

#### ReST^EM: Beyond Human Data — Scaling Self-Training
- **Year:** 2023/2024 (arXiv Dec 2023, TMLR 2024)
- **Authors:** Avi Singh, John D. Co-Reyes, Rishabh Agarwal et al. (Google)
- **Link:** [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
- **Type:** Paper
- **Method Family:** Iterative self-training with EM
- **Teacher Access:** Verifier (correctness check for math/code)
- **Supervision Signal:** Binary correctness filter
- **Rollout Source:** Student generates multiple solutions per problem
- **Loss/Objective:** SFT on correct solutions (E-step: generate & filter; M-step: fine-tune)
- **Training Stage:** Post-training
- **Model Family:** PaLM-2
- **Tasks/Benchmarks:** MATH, APPS (code), GSM8K, HumanEval
- **Code:** N/A
- **Strictness:** **Partial OPD** — student generates, but supervision is binary correctness rather than teacher distribution
- **Summary:** EM-based self-training: generate solutions from current policy, filter correct ones, fine-tune, repeat. Scales favorably with model size and significantly surpasses fine-tuning on human data alone.
- **Why It Matters:** Demonstrated that self-training with verifiable rewards can surpass human data quality. Key precursor to RLVR approaches.
- **Limitations:** Only applicable to tasks with verifiable answers; diminishing returns after 2-3 iterations.

---

#### RFT: Rejection Sampling Fine-Tuning
- **Year:** 2023 (arXiv Aug 2023)
- **Authors:** Hongyi Yuan et al.
- **Link:** [arXiv:2308.01825](https://arxiv.org/abs/2308.01825)
- **Type:** Paper
- **Method Family:** Rejection sampling with correctness filtering
- **Teacher Access:** Verifier (answer correctness)
- **Supervision Signal:** Binary correctness + diversity of reasoning paths
- **Rollout Source:** Student/supervised model generates multiple solutions
- **Loss/Objective:** SFT on correct, diverse reasoning paths
- **Training Stage:** Post-training (math reasoning)
- **Model Family:** LLaMA-7B/13B
- **Tasks/Benchmarks:** GSM8K, MATH
- **Code:** N/A
- **Strictness:** **Partial OPD** — student generates rollouts filtered by correctness, but no teacher distribution matching
- **Summary:** Generates multiple reasoning paths from the model, filters by correctness, and fine-tunes on correct ones. Key finding: the number of distinct reasoning paths matters more than total samples. Combining rejection samples from multiple models pushes LLaMA-7B to 49.3% on GSM8K (vs. 35.9% SFT).
- **Why It Matters:** Established rejection sampling as a simple, effective OPD-adjacent method for math reasoning.
- **Limitations:** Only works for verifiable tasks; diversity of reasoning paths is crucial but hard to control.

---

#### RAFT: Reward rAnked FineTuning
- **Year:** 2023 (arXiv Apr 2023, TMLR 2023)
- **Authors:** Hanze Dong, Wei Xiong et al.
- **Link:** [arXiv:2304.06767](https://arxiv.org/abs/2304.06767) | [GitHub](https://github.com/RLHFlow/RAFT)
- **Type:** Paper (TMLR)
- **Method Family:** Iterative reward-ranked fine-tuning
- **Teacher Access:** Reward model
- **Supervision Signal:** Reward model ranking of student-generated outputs
- **Rollout Source:** Student generates multiple candidates per prompt
- **Loss/Objective:** SFT on top-ranked responses
- **Training Stage:** Post-training (alignment)
- **Model Family:** LLaMA-7B
- **Tasks/Benchmarks:** HH-RLHF
- **Code:** [GitHub](https://github.com/RLHFlow/RAFT)
- **Strictness:** **Partial OPD** — student generates, reward model ranks, but no teacher distribution matching
- **Summary:** Iterative best-of-n fine-tuning: generate candidates, rank by reward model, fine-tune on best. Decouples data generation from model updates (off-policy), enabling memory-efficient training.
- **Why It Matters:** Simplified RLHF into an intuitive, stable alternative. Demonstrated robust reward maximization with low perplexity degradation.
- **Limitations:** Reward model quality is a bottleneck; no distribution-level matching.

---

#### RLKD: Distilling LLMs' Reasoning via Reinforcement Learning
- **Year:** 2025 (AAAI 2025)
- **Authors:** Shicheng Xu, Liang Pang et al.
- **Link:** [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40710)
- **Type:** Paper (AAAI 2025)
- **Method Family:** RL-based KD with structure-aware reward
- **Teacher Access:** White-box (teacher reasoning traces)
- **Supervision Signal:** Generative Structure Reward Model (GSRM) measuring alignment of reasoning structure
- **Rollout Source:** Student on-policy (via GRPO)
- **Loss/Objective:** GRPO with combined GSRM reward + task outcome reward
- **Training Stage:** Post-training (reasoning)
- **Model Family:** Qwen2.5-Math-7B
- **Tasks/Benchmarks:** Math reasoning (OpenR1-Math)
- **Code:** N/A
- **Strictness:** **Strict OPD** — student generates via GRPO, teacher's reasoning structure provides reward signal
- **Summary:** First RL-based KD method for reasoning. Introduces GSRM that converts reasoning paths into meta-reasoning steps and rewards structural alignment between student and teacher. Even with 0.1% of data under RL-only, surpasses standard SFT-RL pipelines.
- **Why It Matters:** Showed that reasoning structure transfer requires RL, not just supervised mimicry. SFT collapses the multi-branch reasoning structure.
- **Limitations:** GSRM adds complexity; sensitive to reward balancing.

---

#### KDRL: Post-Training Reasoning LLMs via Unified KD + RL
- **Year:** 2025 (arXiv Jun 2025)
- **Authors:** Hongling Xu, Qi Zhu et al.
- **Link:** [arXiv:2506.02208](https://arxiv.org/abs/2506.02208)
- **Type:** Paper
- **Method Family:** Unified KD + RL objective
- **Teacher Access:** White-box (teacher logits)
- **Supervision Signal:** Reverse KL to teacher + rule-based outcome rewards
- **Rollout Source:** Student on-policy
- **Loss/Objective:** Joint GRPO + reverse KL minimization (policy gradient for both objectives)
- **Training Stage:** Post-training (reasoning)
- **Model Family:** Qwen2.5-Math (1.5B-7B)
- **Tasks/Benchmarks:** Math reasoning benchmarks
- **Code:** N/A
- **Strictness:** **Strict OPD**
- **Summary:** Unifies KD and RL into a single policy gradient objective that simultaneously minimizes teacher-student KL divergence and maximizes outcome rewards. Systematically explores KL approximations, coefficients, and reward-guided KD strategies.
- **Why It Matters:** First principled unification of KD and RL objectives for LLM post-training. Achieves better accuracy-efficiency tradeoff than either alone.
- **Limitations:** KD and RL objectives can compete; sensitive to weighting hyperparameters.

---

#### RLAD: Reinforcement-Aware Knowledge Distillation
- **Year:** 2026 (arXiv Feb 2026)
- **Authors:** Zhaoyang Zhang et al.
- **Link:** [arXiv:2602.22495](https://arxiv.org/abs/2602.22495)
- **Type:** Paper
- **Method Family:** Trust-region ratio distillation during RL
- **Teacher Access:** White-box (teacher logits)
- **Supervision Signal:** Advantage-aware, trust-region-bounded teacher imitation
- **Rollout Source:** Student on-policy
- **Loss/Objective:** Trust Region Ratio Distillation (TRRD) — PPO/GRPO-style likelihood ratio anchored to teacher-old-policy mixture
- **Training Stage:** Post-training (RL with distillation)
- **Model Family:** Qwen3 (0.6B-7B), DeepSeek-R1
- **Tasks/Benchmarks:** Math reasoning, logic reasoning (K&K Logistics)
- **Code:** N/A
- **Strictness:** **Strict OPD**
- **Summary:** Addresses the objective interference problem of KDRL by replacing the KL regularizer with a trust-region ratio objective that selectively imitates the teacher only when it improves the current policy update. Naturally balances exploration, exploitation, and imitation.
- **Why It Matters:** Solves the practical challenge of balancing KD and RL objectives. Consistently outperforms KDRL and GRPO across diverse benchmarks.
- **Limitations:** More complex implementation than simple KL regularization.

---

#### PRIME: Process Reinforcement through Implicit Rewards
- **Year:** 2025 (arXiv Feb 2025)
- **Authors:** Sun et al.
- **Link:** [arXiv:2502.01456](https://arxiv.org/abs/2502.01456) | [GitHub](https://github.com/PRIME-RL/PRIME)
- **Type:** Paper
- **Method Family:** Implicit process rewards for on-policy RL
- **Teacher Access:** Implicit PRM (updated online with outcome labels)
- **Supervision Signal:** Dense process rewards derived from implicit PRM
- **Rollout Source:** Student on-policy
- **Loss/Objective:** RL with implicit process rewards
- **Training Stage:** Post-training (reasoning)
- **Model Family:** Qwen2.5-Math-7B
- **Tasks/Benchmarks:** Math reasoning benchmarks
- **Code:** [GitHub](https://github.com/PRIME-RL/PRIME)
- **Strictness:** **Adjacent** — on-policy RL with process rewards, not teacher-based distillation
- **Summary:** Enables online PRM updates using only policy rollouts and outcome labels. The implicit PRM provides dense process rewards without expensive human step-level annotations, mitigating distribution shift issues.
- **Why It Matters:** Solved the stale-PRM problem in process-supervised RL. 15.1% average improvement over SFT on reasoning benchmarks.
- **Limitations:** Requires outcome verifiers; implicit rewards may be noisy.

---

### 4. Iterative Self-Distillation & Self-Play

---

#### SPIN: Self-Play Fine-Tuning
- **Year:** 2024 (arXiv Jan 2024)
- **Authors:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA)
- **Link:** [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) | [GitHub](https://github.com/uclaml/SPIN)
- **Type:** Paper
- **Method Family:** Self-play fine-tuning (no external teacher)
- **Teacher Access:** None (self-play against previous iteration)
- **Supervision Signal:** Discriminate self-generated responses from human references
- **Rollout Source:** Student generates (previous iteration outputs)
- **Loss/Objective:** DPO-style objective distinguishing self vs. human responses
- **Training Stage:** Post-training
- **Model Family:** Mistral-7B (zephyr-7b-sft)
- **Tasks/Benchmarks:** HuggingFace Open LLM Leaderboard, MT-Bench, Big-Bench
- **Code:** [GitHub](https://github.com/uclaml/SPIN)
- **Strictness:** **Adjacent** — student generates, but "teacher" is just the human reference data (no external teacher)
- **Summary:** Self-play mechanism where the LLM refines by distinguishing its own generations from human-annotated data. Theoretically proves convergence when the policy matches the target distribution. Outperforms DPO with extra GPT-4 data.
- **Why It Matters:** Showed that self-improvement through self-play can eliminate the need for external preference data. Strong theoretical foundations.
- **Limitations:** Limited by quality of initial SFT data; diminishing returns after 3 iterations.

---

#### Self-Rewarding Language Models
- **Year:** 2024 (arXiv Jan 2024)
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang et al. (Meta)
- **Link:** [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
- **Type:** Paper
- **Method Family:** Iterative self-improvement with LLM-as-a-Judge
- **Teacher Access:** Self (LLM-as-a-Judge scoring its own outputs)
- **Supervision Signal:** Self-generated reward scores via LLM-as-a-Judge prompting
- **Rollout Source:** Student generates both responses and rewards
- **Loss/Objective:** Iterative DPO using self-generated preference pairs
- **Training Stage:** Post-training (iterative alignment)
- **Model Family:** LLaMA-2 70B
- **Tasks/Benchmarks:** AlpacaEval 2.0
- **Code:** N/A
- **Strictness:** **Partial OPD** — student generates rollouts and provides its own supervision, but no external teacher
- **Summary:** The model generates responses, scores them via LLM-as-a-Judge prompting, constructs preference pairs, and trains with DPO. Both instruction-following and reward-modeling abilities improve across iterations. Outperforms Claude 2, Gemini Pro, and GPT-4 0613 on AlpacaEval.
- **Why It Matters:** Demonstrated the possibility of continual self-improvement without external feedback. Both the policy and reward model improve simultaneously.
- **Limitations:** Potential for reward hacking; self-evaluation biases may accumulate.

---

#### Weak-to-Strong Generalization
- **Year:** 2023 (arXiv Dec 2023)
- **Authors:** Collin Burns, Pavel Izmailov et al. (OpenAI)
- **Link:** [arXiv:2312.09390](https://arxiv.org/abs/2312.09390)
- **Type:** Paper
- **Method Family:** Inverse distillation (weak teacher, strong student)
- **Teacher Access:** Weak model labels (opposite of traditional KD)
- **Supervision Signal:** Labels from a smaller, weaker model
- **Rollout Source:** N/A (classification setting)
- **Loss/Objective:** Fine-tuning strong model on weak model labels + confidence-boosting methods
- **Training Stage:** Fine-tuning
- **Model Family:** GPT-4 family
- **Tasks/Benchmarks:** NLP classification, chess, reward modeling
- **Code:** N/A
- **Strictness:** **Adjacent** — inverse distillation direction; strong student generalizes beyond weak teacher
- **Summary:** Studies whether weak model supervision can elicit strong model capabilities. Finds that strong models consistently perform better than their weak supervisors — a phenomenon they call "weak-to-strong generalization." GPT-2 can elicit most of GPT-4's capabilities.
- **Why It Matters:** Addresses the fundamental alignment problem of supervising superhuman models. Suggests that distillation dynamics are richer than simple teacher-to-student transfer.
- **Limitations:** Classification-focused; NLG generalization untested; disanalogy with superhuman alignment.

---

### 5. Black-Box OPD

---

#### GAD: Generative Adversarial Distillation (Black-Box On-Policy Distillation)
- **Year:** 2025 (arXiv Nov 2025)
- **Authors:** Tianzhu Ye, Li Dong et al. (Microsoft Research)
- **Link:** [arXiv:2511.10643](https://arxiv.org/abs/2511.10643) | [GitHub](https://github.com/microsoft/LMOps/tree/main/gad)
- **Type:** Paper
- **Method Family:** Adversarial on-policy distillation (black-box)
- **Teacher Access:** Black-box (text outputs only)
- **Supervision Signal:** Discriminator reward distinguishing student vs. teacher outputs
- **Rollout Source:** Student generates (generator in GAN framework)
- **Loss/Objective:** Minimax adversarial game between student (generator) and discriminator
- **Training Stage:** Post-training
- **Model Family:** Qwen2.5 (3B-14B), LLaMA-3 (3B-8B)
- **Tasks/Benchmarks:** LMSYS-Chat, MT-Bench
- **Code:** [GitHub](https://github.com/microsoft/LMOps/tree/main/gad) | [HuggingFace models](https://huggingface.co/papers/2511.10643)
- **Strictness:** **Strict OPD** — student generates, co-evolving discriminator provides on-policy reward
- **Summary:** Frames student as generator and trains a discriminator to distinguish student outputs from teacher (GPT-5-Chat) outputs. The discriminator serves as an on-policy reward model that co-evolves with the student, providing adaptive feedback without needing teacher logits.
- **Why It Matters:** First effective method for on-policy distillation from proprietary black-box teachers. Qwen2.5-14B matches GPT-5-Chat on LMSYS-Chat via GAD.
- **Limitations:** Requires discriminator warmup; GAN training instability applies; discriminator adds compute cost.

---

#### Zephyr: Direct Distillation of LM Alignment
- **Year:** 2023 (arXiv Oct 2023)
- **Authors:** Lewis Tunstall, Edward Beeching et al. (HuggingFace)
- **Link:** [arXiv:2310.16944](https://arxiv.org/abs/2310.16944) | [HuggingFace](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- **Type:** Paper / Technical Report
- **Method Family:** Distilled SFT + Distilled DPO (dSFT + dDPO)
- **Teacher Access:** Black-box (ChatGPT for SFT data, GPT-4 for preference ratings)
- **Supervision Signal:** Teacher-generated SFT data + AI preference feedback
- **Rollout Source:** Teacher generates (offline distillation)
- **Loss/Objective:** SFT on teacher data + DPO on AI-ranked preferences
- **Training Stage:** Post-training (SFT + DPO)
- **Model Family:** Mistral-7B
- **Tasks/Benchmarks:** MT-Bench, AlpacaEval
- **Code:** [GitHub (alignment-handbook)](https://github.com/huggingface/alignment-handbook)
- **Strictness:** **Not OPD** — fully offline; student trains on teacher-generated data. Included for context as a widely-adopted distillation baseline.
- **Summary:** Three-step pipeline: (1) distilled SFT on ChatGPT-generated data, (2) AI feedback using GPT-4 ratings, (3) distilled DPO. Zephyr-7B surpasses LLaMA-2-Chat-70B on MT-Bench without any human annotation.
- **Why It Matters:** Demonstrated that effective alignment distillation is possible with only AI feedback, no human labels. Widely used as a baseline for OPD comparisons.
- **Limitations:** Fully offline — suffers from distribution mismatch. Not on-policy.

---

### 6. Speculative Decoding + Distillation

---

#### DistillSpec: Improving Speculative Decoding via Knowledge Distillation
- **Year:** 2024 (ICLR 2024)
- **Authors:** Zhou et al. (Google)
- **Link:** [arXiv:2310.08461](https://arxiv.org/abs/2310.08461) | [OpenReview](https://openreview.net/forum?id=rsY6J3ZaTF)
- **Type:** Paper (ICLR 2024)
- **Method Family:** KD for speculative decoding alignment
- **Teacher Access:** White-box (target model logits)
- **Supervision Signal:** Divergence between draft and target model distributions
- **Rollout Source:** Draft model generates on-policy
- **Loss/Objective:** On-policy KD with divergence tailored to decoding strategy
- **Training Stage:** Inference optimization (training draft model)
- **Model Family:** Various LLMs
- **Tasks/Benchmarks:** Standard NLG benchmarks, latency measurements
- **Code:** N/A
- **Strictness:** **Strict OPD** — draft model generates, target model provides on-policy logit supervision
- **Summary:** Uses on-policy data generation from the draft model + tailored divergence functions to align draft models with target models for speculative decoding. Achieves 10-45% speedups over standard SD.
- **Why It Matters:** Applied OPD to inference optimization, showing its versatility beyond training quality improvement.
- **Limitations:** Focused on inference speed rather than model quality.

---

#### DVI: Draft, Verify, and Improve
- **Year:** 2025 (arXiv Oct 2025)
- **Authors:** Not specified
- **Link:** [arXiv:2510.05421](https://arxiv.org/abs/2510.05421)
- **Type:** Paper
- **Method Family:** Training-aware self-speculation
- **Teacher Access:** White-box (frozen verifier within same model)
- **Supervision Signal:** Accept/reject feedback from verifier
- **Rollout Source:** Lightweight LoRA drafter generates on-policy
- **Loss/Objective:** Online LoRA drafter update from accept/reject feedback
- **Training Stage:** Inference optimization
- **Model Family:** Single backbone split into drafter/verifier
- **Tasks/Benchmarks:** Latency benchmarks
- **Code:** N/A
- **Strictness:** **Strict OPD** — drafter generates, frozen verifier provides supervision
- **Summary:** Closes the loop between speculative decoding inference and training. A single backbone is split into shallow drafting and deep verifying paths, with the verifier frozen and a LoRA drafter updated online from accept/reject feedback.
- **Why It Matters:** Eliminates offline drafter training entirely — the drafter improves during inference.
- **Limitations:** Self-speculation within one model limits the student-teacher gap.

---

### 7. Cross-Tokenizer OPD

---

#### DSKD: Dual-Space Knowledge Distillation
- **Year:** 2024 (EMNLP 2024)
- **Authors:** Songming Zhang et al.
- **Link:** [arXiv:2406.17328](https://arxiv.org/abs/2406.17328) | [GitHub](https://github.com/songmzhang/DSKD)
- **Type:** Paper (EMNLP 2024)
- **Method Family:** Cross-tokenizer white-box KD
- **Teacher Access:** White-box (shared output space via projectors)
- **Supervision Signal:** KL divergence in unified output spaces
- **Rollout Source:** Offline (with DSKDv2 adding on-policy support)
- **Loss/Objective:** KL divergence with exact token alignment (ETA) algorithm
- **Training Stage:** Post-training
- **Model Family:** LLaMA, Qwen (different tokenizers)
- **Tasks/Benchmarks:** Instruction following, downstream NLP
- **Code:** [GitHub](https://github.com/songmzhang/DSKD)
- **Strictness:** **Partial OPD** (v2 supports on-policy)
- **Summary:** Introduces projectors to map teacher/student hidden states into each other's representation spaces, enabling white-box KD across different tokenizers. ETA algorithm aligns tokens in differently-tokenized sequences.
- **Why It Matters:** Unlocked cross-model-family distillation (e.g., LLaMA → Qwen), which was previously impossible with standard logit-level KD.
- **Limitations:** Projector initialization and training adds complexity; not fully on-policy in v1.

---

#### GOLD: General Online Logit Distillation
- **Year:** 2026 (TRL, HuggingFace)
- **Authors:** Lewis Tunstall et al. (HuggingFace), inspired by ULD (Boizard et al., 2025)
- **Link:** [HuggingFace Docs](https://huggingface.co/docs/trl/main/en/gold_trainer) | [Blog](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- **Type:** Framework / Trainer
- **Method Family:** Cross-tokenizer on-policy distillation
- **Teacher Access:** White-box (logits with tokenizer alignment)
- **Supervision Signal:** ULD loss with cross-tokenizer span alignment
- **Rollout Source:** Student on-policy (GKD-style scheduling)
- **Loss/Objective:** Hybrid ULD loss with exact matches + sorted-probability fallback
- **Training Stage:** Post-training
- **Model Family:** Any HuggingFace model pair
- **Tasks/Benchmarks:** Math reasoning, instruction following
- **Code:** [TRL (trl.experimental)](https://huggingface.co/docs/trl/main/en/gold_trainer)
- **Strictness:** **Strict OPD** (when used with on-policy scheduling)
- **Summary:** Extends ULD to support any teacher-student pair regardless of tokenizer differences. Incrementally decodes and aligns textual spans, merging logits so no tokens are dropped. Outperforms ULD and even GRPO on math reasoning.
- **Why It Matters:** Democratizes on-policy distillation — any model can be distilled into any other model, regardless of tokenizer.
- **Limitations:** Experimental API; cross-tokenizer alignment adds latency.

---

### 8. Preference-Aligned Distillation

---

#### DPKD: Direct Preference Knowledge Distillation
- **Year:** 2024 (arXiv Jun 2024)
- **Authors:** Li et al. (Microsoft)
- **Link:** [arXiv:2406.19774](https://arxiv.org/abs/2406.19774) | [Code](https://aka.ms/dpkd)
- **Type:** Paper
- **Method Family:** Preference-aligned KD
- **Teacher Access:** White-box
- **Supervision Signal:** Implicit reward from DPO + KL divergence to teacher
- **Rollout Source:** Offline (teacher outputs as preferred, student as dispreferred)
- **Loss/Objective:** Two-stage: implicit reward + reverse KL; then preference probability optimization
- **Training Stage:** Post-training (alignment)
- **Model Family:** GPT-2 (120M-1.5B), OPT
- **Tasks/Benchmarks:** Instruction following, downstream NLP
- **Code:** [GitHub](https://aka.ms/dpkd)
- **Strictness:** **Not OPD** — fully offline, uses pre-collected teacher/student outputs
- **Summary:** Bridges KD and DPO by treating teacher outputs as preferred and student outputs as dispreferred. Uses implicit LLM reward function to supplement KL divergence.
- **Why It Matters:** Connected preference alignment with distillation, opening a new research direction.
- **Limitations:** Fully offline; does not use student on-policy rollouts during training.

---

#### AlignDistil: Token-Level Alignment as Adaptive Policy Distillation
- **Year:** 2025 (ACL 2025)
- **Authors:** Songming Zhang et al.
- **Link:** [arXiv:2503.02832](https://arxiv.org/abs/2503.02832) | [GitHub](https://github.com/songmzhang/AlignDistil)
- **Type:** Paper (ACL 2025)
- **Method Family:** RLHF-equivalent distillation for token-level rewards
- **Teacher Access:** White-box (DPO model + reference model logits)
- **Supervision Signal:** Token-level reward from contrastive DPO, distilled into policy
- **Rollout Source:** Flexible (on-policy or off-policy switching)
- **Loss/Objective:** Distillation objective equivalent to token-level RLHF optimization
- **Training Stage:** Post-training (alignment)
- **Model Family:** LLaMA, Mistral
- **Tasks/Benchmarks:** AlpacaEval 2.0, MT-Bench, Arena-Hard
- **Code:** [GitHub](https://github.com/songmzhang/AlignDistil)
- **Strictness:** **Partial OPD** — supports on-policy mode, but also works off-policy
- **Summary:** Proves theoretical equivalence between token-level RLHF and a specific distillation process. Constructs teacher distribution by linearly combining DPO and reference model logits. Token-adaptive logit extrapolation prevents under/over-optimization.
- **Why It Matters:** Unified RLHF and KD into a single framework with fast convergence due to token-level rewards.
- **Limitations:** Requires a trained DPO model as intermediate step.

---

## Industrial Systems & Technical Reports

| System | Org | Year | OPD Use | Distillation Details | Link |
|---|---|---|---|---|---|
| **DeepSeek-R1** | DeepSeek | 2025 | **Offline KD** (teacher-generated 800K samples) | R1 generates long-CoT data, distilled into Qwen-1.5B to 70B via SFT. Distillation outperforms direct RL on small models. | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **DeepSeek-V3** | DeepSeek | 2024 | **Partial OPD** | Distills R1 reasoning patterns into V3 while controlling output style/length. Multi-token prediction objective. | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| **Qwen3** | Alibaba/Qwen | 2025 | **Strict OPD** | Strong-to-weak distillation for lightweight models. On-policy variant: student generates, teacher provides logit supervision via KL minimization. Four-stage post-training. | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| **DistilQwen2.5** | Alibaba/Qwen | 2025 | **Mixed (Black-box + White-box)** | Two-stage: black-box SFT from proprietary teachers, then white-box KD from Qwen2.5-72B-Instruct. CoT rewriting for reasoning. | [ACL 2025](https://arxiv.org/html/2504.15027v1) |
| **Gemma 2** | Google | 2024 | **Strict OPD** | 2B and 9B models trained with on-policy distillation: student generates completions, KL divergence minimized against teacher logits during pre-training. Up to 10% benchmark improvement. | [arXiv:2408.00118](https://arxiv.org/html/2408.00118v1) |
| **Gemma 3** | Google | 2025 | **Strict OPD** | All models (1B-27B) trained with KD, expanding from Gemma 2 where only small models used it. Advanced RL techniques (BOND, WARM, WARP) combined. | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| **LLaMA 3.1/3.2** | Meta | 2024 | **Offline KD + Pruning** | 1B/3B models created via pruning from 8B + KD using 8B/70B teacher logits during pre-training. Post-training: SFT, rejection sampling, DPO. 405B used for smaller model post-training quality improvement. | [Meta Blog](https://ai.meta.com/blog/meta-llama-3-1/) |
| **Ministral 3** | Mistral | 2026 | **Strict OPD (Cascade)** | Cascade Distillation: iteratively prune and distill from Mistral Small 3.1 (24B) → 14B → 8B → 3B. Logit distillation from parent. Post-training: SFT + ODPO + GRPO. 1-3T tokens vs. 15-36T for training from scratch. | [arXiv:2601.08584](https://arxiv.org/abs/2601.08584) |
| **Phi-4** | Microsoft | 2024 | **Adjacent** (synthetic data from GPT-4) | Centrally focused on synthetic data quality. 50+ synthetic datasets (~400B tokens). Surpasses GPT-4 teacher on STEM, suggesting going "beyond distillation." | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **Phi-4-Reasoning** | Microsoft | 2025 | **Offline KD** | SFT on reasoning traces from o3-mini (1.4M prompts). Phi-4-reasoning-plus adds RL phase. Approaches DeepSeek-R1 performance at 14B. | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/phi-4-reasoning-technical-report/) |
| **Mistral-NeMo-Minitron 8B** | NVIDIA × Mistral | 2024 | **Offline KD** | Width-pruned from Mistral NeMo 12B + KD with 380B tokens. Teacher distribution-shift correction first. | [NVIDIA Blog](https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/) |
| **Llama-Nemotron** | NVIDIA | 2025 | **Offline KD** | NAS + recovery training with KD from DeepSeek-R1 reasoning traces + continued pre-training. | [arXiv:2505.00949](https://arxiv.org/abs/2505.00949) |
| **Zephyr** | HuggingFace | 2023 | **Offline KD** | dSFT (ChatGPT data) + dDPO (GPT-4 ratings). No student rollouts. | [arXiv:2310.16944](https://arxiv.org/abs/2310.16944) |
| **Apple Intelligence (iTeC)** | Apple | 2024 | **Partial OPD** | Iterative Teaching Committee: model generates, committee of RM/evaluators scores, iteratively refined. On-device (3B) + server models. | [arXiv:2407.21075](https://arxiv.org/abs/2407.21075) |
| **InternLM2 (COOL RLHF)** | Shanghai AI Lab | 2024 | **Adjacent** | Conditional Online RLHF: model generates own outputs, conditional reward modeling with PPO. Explicitly "online" over offline. | [arXiv:2403.17297](https://arxiv.org/abs/2403.17297) |
| **Open-R1** | HuggingFace (community) | 2025 | **Partial OPD** | Community reproduction of DeepSeek-R1: offline SFT on R1 reasoning traces + GRPO RL (on-policy). | [GitHub](https://github.com/huggingface/open-r1) |

### The OPD Spectrum in Industry

From least to most on-policy, industrial systems fall on a spectrum:

1. **Pure Offline KD:** Teacher generates all data, student SFTs (Phi-1/2/3, R1-distilled models)
2. **Rejection Sampling:** Student/teacher generates candidates, verifier selects (LLaMA 3.1, RFT)
3. **Iterative Offline KD:** Multiple rounds with model updates between rounds (LLaMA 3.1 6-round)
4. **On-policy DPO:** Student generates responses, teacher judges preferences (Zephyr dDPO)
5. **Online RL with RM:** Student generates, learned reward model scores (RLHF/PPO)
6. **Online RL with Verifier:** Student generates, rule-based verifier rewards (DeepSeek-R1, Qwen3)
7. **Strict OPD (GKD-style):** Student generates, teacher provides token-level KL supervision (Gemma 2/3, Qwen3 small models, Ministral Cascade)

> **Key insight:** Gemma 2/3 and Qwen3 are the clearest industrial examples of strict on-policy distillation. Most other systems use offline KD for bootstrapping + online RL for refinement — a pattern where the RL stage is structurally OPD-like but uses reward signals rather than teacher logits.

---

## Frameworks & Tools Supporting OPD

| Framework | Org | OPD Support | Key Methods | Link |
|---|---|---|---|---|
| **TRL (GKDTrainer + GOLDTrainer + MiniLLMTrainer + DistillationTrainer)** | HuggingFace | **Native** — GKDTrainer, GOLDTrainer (cross-tokenizer), MiniLLMTrainer, DistillationTrainer with 40x speedup buffer | GKD, MiniLLM, GOLD/ULD, on-policy generation, cross-tokenizer alignment, vLLM support, external teacher server | [GitHub](https://github.com/huggingface/trl) / [Docs](https://huggingface.co/docs/trl/gkd_trainer) |
| **OpenRLHF** | OpenRLHF | **Partial** — KD via MiniLLM implementation | MiniLLM-based KD, PPO/GRPO/REINFORCE++ for RL, vLLM generation, Ray-based distributed training | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| **KDFlow** | songmzhang | **Native** — built on OpenRLHF for on-policy KD | On-policy KD loop, cross-tokenizer KD (DSKD), pluggable algorithms, 1.4-6x speedup over alternatives | [GitHub](https://github.com/songmzhang/KDFlow) |
| **EasyDistill** | Alibaba/ModelScope | **Native** — comprehensive KD toolkit | Data synthesis, SFT, logits distillation, ranking optimization, RL for KD, AgentKD, CogPO | [GitHub](https://github.com/modelscope/easydistill) |
| **DistillKit** | Arcee AI | **Partial** — online and offline distillation | Logit-based KD, hidden-state KD, advanced logit compression, flexible loss functions | [GitHub](https://github.com/arcee-ai/DistillKit) |
| **NeMo RL** (formerly NeMo-Aligner) | NVIDIA | **Native** — on-policy distillation supported | On-policy distillation, GRPO/GSPO/DAPO, SFT+KD, Megatron parallelism, sequence packing | [GitHub](https://github.com/NVIDIA-NeMo/RL) |
| **verl (HybridFlow)** | verl-project | **Partial** (via extensions like rLLM, SDPO) | GRPO/PPO/REINFORCE++, on-policy generation, Megatron + vLLM + FSDP2, 671B scale | [GitHub](https://github.com/verl-project/verl) |
| **rLLM** | rllm-org | **Native** — on-policy distillation with verl backend | On-policy distillation trainer, multi-GPU via verl, single-GPU via tinker | [GitHub](https://github.com/rllm-org/rllm) |
| **Alignment Handbook** | HuggingFace | **Partial** — recipes for SFT+DPO | Zephyr reproduction, DeepSpeed ZeRO-3, LoRA/QLoRA, ORPO | [GitHub](https://github.com/huggingface/alignment-handbook) |
| **LMOps (MiniLLM, GAD)** | Microsoft | **Native** — reference implementations | MiniLLM, GAD adversarial distillation | [GitHub](https://github.com/microsoft/LMOps) |
| **LLaMA-Factory + Logits-Based FT** | hiyouga + dvlab | **Partial** — via extension | Logits-based fine-tuning with teacher logits, distillation loss integration | [GitHub](https://github.com/hiyouga/LlamaFactory) / [Extension](https://github.com/dvlab-research/Logits-Based-Finetuning) |

---

## Summary Table

| Paper | Year | Method | Teacher Access | Rollout | OPD Type | Key Innovation |
|---|---|---|---|---|---|---|
| [Hinton et al.](https://arxiv.org/abs/1503.02531) | 2015 | KD | White-box | Teacher | Not OPD | Soft targets, temperature scaling |
| [Kim & Rush](https://arxiv.org/abs/1606.07947) | 2016 | SeqKD | White/Black | Teacher | Not OPD | Sequence-level KD for NMT |
| [DAgger](https://arxiv.org/abs/1011.0686) | 2011 | Imitation | Expert | Student | Adjacent | On-policy correction concept |
| [STaR](https://arxiv.org/abs/2203.14465) | 2022 | Self-train | Self+verifier | Student | Partial OPD | Iterative reasoning bootstrap |
| [CAI](https://arxiv.org/abs/2212.08073) | 2022 | RLAIF | Self | Student | Partial OPD | AI feedback for alignment |
| [**GKD**](https://arxiv.org/abs/2306.13649) | 2023 | On-policy KD | White-box | Student | **Strict OPD** | Mixture sampling, flexible divergences |
| [**MiniLLM**](https://arxiv.org/abs/2306.08543) | 2023 | Reverse KL OPD | White-box | Student | **Strict OPD** | Reverse KL, RLHF-style pipeline |
| [f-DISTILL](https://arxiv.org/abs/2307.15190) | 2023 | f-div KD | White-box | Mixed | Partial OPD | f-divergence framework |
| [ReST](https://arxiv.org/abs/2308.08998) | 2023 | Self-training | Reward model | Student | Strict OPD | Grow-Improve loops |
| [RFT](https://arxiv.org/abs/2308.01825) | 2023 | Rejection sampling | Verifier | Student | Partial OPD | Diverse reasoning paths |
| [RAFT](https://arxiv.org/abs/2304.06767) | 2023 | Reward ranking | Reward model | Student | Partial OPD | Iterative best-of-n |
| [Zephyr](https://arxiv.org/abs/2310.16944) | 2023 | dSFT+dDPO | Black-box | Teacher | Not OPD | AI feedback alignment |
| [Weak→Strong](https://arxiv.org/abs/2312.09390) | 2023 | Inverse KD | Weak teacher | N/A | Adjacent | Generalization beyond teacher |
| [ReST^EM](https://arxiv.org/abs/2312.06585) | 2024 | EM self-train | Verifier | Student | Partial OPD | Scaling self-training |
| [SPIN](https://arxiv.org/abs/2401.01335) | 2024 | Self-play | Self | Student | Adjacent | Self vs. human discrimination |
| [Self-Rewarding](https://arxiv.org/abs/2401.10020) | 2024 | Self-reward | Self | Student | Partial OPD | LLM-as-Judge self-improvement |
| [DistiLLM](https://arxiv.org/abs/2402.03898) | 2024 | Skewed KL | White-box | Adaptive | Partial OPD | 4.3x speedup, skewed KL |
| [DSKD](https://arxiv.org/abs/2406.17328) | 2024 | Cross-tokenizer KD | White-box | Offline→On | Partial OPD | Dual-space, token alignment |
| [DPKD](https://arxiv.org/abs/2406.19774) | 2024 | Preference KD | White-box | Offline | Not OPD | DPO-KD bridge |
| [DistillSpec](https://arxiv.org/abs/2310.08461) | 2024 | SD alignment | White-box | Student (draft) | Strict OPD | On-policy draft alignment |
| [**RLKD**](https://ojs.aaai.org/index.php/AAAI/article/view/40710) | 2025 | RL+KD | White-box | Student | Strict OPD | Structure-aware RL reward |
| [**KDRL**](https://arxiv.org/abs/2506.02208) | 2025 | Unified KD+RL | White-box | Student | Strict OPD | Joint GRPO + KL objective |
| [PRIME](https://arxiv.org/abs/2502.01456) | 2025 | Implicit PRM | Process RM | Student | Adjacent | Online PRM with outcome labels |
| [AlignDistil](https://arxiv.org/abs/2503.02832) | 2025 | RLHF≡Distill | White-box | Flexible | Partial OPD | RLHF-distillation equivalence |
| [ToDi](https://arxiv.org/abs/2505.16297) | 2025 | Token-adaptive | White-box | Offline | Adjacent | Per-token FKL/RKL balance |
| [**GAD**](https://arxiv.org/abs/2511.10643) | 2025 | Adversarial | Black-box | Student | **Strict OPD** | GAN-style black-box OPD |
| [**OPSD**](https://arxiv.org/abs/2601.18734) | 2026 | Self-distill | Self (privileged) | Student | **Strict OPD** | Teacher-free, privileged context |
| [OPSDL](https://arxiv.org/abs/2604.17535) | 2026 | Self-distill | Self (short-ctx) | Student | Strict OPD | Long-context via short-context self |
| [RLAD](https://arxiv.org/abs/2602.22495) | 2026 | Trust-region KD | White-box | Student | Strict OPD | TRRD: trust-region ratio distill |
| [EOPD](https://arxiv.org/abs/2603.07079) | 2026 | Entropy-aware | White-box | Student | Strict OPD | Adaptive FKL on high-entropy tokens |
| [Rethinking OPD](https://arxiv.org/abs/2604.13016) | 2026 | Analysis + fix | White-box | Student | Strict OPD | Failure mode diagnosis + recipes |
| [OPD Survey](https://arxiv.org/abs/2604.00626) | 2026 | Survey | — | — | — | First comprehensive OPD survey |

---

## Markdown-Table Comprehensive Table

```text
Title,Year,Authors/Org,Link,Type,Method Family,Teacher Access,Supervision Signal,Rollout Source,Loss/Objective,Training Stage,Model Family,Tasks/Benchmarks,Code Available,Strictness,Summary
"Distilling the Knowledge in a Neural Network",2015,"Hinton, Vinyals, Dean / Google",https://arxiv.org/abs/1503.02531,Paper,Knowledge Distillation,White-box (logits),KL divergence with temperature,Teacher-generated,Weighted CE + KL soft targets,General,Various NNs,"MNIST, speech",Widely reimplemented,Not OPD,"Foundational KD paper. Soft targets transfer richer knowledge than hard labels."
"Sequence-Level Knowledge Distillation",2016,"Kim, Rush / Harvard",https://arxiv.org/abs/1606.07947,Paper,Sequence-level KD,White/Black-box,Sequence distribution matching,Teacher beam search,Word-level + sequence-level KD,NMT training,Seq2seq,WMT translation,N/A,Not OPD,"First KD for sequence generation. Teacher beam search outputs as training data."
"DAgger",2011,"Ross, Gordon, Bagnell / CMU",https://arxiv.org/abs/1011.0686,Paper,On-policy imitation learning,Expert oracle,Expert corrections on learner states,Student rollouts,Supervised on aggregated data,Imitation learning,General policies,"SuperTux, Mario",Multiple reimpl.,Adjacent,"Conceptual ancestor of OPD. On-policy correction addresses covariate shift."
"Born Again Neural Networks",2018,"Furlanello et al.",https://arxiv.org/abs/1805.04770,Paper,Self-distillation,White-box (same arch),Soft targets (identical architecture),N/A (classification),KD loss,Training,"DenseNets, ResNets","CIFAR-10/100, LM",Unofficial,Not OPD,"Self-distillation improves same-architecture models. Inspired iterative self-distillation."
"STaR: Self-Taught Reasoner",2022,"Zelikman et al. / Stanford, Google",https://arxiv.org/abs/2203.14465,Paper,Iterative self-training,Self + verifier,Correctness filter + rationalization,Student-generated rationales,SFT on correct rationales,Post-training,GPT-J,"Arithmetic, math, CommonsenseQA",https://github.com/ezelikman/STaR,Partial OPD,"Iterative reasoning bootstrap via self-generated CoT. Introduced rationalization."
"Constitutional AI",2022,"Bai et al. / Anthropic",https://arxiv.org/abs/2212.08073,Paper,RLAIF,Self (self-critique),AI preference judgments,Student-generated + self-critiques,"SFT on revisions + RLHF with AI prefs",Post-training,Anthropic ~52B,"Helpfulness, harmlessness",N/A (proprietary),Partial OPD,"Introduced RLAIF. Model provides its own alignment supervision via constitutional principles."
"GKD: On-Policy Distillation of Language Models",2023,"Agarwal et al. / Google DeepMind",https://arxiv.org/abs/2306.13649,Paper (ICLR 2024),On-policy KD,White-box,KL/RKL/JSD on student outputs,Student (mixture λ),"f-divergence on on-policy samples",Post-training,"T5, PaLM","Summarization, translation, reasoning",https://huggingface.co/docs/trl/gkd_trainer,Strict OPD,"Canonical OPD paper. Mixture sampling addresses distribution mismatch. RKL/JSD >> FKL."
"MiniLLM",2023,"Gu et al. / Tsinghua, Microsoft",https://arxiv.org/abs/2306.08543,Paper (ICLR 2024),Reverse KL OPD,White-box,Reverse KL divergence,Student-generated,"Reverse KLD with RLHF-style optimization",Post-training,"GPT-2, GPT-J, OPT",Instruction following,https://github.com/microsoft/LMOps/tree/main/minillm,Strict OPD,"Reverse KL avoids mode-covering. RLHF-like on-policy training scales to 13B."
"f-DISTILL",2023,"Wen et al.",https://arxiv.org/abs/2307.15190,Paper (ACL 2023),f-divergence KD,White-box,"TVD, JSD, etc.",Mixed (step-wise),"f-divergence minimization",Post-training,Transformer,"Translation, summarization, dialogue",N/A,Partial OPD,"f-divergence framework for sequence KD. Symmetric divergences outperform asymmetric."
"ReST: Reinforced Self-Training",2023,"Gulcehre et al. / Google",https://arxiv.org/abs/2308.08998,Paper,On-policy self-training,Reward model,"Reward filtering of student outputs",Student generates,"Offline RL on filtered data",Post-training,Various,Machine translation,N/A,Strict OPD,"Grow-Improve loops for reward-filtered self-training. Template for iterative OPD."
"RFT: Rejection Sampling Fine-Tuning",2023,"Yuan et al.",https://arxiv.org/abs/2308.01825,Paper,Rejection sampling,Verifier,"Correctness filter",Student generates,"SFT on correct reasoning paths",Post-training,"LLaMA-7B/13B","GSM8K, MATH",N/A,Partial OPD,"Simple rejection sampling for math reasoning. Distinct paths matter more than total samples."
"RAFT: Reward rAnked FineTuning",2023,"Dong et al.",https://arxiv.org/abs/2304.06767,Paper (TMLR),Reward ranking,Reward model,Reward model ranking,Student generates,"SFT on top-ranked responses",Post-training,LLaMA-7B,HH-RLHF,https://github.com/RLHFlow/RAFT,Partial OPD,"Iterative best-of-n fine-tuning. Simpler than PPO with comparable results."
"Zephyr",2023,"Tunstall et al. / HuggingFace",https://arxiv.org/abs/2310.16944,Paper,dSFT+dDPO,"Black-box (ChatGPT, GPT-4)","Teacher-generated data + AI preferences",Teacher-generated,"SFT + DPO",Post-training,Mistral-7B,"MT-Bench, AlpacaEval",https://github.com/huggingface/alignment-handbook,Not OPD,"Offline distillation baseline. AI feedback alignment without human labels."
"Weak-to-Strong Generalization",2023,"Burns et al. / OpenAI",https://arxiv.org/abs/2312.09390,Paper,Inverse KD,Weak model labels,Weak supervision,N/A (classification),"Fine-tuning + confidence boosting",Fine-tuning,GPT-4 family,"NLP, chess, reward modeling",N/A,Adjacent,"Strong students generalize beyond weak teachers. Implications for superhuman alignment."
"ReST^EM",2024,"Singh et al. / Google",https://arxiv.org/abs/2312.06585,Paper (TMLR),EM self-training,Verifier,Binary correctness,Student generates,"SFT on correct solutions (EM iterations)",Post-training,PaLM-2,"MATH, APPS, GSM8K, HumanEval",N/A,Partial OPD,"EM-based self-training scales beyond human data. Favorable model size scaling."
"SPIN",2024,"Chen et al. / UCLA",https://arxiv.org/abs/2401.01335,Paper,Self-play,Self (previous iteration),Self vs. human discrimination,Student generates,"DPO-style (self vs. human)",Post-training,Mistral-7B,"Open LLM Leaderboard, MT-Bench",https://github.com/uclaml/SPIN,Adjacent,"Self-play without external teacher. Theoretical convergence guarantee."
"Self-Rewarding LMs",2024,"Yuan et al. / Meta",https://arxiv.org/abs/2401.10020,Paper,Self-reward,Self (LLM-as-Judge),Self-generated reward scores,Student generates,"Iterative DPO with self-rewards",Post-training,LLaMA-2 70B,AlpacaEval 2.0,N/A,Partial OPD,"Self-improving instruction following and reward modeling. Outperforms GPT-4 0613."
"DistiLLM",2024,"Ko et al. / KAIST, Microsoft",https://arxiv.org/abs/2402.03898,Paper (ICML 2024),Skewed KL OPD,White-box,Skewed KL divergence,Adaptive mixture,"Skewed KL + adaptive off-policy",Post-training,"GPT-2, OPT, OpenLLaMA",Instruction following,https://github.com/jongwooko/distillm,Partial OPD,"4.3x speedup over MiniLLM via skewed KL and adaptive off-policy approach."
"DSKD",2024,"Zhang et al.",https://arxiv.org/abs/2406.17328,Paper (EMNLP 2024),Cross-tokenizer KD,White-box,KL in unified output spaces,Offline (v2: on-policy),"KL with exact token alignment",Post-training,"LLaMA, Qwen",Instruction following,https://github.com/songmzhang/DSKD,Partial OPD,"Cross-tokenizer white-box KD via dual-space projectors. ETA algorithm for token alignment."
"DPKD",2024,"Li et al. / Microsoft",https://arxiv.org/abs/2406.19774,Paper,Preference KD,White-box,"Implicit reward + KL divergence",Offline,"Implicit reward + RKL; preference optimization",Post-training,"GPT-2, OPT",Instruction following,https://aka.ms/dpkd,Not OPD,"Bridges KD and DPO. Teacher outputs as preferred responses."
"DistillSpec",2024,"Zhou et al. / Google",https://arxiv.org/abs/2310.08461,Paper (ICLR 2024),SD alignment KD,White-box,Draft-target divergence,"Draft model (on-policy)","On-policy KD with task-specific divergence",Inference,Various LLMs,Latency benchmarks,N/A,Strict OPD,"On-policy KD for speculative decoding. 10-45% inference speedup."
"RLKD",2025,"Xu et al.",https://ojs.aaai.org/index.php/AAAI/article/view/40710,Paper (AAAI 2025),RL-based KD,White-box,"GSRM (structure-aware reward)",Student (GRPO),"GRPO + GSRM reward",Post-training,Qwen2.5-Math-7B,Math reasoning,N/A,Strict OPD,"First RL-based KD for reasoning. Structure-aware reward captures multi-branch reasoning."
"KDRL",2025,"Xu et al.",https://arxiv.org/abs/2506.02208,Paper,Unified KD+RL,White-box,"RKL to teacher + outcome rewards",Student on-policy,"Joint GRPO + RKL minimization",Post-training,"Qwen2.5-Math 1.5B-7B",Math reasoning,N/A,Strict OPD,"Unified KD+RL objective. Better accuracy-efficiency tradeoff than either alone."
"PRIME",2025,"Sun et al.",https://arxiv.org/abs/2502.01456,Paper,Implicit PRM,Implicit PRM,Dense process rewards,Student on-policy,"RL with implicit process rewards",Post-training,Qwen2.5-Math-7B,Math reasoning,https://github.com/PRIME-RL/PRIME,Adjacent,"Online PRM updates without human step labels. 15.1% improvement over SFT."
"AlignDistil",2025,"Zhang et al.",https://arxiv.org/abs/2503.02832,Paper (ACL 2025),RLHF≡Distill,White-box,"Token-level contrastive DPO reward",Flexible,"Token-level distillation ≡ RLHF",Post-training,"LLaMA, Mistral","AlpacaEval, MT-Bench, Arena-Hard",https://github.com/songmzhang/AlignDistil,Partial OPD,"RLHF-distillation equivalence proof. Fast convergence via token-level rewards."
"ToDi",2025,"Jung et al.",https://arxiv.org/abs/2505.16297,Paper (EMNLP 2025),Token-adaptive KD,White-box,"Per-token FKL/RKL blend",Offline,"Sigmoid-weighted FKL+RKL",Post-training,"GPT-2, TinyLLaMA, LLaMA-2",Instruction following,N/A,Adjacent,"Per-token adaptive divergence. O(V) complexity. Inspired EOPD."
"GAD",2025,"Ye et al. / Microsoft",https://arxiv.org/abs/2511.10643,Paper,Adversarial OPD,Black-box,"Discriminator reward",Student (generator),"Minimax adversarial game",Post-training,"Qwen2.5, LLaMA-3","LMSYS-Chat, MT-Bench",https://github.com/microsoft/LMOps/tree/main/gad,Strict OPD,"GAN-style black-box OPD. Qwen2.5-14B matches GPT-5-Chat."
"OPSD",2026,"Zhao et al.",https://arxiv.org/abs/2601.18734,Paper,Self-distillation,Self (privileged context),KL (privileged → standard policy),Student on-policy,"Reverse KL",Post-training,Various,Math reasoning,Project page,Strict OPD,"Teacher-free OPD via privileged context. Single model as both teacher and student."
"OPSDL",2026,"Zhang et al. / Baidu",https://arxiv.org/abs/2604.17535,Paper,Self-distillation,Self (short-context),RKL (short-ctx → long-ctx),Student on-policy (long-ctx),"Reverse KL on-policy",Post-training,"Qwen2.5 7B-32B",Long-context benchmarks,N/A,Strict OPD,"Short-context self-distillation for long-context extension."
"RLAD",2026,"Zhang et al.",https://arxiv.org/abs/2602.22495,Paper,Trust-region KD,White-box,"Advantage-aware teacher imitation",Student on-policy,"TRRD: trust-region ratio distillation",Post-training,"Qwen3 0.6B-7B","Math, logic reasoning",N/A,Strict OPD,"Trust-region ratio distillation. Selective teacher imitation during RL."
"EOPD",2026,"Jin et al.",https://arxiv.org/abs/2603.07079,Paper,Entropy-aware OPD,White-box,"Adaptive RKL + FKL (entropy-based)",Student on-policy,"RKL + FKL on high-entropy tokens",Post-training,"Qwen2.5, LLaMA-3","GSM8K, MATH, AIME",N/A,Strict OPD,"Entropy-adaptive divergence preserves diversity. Strong out-of-domain transfer."
"Rethinking OPD",2026,"Li et al. / THUNLP",https://arxiv.org/abs/2604.13016,Paper,OPD analysis,White-box,Token-level KL,Student on-policy,"Standard OPD + proposed fixes",Post-training,Various,Math reasoning,https://github.com/thunlp/OPD,Strict OPD,"Diagnoses OPD failure modes. Compatible thinking patterns + new capabilities required."
"OPD Survey",2026,"Song et al.",https://arxiv.org/abs/2604.00626,Survey,Survey,—,—,—,—,—,—,—,N/A,Survey,"First comprehensive OPD survey. f-divergence framework, 3-axis taxonomy."
```

---

## Open Problems in LLM OPD

### 1. Distillation Scaling Laws
- How does OPD performance scale with student size, teacher size, data quantity, and compute?
- Are there optimal student/teacher size ratios for OPD?
- What is the compute-optimal allocation between on-policy generation and gradient updates?
- **Current status:** No systematic scaling law study exists for OPD (analogous to Chinchilla for pre-training).

### 2. When and Why OPD Fails
- OPD can fail when teacher and student have incompatible "thinking patterns" ([Rethinking OPD, 2026](https://arxiv.org/abs/2604.13016)).
- A stronger teacher can paradoxically fail to improve a student when a weaker teacher succeeds.
- **Open:** Formal characterization of student-teacher compatibility; automatic teacher selection.

### 3. Breaking the Teacher Ceiling
- Standard OPD is bounded by teacher quality — the student cannot exceed the teacher.
- G-OPD (Yang et al., 2026) shows that reward scaling (α > 1) can incentivize students to exceed teachers.
- **Open:** Practical methods for reliably surpassing teachers; understanding when extrapolation is safe.

### 4. Uncertainty-Aware Feedback
- Current OPD applies uniform supervision regardless of teacher confidence.
- High-entropy teacher tokens may provide noisy or misleading signal.
- EOPD begins addressing this but the problem is far from solved.
- **Open:** Token-level uncertainty estimation for adaptive supervision strength.

### 5. Agent-Level and Tool-Use Distillation
- Current OPD focuses on single-turn text generation.
- Distilling agentic behaviors (multi-step tool use, environment interaction) requires trajectory-level OPD.
- **Open:** OPD for agentic LLMs with tool calling, multi-turn reasoning, and environment feedback.

### 6. Cross-Tokenizer OPD
- Different model families use incompatible tokenizers (LLaMA vs. Qwen vs. Gemma).
- GOLD, ULD, DSKD, ALM, and BLD offer solutions but "consistent improvements across all tasks remain elusive."
- **Open:** Principled, efficient cross-tokenizer alignment without performance degradation.

### 7. Multi-Teacher and Ensemble OPD
- Using multiple teachers could combine complementary strengths.
- Challenges: conflicting teacher signals, teacher selection, weighting strategies.
- **Open:** Principled multi-teacher OPD frameworks; automatic teacher weighting.

### 8. OPD for Long Chain-of-Thought Reasoning
- Long-CoT models (DeepSeek-R1, Qwen3 thinking mode) generate reasoning traces of thousands of tokens.
- Dense token-level supervision on long traces is computationally expensive.
- Prefix-based OPD (PPD) reduces cost but may miss important signal in later tokens.
- **Open:** Efficient OPD for long-CoT without sacrificing reasoning quality.

### 9. Catastrophic Forgetting During OPD
- OPD may cause the student to forget previously learned capabilities while acquiring new ones.
- Unlike SFT, on-policy learning may help preserve prior knowledge, but this is not guaranteed.
- **Open:** OPD methods that provably preserve existing capabilities while learning new ones.

### 10. Reproducibility and Standardized Benchmarks
- OPD results are hard to compare across papers due to different models, datasets, and evaluation protocols.
- No standardized OPD benchmark suite exists.
- **Open:** Community-standard benchmarks and evaluation protocols for OPD research.

---

## Citation

If you find this survey useful, please consider citing:

```bibtex
@misc{On Policy Distillation Landscape-2026,
  title={Awesome On-Policy Distillation (OPD) for LLMs — Research Landscape},
  year={2026},
  url={https://github.com/YOUR_USERNAME/on-policy-distillation-landscape}
}
```

## Key Survey References

- Song et al. (2026). [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626)
- Li et al. (2026). [Rethinking On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2604.13016)
- Xu et al. (2024). [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116)
- Tebmer et al. (2024). [Awesome-Knowledge-Distillation-of-LLMs](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)

---

*This document was compiled through systematic search of arXiv, OpenReview, ACL Anthology, Google Scholar, GitHub, HuggingFace, and major lab technical reports. Every entry has been verified against its source. Last verified: May 2026.*
