# On Policy Distillation Landscape: A Taxonomy-First Survey of On-Policy Distillation for LLMs

**On-policy distillation (OPD) has emerged as the dominant post-training paradigm for frontier-class open LLMs in 2025–2026**, achieving the quality of RL with roughly an order of magnitude less compute. This survey maps ~80 entries spanning foundational methods (GKD, MiniLLM, DistiLLM), industrial pipelines (Qwen3, Gemma 2/3, Nemotron-Cascade 2, MiMo-V2-Flash, GLM-4.5), self-distillation variants (OPSD, π-Distill), agentic distillation, and supporting frameworks. We apply a strict definition throughout — student rollouts during training **plus** distributional/token/trajectory-level teacher supervision on those rollouts — and explicitly flag the many works (DeepSeek-R1-Distill, SPIN, Self-Rewarding, Tülu 3 RLVR, Constitutional AI RLAIF, Phi-4 PTS-DPO) that are routinely but incorrectly grouped under OPD. The clearest finding: the entire industrial OPD wave traces back to a handful of academic papers — Agarwal et al.'s GKD (DeepMind 2023/24), Gu et al.'s MiniLLM (Tsinghua/MSRA 2024), and Ko et al.'s DistiLLM (KAIST 2024) — operationalized at scale first by Alibaba's Qwen3 (May 2025), then popularized by Thinking Machines Lab's October 2025 blog.

## What counts as OPD, and what doesn't

Throughout this survey we use a two-clause test. **Clause 1 (rollouts):** the student/policy generates its own trajectories or completions during training. **Clause 2 (supervision):** a teacher LM, verifier, discriminator, privileged-context model, reward model, or reference model provides supervision on those rollouts at the **distribution, token, or trajectory level** — not as a scalar reward consumed by PPO/GRPO/DPO. Both clauses must hold.

This definition excludes four classes that are commonly conflated with OPD. **Offline distillation** (DeepSeek-R1-Distill, Sky-T1, s1, LIMO, Open-R1's SFT phase) trains the student on teacher-generated text via cross-entropy; the student never rolls out during training, failing Clause 1. **RLVR** (Tülu 3, OLMo 2/3, Skywork-OR1, SimpleRL-Zoo, Magistral, AceMath-RL, SWE-RL, Search-R1) uses verifiable scalar rewards in PPO/GRPO; supervision is a scalar, failing Clause 2. **RLAIF** (Anthropic Constitutional AI, Self-Rewarding LMs, Snorkel-Mistral-PairRM-DPO, Kimi K2's self-critic rubric) is also a scalar-reward pipeline. **Iterative SFT/DPO on self-generated data** (STaR, RFT, ReST^EM, SPIN, Iterative RPO, SPPO) uses binary correctness or preference labels, which are scalar — these are at best Partial OPD. The most important misclassification to flag: **DeepSeek-R1-Distill-Qwen/Llama is offline SFT on R1-generated reasoning traces** — the R1 paper explicitly states this — and is **not OPD** despite the name.

## A taxonomy of LLM OPD methods

The space organizes naturally into six branches.

**White-box OPD with logit access** is the largest and most rigorous branch. Sub-branches split by divergence: **forward KL on student rollouts** (ImitKD's DAgger-style imitation, the λ=1, β=0 setting of GKD); **reverse KL** (MiniLLM with policy gradient; Thinking Machines' Tinker recipe; OPSD); **JSD / generalized divergences** (GKD's interpolating JSD parameterized by β; AKL's adaptive head-tail mixing; ToDi's per-token sigmoid mixing); **skew and contrastive variants** (DistiLLM's skew forward/reverse KL with adaptive off-policy replay; DistiLLM-2's CALD asymmetric contrastive loss); **f-divergence generalizations** (Wen et al.'s f-distill; DistillSpec's task-tailored choice); **speculative-decoding-flavored OPD** (Speculative KD's teacher-replaced student tokens; DistillSpec's draft-model alignment); and **multi-teacher OPD (MOPD)**, which appears in two distinct senses — Xiaomi's MiMo-V2-Flash uses multiple domain-specialist teachers each providing dense token-level rewards to a single student, while NVIDIA's Nemotron-Cascade 2 uses the strongest intermediate RL checkpoint per domain as the OPD teacher.

**Black-box OPD with sample access only** is sparser. PLaD constructs pseudo-preference pairs (assuming teacher > student) and trains via ranking loss on student rollouts. The recent **GAD (Generative Adversarial Distillation)** from Microsoft trains an on-policy discriminator to distinguish student from black-box teacher samples (e.g., GPT-5-Chat), then drives the student via GRPO — trajectory-level adversarial supervision rather than token-level. Black-box sequence-level KD on student rollouts (the modern \"on-policy SeqKD\") is the no-logit baseline.

**OPD + RL/RLVR hybrids** combine token-level teacher supervision with reward signals. GKD's RL extension adds a reward term J(θ) = E[r(x,y)] − α·D(p_T ‖ π_θ). BOND treats the implicit Best-of-N reward-induced policy as the teacher and minimizes Jeffreys divergence. AlignDistil proves RLHF is equivalent to token-level distillation against a synthetic teacher constructed from DPO + reverse-DPO logits. PRIME and Math-Shepherd-style methods use process reward models as dense token-level rewards inside policy gradient — strictly RLVR with shaped reward, but the boundary with OPD is thin when the PRM signal becomes truly distributional.

**On-Policy Self-Distillation (OPSD)** uses the same model (or its EMA) as teacher and student, exploiting privileged context as the asymmetry. Zhao et al.'s OPSD and Hübotter et al.'s π-Distill (both early 2026) condition the teacher on the ground-truth answer or reference CoT; the student rolls out without that privilege and is supervised at the token level — a textbook strict OPD configuration. GLM-4.5's \"unified training\" stage and GLM-5's \"on-policy cross-stage distillation\" are the industrial counterpart, merging domain expert checkpoints back into a single hybrid model.

**Agentic and tool-use OPD** is the thinnest branch. Most agent fine-tuning (FireAct, AgentTuning, ToolBench/ToolLLaMA, Gorilla, Agent-FLAN, xLAM, AgentInstruct, Hammer, NexusRaven) is offline SFT on teacher trajectories — not OPD. Recent RL agents (Search-R1, WebRL, SWE-RL, Agent-R1) are RLVR. The single clearest strict-OPD agent paper is **Structured Agent Distillation** (Liu et al., 2025), which queries an online teacher's logits on student-rolled ReAct trajectories and applies span-aware token-level KD over `[REASON]` and `[ACT]` segments. ETO's trajectory-level DPO on student failures vs. expert successes, ToRA's output-space shaping with teacher correction, and UltraInteract/Eurus's tree-structured preference data sit at Partial OPD.

**Industrial post-training pipelines** form a final practical branch. Two Western reports meet strict OPD: **Gemma 2** (the canonical Western citation, post-training SFT with student rollouts and teacher KL on a sampled subset of the 256k vocab, citing Agarwal and Gu) and **Nemotron-Cascade 2** (explicit §4.4 \"Multi-Domain On-Policy Distillation\" as a stabilization step after Cascade RL). Four Chinese reports meet strict OPD: **Qwen3** (the publicly canonical recipe — off-policy SFT cold-start followed by on-policy logit-KL against Qwen3-32B or Qwen3-235B), **Qwen3-Omni** (multimodal extension), **MiMo-V2-Flash** (MOPD as RL with dense token-level rewards from multiple domain teachers), **GLM-4.5/4.6/4.7** (iterative self-distillation merging expert RL checkpoints), and **Tencent HY-MT1.5** (translation-specialist OPD). Gemma 3 inherits the Gemma 2 recipe but is less explicit.

## Historical timeline

The field's milestones cluster in three waves. The **pre-LLM foundations** include Hinton et al.'s 2015 KD, Kim & Rush's 2016 SeqKD (offline, teacher rollouts only — the contrast point for OPD), and **ImitKD** (Lin et al., EMNLP 2020) which first applied DAgger-style imitation to sequence-model distillation on student-visited prefixes. The **academic OPD wave** runs 2023–2024: **MiniLLM** (Gu et al., arXiv June 2023, ICLR 2024) introduces reverse KL with policy gradient; **GKD** (Agarwal et al., arXiv June 2023, ICLR 2024) formalizes the general framework with generalized JSD and λ-mixing; **f-distill** (Wen et al., ACL 2023) provides the f-divergence theoretical lens; **DistillSpec** (Zhou et al., ICLR 2024) extends to speculative decoding alignment; **DistiLLM** (Ko et al., ICML 2024) adds skew KL and adaptive off-policy replay; **BOND** (Sessa et al., DeepMind 2024) reframes RLHF as Best-of-N distillation. **SKD** (Xu et al., Oct 2024) interleaves teacher token replacement. The **industrial wave** begins June 2024 with **Gemma 2** as the first explicit production OPD step. **Qwen3** (May 2025) demonstrates OPD beats RL at ~1/10 the GPU hours on AIME, becoming the empirical anchor cited everywhere. **DistiLLM-2** (Ko et al., ICML 2025) adds contrastive CALD. The **Thinking Machines blog** (October 27, 2025) consolidates the narrative and releases the Tinker cookbook. **Nemotron-Cascade 2** (NVIDIA, late 2025/2026), **MiMo-V2-Flash** (Xiaomi, 2026), **GLM-4.5/4.6** (Z.ai, 2025–26), **Qwen3-Omni** (September 2025), and **Tencent HY-MT1.5** (late 2025/26) extend OPD to multi-domain, multi-teacher, and multimodal settings. **OPSD / π-Distill** (early 2026) introduces privileged-context self-distillation as a no-external-teacher variant.

## The first ten papers to read

A reader new to OPD should start with these in this order. **Agarwal et al. 2023/24 (GKD)** defines the framework and the on-policy ↔ off-policy and forward ↔ reverse KL axes. **Gu et al. 2024 (MiniLLM)** is the reverse-KL policy-gradient counterpart with all the stabilization tricks. **Lin et al. 2020 (ImitKD)** provides the DAgger lineage. **Ko et al. 2024 (DistiLLM)** introduces skew KL and the adaptive replay buffer that makes OPD efficient. **Ko et al. 2025 (DistiLLM-2)** is the current SOTA at 7–9B with contrastive CALD. **Xu et al. 2024 (Speculative KD)** robustifies OPD against weak student initializations. **Sessa et al. 2024 (BOND)** bridges OPD and RLHF. **The Gemma 2 tech report** (Team, 2024) is the cleanest industrial application. **The Qwen3 tech report** (Team, May 2025) is the empirical anchor proving OPD beats RL. **Thinking Machines Lab's \"On-Policy Distillation\" blog** (Lu et al., October 2025) is the most accessible synthesis with reproducible Tinker code.

## Industrial systems using OPD-like training

The frontier-class industrial picture, ordered by strictness, is as follows. **Strict OPD, explicitly named:** Qwen3 (Alibaba, arXiv:2505.09388, §3.4 Strong-to-Weak Distillation, *\"the student model generates on-policy sequences and is fine-tuned by aligning its logits with those of a teacher model to minimize KL divergence\"*); Qwen3-Omni (arXiv:2509.17765, multimodal extension); MiMo-V2-Flash (Xiaomi, arXiv:2601.02780, *\"domain-specialized teachers... provide dense and token-level reward, enabling the student model to perfectly master teacher expertise\"*); GLM-4.5/4.6 (Z.ai, arXiv:2508.06471, expert-merge self-distillation); Tencent HY-MT1.5 (arXiv:2512.24092); Gemma 2 (Google DeepMind, arXiv:2408.00118, §4 Post-Training, *\"distillation from the teacher on the student's distribution [Agarwal et al., 2024, Gu et al., 2024]\"*); Nemotron-Cascade 2 (NVIDIA, arXiv:2603.19220, §4.4 Multi-Domain On-Policy Distillation). **Partial OPD or inherits prior recipe:** Gemma 3 (arXiv:2503.19786). **Frequently misclassified — actually not OPD:** DeepSeek-R1-Distill (offline SFT on R1 traces); Phi-4 (PTS-DPO uses on-policy rollouts but scalar preference, not distributional — common misconception); Phi-4-reasoning (SFT on o3-mini traces + GRPO); Llama 3/3.1 (synthetic data SFT + DPO); Llama 3.2 (offline logit KD on pretraining corpus); Llama 4 codistillation (pretraining KD only); Mistral Ministral 3 Cascade Distillation (pretraining/teacher-rollout KD); Llama-Nemotron / Magistral / Skywork-OR1 / AceMath-RL (RLVR); Tülu 3 / OLMo 2 / OLMo 3 (RLVR); Kimi K1.5 long-to-short (offline SeqKD); Kimi K2 / DeepSeek-V3 / MiniMax-M1 / Hunyuan-Large / Yi-Lightning / Seed-Thinking / Doubao 1.5 / InternLM (no OPD step described); Constitutional AI / Claude (RLAIF, per task exclusion).

## Frameworks supporting OPD

Eight frameworks meet the strict OPD definition; the rest support adjacent capabilities. **TRL (Hugging Face)** is the canonical reference via `GKDTrainer` and the newer `GOLDTrainer` for cross-tokenizer settings, documented at huggingface.co/docs/trl/main/en/gkd_trainer. **veRL (ByteDance)** ships a dedicated `recipe/gkd/` with async on-policy distillation supporting `one_step_off` and `two_step_off` schedulers to overlap rollout, teacher logit serving, and actor updates; see verl.readthedocs.io. **ms-swift (Alibaba ModelScope)** wraps TRL's GKDTrainer with VLM/MLLM support and an OPSD privileged-context column; the docs explicitly cite the Thinking Machines recipe. **NeMo-RL (NVIDIA)** treats OPD as a first-class algorithm alongside GRPO/GSPO/DAPO; see docs.nvidia.com/nemo/rl. **Tinker cookbook (Thinking Machines Lab)** is the cleanest reverse-KL reference, used by Thinking Machines' own blog. **SkyRL (UC Berkeley NovaSky)** implements the Tinker API on user-owned GPUs and added an on-policy distillation example in v0.3. **slime (THUDM/Z.ai)** lists OPD as an Advanced Feature and powers GLM-4.5 onward. **Unsloth** wraps TRL's GKDTrainer with faster LoRA kernels (experimental).

Frameworks without native OPD support include **OpenRLHF, OpenInstruct (AI2), DeepSpeed-Chat, LlamaFactory, axolotl, TRLX (archived), AReaL, Llama-Recipes**, and **OpenPipe ART** (whose \"SFT distillation\" is off-policy teacher-data SFT, failing Clause 1). Several known limitations cut across the supporting frameworks: cross-tokenizer OPD has a correctness bug in TRL's GKD/MiniLLM trainers (Issue #4562, Nov 2025) — TRL's new GOLDTrainer and DSKDv2 research code address this. Multi-teacher OPD is not yet first-class in any production framework (NeMo-RL roadmap mentions it). Memory pressure from running a teacher forward each step is the main practical bottleneck; veRL's async schedulers and cached-teacher approaches are the main mitigations.

## README-style table

| Title | Year | Org | Strictness | Method Family | Code | Link |
|---|---|---|---|---|---|---|
| GKD / On-Policy Distillation of LMs | 2023/24 | Google DeepMind | Strict OPD | White-box OPD (gen. JSD) | TRL | https://arxiv.org/abs/2306.13649 |
| MiniLLM | 2023/24 | Tsinghua/MSRA | Strict OPD | White-box OPD (reverse KL, PG) | Yes | https://arxiv.org/abs/2306.08543 |
| DistiLLM | 2024 | KAIST/MS | Strict OPD | Skew KL + adaptive replay | Yes | https://arxiv.org/abs/2402.03898 |
| DistiLLM-2 | 2025 | KAIST/MS | Strict OPD | Contrastive CALD | Yes | https://arxiv.org/abs/2503.07067 |
| ImitKD | 2020 | ASAPP | Strict OPD | DAgger-style imitation KD | Limited | https://arxiv.org/abs/2009.07253 |
| f-distill | 2023 | U. Alberta/Mila | Partial OPD | f-divergence KD | Yes | https://arxiv.org/abs/2307.15190 |
| BOND | 2024 | Google DeepMind | Partial OPD | OPD+RL (BoN distillation) | No | https://arxiv.org/abs/2407.14622 |
| Speculative KD | 2024 | UCSB/Google | Strict OPD | Interleaved teacher-student | Yes | https://arxiv.org/abs/2410.11325 |
| DistillSpec | 2024 | Google DeepMind | Strict OPD | OPD for spec. decoding | No | https://arxiv.org/abs/2310.08461 |
| AKL | 2024 | HKU/Tencent | Adjacent | Adaptive KL divergence | Yes | https://arxiv.org/abs/2404.02657 |
| ToDi | 2025 | — | Strict OPD | Per-token adaptive KL | — | https://arxiv.org/abs/2505.16297 |
| DPKD | 2024 | MS/SJTU | Partial OPD | DPO-style KD | Yes | https://arxiv.org/abs/2406.19774 |
| PLaD | 2024 | Google Research | Partial OPD | Black-box ranking KD | No | https://arxiv.org/abs/2406.02886 |
| AlignDistil | 2025 | BJTU/Tencent | Strict OPD | Token-level RLHF=KD | Yes | https://arxiv.org/abs/2503.02832 |
| GAD (Generative Adv. Distill) | 2025 | Microsoft | Strict OPD | Black-box adversarial OPD | Yes | https://arxiv.org/abs/2511.10643 |
| TAID | 2025 | Sakana AI | Partial OPD | Adaptive interpolated KD | Yes | https://arxiv.org/abs/2501.16937 |
| ULD | 2024/25 | Diabolocom | Adjacent | Cross-tokenizer logit KD | Yes | https://arxiv.org/abs/2402.12030 |
| DSKD / DSKDv2 | 2024/25 | BJTU | Partial→Strict (v2) | Dual-space cross-tok. KD | Yes | https://arxiv.org/abs/2406.17328 |
| Distillation Scaling Laws | 2025 | Apple | Adjacent | Empirical scaling theory | No | https://arxiv.org/abs/2502.08606 |
| Rethinking On-Policy Distillation | 2026 | THU NLP | Strict OPD | Diagnostic + recipe | Yes | https://arxiv.org/abs/2604.13016 |
| OPSD / Self-Distilled Reasoner | 2026 | UCLA/Meta | Strict OPD | OPSD privileged-context | Yes | https://arxiv.org/abs/2601.18734 |
| π-Distill | 2026 | — | Strict OPD | OPSD privileged-context | — | https://arxiv.org/abs/2602.04942 |
| Entropy-Aware OPD | 2026 | — | Strict OPD | RKL+FKL hybrid | — | https://arxiv.org/abs/2603.07079 |
| Near-Policy Distillation | 2026 | — | Strict OPD | Async OPD acceleration | — | https://arxiv.org/abs/2605.05940 |
| Survey of OPD for LLMs | 2026 | — | Survey | — | — | https://arxiv.org/abs/2604.00626 |
| Thinking Machines OPD blog | 2025 | TML | Strict OPD | Tinker reverse-KL recipe | Yes | https://thinkingmachines.ai/blog/on-policy-distillation/ |
| HF GOLD blog | 2025 | Hugging Face | Strict OPD | Cross-tokenizer OPD | TRL | https://huggingface.co/docs/trl/main/en/gold_trainer |
| HF GKD docs | 2024/25 | Hugging Face | Strict OPD | Production GKD trainer | TRL | https://huggingface.co/docs/trl/gkd_trainer |
| NVIDIA Minitron blog | 2024 | NVIDIA | Adjacent | Prune+KD recipe | Yes | https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-... |
| Qwen3 | 2025 | Alibaba | Strict OPD | Industrial S2W OPD | Weights | https://arxiv.org/abs/2505.09388 |
| Qwen3-Omni | 2025 | Alibaba | Strict OPD | Multimodal OPD | Weights | https://arxiv.org/abs/2509.17765 |
| MiMo-V2-Flash | 2026 | Xiaomi | Strict OPD | MOPD multi-teacher | Yes | https://arxiv.org/abs/2601.02780 |
| GLM-4.5 / 4.6 | 2025–26 | Z.ai | Strict OPD | Self-distill expert merge | slime | https://arxiv.org/abs/2508.06471 |
| HY-MT1.5 | 2025–26 | Tencent | Strict OPD | Translation OPD | No | https://arxiv.org/abs/2512.24092 |
| Gemma 2 | 2024 | Google DeepMind | Strict OPD | Industrial OPD (post-SFT) | Weights | https://arxiv.org/abs/2408.00118 |
| Gemma 3 | 2025 | Google DeepMind | Partial OPD | Inherits Gemma 2 recipe | Weights | https://arxiv.org/abs/2503.19786 |
| Nemotron-Cascade 2 | 2026 | NVIDIA | Strict OPD | Cascade RL + MOPD | Weights | https://arxiv.org/abs/2603.19220 |
| Nemotron Nano 2 | 2025 | NVIDIA | Not OPD | Prune+KD+GRPO | Yes | https://arxiv.org/abs/2508.14444 |
| Llama-Nemotron | 2025 | NVIDIA | Not OPD | Reasoning-SFT+RLVR | Yes | https://arxiv.org/abs/2505.00949 |
| Phi-4 | 2024 | Microsoft | Not OPD (≠) | SFT+PTS-DPO | No | https://arxiv.org/abs/2412.08905 |
| Phi-4-reasoning | 2025 | Microsoft | Not OPD | SFT-on-o3 traces+RLVR | Weights | https://arxiv.org/abs/2504.21318 |
| Llama 3.1 Herd | 2024 | Meta | Not OPD | SFT+RS+DPO | Weights | https://arxiv.org/abs/2407.21783 |
| Llama 4 Codistillation | 2025 | Meta | Not OPD | Pretraining KD | Weights | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ |
| Mistral Ministral 3 | 2026 | Mistral | Not OPD | Cascade pretraining KD | Weights | https://arxiv.org/abs/2601.08584 |
| Magistral | 2025 | Mistral | Not OPD | Pure RLVR | Weights | https://arxiv.org/abs/2506.10910 |
| Tülu 3 | 2024 | AI2 | Not OPD | SFT+DPO+RLVR | Yes | https://arxiv.org/abs/2411.15124 |
| OLMo 2 / 3 | 2024–25 | AI2 | Not OPD | Tülu 3 recipe | Yes | https://allenai.org/blog/olmo3 |
| Constitutional AI / Claude | 2022 | Anthropic | Not OPD (RLAIF) | RLAIF | No | https://arxiv.org/abs/2212.08073 |
| DeepSeek-R1 + Distill | 2025 | DeepSeek | Not OPD (≠) | Offline SFT on R1 traces | Weights | https://arxiv.org/abs/2501.12948 |
| DeepSeek-V3 | 2024 | DeepSeek | Adjacent | R1-SFT + GRPO | Weights | https://arxiv.org/abs/2412.19437 |
| Kimi K1.5 | 2025 | Moonshot | Not OPD | RL + long-to-short SeqKD | No | https://arxiv.org/abs/2501.12599 |
| Kimi K2 | 2025 | Moonshot | Not OPD | Agentic RL + self-critic | Weights | https://arxiv.org/abs/2507.20534 |
| Hunyuan-Large | 2024 | Tencent | Not OPD | SFT+RLHF | Weights | https://arxiv.org/abs/2411.02265 |
| Yi-Lightning | 2024 | 01.AI | Not OPD | SFT+RLHF+RAISE | Weights | https://arxiv.org/abs/2412.01253 |
| MiniMax-M1 | 2025 | MiniMax | Not OPD | CISPO RL | Weights | https://arxiv.org/abs/2506.13585 |
| Seed-Thinking v1.5 | 2025 | ByteDance | Not OPD | RL with dual rewards | No | https://arxiv.org/abs/2504.13914 |
| InternLM2 | 2024 | Shanghai AI Lab | Not OPD | COOL RLHF | Weights | https://arxiv.org/abs/2403.17297 |
| Ernie 4.5 / 5.0 | 2025–26 | Baidu | Not OPD | Self-distill for pretrain data | Weights | https://arxiv.org/abs/2602.04705 |
| AI21 Jamba 1.5 | 2024 | AI21 | Not OPD | DPO+RL with model reward | Weights | https://arxiv.org/abs/2408.12570 |
| OpenChat (C-RLFT) | 2023 | Tsinghua | Not OPD | Class-conditioned SFT | Yes | https://arxiv.org/abs/2309.11235 |
| SPIN | 2024 | UCLA | Partial OPD | Iterative self-play DPO | Yes | https://arxiv.org/abs/2401.01335 |
| Self-Rewarding LMs | 2024 | Meta/NYU | Not OPD (RLAIF) | LLM-as-judge iter. DPO | No | https://arxiv.org/abs/2401.10020 |
| Snorkel-Mistral-PairRM-DPO | 2024 | Snorkel | Not OPD | Iter. DPO + PairRM | Weights | https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO |
| ReST | 2023 | Google DeepMind | Partial OPD | Grow-improve self-train | No | https://arxiv.org/abs/2308.08998 |
| ReST^EM | 2024 | Google DeepMind | Partial OPD | EM-style self-train | No | https://arxiv.org/abs/2312.06585 |
| STaR | 2022 | Stanford/Google | Partial OPD | Iter. SFT on self CoT | No | https://arxiv.org/abs/2203.14465 |
| Quiet-STaR | 2024 | Stanford | Partial OPD | Token-wise rationale RL | Community | https://arxiv.org/abs/2403.09629 |
| V-STaR | 2024 | Mila/Google | Partial OPD | STaR + DPO verifier | No | https://arxiv.org/abs/2402.06457 |
| rStar-Math | 2025 | MSR Asia | Partial OPD | MCTS + PPM self-evolve | Yes | https://arxiv.org/abs/2501.04519 |
| ReST-MCTS* | 2024 | Tsinghua | Partial OPD | MCTS + PRM self-train | Yes | https://arxiv.org/abs/2406.03816 |
| AlphaLLM / AlphaLLM-CPL | 2024 | Tencent | Partial OPD | MCTS + critics | No | https://arxiv.org/abs/2404.12253 |
| AlphaMath Almost Zero | 2024 | Alibaba | Partial OPD | MCTS + value head | Yes | https://arxiv.org/abs/2405.03553 |
| RFT (Rejection-sample FT) | 2023 | DAMO/Alibaba | Partial OPD | Iter. SFT on filtered self | Yes | https://arxiv.org/abs/2308.01825 |
| SCoRe | 2024 | Google DeepMind | Not OPD | Multi-turn online RL | No | https://arxiv.org/abs/2409.12917 |
| RISE | 2024 | CMU/Google | Partial OPD | Multi-turn RWR | Yes | https://arxiv.org/abs/2407.18219 |
| Iterative RPO | 2024 | Meta/NYU | Partial OPD | Iter. DPO + NLL | No | https://arxiv.org/abs/2404.19733 |
| SPPO | 2024 | UCLA/CMU | Partial OPD | Self-play preference | Yes | https://arxiv.org/abs/2405.00675 |
| EVA | 2024 | Google DeepMind | Not OPD | Asymmetric self-play | No | https://arxiv.org/abs/2411.00062 |
| Self-Refine | 2023 | CMU et al. | Adjacent | Inference-time only | Yes | https://arxiv.org/abs/2303.17651 |
| SDFT | 2024 | Sea AI Lab | Adjacent | Off-policy self-rewrite | Yes | https://arxiv.org/abs/2402.13669 |
| Distilling System 2 into 1 | 2024 | Meta FAIR | Not OPD | Self-S2 → SFT | No | https://arxiv.org/abs/2407.06023 |
| PRIME | 2025 | Tsinghua/UIUC | Partial OPD | Implicit PRM token RL | Yes | https://arxiv.org/abs/2502.01456 |
| Implicit PRM | 2024 | Tsinghua/UIUC | Adjacent | Free PRM via log-ratio | Yes | https://arxiv.org/abs/2412.01981 |
| Math-Shepherd | 2023 | DeepSeek/PKU | Not OPD | PRM + step PPO | Partial | https://arxiv.org/abs/2312.08935 |
| PRM800K | 2023 | OpenAI | Adjacent | Human PRM data | Data | https://arxiv.org/abs/2305.20050 |
| OmegaPRM | 2024 | Google DeepMind | Adjacent | MCTS PRM data | No | https://arxiv.org/abs/2406.06592 |
| Qwen2.5-Math-PRM | 2025 | Alibaba | Adjacent | Open SOTA PRM | Weights | https://arxiv.org/abs/2501.07301 |
| Step-DPO | 2024 | CUHK | Not OPD | Step-pref DPO | Yes | https://arxiv.org/abs/2406.18629 |
| Step-KTO | 2025 | Meta | Not OPD | Step-binary KTO | No | https://arxiv.org/abs/2501.10799 |
| PAV | 2024 | Google/CMU | Not OPD | Process-advantage RL | Community | https://arxiv.org/abs/2410.08146 |
| Generative Verifiers (GenRM) | 2024 | Google DeepMind | Adjacent | LM-as-verifier | No | https://arxiv.org/abs/2408.15240 |
| Distilling Step-by-Step | 2023 | Google | Adjacent | Offline rationale KD | Yes | https://arxiv.org/abs/2305.02301 |
| ReFT | 2024 | ByteDance | Not OPD | SFT + outcome PPO | Yes | https://arxiv.org/abs/2401.08967 |
| DeepSeek-Prover-V1.5 | 2024 | DeepSeek | Not OPD | RLPAF + RMaxTS | Yes | https://arxiv.org/abs/2408.08152 |
| Sky-T1-32B-Preview | 2025 | NovaSky/Berkeley | Not OPD | Offline SFT from QwQ | Yes | https://novasky-ai.github.io/posts/sky-t1/ |
| s1 / s1.1 | 2025 | Stanford et al. | Not OPD | Offline SFT, 1k samples | Yes | https://arxiv.org/abs/2501.19393 |
| LIMO | 2025 | GAIR/SJTU | Not OPD | Offline SFT, 817 samples | Yes | https://arxiv.org/abs/2502.03387 |
| LIMR | 2025 | GAIR | Not OPD | RLVR + sample selection | Yes | https://arxiv.org/abs/2502.11886 |
| TinyR1-32B | 2025 | — | Not OPD | Branch-merge SFT | Yes | https://arxiv.org/abs/2503.04872 |
| Light-R1 | 2025 | Qihoo 360 | Not OPD | SFT+DPO+GRPO | Yes | https://arxiv.org/abs/2503.10460 |
| Skywork-OR1 | 2025 | Skywork | Not OPD | RLVR on R1-Distill | Yes | https://arxiv.org/abs/2505.22312 |
| SimpleRL-Zoo | 2025 | HKUST | Not OPD | RLVR cross-base study | Yes | https://arxiv.org/abs/2503.18892 |
| STILL-2 | 2024 | RUC AI Box | Partial OPD | Imitate-explore-improve | Yes | https://arxiv.org/abs/2412.09413 |
| Open-R1 | 2025 | Hugging Face | Not OPD | SFT + GRPO replication | Yes | https://github.com/huggingface/open-r1 |
| NuminaMath dataset | 2024–25 | Project Numina | Not OPD | Distilled SFT dataset | Yes | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT |
| OpenMathInstruct-2 | 2024 | NVIDIA | Not OPD | 14M SFT dataset | Yes | https://arxiv.org/abs/2410.01560 |
| OpenCodeReasoning | 2025 | NVIDIA | Not OPD | Code SFT distillation | Yes | https://arxiv.org/abs/2504.01943 |
| MetaMath | 2023 | Cambridge/HKUST | Not OPD | Question-bootstrap SFT | Yes | https://arxiv.org/abs/2309.12284 |
| AceMath / AceMath-RL | 2024–25 | NVIDIA | Not OPD | SFT + RLVR | Weights | https://arxiv.org/abs/2412.15084 |
| Magicoder / WizardCoder | 2023 | UIUC / MS | Not OPD | OSS-Instruct/Evol-Inst | Yes | https://arxiv.org/abs/2312.02120 |
| TokenSkip | 2025 | HKPU/MSRA | Not OPD | CoT compression SFT | Yes | https://arxiv.org/abs/2502.12067 |
| FireAct | 2023 | Princeton et al. | Not OPD | Agent SFT from GPT-4 | Yes | https://arxiv.org/abs/2310.05915 |
| AgentTuning / AgentLM | 2023 | THUDM | Not OPD | Multi-task agent SFT | Yes | https://arxiv.org/abs/2310.12823 |
| ToolLLM / ToolBench | 2023 | OpenBMB | Not OPD | DFSDT agent SFT | Yes | https://arxiv.org/abs/2307.16789 |
| Gorilla | 2023 | UC Berkeley | Not OPD | API SFT | Yes | https://arxiv.org/abs/2305.15334 |
| Agent-FLAN | 2024 | Shanghai AI Lab | Not OPD | Decomposed agent SFT | Yes | https://arxiv.org/abs/2403.12881 |
| AgentInstruct (Orca) | 2024 | Microsoft | Not OPD | Agentic data synthesis | Data | https://arxiv.org/abs/2407.03502 |
| ETO | 2024 | AI2 et al. | Partial OPD | Trajectory DPO on failures | Yes | https://arxiv.org/abs/2403.02502 |
| AutoAct | 2024 | ZJU/Alibaba | Partial OPD | Self-synthesized agent | Yes | https://arxiv.org/abs/2401.05268 |
| AgentGym / AgentEvol | 2024 | Fudan NLP | Partial OPD | 14-env self-evolve | Yes | https://arxiv.org/abs/2406.04151 |
| UltraInteract / Eurus | 2024 | OpenBMB/Tsinghua | Partial OPD | Tree-preference KTO | Yes | https://arxiv.org/abs/2404.02078 |
| xLAM / APIGen-MT | 2024–25 | Salesforce | Not OPD | Function-calling SFT | Yes | https://arxiv.org/abs/2409.03215 |
| ToRA | 2023 | MS/Tsinghua | Partial OPD | Tool reasoning + shaping | Yes | https://arxiv.org/abs/2309.17452 |
| Structured Agent Distillation | 2025 | CMU et al. | Strict OPD | Span-aware online KD | — | https://arxiv.org/abs/2505.13820 |
| WebRL | 2024 | THUDM | Not OPD | Curriculum web RL | Yes | https://arxiv.org/abs/2411.02337 |
| SWE-RL | 2025 | Meta | Not OPD | Similarity-reward RL | No | https://arxiv.org/abs/2502.18449 |
| Search-R1 | 2025 | UIUC | Not OPD | RLVR search agent | Yes | https://arxiv.org/abs/2503.09516 |
| SWE-Gym | 2024 | Berkeley/Apple | Partial OPD | Verifier-guided SWE | Yes | https://arxiv.org/abs/2412.21139 |
| WizardLM / Evol-Instruct | 2023 | MS | Adjacent | Data synthesis | Yes | https://arxiv.org/abs/2304.12244 |
| Magpie | 2024 | — | Adjacent | Self-prompted data | Yes | https://arxiv.org/abs/2406.08464 |
| UltraChat / UltraFeedback | 2023 | OpenBMB | Adjacent | Synthetic datasets | Yes | https://arxiv.org/abs/2305.14233 |

## Markdown-table master entry table (pipe-delimited)

```
Title|Year|Org|Type|MethodFamily|TeacherAccess|SupervisionSignal|RolloutSource|Loss|Stage|Backbone|Benchmarks|Code|Repro|Strictness
GKD (On-Policy Distillation of LMs)|2023/24|Google DeepMind|Paper|White-box OPD|White-box logits|Generalized JSD/FKL/RKL|Mixed (lambda)|GKD loss|Post-SFT|T5/FLAN-T5|XSum,WMT,GSM8K|TRL impl|High|Strict
MiniLLM|2023/24|Tsinghua+MSRA|Paper+Code|White-box OPD (PG)|White-box logits|Reverse KL via PG|Student on-policy (teacher-mixed)|RKL+REINFORCE|Post-SFT|GPT-2,OPT,LLaMA-7B|Dolly,SelfInst,Vicuna|Yes|High|Strict
DistiLLM|2024|KAIST+MS|Paper+Code|Skew KL + replay|White-box|SKL/SRKL|Adaptive student+replay|SKL/SRKL|Post-SFT|GPT-2,OpenLLaMA,LLaMA-7B|Inst-following,summ,MT|Yes|High|Strict
DistiLLM-2|2025|KAIST+MS|Paper+Code|Contrastive CALD|White-box|SKL(teacher)+SRKL(student)|Both|CALD asymmetric|Post-SFT|Qwen2-7B,Mistral-7B,Gemma2-9B|AlpacaEval LC,MT-Bench,HumanEval|Yes|High|Strict
ImitKD|2020|ASAPP|Paper|DAgger imitation KD|White-box|NLL on student-visited|DAgger mixed|FKL/NLL|SFT|Transformer NMT|WMT,IWSLT,CNN/DM|Limited|Med|Strict
f-distill|2023|U.Alberta/Mila|Paper+Code|f-divergence KD|White-box|Any f-divergence|Variant-dependent|D_f|Post-SFT|T5,BART,GPT-2|DART,WebNLG,XSum|Yes|Med-High|Partial
BOND|2024|Google DeepMind|Paper|OPD+RL|Reward model|Jeffreys KL to BoN|Student on-policy|D_Jeffreys|RL|Gemma 2B/7B|XSum,arena|No|Low-Med|Partial
Speculative KD|2024|UCSB+Google|Paper+Code|Interleaved OPD|White-box+verifier|Token KL on interleaved|Mixed student-propose+teacher-replace|KL|Post-SFT|GPT-2,T5,Gemma,mT5|WMT,XSum,GSM8K|Yes|Med|Strict
DistillSpec|2024|Google DeepMind|Paper|White-box OPD (spec dec)|White-box target|f-divergence|Draft on-policy|D_f|Spec decoding|T5 v1.1|XSum,GSM8K,WMT|No|Low-Med|Strict
AKL|2024|HKU+Tencent|Paper+Code|Adaptive KL|White-box|Adaptive FKL+RKL|Teacher data|AKL|Post-SFT|GPT-2,LLaMA-6.7B|Inst-following|Yes|Med|Adjacent
ToDi|2025|—|Paper|Per-token adaptive KL|White-box|Sigmoid-mixed FKL/RKL|Student on-policy|Per-token mix|Post-SFT|Small LMs|Inst-following|—|Med|Strict
DPKD|2024|MS+SJTU|Paper+Code|DPO-style KD|White-box+ref|RKL+DPO|Both|2-stage|Post-SFT|GPT-2,OPT|Dolly,SelfInst,S-NI|Yes|High|Partial
PLaD|2024|Google Research|Paper|Black-box ranking|Black-box samples|Pairwise ranking|Both|Pairwise|Post-SFT|PaLM2->T5|TL;DR,inst|No|Med|Partial
AlignDistil|2025|BJTU+Tencent|Paper+Code|Token-level RLHF=KD|DPO+ref logits|Token KL to synth teacher|Both|Token KL|Alignment|Llama-3,Mistral-7B|AlpacaEval2,Arena-Hard|Yes|High|Strict
GAD|2025|Microsoft|Paper+Code|Black-box adversarial OPD|Black-box samples|Discriminator on-policy reward|Student on-policy|GRPO+adv|Post-SFT|Qwen2.5-14B|LMSYS-Chat|Yes|High|Strict
TAID|2025|Sakana AI|Paper+Code|Adaptive interpolated KD|White-box|Time-varying interp dist|Mixed|TAID|Pretrain+SFT|Various|Inst-following|Yes|High|Partial
ULD|2024/25|Diabolocom|Paper+Code|Cross-tokenizer logit KD|Cross-tok logits|Wasserstein on sorted logits|Off-policy|OT loss|Post-SFT|Cross-family|Inst-following|Yes|High|Adjacent
DSKD/v2|2024/25|BJTU|Paper+Code|Dual-space cross-tok KD|White-box (proj)|Token KL via projector|Off->On (v2)|Projector+KD|Post-SFT|Cross-family|Inst-following|Yes|High|Partial(v1)/Strict(v2)
Distillation Scaling Laws|2025|Apple|Paper|Empirical theory|White-box|Power-law analysis|Pretrain|—|Pretrain|Multiple|—|No|—|Adjacent
Rethinking OPD|2026|THU NLP|Paper+Code|Diagnostic+recipe|White-box|Token KL+cold-start|Student on-policy|GKD+SFT seed|Post-SFT|Qwen3 fam|AIME,MATH|Yes|High|Strict
OPSD / Self-Distilled Reasoner|2026|UCLA+Meta|Paper+Code|OPSD privileged-context|Self+privileged|Token-level KL|Student on-policy|FKL/RKL|Post-SFT|Qwen3-1.7B/4B|AIME,AMC|Yes|High|Strict
pi-Distill|2026|—|Paper|OPSD|Self+privileged|Per-token KL|Student|FKL|Post-SFT|Qwen/Llama|Math|—|Med|Strict
Entropy-Aware OPD|2026|—|Paper|RKL+FKL hybrid|White-box|Entropy-gated mix|Student|Mix|Post-SFT|Qwen3-0.6/1.7/4B|Math Pass@8|—|Med|Strict
Near-Policy Distillation|2026|—|Paper|Async OPD acceleration|White-box|Async KL|Near on-policy|—|Post-SFT|Various|—|—|Med|Strict
Survey of OPD for LLMs|2026|—|Survey|Taxonomy|—|—|—|—|—|—|—|—|—|Survey
Thinking Machines OPD blog|2025|TML|Blog+Code|Tinker reverse-KL recipe|White-box|Per-token RKL|Student on-policy|RKL|Post-SFT|Qwen3-8B/32B/235B|AIME,IF-eval|Yes|High|Strict
HF GOLD blog|2025|HuggingFace|Blog+TRL code|Cross-tokenizer OPD|Cross-tok logits|JSD+ULD hybrid|Student on-policy|GKD+ULD|Post-SFT|Llama-3.2,Qwen2.5|Countdown,OpenR1-Math|Yes|High|Strict
HF GKD docs|2024/25|HuggingFace|Docs+Code|Production GKD trainer|White-box|JSD/FKL/RKL/seqKD|Mixed (lmbda)|GKD|Post-SFT|Any HF causal LM|—|Yes|High|Strict
NVIDIA Minitron blog|2024|NVIDIA|Blog+Code|Prune+KD recipe|White-box|FKL on teacher data|Teacher data|FKL|Pretrain|Llama-3.1 8B->4B|MMLU,etc|Yes|High|Adjacent
Qwen3|2025|Alibaba|Tech report|Industrial S2W OPD|White-box|Token KL|Student on-policy|KL|Post-SFT|Qwen3 0.6B-30B-A3B|AIME,LiveCodeBench|Weights|Med|Strict
Qwen3-Omni|2025|Alibaba|Tech report|Multimodal OPD|White-box|Token KL|Student on-policy|KL+GSPO|Post-SFT|Qwen3-Omni|Multimodal|Weights|Med|Strict
MiMo-V2-Flash|2026|Xiaomi|Tech report+Code|MOPD multi-teacher|Multiple white-box|Token-level dense reward|Student on-policy|Multi-teacher KL+RL|Post-SFT|309B/15B-active MoE|Reasoning,agentic|Yes|Med|Strict
GLM-4.5/4.6|2025-26|Z.ai|Tech report+Code|Self-distill expert merge|Internal experts|Logit-level + outcome|Student rollouts|Self-distill loss|Unified stage|GLM-4.5 355B-A32B|Agentic,reasoning|slime|Med|Strict
HY-MT1.5|2025-26|Tencent|Tech report|Translation OPD|White-box|—|Student rollouts|—|Pipeline stage|HY-MT1.5 1.8B/7B|MT|No|Low|Strict
Gemma 2|2024|Google DeepMind|Tech report|Industrial OPD (post-SFT)|White-box (256k vocab sample)|Token KL|Student|KL|Post-SFT|Gemma2 2B/9B/27B|MMLU,GSM8K,MATH,HumanEval|Weights|Med|Strict
Gemma 3|2025|Google DeepMind|Tech report|Inherits Gemma 2 recipe|White-box|—|—|—|Post-training|Gemma3 1-27B|Various|Weights|Low-Med|Partial
Nemotron-Cascade 2|2026|NVIDIA|Tech report|Cascade RL + MOPD|Multiple white-box|Token KL on student rollouts|Student on-policy|MOPD+GRPO|Post-RL stabilize|Nemotron-3-Nano-30B-A3B|AIME,IMO,IOI,SWE-V|Weights|Med|Strict
Nemotron Nano 2|2025|NVIDIA|Tech report|Prune+KD+GRPO|White-box pretrain|FKL pretrain|Pretrain|—|Pretrain+RL|12B->9B|Various|Yes|Med|NotOPD
Llama-Nemotron|2025|NVIDIA|Tech report|Reasoning-SFT+RLVR|R1 traces|—|Teacher SFT|—|SFT+RLVR|Llama-3 NAS|AIME,GPQA|Yes|Med|NotOPD
Phi-4|2024|Microsoft|Tech report|SFT+PTS-DPO|—|Token-pivotal preference|Student on-policy|DPO|SFT+DPO|14B|MATH,MMLU,etc|No|Low|NotOPD
Phi-4-reasoning|2025|Microsoft|Tech report|SFT-on-o3+RLVR|o3-mini traces|—|Teacher SFT then student|SFT+GRPO|SFT+RL|14B|AIME,GPQA|Weights|Low-Med|NotOPD
Llama 3.1|2024|Meta|Tech report|SFT+RS+DPO|—|—|Synth+student|—|SFT+DPO|8B/70B/405B|Various|Weights|Med|NotOPD
Llama 4 Codistillation|2025|Meta|Blog|Pretraining KD|White-box pretrain|Soft+hard mix|Pretrain data|—|Pretrain|Scout/Maverick MoE|—|Weights|Low|NotOPD
Mistral Ministral 3|2026|Mistral|Tech report|Cascade pretrain KD|White-box|FKL|Teacher rollouts|FKL|Pretrain+post|3/8/14B|Various|Weights|Med|NotOPD
Magistral|2025|Mistral|Tech report|Pure RLVR|—|—|Student on-policy|GRPO|RL|Mistral S/M|Reasoning|Weights|Med|NotOPD
Tulu 3|2024|AI2|Tech report+Code|SFT+DPO+RLVR|—|Verifiable reward|Student on-policy|RLVR|RL|Llama-3.1 8/70/405B|Tulu eval|Yes|High|NotOPD
OLMo 2/3|2024-25|AI2|Tech report+Code|Tulu 3 recipe|—|—|Student|—|SFT+DPO+RLVR|OLMo 7B-32B|Various|Yes|High|NotOPD
Constitutional AI|2022|Anthropic|Paper|RLAIF|RLAIF preference|Scalar reward via PM|Student|PPO|RL|Claude|—|No|Low|NotOPD
DeepSeek-R1 + Distill|2025|DeepSeek|Tech report|Offline SFT on R1 traces|R1 black-box|NLL on traces|Teacher rollouts|SFT NLL|Post-SFT|Qwen/Llama base|AIME,MATH,GPQA|Weights|Med|NotOPD
DeepSeek-V3|2024|DeepSeek|Tech report|R1-SFT + scalar RL|R1 black-box|—|Mixed|SFT+RL|Post-train|671B MoE|Various|Weights|Med|Adjacent
Kimi K1.5|2025|Moonshot|Tech report|RL + long-to-short SeqKD|Long-CoT Kimi|SFT NLL|Teacher rollouts|SeqKD+RL|Post-train|—|Reasoning|No|Low|NotOPD
Kimi K2|2025|Moonshot|Tech report|Agentic RL + self-critic|—|Verifiable+rubric|Student|MuonClip RL|RL|1T/32B MoE|Agentic|Weights|Med|NotOPD
Hunyuan-Large|2024|Tencent|Tech report|SFT+RLHF|—|—|Student|—|RL|389B/52B MoE|Various|Weights|Med|NotOPD
Yi-Lightning|2024|01.AI|Tech report|SFT+RLHF+RAISE|—|—|Student|—|RL|MoE|Various|Weights|Med|NotOPD
MiniMax-M1|2025|MiniMax|Tech report|CISPO RL|—|Verifiable|Student|CISPO|RL|456B/45.9B|TestCompute|Weights|Med|NotOPD
Seed-Thinking v1.5|2025|ByteDance|Tech report|RL with dual rewards|—|—|Student|—|RL|200B/20B MoE|Reasoning|No|Low|NotOPD
InternLM2|2024|Shanghai AI Lab|Tech report|COOL RLHF|—|—|Student|—|RL|Various|—|Weights|Med|NotOPD
Ernie 4.5/5.0|2025-26|Baidu|Tech report|Self-distill for pretrain data|—|—|—|—|Pretrain|—|—|Weights|Low|NotOPD
AI21 Jamba 1.5|2024|AI21|Tech report|DPO+RL|—|—|—|—|—|SSM-Transformer|—|Weights|Low|NotOPD
OpenChat C-RLFT|2023|Tsinghua|Paper+Code|Class-conditioned SFT|—|—|—|—|SFT|—|—|Yes|High|NotOPD
SPIN|2024|UCLA|Paper+Code|Iterative self-play DPO|Self vs human|DPO logistic|Student vs SFT data|DPO|Iter alignment|Zephyr-7B|MT-Bench,BigBench|Yes|High|Partial
Self-Rewarding LMs|2024|Meta+NYU|Paper|LLM-as-judge iter DPO|Self|Scalar 1-5|Student|DPO|Iter|Llama2-70B|AlpacaEval2|No|Med|NotOPD
Snorkel-Mistral-PairRM-DPO|2024|Snorkel|Model card|Iter DPO + PairRM|PairRM|Pairwise pref|Student|DPO|Iter|Mistral-7B|AlpacaEval2|Weights|Med-High|NotOPD
ReST|2023|Google DeepMind|Paper|Grow-improve self-train|RM scalar|Reward filter|Student|BC|Iter|MT|WMT|No|Low|Partial
ReST^EM|2024|Google DeepMind|Paper|EM-style self-train|Verifier|Binary filter|Student|SFT on correct|Iter|PaLM-2 S/L|MATH,APPS|No|Low|Partial
STaR|2022|Stanford/Google|Paper|Iter SFT on self CoT|Ground-truth|Binary filter|Student|SFT|Iter|GPT-J 6B|CSQA,GSM8K,ARC|No|Med|Partial
Quiet-STaR|2024|Stanford|Paper+Community|Token-wise rationale RL|Likelihood diff|Scalar|Student|REINFORCE|Iter|Mistral-7B|GSM8K,CSQA|Community|Med|Partial
V-STaR|2024|Mila/DeepMind|Paper|STaR + DPO verifier|Self verifier|Binary+DPO|Student|SFT+DPO|Iter|Llama-2|GSM8K,MATH|No|Low-Med|Partial
rStar-Math|2025|MSR Asia|Paper+Code|MCTS + PPM self-evolve|PPM|Step-scalar|Student MCTS|SFT|Iter|Qwen2.5-Math-7B,Phi3-mini|MATH,AIME|Yes|Med|Partial
ReST-MCTS*|2024|Tsinghua|Paper+Code|MCTS + PRM self-train|PRM|Step-value|Student MCTS|SFT+regr|Iter|Mistral-7B,Llama3|MATH,SciBench|Yes|High|Partial
AlphaLLM|2024|Tencent|Paper|MCTS + critics|Value+PRM+ORM|Step|Student MCTS|SFT|Iter|Llama-2 70B|GSM8K,MATH|No|Low-Med|Partial
AlphaMath Almost Zero|2024|Alibaba|Paper+Code|MCTS + value head|Co-trained value|Step-value+SFT|Student MCTS|Joint CE+regression|Iter|DeepSeekMath-7B|MATH,GSM8K|Yes|High|Partial
RFT|2023|DAMO/Alibaba|Paper|Iter SFT on filtered self|Ground-truth|Binary filter|Student|SFT|Iter|LLaMA-7B|GSM8K|Yes|High|Partial
SCoRe|2024|Google DeepMind|Paper|Multi-turn online RL|Outcome reward|Scalar+KL|Student|REINFORCE+KL|RL|Gemini 1.x|MATH,HumanEval|No|Low|NotOPD
RISE|2024|CMU+Google|Paper+Code|Multi-turn RWR|Reward or oracle|RWR/DAgger|Student multi-turn|RWR|Iter|Llama-2/3,Mistral|GSM8K,MATH|Yes|Med|Partial
Iterative RPO|2024|Meta+NYU|Paper|Iter DPO + NLL|Correctness|Pref+NLL|Student|DPO+NLL|Iter|Llama-2-70B-Chat|GSM8K,MATH|No|Low-Med|Partial
SPPO|2024|UCLA+CMU|Paper+Code|Self-play preference|PairRM scalar|Squared loss|Student|SPPO|Iter|Mistral-7B,Llama-3-8B,Gemma2-9B|AlpacaEval2|Yes|High|Partial
EVA|2024|Google DeepMind|Paper|Asymmetric self-play|RM|Reward|Solver+creator|DPO/RLHF|Iter|Gemma-2-9B-it|Arena-Hard|No|Low|NotOPD
Self-Refine|2023|CMU et al.|Paper+Code|Inference-time only|—|—|None (no training)|—|Inference|—|Various|Yes|—|Adjacent
SDFT|2024|Sea AI Lab|Paper+Code|Off-policy self-rewrite|Seed self|CE on rewrites|Off-policy|CE|SFT|—|—|Yes|High|Adjacent
Distilling System 2 into 1|2024|Meta FAIR|Paper|Self-S2 -> SFT|Self|SFT NLL|Teacher rollouts|SFT|SFT|Llama-2-70B|Various|No|Med|NotOPD
PRIME|2025|Tsinghua+UIUC|Paper+Code|Implicit PRM token RL|Implicit PRM|Token-level dense reward|Student on-policy|RLOO/PPO|RL|Qwen2.5-Math-7B|AIME,MATH,AMC,LCB|Yes|High|Partial
Implicit PRM|2024|Tsinghua+UIUC|Paper+Code|Free PRM via log-ratio|Implicit|Token reward|Offline data|CE or DPO|Verifier train|Llama-3.1-Inst|MATH BoN|Yes|High|Adjacent
Math-Shepherd|2023|DeepSeek+PKU|Paper|PRM + step PPO|PRM|Step scalar|Student|PPO|RL|Mistral-7B,DeepSeekMath-7B|GSM8K,MATH|Partial|Med|NotOPD
PRM800K|2023|OpenAI|Paper+Data|Human PRM data|Human|Step CE|Pre-generated|CE|Verifier|GPT-4|MATH|Data|High|Adjacent
OmegaPRM|2024|Google DeepMind|Paper|MCTS PRM data|PRM|Step CE|Pre-generated|CE|Verifier|Gemini,Gemma2-27B|MATH,GSM8K|No|Low|Adjacent
Qwen2.5-Math-PRM|2025|Alibaba|Paper+Weights|Open SOTA PRM|PRM|Step CE|Pre-generated|CE|Verifier|Qwen2.5-Math 7/72B|ProcessBench|Weights|High|Adjacent
Step-DPO|2024|CUHK|Paper+Code|Step-pref DPO|Self|Step-pref|Self-mined|DPO|Alignment|Qwen2-72B-Inst|MATH,GSM8K|Yes|High|NotOPD
Step-KTO|2025|Meta|Paper|Step-binary KTO|PRM+ORM|Step-binary|Self-iter|KTO|Iter|Llama|MATH-500|No|Med|NotOPD
PAV|2024|Google+CMU|Paper+Community|Process-advantage RL|PAV|Step-advantage|Student|PG|RL|Gemma 2/9/27B|MATH|Community|Med|NotOPD
Generative Verifiers (GenRM)|2024|Google DeepMind|Paper|LM-as-verifier|GenRM|Next-token CoT|Pre-generated|CE|Verifier|Gemma|GSM8K,MATH,algorithmic|No|Med|Adjacent
Distilling Step-by-Step|2023|Google|Paper+Code|Offline rationale KD|Privileged-context teacher|Token CE on rationales|Teacher offline|alpha CE+rationale|SFT|T5|e-SNLI,ANLI,CQA,SVAMP|Yes|High|Adjacent
ReFT|2024|ByteDance|Paper+Code|SFT + outcome PPO|Verifier|Scalar outcome|Student|PPO|RL|—|GSM8K,MathQA,SVAMP|Yes|High|NotOPD
DeepSeek-Prover-V1.5|2024|DeepSeek|Paper|RLPAF + RMaxTS|Lean verifier|Verified pass/fail|Student MCTS|PPO+intrinsic|RL|—|miniF2F,ProofNet|Yes|High|NotOPD
Sky-T1-32B-Preview|2025|NovaSky/Berkeley|Blog+Code|Offline SFT from QwQ|QwQ+GPT-4o|NLL|Teacher rollouts|SFT|SFT|Qwen2.5-32B-Inst|AIME,MATH-500,LCB|Yes|High|NotOPD
s1/s1.1|2025|Stanford et al.|Paper+Code|Offline SFT 1k samples|Gemini/R1|NLL|Teacher rollouts|SFT|SFT|Qwen2.5-32B-Inst|AIME,MATH-500,GPQA|Yes|High|NotOPD
LIMO|2025|GAIR/SJTU|Paper+Code|Offline SFT 817 samples|R1/R1-Distill|NLL|Teacher rollouts|SFT|SFT|Qwen2.5-32B-Inst|AIME,MATH|Yes|High|NotOPD
LIMR|2025|GAIR|Paper|RLVR + sample selection|Verifier|Outcome|Student|GRPO|RL|Qwen-2.5-Math-7B|MATH|Yes|High|NotOPD
TinyR1-32B|2025|—|Paper|Branch-merge SFT|R1|NLL|Teacher|SFT+merge|SFT|R1-Distill-Qwen-32B|AIME|Yes|High|NotOPD
Light-R1|2025|Qihoo 360|Paper|SFT+DPO+GRPO|R1|NLL+DPO+RL|Mixed|SFT+DPO+GRPO|Multi-stage|Qwen2.5-32B-Inst|AIME,MATH|Yes|High|NotOPD
Skywork-OR1|2025|Skywork|Paper+Code|RLVR on R1-Distill|Verifier|Outcome|Student|GRPO|RL|R1-Distill-Qwen-7B/32B|AIME,LCB|Yes|High|NotOPD
SimpleRL-Zoo|2025|HKUST|Paper|RLVR cross-base study|Verifier|Outcome|Student|RLVR|RL|10 bases|MATH|Yes|High|NotOPD
STILL-2|2024|RUC AI Box|Paper+Code|Imitate-explore-improve|R1/QwQ + verifier|NLL+filter|Mixed|SFT|Multi-stage|Qwen2.5-32B-Inst|MATH,AIME|Yes|High|Partial
Open-R1|2025|HuggingFace|Repo+Blog|SFT + GRPO replication|R1|NLL+verifier|Mixed|SFT+GRPO|Multi-stage|Qwen|MATH,AIME,LCB|Yes|High|NotOPD
NuminaMath dataset|2024-25|Project Numina|Paper+Data|Distilled SFT dataset|GPT-4/o|NLL|Teacher|SFT|—|DeepSeek-Math-7B|AIMO,MATH,GSM8K|Yes|High|NotOPD
OpenMathInstruct-2|2024|NVIDIA|Paper+Data|14M SFT dataset|Llama-3.1-405B|NLL|Teacher|SFT|—|Llama-3.1-8/70B|MATH|Yes|High|NotOPD
OpenCodeReasoning|2025|NVIDIA|Paper+Data|Code SFT distillation|R1|NLL|Teacher|SFT|—|Qwen2.5 7/14/32B|LiveCodeBench|Yes|High|NotOPD
MetaMath|2023|Cambridge/HKUST|Paper+Code|Question-bootstrap SFT|GPT-3.5|NLL|Synth|SFT|—|LLaMA-2|GSM8K,MATH|Yes|High|NotOPD
AceMath/AceMath-RL|2024-25|NVIDIA|Paper+Weights|SFT + RLVR|GPT-4 + RM|NLL+ORM|Mixed|SFT+GRPO|Multi-stage|Qwen2.5-Math|MATH,AIME|Weights|High|NotOPD
Magicoder/WizardCoder|2023|UIUC/MS|Paper+Code|OSS/Evol-Instruct|GPT-3.5/4|NLL|Teacher|SFT|—|CodeLlama,DeepSeek-Coder|HumanEval,MBPP|Yes|High|NotOPD
TokenSkip|2025|HKPU/MSRA|Paper+Code|CoT compression SFT|Importance+R1|NLL on compressed|Offline|SFT|—|Qwen2.5 7/14B|GSM8K|Yes|High|NotOPD
FireAct|2023|Princeton et al.|Paper+Code|Agent SFT from GPT-4|GPT-4|NLL|Teacher|SFT|—|Llama-2/CodeLlama|HotpotQA,StrategyQA|Yes|High|NotOPD
AgentTuning|2023|THUDM|Paper+Code|Multi-task agent SFT|GPT-3.5/4|NLL|Teacher|SFT|—|Llama-2 7/13/70B|6 envs+held-out|Yes|High|NotOPD
ToolLLM/ToolBench|2023|OpenBMB|Paper+Code|DFSDT agent SFT|ChatGPT DFSDT|NLL|Teacher|SFT|—|LLaMA-7B|ToolEval|Yes|High|NotOPD
Gorilla|2023|UC Berkeley|Paper+Code|API SFT|GPT-4 self-instruct|NLL|Teacher|SFT|—|LLaMA-7B|APIBench,BFCL|Yes|High|NotOPD
Agent-FLAN|2024|Shanghai AI Lab|Paper+Code|Decomposed agent SFT|GPT-4|NLL+negatives|Teacher|SFT|—|Llama-2|Held-in+out|Yes|High|NotOPD
AgentInstruct|2024|Microsoft|Paper+Data|Agentic data synthesis|GPT-4 flows|NLL|Teacher synth|SFT|—|Mistral-7B|AGIEval,MMLU,etc|Data|High|NotOPD
ETO|2024|AI2 et al.|Paper+Code|Trajectory DPO on failures|GPT-4 expert|Trajectory pref|Student fails+expert|SFT+DPO|Iter|Llama-2-7B|WebShop,SciWorld,ALFWorld|Yes|High|Partial
AutoAct|2024|ZJU+Alibaba|Paper+Code|Self-synthesized agent|Self meta-agent|Self-filter|Self|SFT|—|Llama-2|HotpotQA,SciQA|Yes|High|Partial
AgentGym/AgentEvol|2024|Fudan NLP|Paper+Code|14-env self-evolve|Env reward|Filter|Student|SFT+self-imit|Iter|Llama-2-7B|14 envs|Yes|High|Partial
UltraInteract/Eurus|2024|OpenBMB|Paper+Code|Tree-preference KTO|GPT-3.5+Python|Pref+CE|Mixed actor|SFT+KTO/NCA|—|Mistral-7B,CL-70B,Mixtral|LeetCode,TheoremQA|Yes|High|Partial
xLAM/APIGen-MT|2024-25|Salesforce|Paper+Code|Function-calling SFT|Strong LLM+exec verif|NLL|Teacher|SFT|—|Llama-3,Qwen-2.5 1-70B|BFCL,tau-bench|Yes|High|NotOPD
ToRA|2023|MS+Tsinghua|Paper+Code|Tool reasoning + shaping|GPT-4 + teacher correction|NLL+shaping|Student-resample+teacher-fix|SFT+shaping|Multi-stage|Llama-2,CodeLlama|MATH,GSM8K|Yes|High|Partial
Structured Agent Distillation|2025|CMU et al.|Paper|Span-aware online KD|Online teacher logits|Span-KL on student rollouts|Student|Span KL+contrastive|—|Llama,Qwen|ALFWorld,HotpotQA-ReAct,WebShop|—|Med|Strict
WebRL|2024|THUDM|Paper+Code|Curriculum web RL|ORM|Outcome|Student|RL+KL|RL|Llama-3.1 8/70B,GLM-4|WebArena-Lite|Yes|High|NotOPD
SWE-RL|2025|Meta|Paper|Similarity-reward RL|Patch sim|Scalar|Student|RL|RL|Llama-3-70B|SWE-Verified|No|Low-Med|NotOPD
Search-R1|2025|UIUC|Paper+Code|RLVR search agent|Outcome|Scalar|Student|RL|RL|Qwen-2.5|7 QA|Yes|High|NotOPD
SWE-Gym|2024|Berkeley/Apple|Paper+Code|Verifier-guided SWE|Unit tests|Verifier+rejection-FT|Mixed|SFT+RFT|SFT+Iter|Qwen-2.5-Coder-32B|SWE-Bench Verified/Lite|Yes|High|Partial
WizardLM/Evol-Instruct|2023|MS|Paper+Code|Data synthesis|GPT|NLL|Teacher|SFT|—|—|Inst-following|Yes|High|Adjacent
Magpie|2024|—|Paper+Code|Self-prompted data|Aligned LM|NLL|Teacher|SFT|—|—|Inst-following|Yes|High|Adjacent
UltraChat/UltraFeedback|2023|OpenBMB|Paper+Code|Synthetic datasets|GPT|—|Teacher|SFT/DPO|—|—|Various|Yes|High|Adjacent
```

## Open problems in LLM OPD

Despite OPD's empirical dominance, the field still faces substantial unresolved challenges. **First, the divergence question remains open**: forward KL (mode-covering), reverse KL (mode-seeking), JSD, skew KL, and adaptive per-token mixtures (ToDi, AKL, Entropy-Aware OPD) all show empirical advantages in different settings, but no principled selection criterion exists. The Thinking Machines blog defaults to reverse KL; TRL defaults to JSD; veRL uses top-k sparse forward KL; ms-swift exposes all. **Second, cross-tokenizer OPD is unsolved at production scale**: TRL's GKDTrainer has a documented correctness bug (Issue #4562), GOLD and DSKDv2 propose remedies, but the alignment between mismatched vocabularies remains a heuristic merge-and-sort process. **Third, multi-teacher OPD is in its infancy**: MiMo-V2-Flash and Nemotron-Cascade 2 demonstrate value, but how to weight teachers, schedule curricula across domains, and avoid catastrophic conflict between expert teachers lacks theory and benchmarks.

**Fourth, the student-teacher gap problem**: weak students cannot meaningfully sample from regions where teacher logits are informative, leading to noisy gradients early in training; Speculative KD addresses this via teacher token replacement, but ablations across initialization quality are sparse. **Fifth, OPD compute economics are not yet characterized**: Qwen3 and Thinking Machines report ~10× savings vs. RL, but no scaling law exists analogous to Apple's distillation scaling laws. When does on-policy beat off-policy as a function of student/teacher gap, data, and steps? Sixth, **mode collapse from reverse-KL OPD is empirically observed but not systematically studied** — diversity-preserving objectives are needed for creative tasks.

**Seventh, OPD for agentic and long-horizon tasks is barely explored** — Structured Agent Distillation is essentially the only strict-OPD agent paper, leaving open how to handle exposure bias across thousand-step ReAct trajectories with sparse teacher coverage. **Eighth, verifier-OPD boundaries are blurred**: PRIME, Math-Shepherd, and PAV provide dense token-level rewards that some readers count as distributional supervision; clearer empirical and theoretical contrasts between dense-reward RL and token-KL OPD are needed. **Ninth, OPSD vs. external-teacher OPD comparisons are immature**: π-Distill and OPSD claim 4–8× efficiency over GRPO, but head-to-head OPD-with-external-teacher comparisons at matched compute are missing. **Tenth, RL+OPD hybrids are under-studied**: GKD's RL extension, AlignDistil, and BOND each propose different formulations; the right way to combine reward signals with KL anchors for reasoning tasks is open. **Eleventh, OPD for safety/alignment** has not been seriously explored — most safety post-training is RLHF/RLAIF, but OPD against a refusal-trained teacher might be Pareto-superior. **Twelfth, reproducibility of industrial OPD is poor**: Qwen3, MiMo-V2-Flash, GLM-4.5, and Nemotron-Cascade 2 disclose loss types but not divergence choices, sampling temperatures, top-k truncations, or data mixtures — exact replication remains impossible. **Thirteenth, OPD against black-box teachers** (GPT-5, Claude, Gemini): GAD provides one approach, PLaD another, but the field lacks consensus on the best way to do OPD when only API access is available. **Fourteenth, evaluation of OPD-trained models** beyond AIME and MATH is thin; instruction-following, agentic, safety, and multilingual evaluations of OPD vs. RL vs. SFT at matched compute are largely absent. **Fifteenth, theoretical understanding** of why OPD converges faster than RL — beyond the dense-supervision argument — remains qualitative; rigorous sample-complexity bounds are an open question.

## Conclusion

On-policy distillation is no longer a niche technique. As of mid-2026 it is the dominant post-training strategy at Alibaba, Xiaomi, Z.ai, Tencent, NVIDIA, and Google DeepMind, and the most likely candidate for Phi/Llama-style labs to adopt next. The intellectual lineage runs cleanly from imitation learning (DAgger 2011, ImitKD 2020) through the academic OPD wave (GKD, MiniLLM, DistiLLM 2023–24) to industrial canonization (Gemma 2 mid-2024, Qwen3 May 2025, Thinking Machines October 2025). Most of what is colloquially called \"distillation\" in 2025 — DeepSeek-R1-Distill, Sky-T1, s1, LIMO, Open-R1's SFT phase — is **not** OPD by any strict definition, and most of what is called \"on-policy\" — Tülu 3's RLVR, Phi-4's PTS-DPO, Constitutional AI's RLAIF — is RL with scalar rewards. The strict OPD line — student rollouts plus distributional/token/trajectory teacher supervision — is narrow but productive: it explains why Qwen3 beats RL at one-tenth the cost, why Gemma 2 cites only two academic papers (Agarwal and Gu), and why Nemotron-Cascade 2 introduces MOPD as a stabilization step rather than a primary training stage. The next frontier sits at the intersection of multi-teacher OPD, cross-tokenizer OPD, OPSD with privileged context, and verifier-OPD hybrids. The open infrastructure (TRL GOLD, veRL recipe/gkd, NeMo-RL, ms-swift, Tinker, SkyRL, slime) is sufficient that any reader of this survey can run a strict OPD experiment in a weekend — which is itself a notable change from a year ago.