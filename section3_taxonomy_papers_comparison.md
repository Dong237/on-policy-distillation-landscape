# Section 3 Taxonomy Papers: Comprehensive Comparison

> 14 papers cited in Section 3 (Taxonomy of On-Policy Distillation) of Song et al. (2026), "A Survey of On-Policy Distillation for Large Language Models" (arXiv:2604.00626). Each paper was individually researched from its primary source.

---

## Table 1: Master Method Comparison

| # | Method | Year | Venue | Category | Teacher Access | On-Policy? | Divergence / Loss | Granularity | Key Innovation |
|---|--------|------|-------|----------|---------------|------------|-------------------|-------------|----------------|
| 1 | **GKD** | 2024 | ICLR | White-box KD | White-box | Yes (configurable lambda) | FKL / RKL / JSD(beta) | Token-level | On-policy sampling + flexible divergence; unifies SeqKD, KD, ImitKD as special cases |
| 2 | **MiniLLM** | 2024 | ICLR | White-box KD | White-box | Yes (student rollouts) | Reverse KL + REINFORCE | Sequence-level | Mode-seeking reverse KL with three variance-reduction tricks |
| 3 | **ToDi** | 2025 | EMNLP | Divergence design | White-box | No (off-policy) | Adaptive FKL+RKL per token | Token-level | Per-(position, vocab-entry) sigmoid-weighted KL switching |
| 4 | **RLKD** | 2026 | AAAI | RL-based KD | Black-box (text only) | Yes (GRPO rollouts) | Structural reward + GRPO | Hybrid (step-level) | Generative Structure Reward Model decomposing reasoning into meta-reasoning/solving |
| 5 | **SPIN** | 2024 | ICML | Self-play | None (self-play) | Iterative | IPM-based logistic loss | Sequence-level | Two-player self-play game; no teacher, no preference data needed |
| 6 | **OPSD** | 2026 | ICML | Self-distillation | Self (privileged info) | Yes (student rollouts) | Forward KL (soft distillation) | Token-level | Same model as teacher (conditioned on ground-truth) and student; 8-16x token-efficient vs GRPO |
| 7 | **DistiLLM** | 2024 | ICML | White-box KD | White-box | Adaptive off-policy | Skewed KL (SKL / SRKL) | Token-level (sequence decomposed) | Skewed KL for stable gradients + adaptive replay buffer; 2.5-4.3x faster training |
| 8 | **Lion** | 2023 | EMNLP | Black-box KD | Black-box (API) | Iterative (3-stage loop) | Cross-entropy (SFT) | Sequence-level | Adversarial Imitation-Discrimination-Generation loop; teacher as referee + curriculum designer |
| 9 | **GAD** | 2025 | -- | Black-box KD | Black-box (API) | Yes (GRPO rollouts) | Bradley-Terry + GRPO | Sequence-level | Co-evolutionary discriminator avoids reward hacking; student can surpass teacher |
| 10 | **GATES** | 2026 | -- | Self-distillation | Self (privileged context) | Hybrid (off+on) | NLL + advantage-weighted | Token-level | Consensus gating: self-agreement as proxy for correctness; no labels needed |
| 11 | **SDPO** | 2026 | -- | Self-distillation + RL | Self (feedback-conditioned) | Yes (on-policy) | JSD (symmetric KL) | Logit-level (per-token, per-vocab) | In-context self-retrospection as teacher; dense logit-level credit assignment |
| 12 | **pi-Distill** | 2026 | Workshop | Self-distillation + RL | Self (PI-conditioned) | Yes (on-policy) | Joint teacher-student KL + GRPO | Token-level | Shared-parameter teacher-student with privileged information; single-phase training for agentic tasks |
| 13 | **PRMs** | 2023 | -- | Reward model | N/A (verifier) | N/A | Supervised (step labels) | Step-level | Process-level reward > outcome reward; enables hybrid OPD-RL methods |
| 14 | **SuperCorrect** | 2025 | ICLR | Hybrid KD + DPO | White-box (proprietary API for data) | Partial (Stage 2 only) | SFT + Cross-model DPO | Step-level (error-step DPO) | Hierarchical thought templates + error-step-level correction DPO |

---

## Table 2: Models & Scale

| Method | Teacher | Student(s) | Largest Scale Tested | Cross-Architecture? |
|--------|---------|------------|---------------------|---------------------|
| GKD | T5-XL (3B), FLAN T5-XL | T5-small (77M), T5-base (250M), T5-large (800M) | 3B -> 800M | No (T5 family only) |
| MiniLLM | GPT-2-1.5B, OPT-13B, LLaMA-13B | GPT-2-120M/340M/760M, OPT-1.3B/2.7B/6.7B, LLaMA-7B | 13B -> 7B | No (within-family) |
| ToDi | GPT-2-1.5B, LLaMA2-7B | GPT-2-120M, TinyLLaMA-1.1B | 7B -> 1.1B | No (shared tokenizer required) |
| RLKD | DeepSeek-R1 (671B) | Qwen2.5-Math-7B, DeepSeek-R1-Distill-Qwen-7B | 671B -> 7B | Yes (text-only, black-box) |
| SPIN | None | Mistral-7B (zephyr-7b-sft-full) | 7B (self-play) | N/A |
| OPSD | Self (Qwen3) | Qwen3-1.7B, Qwen3-4B, Qwen3-8B | 8B (self) | N/A |
| DistiLLM | GPT-2-XL, OPT-6.7B, OpenLLaMA-7B, T5-XL | GPT-2, OPT-1.3B, OpenLLaMA-3B, T5-Base/Small | 7B -> 3B | No (within-family) |
| Lion | ChatGPT (API) | LLaMA-7B, LLaMA-13B | API -> 13B | Yes (black-box) |
| GAD | GPT-5-Chat, Qwen2.5-14B | Qwen2.5-3B/7B/14B, Llama-3.2-3B, Llama-3.1-8B | API -> 14B | Yes (cross-tokenizer works) |
| GATES | Self (Qwen3-4B) | Qwen3-4B | 4B (self) | N/A |
| SDPO | Self (feedback-conditioned) | Qwen3-0.6B/8B/32B, OLMo3-7B | 32B (self) | N/A |
| pi-Distill | Self + DeepSeek-v3.1 PI | Qwen3-4B/8B, R1-Distill-Llama-8B | 8B (self) | N/A |
| PRMs | GPT-4 (base) | GPT-4 variants (small-scale) | GPT-4 scale (undisclosed) | N/A |
| SuperCorrect | o1-preview, o1-mini, GPT-4o | Qwen2.5-Math-7B, DeepSeek-Math-7B, Llama-3.1-8B | API -> 7B/8B | Yes (cross-family) |

---

## Table 3: Best Quantitative Results by Benchmark Domain

### A. Math Reasoning

| Method | Benchmark | Best Result | Student Model | Baseline | Delta |
|--------|-----------|-------------|---------------|----------|-------|
| **RLKD** | AIME24 pass@1 | **53.3%** | DeepSeek-R1-Distill-Qwen-7B + RLKD | GRPO: 50.0% | **+3.3** |
| **RLKD** | AIME24 pass@64 | **86.7%** | DeepSeek-R1-Distill-Qwen-7B + RLKD | GRPO: 83.3% | +3.4 |
| **RLKD (Zero)** | AIME24 pass@1 | **23.3%** | Qwen2.5-Math-7B (RL-only, 3.2K data) | Instruct (2.9M SFT): 16.7% | **+6.6 with 0.1% data** |
| **OPSD** | AIME24 Avg@12 | **77.8%** | Qwen3-8B + OPSD | GRPO: 76.4% | +1.4 |
| **OPSD** | AIME24 Avg@12 | **57.2%** | Qwen3-1.7B + OPSD | GRPO: 51.1% | **+6.1** |
| **SuperCorrect** | MATH | **70.2%** | SuperCorrect-Qwen-7B | Qwen2.5-Math-7B: 55.1% | **+15.1** |
| **SuperCorrect** | GSM8K | **89.5%** | SuperCorrect-Qwen-7B | Qwen2.5-Math-7B: 83.2% | +6.3 |
| **PRMs** | MATH best-of-1860 | **78.2%** | GPT-4 + PRM | ORM: 72.4% | +5.8 |
| **GATES** | MATH maj@8 | **65.6%** | Qwen3-4B + GATES | Base: 42.6% | +23.0 |
| **SPIN** | GSM8K | **38.97%** | Mistral-7B (SPIN iter 3) | SFT: 26.76% | +12.2 |

### B. Code / Competitive Programming

| Method | Benchmark | Best Result | Student Model | Baseline | Delta |
|--------|-----------|-------------|---------------|----------|-------|
| **SDPO** | LiveCodeBench v6 | **48.8%** | Qwen3-8B + SDPO | GRPO: 41.2% | **+7.6** |
| **SDPO** | LCBv6 (aggregate) | **68.8%** | Qwen3-8B + SDPO | GRPO: 64.1% | +4.7 |

> SDPO 48.8% surpasses Claude Sonnet 4 (40.5%) and Claude Opus 4 (39.7%) on LCBv6.

### C. Instruction Following (ROUGE-L)

| Method | Benchmark | Best Result | Setup | vs Best Baseline | Delta |
|--------|-----------|-------------|-------|-----------------|-------|
| **DistiLLM** | Dolly Eval | **26.37** | GPT-2-XL -> GPT-2 (SRKL + off-policy) | GKD: 23.75 | **+2.62** |
| **DistiLLM** | S-NI | **28.24** | GPT-2-XL -> GPT-2 | GKD: 26.05 | +2.19 |
| **DistiLLM** | UnNI | **30.11** | GPT-2-XL -> GPT-2 | GKD: -- | -- |
| **MiniLLM** | Dolly GPT4-score | **76.4** | LLaMA-13B -> 7B | KD: 73.7 | +2.7 |
| **MiniLLM** | SelfInst GPT4-score | **73.1** | LLaMA-13B -> 7B | KD: 70.5 | +2.6 |
| **ToDi** | Avg ROUGE-L | **24.83** | LLaMA2-7B -> TinyLLaMA-1.1B | AKL: 24.15 | +0.68 |

### D. Conversational / Open-Ended Quality (GPT-4 judged)

| Method | Benchmark | Best Result | Setup | Baseline | Delta |
|--------|-----------|-------------|-------|----------|-------|
| **GAD** | LMSYS-Chat | **52.1** | Qwen2.5-14B + GAD | Teacher GPT-5-Chat: 51.7 | **+0.4 (surpasses teacher)** |
| **Lion** | Vicuna-Instr | **98.38%** of ChatGPT | Lion-13B | Vicuna-13B: 92.61% | +5.77 |
| **Lion** | BBH avg | **36.2%** | Lion-13B | Vicuna-13B: 23.3% | +12.9 |

### E. Scientific Reasoning

| Method | Benchmark | Best Result | Student Model | Baseline | Delta |
|--------|-----------|-------------|---------------|----------|-------|
| **SDPO** | Chemistry (SciKnowEval) | **70.1%** | Qwen3-8B + SDPO | GRPO: 60.0% | **+10.1** |
| **SDPO** | Physics | **75.6%** | Qwen3-8B + SDPO | GRPO: 72.7% | +2.9 |

### F. Agentic / Multi-Turn

| Method | Benchmark | Best Result | Student Model | Baseline | Delta |
|--------|-----------|-------------|---------------|----------|-------|
| **pi-Distill** | Travel Planner | **44.1%** | Qwen3-8B + pi-Distill | SFT+RL: 32.3% | **+11.8** |
| **pi-Distill** | tau-Bench Retail | **31.1%** | Qwen3-8B + pi-Distill | SFT+RL: 29.1% | +2.0 |

---

## Table 4: Training Efficiency Comparison

| Method | Training Cost vs Standard KD | Data Efficiency | Notable |
|--------|------------------------------|-----------------|---------|
| **DistiLLM** | **1.6x standard KD** (2.5-4.3x faster than GKD/MiniLLM on-policy) | Standard | Fastest on-policy method |
| **GKD** | 1.8-2.2x standard KD | **5% data matches 100% KD** | Exceptional data efficiency |
| **MiniLLM** | Not reported; requires teacher + student in memory | Standard | Most complex training (4 stabilization techniques) |
| **RLKD** | Not reported | **3.2K samples (0.1% of data)** outperforms 2.9M SFT | Extreme data efficiency |
| **SPIN** | ~1.45h generation + 4-8h training per iter on 8xA100 | Standard SFT dataset only | No additional data needed |
| **OPSD** | **8-16x fewer tokens than GRPO** | Standard | 100 steps to peak; 1024 tokens/problem vs 16K for GRPO |
| **SDPO** | 10-25% wall-clock overhead vs GRPO; **4x fewer generations** | Standard | 6-10x wall-clock speedup on some tasks |
| **DistiLLM** off-policy degradation | **-0.04 to -0.25 pts** (vs GKD -0.86, ImitKD -1.90) | -- | Most robust to off-policy approximation |
| **GATES** | 16 rollouts per question (expensive) | 551 questions, 1 epoch | Very data-efficient |
| **Lion** | ~$900 API cost, 450K API calls | 70K samples total | Lower API cost than WizardLM |
| **GAD** | 30h on 16xH100 for 14B | 200K training samples | Discriminator doubles memory |

---

## Table 5: Strengths, Weaknesses & Practical Considerations

| Method | Primary Strength | Primary Weakness | When to Use |
|--------|-----------------|------------------|-------------|
| **GKD** | Simple, stable, flexible divergence; no backprop through sampling | Task-dependent divergence choice; limited to T5-scale experiments | Default starting point for white-box OPD |
| **MiniLLM** | Strongest consistent gains on instruction following; reduces exposure bias | Complex training (4 stabilization tricks); reward hacking without teacher-mixed sampling | When instruction-following quality matters most |
| **ToDi** | Theoretically elegant per-token weighting; O(V) efficient | Off-policy only; modest gains (+0.6 avg); same tokenizer required | As a drop-in divergence improvement for any KD method |
| **RLKD** | Extreme data efficiency (3.2K samples); transfers reasoning structure | No comparison with KD baselines; math-only evaluation; GSRM dependency | Math/reasoning distillation from frontier models with minimal data |
| **SPIN** | No teacher, no extra data needed; theoretically grounded convergence | Saturates after 2-3 iterations; bounded by SFT data quality | When no teacher model is available and only SFT data exists |
| **OPSD** | 8-16x token-efficient; no external teacher; dense signal | Information leakage risk in long training; math-only; scale ceiling unknown | Self-improvement on reasoning when ground-truth solutions exist |
| **DistiLLM** | Fastest training (2.5-4.3x faster); robust off-policy; no SFT pre-training needed | White-box only; alpha needs tuning; weaker on task-specific settings | When training budget is limited; production deployment |
| **Lion** | Strong BBH reasoning gains; principled adversarial curriculum | Far behind teacher on reasoning; AGIEval anomaly at 7B | Black-box distillation with iterative improvement |
| **GAD** | Student can surpass teacher; works cross-tokenizer; strong OOD generalization | Longer responses inflate scores; warmup-sensitive; no reasoning benchmarks | Black-box distillation from proprietary models |
| **GATES** | No labels, no rewards, no external teacher needed at all | Consensus errors pass the gate; math-only; 4B only | When absolutely no supervision signal exists |
| **SDPO** | Dense logit-level credit assignment; surpasses Claude on code; 3-11x shorter responses | Fails on small models (<1B); collapses in long training | Code/scientific reasoning with rich environment feedback |
| **pi-Distill** | Single-phase training; works without CoT; agentic environments | Model-dependent; high variance; modified benchmarks | Agentic multi-turn tool-use distillation |
| **PRMs** | 78.2% MATH best-of-1860; better credit assignment than ORM | High annotation cost; math-specific; not distillation per se | As reward component in hybrid OPD+RL systems |
| **SuperCorrect** | +15.1% MATH; teaches genuine self-correction ability | Depends on proprietary teachers (o1, GPT-4o); narrow benchmark scope | When self-correction capability is the goal |

---

## Table 6: Key Tricks & Implementation Notes

| Method | Critical Trick | Impact if Omitted |
|--------|---------------|-------------------|
| GKD | At least 25% on-policy data (lambda >= 0.25) | Inconsistent gains below this threshold |
| GKD | Student temperature 1.0 during training (even for greedy eval) | Reduced exploration |
| MiniLLM | Teacher-mixed sampling (alpha=0.2) | Reward hacking: degenerate repeated phrases |
| MiniLLM | Length normalization | Bias toward generating very short responses |
| MiniLLM | Single-step decomposition | Higher gradient variance |
| ToDi | Stop-gradient on sigmoid weight alpha | Optimization instability |
| ToDi | beta=1 for sigmoid scaling | beta<1: too flat; beta>1: discontinuous |
| RLKD | R_gsrm / R_acc ratio = 1.0 | Too much structure ignores accuracy; too much accuracy ignores structure |
| RLKD | Early termination in reward at first meta-reasoning mismatch | Reward hacking |
| SPIN | Learning rate decay from 5e-7 to 1e-7 at later iterations | Instability near convergence |
| SPIN | Increase beta from 0.1 to 5.0 at final iteration | Instability at convergence |
| OPSD | Per-token pointwise KL clipping (tau) | Training collapse: style tokens dominate gradient |
| OPSD | Forward KL only (not reverse KL or JSD) | Reverse KL degrades below baseline |
| OPSD | TM-off student / TM-on teacher (Qwen3) | Suboptimal KL distribution |
| DistiLLM | Skew parameter alpha=0.1 | alpha=0: standard unstable KL; alpha>0.5: too soft |
| DistiLLM | Adaptive off-policy scheduler (phi increases on val loss rise) | 2.5-4.3x slower if always on-policy |
| GAD | Both generator AND discriminator warmup | Without disc warmup: drops below SeqKD |
| GAD | Matched discriminator-generator size | Oversized discriminator hurts |
| GAD | Bradley-Terry loss (not cross-entropy) for discriminator | -1.0 point on LMSYS |
| GATES | Consensus gate (>=4/8 agreement) | -4.3 benchmark avg, -8.0 in-domain |
| GATES | Off-policy dominates (lambda_off=1.0, lambda_on=0.1) | On-policy dominant: -4.8 avg |
| SDPO | Teacher regularization (EMA or trust-region) | Catastrophic divergence (36.1% vs 48.8%) |
| SDPO | Do NOT include student's original attempt in reprompt | Kills exploration (entropy drops 38%) |
| SDPO | Top-K=100 logit approximation | Full vocab: memory explosion; negligible quality loss |
| pi-Distill | alpha=0.5 as default | Safest; worst in only 1/16 scenarios |
| pi-Distill | Non-zero beta KL regularization | Important in 17/21 configurations |
| SuperCorrect | Inspector LLM validation (up to 3 rounds) | Correction accuracy: 92% -> 99% |
| PRMs | Convincing wrong-answer active learning strategy | 2.6x less data-efficient without it |

---

## Verdict: Research-Agreed Best Methods

### By Scenario

#### 1. White-Box Distillation for Instruction Following
**Winner: DistiLLM** (if training budget matters) or **MiniLLM** (if absolute quality matters)

- DistiLLM is 2.5-4.3x faster than MiniLLM/GKD while achieving higher ROUGE-L scores
- MiniLLM gives the largest GPT-4 quality gains (up to +10 points) but requires complex training
- GKD is the simplest and most principled starting point, with exceptional 5% data efficiency
- **Recommendation**: Start with GKD as baseline, upgrade to DistiLLM for speed or MiniLLM for quality

#### 2. Math / Reasoning Distillation
**Winner: RLKD + OPSD** (complementary)

- RLKD achieves the highest absolute scores on AIME24 (53.3% pass@1) with extreme data efficiency (3.2K samples)
- OPSD provides 8-16x token efficiency over GRPO and works without any external teacher
- SuperCorrect achieves 70.2% on MATH (highest among 7B models) but depends on proprietary APIs
- **Recommendation**: Use RLKD when a frontier teacher's reasoning traces are available; use OPSD for self-improvement when ground-truth solutions exist

#### 3. Code Generation
**Winner: SDPO**

- 48.8% on LCBv6, surpassing Claude Sonnet 4 (40.5%) and Opus 4 (39.7%)
- 4x sample efficiency over GRPO; 3-11x shorter reasoning traces
- Requires rich environment feedback (runtime errors, test cases)
- **Caveat**: Fails on models < 1B parameters; collapses in very long training

#### 4. Black-Box Distillation (API-only teacher)
**Winner: GAD**

- Student can surpass the teacher (Qwen2.5-14B+GAD 52.1 > GPT-5-Chat 51.7)
- Works cross-tokenizer (Qwen teacher -> Llama student)
- Co-evolutionary discriminator avoids reward hacking
- Lion is simpler but results are weaker on structured reasoning

#### 5. No Teacher Available (Self-Improvement)
**Winner: SDPO** (with feedback) or **SPIN** (without feedback)

- SDPO: Best when rich environment feedback exists (code, science)
- SPIN: Best when only SFT data exists; no extra data, no teacher, proven convergence
- OPSD: Best for math when ground-truth solutions exist
- GATES: Best when absolutely no supervision signal exists (novel domain)

#### 6. Distillation + RL Combined
**Winner: SDPO > RLKD > pi-Distill** (by evidence strength)

- SDPO replaces GRPO's scalar reward with dense logit-level self-distillation signal: +7.6% on LCBv6
- RLKD adds structural reward to GRPO: +3.3% on AIME24 over GRPO
- pi-Distill combines KL-regularized RL with privileged information: +11.8% on Travel Planner over SFT+RL
- **The consensus is clear**: combining distillation signals with RL consistently outperforms pure RL (GRPO/PPO) or pure distillation (SFT/KD)

#### 7. Production / Industry Deployment
**Winner: DistiLLM**

- Only 1.6x cost of naive KD (vs 1.8-2.2x for GKD, much more for MiniLLM)
- No SFT pre-training needed (one-stage pipeline)
- Minimal off-policy degradation (-0.04 pts) enables practical replay buffer usage
- Works across GPT-2, OPT, LLaMA, T5 families

---

## Overall Ranking (Research-Agreed, by Evidence Breadth)

| Rank | Method | Justification |
|------|--------|---------------|
| 1 | **GKD** | Most validated foundational method; ICLR 2024; unifying framework; exceptional data efficiency; adopted industrially (Gemma 2, referenced by Qwen3). Start here. |
| 2 | **DistiLLM** | Best speed-quality tradeoff; ICML 2024; 2.5-4.3x faster with higher scores than GKD; robust off-policy. Best for production. |
| 3 | **SDPO** | Strongest combined KD+RL method; surpasses Claude on code; dense credit assignment; 4x sample efficient. Best for code/science with feedback. |
| 4 | **MiniLLM** | Largest absolute quality gains on instruction following; ICLR 2024; tested at 13B->7B scale. Best for quality-first scenarios. |
| 5 | **RLKD** | Extreme data efficiency (3.2K samples); AAAI 2026; best for math reasoning from frontier teachers. |
| 6 | **OPSD** | Most token-efficient self-distillation; ICML 2026; no external teacher needed. Best for math self-improvement. |
| 7 | **SuperCorrect** | Highest MATH score among 7B models (70.2%); ICLR 2025; teaches self-correction. Requires proprietary APIs. |
| 8 | **GAD** | Best black-box method; student surpasses teacher; cross-tokenizer. Best for API distillation. |
| 9 | **SPIN** | Elegant self-play requiring no teacher or extra data; ICML 2024; proven convergence. Limited by data quality ceiling. |
| 10 | **pi-Distill** | Unique agentic/multi-turn focus; single-phase training. Early-stage research. |
| 11 | **GATES** | Operates without any supervision signal. Very narrow evaluation (math, 4B only). |
| 12 | **ToDi** | Clean per-token weighting theory; orthogonal to sampling strategy. Off-policy, modest gains. |
| 13 | **PRMs** | Not a distillation method per se, but critical component for hybrid OPD+RL systems. |
| 14 | **Lion** | Pioneering black-box work; superseded by GAD in results and methodology. |

---

## Key Takeaways

1. **Combining distillation with RL is strictly better than either alone.** Every paper that tests KD+RL vs pure KD or pure RL finds the combination wins: SDPO > GRPO, RLKD > GRPO, pi-Distill > SFT+RL, GKD+RL > RLEF.

2. **On-policy beats off-policy, but the gap can be narrowed.** DistiLLM shows that skewed KL + replay buffers can achieve near-on-policy quality at off-policy cost. GKD shows 5% on-policy data matches 100% off-policy.

3. **Forward KL is not always best.** Reverse KL wins for instruction following (MiniLLM), Forward KL wins for self-distillation (OPSD), JSD wins for translation (GKD), skewed KL wins for speed (DistiLLM). The optimal divergence is task-dependent.

4. **Self-distillation is surprisingly effective.** OPSD, SDPO, SPIN, and GATES all achieve strong results without external teachers, often competitive with white-box methods that use much larger teachers.

5. **Dense supervision > sparse reward.** SDPO's logit-level advantages beat GRPO's scalar rewards (+7.6%); PRMs beat ORMs (+5.8%); MiniLLM's token-level supervision beats sequence-level SeqKD.

6. **Data efficiency varies by orders of magnitude.** RLKD needs 3.2K samples; GKD needs 5% of data; MiniLLM/DistiLLM use full datasets. Method choice depends heavily on data availability.

7. **Small models need different methods.** SDPO and OPSD are most effective at 4B+ scale; SPIN works at 7B; SDPO fails below 1B. DistiLLM and GKD work at all scales.

8. **Stability tricks are non-optional.** Almost every method has a critical stabilization technique without which training collapses: MiniLLM's teacher-mixed sampling, OPSD's KL clipping, SDPO's teacher regularization, DistiLLM's skew parameter.

---

*Generated 2026-05-12 by systematic paper-by-paper analysis of all 14 works cited in Section 3 of Song et al. (2026).*
