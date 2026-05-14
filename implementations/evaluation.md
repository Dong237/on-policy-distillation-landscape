# Evaluation Methodology for On-Policy Distillation

> Metrics and protocols that capture what OPD does that other paradigms do not.
>
> Last updated: 2026-05-14

---

## Guiding Principle

Strict OPD has three necessary components:

- **C1.** Student generates its own trajectories (on-policy rollouts).
- **C2.** Teacher supervises those exact trajectories with distributional signal.
- **C3.** Training objective consumes that distillation signal.

OPD's theoretical advantages over alternative paradigms:

- **vs. Offline KD:** reduces exposure bias (student trains on its own distribution, not the teacher's).
- **vs. Reward-only RL:** provides dense teacher guidance (token-level KL, not just scalar reward).
- **vs. SFT:** avoids teacher-forcing and static data.

An evaluation protocol for OPD must test these claimed advantages directly. Standard accuracy alone is insufficient: a method can match accuracy by luck while failing on calibration, distributional fidelity, or rollout stability. The metrics and protocols below are designed to separate OPD-specific gains from confounders.

## Reporting Rule

Every serious OPD evaluation should report a capability metric plus at least one OPD-specific diagnostic.

| OPD claim | Required companion diagnostic |
|---|---|
| "Improves reasoning or instruction quality" | Calibration or distributional-fidelity metric, plus SFT/offline-KD/reward-only controls. |
| "Reduces exposure bias" | Rollout drift, corrupted-prefix recovery, first-error recovery, or horizon-conditioned success. |
| "Beats reward-only RLVR" | OPD-only, RLVR-only, OPD+RL, offline KD, and SFT ablations where feasible. |
| "Transfers teacher uncertainty" | ECE, Brier/NLL, teacher-student entropy correlation, KL/JS, or top-k support retention. |
| "Improves VLM/MLLM reasoning" | Separate perception-heavy and reasoning-heavy benchmarks, plus hallucination or teacher-disagreement slices. |
| "Improves GUI/agent/VLA behavior" | Task success plus visited-state recovery, invalid-action rate, action/coordinate KL, and safety or side-effect metrics. |
| "Improves diffusion/flow generation" | Image quality plus teacher score/velocity-field fidelity, trajectory proxy metrics, and diversity/coverage. |

See [Benchmarks](benchmarks.md) for the benchmark bundle map. Audit provenance is archived under [assets/research](../assets/research/).

---

## TASK 1: Metric Taxonomy

### MF-1. Accuracy / Task Completion

| Field | Detail |
|---|---|
| **What it measures** | End-task correctness: does the model produce the right answer, complete the task, or satisfy the user intent? |
| **Applicable OPD families** | All families (white-box, black-box, self-teacher, action-token, diffusion/flow, OPD+RL hybrids). This is a universal baseline metric. |
| **Example specific metrics** | Exact-match accuracy, ANLS (document QA), relaxed accuracy (ChartQA), success rate (robotics, GUI agents), pass@k (code), F1 (QA), BLEU/ROUGE (summarization). |
| **What failure modes it catches** | Catastrophic forgetting, training divergence, teacher-student mismatch severe enough to hurt correctness. |
| **Limitations** | Necessary but far from sufficient. A model can achieve high accuracy while being poorly calibrated, over-verbose, brittle to distribution shift, or exposure-bias-prone on longer sequences. Accuracy conflates perception and reasoning errors in VLMs. Two methods can have identical accuracy but very different distributional quality. |

---

### MF-2. Calibration

| Field | Detail |
|---|---|
| **What it measures** | Whether the model's expressed confidence matches its actual correctness rate. OPD should transfer not just the teacher's predictions but the teacher's uncertainty structure. |
| **Applicable OPD families** | Primarily white-box token-level OPD (GKD, MiniLLM, Entropy-Aware OPD, CaOPD), where the teacher provides calibrated distributional signal. Also relevant for self-teacher OPD (OPSD, OPCD) where the teacher is the model itself. Less directly applicable to black-box OPD (no logit access), diffusion/flow OPD, and action-token OPD. |
| **Example specific metrics** | Expected Calibration Error (ECE) over binned confidence intervals. Brier score (combines calibration and refinement). Reliability diagrams (visual: predicted probability vs. empirical accuracy per bin). Adaptive ECE (equal-mass binning). Token-level calibration (per-position ECE on generated tokens). Selective prediction accuracy (performance when the model is allowed to abstain on low-confidence queries). |
| **What failure modes it catches** | Systematic overconfidence from forward KL collapse to teacher modes. Under-confidence from reverse KL mode-seeking. CaOPD specifically identified and addressed calibration failure in standard OPD. Overconfident hallucination in VLMs. |
| **Limitations** | ECE is sensitive to binning strategy and sample size. Brier score conflates calibration with sharpness. Token-level calibration is expensive to compute at scale. Black-box OPD methods cannot directly optimize for calibration because they lack logit access. Calibration on the training distribution does not guarantee OOD calibration. |

---

### MF-3. Distributional Fidelity

| Field | Detail |
|---|---|
| **What it measures** | How closely the student's output distribution matches the teacher's distribution. This is OPD's core objective: the student should approximate the teacher not just in top-1 predictions but in the full predictive distribution. |
| **Applicable OPD families** | White-box OPD (directly measurable via KL, JSD). Black-box OPD (approximated via agreement rate or output similarity). Self-teacher OPD (measurable between current and privileged-context model). OPD+RL hybrids (measurable but confounded by the RL reward component). Less directly applicable to diffusion/flow OPD (measured via different distribution metrics) and action-token OPD (measured in action space). |
| **Example specific metrics** | Forward KL(teacher || student) on held-out prompts. Reverse KL(student || teacher) on held-out prompts. Jensen-Shannon Divergence (JSD) between teacher and student on sampled completions. Token-level entropy correlation: Pearson or Spearman correlation between teacher and student per-token entropies across held-out sequences. Top-k agreement rate: fraction of tokens where student's top-k set intersects teacher's top-k set. Distribution overlap on full vocabulary: L1 distance between teacher and student probability vectors averaged over positions. Temperature-scaled KL: measure KL at multiple temperatures to check if fidelity holds across sharpness levels. |
| **What failure modes it catches** | Mode collapse (student assigns mass to far fewer modes than teacher). Mode covering without precision (student spreads mass too broadly). Entropy collapse or explosion. Token-level teacher-student misalignment that accuracy cannot detect. Divergence from the teacher distribution that indicates the student learned a systematically different function. |
| **Limitations** | Requires white-box access to both teacher and student for exact computation. KL is asymmetric and can give misleading results if not reported in both directions. High distributional fidelity to a poor teacher is not useful. Entropy correlation can be gamed by trivially matching marginal entropy without matching conditional structure. Distribution fidelity on prompts from the training distribution may not generalize OOD. |

---

### MF-4. Exposure-Bias Reduction

| Field | Detail |
|---|---|
| **What it measures** | Whether training on the student's own distribution (the on-policy property) reduces the accumulation of errors during autoregressive generation. Exposure bias is the core problem that OPD is designed to solve relative to offline KD and SFT: when the model conditions on its own previous tokens (which may contain errors), does quality degrade faster or slower than an offline-trained model? |
| **Applicable OPD families** | All text-generating OPD families. This is the most OPD-discriminative metric family. White-box OPD (GKD, MiniLLM) should show the clearest gains. OPD+RL hybrids (G-OPD, REOPOLD, VOLD) combine both. Black-box OPD (GAD, OVD) should show gains if the response-level signal propagates. Self-teacher OPD (OPSD) should show gains on longer-context tasks. Action-token OPD (VLA-OPD) shows an analog: multi-step policy degradation in embodied tasks. Diffusion/flow OPD has a different analog: denoising quality across diffusion steps. |
| **Example specific metrics** | (a) Rollout quality vs. sequence length: generate completions of varying target lengths (e.g., 64, 128, 256, 512, 1024 tokens) and measure task metric at each length. Plot quality as a function of length. OPD should degrade more gracefully than offline KD. (b) Position-conditional accuracy: accuracy of the k-th generated token as a function of k, averaged over many sequences. Offline KD should show steeper degradation at later positions. (c) Long-sequence benchmark delta: performance gap between short and long tasks. For example, on math, compare problems requiring short (under 100 tokens) vs. long (over 500 tokens) reasoning chains. (d) Prefix perturbation robustness: inject small errors into prefixes at controlled positions, then measure completion quality. OPD-trained models should recover better because they have seen their own imperfect prefixes during training. (e) Self-BLEU degradation: measure diversity of completions at different decoding lengths. Exposure bias often manifests as repetitive or degenerate text at longer lengths. (f) Agentic multi-step compounding: for agent OPD (SOD, MAD-OPD), measure task success rate as a function of the number of interaction steps. |
| **What failure modes it catches** | The fundamental failure mode of offline KD and SFT: performance collapse on long sequences because the model never trained on its own error-prone prefixes. Also catches long-CoT drift (noted in `notes/long-cot-drift.md`), where extended chain-of-thought reasoning degrades over many reasoning steps. |
| **Limitations** | Measuring exposure bias directly requires controlled experiments with matched models (same architecture, same data, different training paradigm). Confounders include model capacity, data quality, and training duration. Long-sequence benchmarks may test other capabilities (attention span, memory) beyond exposure-bias reduction. The prefix-perturbation test requires careful calibration to avoid testing robustness to adversarial noise rather than natural error accumulation. |

---

### MF-5. OOD Robustness

| Field | Detail |
|---|---|
| **What it measures** | Performance on prompts and tasks outside the training distribution. OPD should improve OOD robustness because the student explores its own distribution during training, encountering states that a teacher-generated dataset would not contain. |
| **Applicable OPD families** | All families. White-box OPD should show improved generalization from on-policy exploration. Self-teacher OPD (OPSD, OPCD) should show robustness proportional to the quality of self-exploration. Black-box OPD benefits if the discriminator/response-level signal generalizes. Action-token OPD (VLA-OPD) should show robustness to novel environments. Diffusion/flow OPD should show robustness to novel prompts/styles. |
| **Example specific metrics** | (a) Held-out domain accuracy: train on domain A, evaluate on domain B (e.g., train on math, evaluate on science reasoning). (b) Adversarial prompt robustness: performance on adversarially paraphrased or perturbed versions of in-distribution prompts. (c) Format transfer: train on one output format (e.g., multiple choice), evaluate on another (e.g., free-form). (d) Cross-benchmark generalization: train-time benchmarks vs. held-out benchmarks not seen during training. (e) Instruction-following on novel instructions: IF-Eval-style tests on instruction categories not in training data. (f) Visual domain shift (VLM-specific): train on natural images, evaluate on synthetic, medical, satellite, or artistic images. (g) Temporal OOD: evaluate on data from a later time period than training. |
| **What failure modes it catches** | Overfitting to the training distribution despite on-policy training. Teacher-dependent failures where the student memorizes teacher patterns rather than learning generalizable skills. Narrow exploration where the student's on-policy distribution is too close to the teacher's distribution to provide diversity. |
| **Limitations** | OOD evaluation requires careful selection of what counts as "out of distribution." A test set can be OOD in trivial ways (formatting) or deep ways (reasoning type). Attribution is difficult: improved OOD performance could come from the on-policy property or from other factors (model size, data quality, training duration). No standard OOD benchmark suite exists for OPD evaluation. |

---

### MF-6. Length / Verbosity Control

| Field | Detail |
|---|---|
| **What it measures** | Whether OPD produces outputs of appropriate length, avoiding both over-compression and length inflation. Length inflation is a documented failure mode of OPD (noted in `papers/white-box-distributional-opd.md` under StableOPD) where the student learns to produce longer outputs because the teacher's per-token reward signal implicitly rewards continuation. |
| **Applicable OPD families** | All text-generating families. White-box OPD (particularly reverse KL methods like MiniLLM) is prone to length inflation because generating more tokens provides more opportunities for reward. OPD+RL hybrids inherit this risk and may amplify it if the RL reward also correlates with length. Self-teacher OPD can inherit length biases from the privileged model. Black-box OPD depends on whether the response-level signal penalizes verbosity. |
| **Example specific metrics** | (a) Output length ratio: mean(student output length) / mean(teacher output length) on matched prompts. Ideal is close to 1.0 unless deliberate compression is intended. (b) Length-conditional accuracy: accuracy plotted against output length bins. Reveals whether longer outputs are more accurate (genuine reasoning) or just verbose. (c) Compression ratio for distillation: if the goal is model compression, measure tokens-per-answer and quality jointly. (d) Instruction-following length compliance: on prompts requesting specific lengths ("answer in one sentence"), measure compliance rate. (e) Padding/filler token rate: fraction of generated tokens that are semantically vacuous (e.g., repeated phrases, unnecessary hedging). Manual or classifier-based annotation. (f) StableOPD truncation-collapse rate: fraction of outputs that are truncated at the maximum length limit, indicating the model has not learned to terminate naturally. |
| **What failure modes it catches** | Length inflation (truncation collapse): the model generates until hitting the length limit, producing verbose or repetitive outputs. Over-compression: reverse KL mode-seeking can cause the student to produce very short outputs that cover only the highest-probability mode. Length gaming: the model learns to produce longer outputs because the training signal rewards more tokens. |
| **Limitations** | Optimal length is task-dependent and subjective. Length metrics alone do not distinguish between informative detail and padding. Comparing output lengths across methods requires controlling for task difficulty and prompt format. |

---

### MF-7. Teacher Agreement Rate

| Field | Detail |
|---|---|
| **What it measures** | How often the student's predictions agree with the teacher's predictions on the same input. This is a coarser version of distributional fidelity that can be computed for both white-box and black-box teachers. |
| **Applicable OPD families** | All families with an external teacher. White-box OPD: top-1 agreement and top-k overlap. Black-box OPD: response-level agreement (does the student's answer match the teacher's answer?). Self-teacher OPD: agreement between the base model and the privileged-context model. OPD+RL hybrids: agreement can be measured separately for the distillation component. Not directly applicable to reward-only RL (no teacher to agree with). |
| **Example specific metrics** | (a) Top-1 token agreement: fraction of positions where argmax(student) = argmax(teacher). (b) Top-k token overlap: fraction of positions where student's top-k intersects teacher's top-k. (c) Sequence-level agreement: fraction of prompts where student and teacher produce the same final answer. (d) Agreement conditioned on teacher confidence: agreement rate separately for high-confidence and low-confidence teacher positions. OPD should show higher agreement on high-confidence positions. (e) Agreement on student-generated vs. teacher-generated prefixes: does agreement differ when the prefix was generated by the student vs. the teacher? (f) Disagreement analysis: on positions where student and teacher disagree, is the student or the teacher more often correct? This tests whether selective disagreement is beneficial. |
| **What failure modes it catches** | Student learns surface patterns but not the teacher's decision function. Student agrees with the teacher on easy cases but diverges on hard cases (the cases that matter most). Agreement is artificially high because both models default to the same trivial answers. |
| **Limitations** | High agreement with a poor teacher is not desirable. Agreement does not capture distributional nuance (two models can agree on top-1 while having very different probability masses on alternatives). Sequence-level agreement is brittle to minor phrasing differences. Computing top-k agreement requires white-box access. |

---

### MF-8. Credit Assignment Quality

| Field | Detail |
|---|---|
| **What it measures** | In multi-step, agentic, or chain-of-thought settings, whether the model can identify which step was correct and which was wrong. OPD with dense token-level supervision should improve credit assignment relative to reward-only RL (which provides only a scalar outcome signal). |
| **Applicable OPD families** | Most relevant for: agentic OPD (SOD, MAD-OPD OPAD), OPD+RL hybrids on reasoning tasks (VOLD, G-OPD on math), GUI/agent OPD (GUI-SD, LiteGUI), action-token OPD (VLA-OPD). Less relevant for single-turn QA. Diffusion/flow OPD has an analog in per-step denoising quality. |
| **Example specific metrics** | (a) Step-level error localization: present the model with a multi-step solution containing a planted error at step k. Ask the model to identify the first incorrect step. Measure localization accuracy. (b) Process reward model agreement: if a process reward model (PRM) exists, compare the student's per-step confidence with PRM scores. (c) Self-correction rate: after generating an incorrect multi-step solution, can the model identify and correct the specific failing step when prompted? (d) Agentic trajectory diagnosis: in GUI or tool-use tasks, after a failed trajectory, can the model identify the action that caused failure? (e) Reasoning chain pruning: remove individual reasoning steps and measure the impact on final-answer accuracy. Steps that the model correctly identifies as critical should have larger impact when removed. (f) Step-level advantage correlation (for OPD+RL): correlation between the per-step advantage used during training and the actual per-step contribution to outcome. |
| **What failure modes it catches** | Outcome-hacking: the model gets the right answer through compensating errors rather than correct reasoning at each step. Reward sparsity masking: with scalar reward, the model cannot distinguish between a good plan with a bad final step and a bad plan with a lucky final step. Credit mis-attribution: the model attributes success/failure to the wrong steps. |
| **Limitations** | Measuring credit assignment requires ground-truth step-level annotations, which are expensive to produce. Self-correction tests conflate credit assignment ability with correction ability. The planted-error test may not generalize to natural error patterns. Step-level evaluation is not standardized across benchmarks. |

---

### MF-9. Inference Cost

| Field | Detail |
|---|---|
| **What it measures** | The computational cost of using the distilled model at inference time. OPD's training cost is high (online teacher scoring during training), but a core promise is that the resulting student model has the same inference cost as any fine-tuned model of the same architecture. This metric verifies that promise and enables cost-adjusted comparisons. |
| **Applicable OPD families** | All families. The inference cost should be identical across OPD, offline KD, SFT, and RL for the same student architecture. Deviations indicate problems (e.g., speculative decoding drafters may have different inference profiles). For diffusion/flow OPD, inference cost includes the number of denoising steps and per-step compute. |
| **Example specific metrics** | (a) Throughput: tokens per second (text) or images per second (diffusion) on standard hardware. (b) Latency: time-to-first-token and time-to-completion for representative prompts. (c) Memory footprint: peak GPU memory during inference. (d) Quality-per-FLOP: task metric divided by inference FLOPs, enabling fair comparison across model sizes. (e) Speculative decoding acceptance rate: for OPD methods that produce drafters (DistillSpec, MASSV), measure the acceptance rate, which directly determines speedup. (f) Training cost: total GPU-hours for the OPD training run, reported alongside inference cost for full cost picture. Decompose into student generation cost, teacher scoring cost, and gradient update cost. |
| **What failure modes it catches** | Hidden inference costs from architectural changes introduced during distillation. Training costs that are prohibitive relative to the inference-time gains. Speculative decoding drafters that have low acceptance rates, negating the speedup. |
| **Limitations** | Inference cost depends heavily on hardware, batch size, sequence length, and serving framework. Comparisons across papers require standardized measurement conditions. Training cost is often underreported and hard to compare across codebases. |

---

### MF-10. Visual Perception vs. Reasoning (VLM-Specific)

| Field | Detail |
|---|---|
| **What it measures** | Whether OPD improves the model's reasoning ability without degrading its visual perception capability. This is a VLM-specific concern documented in `notes/opd-failure-modes.md`: text-heavy teacher signals can improve language reasoning while eroding visual grounding. Perception-R1 showed that standard VLM RLVR fails to improve perception (verified by McNemar's test). |
| **Applicable OPD families** | All VLM/MLLM OPD methods. VOLD and Uni-OPD (cross-modal text teacher to VLM student) are at highest risk for perception degradation. Self-teacher OPD (GUI-SD) with visual-only privileged context may be less affected. VLA-OPD with vision-language-action is affected in the visual perception component. Black-box VLM OPD depends on whether the teacher is multimodal. Not applicable to text-only or diffusion/flow OPD. |
| **Example specific metrics** | (a) Perception-reasoning decomposition: evaluate separately on perception-heavy benchmarks (OCRBench, DocVQA, ScreenSpot) and reasoning-heavy benchmarks (MathVista, MMMU-Pro, LogicVista). Report both and compute the delta. (b) McNemar's test for perception change: following Perception-R1's methodology, use McNemar's test to determine whether the OPD training significantly changed perception accuracy (not just reasoning accuracy). (c) Visual grounding accuracy: on tasks requiring spatial localization (e.g., "click on the submit button"), measure grounding accuracy separately from answer accuracy. (d) Hallucination rate: on benchmarks like POPE or MMHal-Bench, measure the rate of visual hallucinations (claims about visual content that is not present). OPD with text-only teachers may increase hallucination. (e) Caption fidelity: generate captions for held-out images and measure CLIP-score or human-judged fidelity. Detects whether OPD preserved visual understanding. (f) Perception-conditioned reasoning: on tasks that require both (e.g., "what is 3 times the number of red objects?"), measure whether errors come from the counting (perception) step or the multiplication (reasoning) step. |
| **What failure modes it catches** | The visual neglect problem: OPD improves language reasoning scores while the model increasingly ignores visual content. Text-only teacher mismatch: the teacher gives confident but visually ungrounded feedback (noted in `notes/opd-failure-modes.md`). Perception regression: OCR, grounding, or spatial understanding degrades after OPD training. Hallucination increase: the model "reasons" correctly from hallucinated visual content. |
| **Limitations** | Disentangling perception from reasoning errors requires benchmarks designed for this purpose, which are limited. CLIP-score and caption-based metrics are noisy proxies. McNemar's test requires item-level paired comparisons. Hallucination benchmarks may not cover all hallucination types relevant to OPD-trained models. The perception-reasoning decomposition assumes the two are separable, which may not always hold. |

---

## TASK 2: Evaluation Protocols

### Protocol A: White-Box Token-Level OPD

**Methods covered:** GKD, MiniLLM, DistiLLM, Entropy-Aware OPD, G-OPD, REOPOLD, vOPD, AOPD, TIP, CaOPD, SOD, Rock Tokens, SimCT, TCOD, VOLD (distillation component), Uni-OPD, LiteGUI (guided stage).

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD | Standard forward-KL KD on teacher-generated data (SeqKD, word-level KD) | Tests whether on-policy rollouts add value over offline distillation |
| Reward-only RL | GRPO or REINFORCE on the same task with verifiable reward only | Tests whether teacher distributional signal adds value over scalar reward |
| SFT | SFT on teacher-generated reasoning traces | Tests whether teacher-forcing is sufficient |
| Base model | Zero-shot or few-shot base model (no fine-tuning) | Establishes the performance floor |
| (Recommended) Offline reverse-KL | Offline reverse-KL KD (if comparing MiniLLM-family methods) | Isolates the on-policy contribution from the reverse-KL contribution |

#### Required metrics (beyond accuracy)

1. **Distributional fidelity (MF-3):** Report forward KL, reverse KL, and entropy correlation between student and teacher on a held-out eval set. This is the primary metric that validates the OPD objective.
2. **Exposure-bias reduction (MF-4):** Report quality as a function of generation length. Use at least three length bins (short / medium / long). Compare degradation slope against offline KD baseline.
3. **Calibration (MF-2):** Report ECE and Brier score. Compare against offline KD and SFT baselines.
4. **Length control (MF-6):** Report output length ratio (student / teacher) and length-conditional accuracy.

#### Protocol for testing exposure-bias reduction

Step 1: Select a task with variable-length outputs (e.g., math reasoning, summarization, multi-step QA).
Step 2: Partition the test set by required output length into at least three bins.
Step 3: For each method (OPD, offline KD, SFT, base), generate completions and measure task metric in each length bin.
Step 4: Plot task metric vs. length bin. Compute the slope of degradation. OPD should have a flatter (less negative) slope.
Step 5: Run the prefix-perturbation test: inject controlled errors at positions {10, 50, 100, 200} in prefixes, measure completion quality. OPD should recover better.
Step 6: Report statistical significance (paired bootstrap or permutation test across prompts).

#### Ablation design

- OPD loss only vs. OPD + MLE/SFT regularization.
- On-policy rollout vs. teacher-mixed rollout (lambda = 0 vs. lambda = 0.5 vs. lambda = 1 in GKD).
- Different divergence objectives on the same on-policy rollouts (FKL, RKL, JSD, skew-KL).
- Token selection strategies (all tokens vs. entropy-gated vs. TIP-selected vs. Rock-Token-frozen).
- Teacher temperature sweeps (tau = 1, 2, 4, 8).

---

### Protocol B: Black-Box Response-Level OPD

**Methods covered:** GAD, PRISM, OVD, SODA, ORPO-Distill.

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD | SFT on teacher-generated responses | Tests whether response-level OPD feedback adds value over simply training on teacher outputs |
| Reward-only RL | GRPO/PPO with a reward model but no teacher/discriminator | Tests whether the discriminator adds value over a learned reward |
| White-box OPD | GKD or MiniLLM (if applicable) | Tests whether response-level signal is competitive with token-level signal |
| SFT | SFT on the same data without response-level feedback | Establishes the SFT baseline |
| Base model | Zero-shot or few-shot | Performance floor |

#### Specific evaluation needs (different from white-box)

1. **Signal granularity test:** Black-box OPD provides response-level, not token-level, signal. Measure distributional fidelity (MF-3) at the sequence level: do student and teacher produce similar answer distributions over sampled completions? Compute answer-level KL via frequency estimation across multiple samples per prompt (e.g., 32 samples).
2. **Discriminator accuracy:** For GAD-style adversarial methods, report the discriminator's accuracy at distinguishing student from teacher outputs on a held-out set. If the discriminator is unreliable, the OPD signal is noisy.
3. **Response-level agreement (MF-7):** Fraction of prompts where student and teacher agree on the final answer. Report separately for easy and hard prompts.
4. **Calibration without logits:** Since black-box methods lack logit access, measure calibration via verbalized confidence ("I am 80% sure...") or via sampling-based confidence (fraction of samples that agree on the same answer out of N samples).
5. **Exposure-bias reduction (MF-4):** Still critical. Even with response-level signal, on-policy training should reduce exposure bias on long generations.

#### Protocol for testing discriminator/response-level signal effectiveness

Step 1: Train the full black-box OPD method.
Step 2: Train an ablation that replaces the discriminator/response-level teacher signal with a random signal of the same magnitude.
Step 3: Train an ablation that replaces the discriminator signal with a simple heuristic (e.g., length penalty, answer-format check).
Step 4: Compare all three. If the full method does not significantly outperform the heuristic ablation, the discriminator is not providing useful signal beyond format enforcement.

---

### Protocol C: Privileged Self-Teacher OPD

**Methods covered:** OPSD, OPCD, GUI-SD, Skill-SD, OPSDL.

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD | SFT on outputs from the model with privileged context | Tests whether on-policy self-distillation adds value over offline self-distillation |
| Reward-only RL | GRPO/PPO without privileged context signal | Tests whether the privileged context adds value over scalar reward |
| SFT | SFT on the same data without privileged context | Tests the contribution of the self-teacher |
| Full-context model | The model with privileged context at inference time (upper bound) | Establishes the ceiling: can the student match the privileged model? |
| Base model | Zero-shot without privileged context | Performance floor |

#### Protocol for testing privileged context internalization

The core claim of self-teacher OPD is that the student internalizes knowledge from the privileged context so that it performs well at inference time without that context.

Step 1: **Context removal test.** Train the model with OPD using privileged context (e.g., GUI-SD uses visible bounding boxes during training). At eval time, remove the privileged context. Measure the performance gap between (a) model with context, (b) OPD-trained model without context, (c) base model without context. The gap (a)-(b) measures internalization failure. The gap (b)-(c) measures internalization success.

Step 2: **IF-Eval-style context retention tests.** Design prompts that can only be answered correctly if the privileged context was internalized. For GUI-SD: ask the model to locate UI elements without bounding-box annotations. For OPSD with chain-of-thought context: ask questions whose answers require intermediate reasoning that was only available during privileged-context training. For OPSDL with short-context teacher: test on long-context prompts that require retrieving information beyond the teacher's short-context window.

Step 3: **Privileged feature probing.** Train a linear probe on the student's internal representations to predict the privileged information (e.g., bounding box coordinates, chain-of-thought steps). Higher probe accuracy indicates more successful internalization.

Step 4: **Progressive context degradation.** Gradually reduce the quality or completeness of the privileged context during training (e.g., add noise to bounding boxes, truncate chain-of-thought). Measure how gracefully the student's inference-time performance degrades. OPD should show a smoother degradation curve than SFT because the student learns from its own distribution.

---

### Protocol D: OPD+RL Hybrids

**Methods covered:** G-OPD, REOPOLD, VOLD, SOD, KDRL, vOPD, AOPD.

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD | Standard forward-KL KD | Tests on-policy contribution |
| Reward-only RL | GRPO/PPO with verifiable reward, no teacher KL | Tests teacher contribution |
| OPD only | Same OPD method without the RL reward component | Isolates the OPD contribution |
| RL + offline KD | RL with KL penalty against a frozen reference (standard RLHF-style) | Tests whether the on-policy KL provides more than a fixed reference KL |
| SFT | SFT on teacher traces | Teacher-forcing baseline |
| Base model | Zero-shot | Performance floor |

#### Ablation design to separate OPD from RL contributions

The central challenge: both OPD and RL improve performance, and they may interact non-linearly. The following ablation structure is necessary.

| Ablation | OPD loss | RL reward | What it tests |
|---|---|---|---|
| Full hybrid | Yes | Yes | Claimed result |
| OPD only | Yes | No | OPD contribution in isolation |
| RL only | No | Yes | RL contribution in isolation |
| OPD + random reward | Yes | Random | Whether RL reward quality matters |
| RL + frozen reference KL | KL to frozen ref (not teacher) | Yes | Whether teacher supervision (vs. fixed reference) matters |
| Reward coefficient sweep | Yes | Scaled (0.1x, 0.5x, 1x, 2x, 5x) | Sensitivity to relative weighting |

For each ablation, report:
1. Task accuracy (MF-1).
2. Distributional fidelity to teacher (MF-3).
3. Reward achieved (how well does the model optimize the RL objective?).
4. Length distribution (MF-6): do hybrid methods show more or less length inflation than pure RL?
5. Credit assignment quality (MF-8): does the OPD component improve step-level credit assignment over pure RL?

#### Interaction tests

- **Conflict detection:** On prompts where the teacher's preferred answer and the reward-optimal answer differ, which does the hybrid model follow? Report the fraction of conflict cases and the model's behavior.
- **Training dynamics:** Plot the OPD loss and RL reward separately over training steps. Do they move in the same direction (synergistic) or opposite directions (conflicting)?
- **Sample efficiency:** Compare the number of training steps required to reach a fixed performance threshold for OPD-only, RL-only, and hybrid.

---

### Protocol E: Action-Token OPD (VLA-OPD)

**Methods covered:** VLA-OPD, embodied OPD variants.

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD (behavioral cloning) | BC on expert demonstrations | Standard offline imitation baseline |
| Reward-only RL | PPO/SAC with environment reward | Tests whether teacher action-token supervision adds value over scalar reward |
| DAgger | Online imitation with expert relabeling | The closest non-OPD online imitation baseline |
| SFT on expert trajectories | SFT on the same demonstrations used for BC | Tests teacher-forcing in the action domain |
| Base model | Zero-shot or pretrained VLA without fine-tuning | Performance floor |

#### Embodied evaluation: success rate vs. sample efficiency

Step 1: **Success rate across task suites.** Evaluate on standardized task suites (LIBERO, RoboTwin2.0). Report success rate with confidence intervals across at least 50 rollouts per task (environment stochasticity requires many trials).

Step 2: **Sample efficiency curve.** Plot success rate as a function of training environment interactions. OPD should achieve a given success rate with fewer interactions than reward-only RL (because the teacher provides denser signal). DAgger may have similar sample efficiency but different asymptotic performance.

Step 3: **Multi-step error compounding.** Measure success rate as a function of task horizon (number of required action steps). OPD should degrade more gracefully than BC (which suffers from compounding errors, the embodied analog of exposure bias).

Step 4: **Generalization to unseen objects/configurations.** Evaluate on held-out object configurations, initial poses, and distractor objects. Report the performance gap between in-distribution and OOD settings.

#### Safety evaluation for on-policy exploration

On-policy exploration in embodied settings carries physical risk. Evaluation must include:

1. **Constraint violation rate:** Fraction of training rollouts that violate safety constraints (collisions, workspace boundary violations, excessive force).
2. **Exploration quality:** Fraction of training rollouts that visit genuinely informative states vs. degenerate or unsafe states.
3. **Sim-to-real transfer fidelity:** If training is in simulation, measure the reality gap in success rate and constraint violation when deploying in the real environment.
4. **Recovery from failure states:** After entering a failure state during exploration, can the policy recover and continue the task?

---

### Protocol F: Diffusion/Flow OPD

**Methods covered:** D-OPSD, Flow-OPD, diffusion-based self-distillation variants.

#### Required baselines

| Baseline type | Minimum required method | Purpose |
|---|---|---|
| Offline KD (progressive distillation) | Standard progressive distillation or consistency distillation | Tests whether on-policy denoising adds value over offline step reduction |
| Reward-only RL (RLHF for diffusion) | DPOK, DDPO, or ReFL with scalar reward | Tests whether teacher velocity/distribution signal adds value over scalar reward |
| SFT | Fine-tuning on curated image datasets | Standard fine-tuning baseline |
| Base model | Pre-trained diffusion/flow model without distillation | Performance floor |
| Teacher model | Full-step teacher model (upper bound) | Performance ceiling |

#### Distributional quality metrics for generated images

1. **FID (Frechet Inception Distance):** Standard distribution-level quality metric. Compute between generated images and a reference set. Lower is better.
2. **FID-CLIP:** FID computed in CLIP feature space rather than Inception feature space. Better captures semantic quality.
3. **CLIP Score:** Alignment between generated image and text prompt. Measures text-image fidelity.
4. **PickScore / ImageReward / HPS v2:** Learned human-preference-aligned scores. Capture aesthetic quality and prompt adherence.
5. **GenEval:** Compositional generation benchmark (object count, spatial relations, attribute binding). Tests whether OPD preserves compositional understanding.
6. **LPIPS diversity:** Average LPIPS distance between pairs of images generated from the same prompt. Tests mode diversity (has the student collapsed to fewer modes than the teacher?).
7. **Step-efficiency curve:** Image quality (FID or CLIP score) as a function of the number of denoising steps. OPD-distilled models should achieve given quality with fewer steps.

#### Per-step denoising quality evaluation

Step 1: For a fixed set of prompts, generate images using both the teacher (full steps) and the student (reduced steps).
Step 2: At matched noise levels (by interpolating the student's schedule to align with the teacher's), compute the velocity field error (L2 between teacher and student predicted velocities).
Step 3: Plot velocity field error across denoising steps. OPD should show lower error at steps the student actually visits during training (on-policy steps) compared to offline distillation.
Step 4: Measure whether error accumulates across steps (the diffusion analog of exposure bias). Compute image quality at intermediate denoising stages, not just the final output.

---

## TASK 3: Minimum Required Baselines per OPD Family

### Family A: White-Box Token-Level OPD

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **Forward-KL offline KD** | Offline KD | Isolate on-policy benefit | SeqKD, word-level KD, standard teacher-forcing distillation |
| 2 | **GRPO with verifiable reward** | Reward-only RL | Isolate teacher distributional signal benefit | GRPO on math, code with ground-truth rewards |
| 3 | **SFT on teacher traces** | SFT | Isolate dynamic supervision benefit | SFT on teacher-generated CoT reasoning |
| 4 | **Zero-shot base model** | Base model | Performance floor | Same architecture with no fine-tuning |
| 5 | **(Recommended) Offline reverse-KL** | Offline KD variant | Isolate on-policy from divergence choice | Offline MiniLLM-style reverse KL without fresh rollouts |

---

### Family B: Black-Box Response-Level OPD

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **SFT on teacher responses** | Offline KD | Isolate on-policy and discriminator benefit | SFT on teacher-generated text responses |
| 2 | **GRPO with reward model** | Reward-only RL | Isolate discriminator/response-level signal vs. scalar reward | GRPO/PPO with learned reward model |
| 3 | **SFT on same training data** | SFT | Isolate feedback signal benefit | Standard SFT without response-level teacher feedback |
| 4 | **Zero-shot base model** | Base model | Performance floor | Same architecture with no fine-tuning |
| 5 | **(Recommended) White-box OPD** | OPD upper bound | Test how much is lost by going black-box | GKD or MiniLLM if both teacher and student support it |

---

### Family C: Privileged Self-Teacher OPD

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **Offline self-distillation** | Offline KD | Isolate on-policy benefit within self-distillation | SFT on model's own outputs generated with privileged context |
| 2 | **GRPO without privileged context** | Reward-only RL | Isolate privileged context contribution | GRPO/PPO with only verifiable reward |
| 3 | **SFT without privileged context** | SFT | Isolate self-teacher contribution | Standard SFT on the same tasks |
| 4 | **Zero-shot base model** | Base model | Performance floor | Same model without fine-tuning or privileged context |
| 5 | **(Recommended) Full-context model** | Upper bound | Performance ceiling for internalization | Same model with privileged context available at inference |

---

### Family D: OPD+RL Hybrids

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **Forward-KL offline KD** | Offline KD | Isolate on-policy and RL benefit jointly | SeqKD, standard distillation |
| 2 | **GRPO / PPO with reward only** | Reward-only RL | Isolate teacher KL contribution | GRPO with verifiable reward, no KL penalty to teacher |
| 3 | **OPD loss only (no reward)** | OPD only | Isolate RL contribution | Same OPD method with RL reward coefficient set to zero |
| 4 | **SFT on teacher traces** | SFT | Teacher-forcing baseline | SFT on teacher-generated data |
| 5 | **Zero-shot base model** | Base model | Performance floor | Same architecture with no fine-tuning |
| 6 | **(Recommended) RL + frozen reference KL** | RL+KL | Test on-policy teacher vs. fixed reference | Standard RLHF with KL to the SFT model (not the teacher) |

---

### Family E: Action-Token OPD (VLA-OPD)

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **Behavioral cloning (BC)** | Offline KD | Standard offline imitation | BC on expert demonstrations |
| 2 | **PPO / SAC with environment reward** | Reward-only RL | Isolate teacher action-token supervision | RL with environment success signal |
| 3 | **SFT on expert action sequences** | SFT | Teacher-forcing in action space | SFT on expert trajectories |
| 4 | **Zero-shot pretrained VLA** | Base model | Performance floor | VLA with no task-specific fine-tuning |
| 5 | **(Recommended) DAgger** | Online imitation | Test expert relabeling vs. OPD-style supervision | Standard DAgger with expert querying |

---

### Family F: Diffusion/Flow OPD

| # | Baseline | Category | Purpose | Example methods |
|---|---|---|---|---|
| 1 | **Progressive distillation** | Offline KD | Isolate on-policy denoising benefit | Standard progressive distillation or consistency distillation |
| 2 | **DPOK / DDPO / ReFL** | Reward-only RL | Isolate teacher velocity/distribution signal | Reward-based RL for diffusion |
| 3 | **Fine-tuning on curated data** | SFT | Standard fine-tuning baseline | Fine-tuning on high-quality images |
| 4 | **Pre-trained model (full steps)** | Base model | Performance floor at inference cost | Base model without distillation (using full denoising steps) |
| 5 | **(Recommended) Teacher model** | Upper bound | Performance ceiling | Full-step teacher model |

---

## Cross-Cutting Recommendations

### Recommendation 1: Always report more than accuracy

Every OPD paper should report at least:
- Task accuracy (MF-1) on the primary benchmarks.
- One distributional metric (MF-3): KL, entropy correlation, or agreement rate.
- One exposure-bias metric (MF-4): quality vs. length or position-conditional accuracy.
- Output length statistics (MF-6): mean, median, and max output length.

### Recommendation 2: Test the on-policy claim directly

The most discriminative test for OPD is whether it outperforms its offline counterpart on long-generation tasks. If an OPD method does not show an advantage over offline KD on tasks requiring long outputs, the on-policy property is not providing its theorized benefit.

### Recommendation 3: Ablate the training paradigm, not just the loss function

Many OPD papers ablate the divergence objective (FKL vs. RKL vs. JSD) but not the rollout source (on-policy vs. off-policy). The most informative ablation is: hold the divergence objective fixed and compare on-policy vs. off-policy rollouts.

### Recommendation 4: For VLMs, always report perception and reasoning separately

Any VLM OPD paper must report at least one perception-focused benchmark (OCRBench, DocVQA, or ScreenSpot) and at least one reasoning-focused benchmark (MathVista, MMMU-Pro, or LogicVista). The delta from baseline should be reported for both.

### Recommendation 5: Report training cost alongside inference cost

OPD's training cost is higher than offline KD. Papers should report total GPU-hours, decomposed into student generation, teacher scoring, and gradient updates. This enables practitioners to assess the cost-benefit tradeoff.

### Recommendation 6: Use matched baselines

All baselines must use the same student architecture, the same training data (prompts), and the same training compute budget (or report compute-matched and step-matched comparisons). Unmatched comparisons confound the training paradigm with other factors.

---

## Benchmark Recommendations per OPD Family

Cross-referencing with `tables/benchmarks.md`:

| OPD Family | Primary Benchmarks | Perception Check | Reasoning Check | OOD Check |
|---|---|---|---|---|
| White-box token-level (LLM) | Task-specific (math, code, summarization) | N/A | Task-specific | Held-out domain |
| White-box token-level (VLM) | MathVista, MMMU-Pro | OCRBench, DocVQA | MathVista, LogicVista | Cross-domain VQA |
| Black-box response-level | Same as above per modality | Same as above | Same as above | Same as above |
| Privileged self-teacher | Task-specific + context-retention tests | If VLM: OCRBench | If VLM: MathVista | Prompts outside privileged context |
| OPD+RL hybrid (VLM) | MathVista, MMMU-Pro, MATH-Vision | OCRBench, ChartQA | MathVista, LogicVista | Cross-domain VQA |
| Action-token (VLA) | LIBERO, RoboTwin2.0 | N/A (embodied) | N/A (embodied) | Unseen object configurations |
| Diffusion/Flow | FID, GenEval, PickScore | N/A | GenEval (compositional) | Novel prompt styles |
| GUI agent | ScreenSpot, OSWorld | ScreenSpot (grounding) | Multi-step planning | Unseen applications |

---

*This evaluation methodology document was designed for the On-Policy Distillation Landscape survey. It should be updated as new OPD families emerge and as community evaluation standards evolve.*
