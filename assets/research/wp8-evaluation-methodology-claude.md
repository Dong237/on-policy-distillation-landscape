# WP8: Evaluation Methodology

**Agent:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13

---

## 1. Evaluation Philosophy

Most OPD papers report accuracy on popular leaderboards and declare victory, but accuracy alone cannot distinguish whether gains come from the OPD mechanism (distributional teacher supervision on student-generated states) or incidental factors (more training, better data, reward shaping). A principled OPD evaluation must probe three mechanistic advantages:

1. **Exposure-bias reduction** — OPD trains on self-generated trajectories, so it should recover better from its own mistakes than offline KD/SFT.
2. **Distributional fidelity** — Dense teacher KL should preserve teacher uncertainty and entropy better than reward-only RL.
3. **Dense guidance beyond scalar reward** — Token-level teacher supervision should provide finer credit assignment than episode-level reward.

**Evaluation rules:**
- Always compare strict OPD against both offline KD and reward-only RL baselines.
- Always report at least one calibration/distributional metric alongside accuracy.
- Always separate OPD-stage gains from RL-stage gains in hybrid methods (e.g., LiteGUI Stage 1 vs Stage 2).
- Do not use GSM8K or AIME 2024 as primary evidence due to contamination/small N.
- Prefer benchmarks with rolling new problems (LiveCodeBench) or simulator-based tasks (LIBERO, ALFWorld) that resist contamination.

---

## 2. Benchmark Bundle Table

| bundle | modality | target_methods | benchmark_names | official_urls | metrics | why_it_tests_opd | adjacent_baselines | contamination_or_private_risk | priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A: LLM Reasoning** | LLM | GKD, MiniLLM, G-OPD, REOPOLD, Fast-OPD, vOPD, TIP, AOPD, Rock-Tokens, SOD, OPSD, CRISP, SDPO, CaOPD | LiveCodeBench v6; AIME 2025; MATH-500; GPQA-Diamond; HumanEval+; MBPP+ | livecodebench.github.io; N/A; github.com/hendrycks/math; N/A; github.com/evalplus | accuracy; pass@1; pass@k diversity; ECE; token-entropy histogram | LiveCodeBench has rolling problems (low contamination); pass@k diversity detects mode collapse; ECE catches overconfidence from forward-KL | GRPO-only; offline KD (SeqKD); SFT-only; base model | LiveCodeBench: low; AIME 2025: medium (N=30); MATH-500: high; GPQA: medium | critical |
| **B: Instruction Following** | LLM | GKD, MiniLLM, DistiLLM, PAD, daDPO | Dolly/S-NI/UnNI (ROUGE-L); MT-Bench (multi-turn); AlpacaEval 2.0 LC; IF-Eval | N/A; github.com/lm-sys/FastChat; github.com/tatsu-lab/alpaca_eval; google/IFEval | ROUGE-L; judge score; win rate; format accuracy | Dolly/S-NI test long-form generation quality (exposure bias visible); MT-Bench tests multi-turn error recovery | GRPO-only; offline KD; SFT; base model | Medium (judge contamination possible) | high |
| **C: Calibration-Aware** | All | CaOPD; all strict OPD methods | Same as method's primary benchmarks + held-out calibration test set | N/A | ECE; Brier score; reliability diagram; confidence-accuracy correlation; entropy preservation ratio (student/teacher) | CaOPD identifies "Scaling Law of Miscalibration" — OPD improves accuracy but causes overconfidence. Universal test. | Any non-OPD method for calibration comparison | N/A (metric overlay, not separate benchmark) | critical |
| **D: Long-Context** | LLM | OPSDL, SDFT-continual | Long-context QA (LongBench, RULER); IF-Eval (OOD) | N/A | accuracy by context length; position-conditional accuracy; ECE | Tests whether privileged short-context self-teacher successfully internalizes long-context behavior | Full-context model (ceiling); SFT on long-context data; base model | Medium | medium |
| **E: Black-Box / API** | LLM | GAD, OVD, SODA, PRISM | AlpacaEval 2.0; MT-Bench; LMSYS-Chat 1M (OOD) | N/A | win rate; judge agreement; discriminator accuracy; sampling-based KL estimate | Tests whether response-level discriminator/verbal signal improves beyond scalar reward | GRPO-only; offline KD; SFT; DPO/RLHF | Medium | medium |
| **F: VLM Reasoning** | VLM/MLLM | VOLD, Uni-OPD, PRISM, KEPO (candidate) | MathVista; MMMU-Pro; MATH-Vision; LogicVista; VisuLogic; DynaMath; WeMath | mathvista.github.io; mmmu-benchmark.github.io; mathllm.github.io/mathvision; logicvista.github.io | accuracy; ECE; perception-reasoning decomposition | Tests whether OPD improves VLM reasoning where RLVR cannot (Perception-R1 shows RLVR fails perception) | VLM-R1 (GRPO-only); offline MLLM KD (LLaVA-KD); SFT-only | Medium (MMMU-Pro private test) | critical |
| **G: VLM Perception** | VLM/MLLM | VOLD, Uni-OPD | OCRBench; ChartQA; DocVQA; InfoVQA; AI2D | github.com/Yuliang-Liu/MultimodalOCR; github.com/vis-nlp/ChartQA; docvqa.org | accuracy; ANLS; hallucination rate; McNemar's test (perception vs reasoning) | Separates perception from reasoning — RLVR improves reasoning but not perception; OPD with VLM teacher should improve both | VLM-R1 (GRPO-only); Perception-R1 (perception reward); LLaVA-KD | Low-medium | high |
| **H: Video Temporal** | MLLM | Video-OPD | Charades-STA; ActivityNet; QVHighlights | N/A | IoU; mAP; credit assignment quality | Tests dense temporal grounding where sparse reward fails (Video-OPD reports 17% vs GRPO's 12% at 20% cost) | GRPO on video; offline KD; SFT | Low | high |
| **I: Speech/Cross-Modal** | Speech/LLM | X-OPD | BIG Bench Audio; Audio Multi-Challenge; VoiceBench; MMAR | N/A | accuracy; cross-modal transfer gap (speech vs text performance delta) | Tests whether text teacher successfully supervises speech student rollouts; "knowledge gap" finding (too-large teacher hurts) | Speech SFT; text-only baseline; speech RLVR | Low | medium |
| **J: GUI Grounding** | VLM | GUI-SD, LiteGUI (Stage 1) | ScreenSpot-Pro; ScreenSpot-v2; UI-Vision; OSWorld-G; MMBench-GUI L2 | github.com/OSU-NLP-Group/SeeClick | click/bbox accuracy; coordinate-token entropy | GUI-SD shows pure reverse-KL OPD > GRPO at 4x lower cost; "Distillation-to-SFT Collapse" finding is GUI-specific | GRPO-Gaussian; GRPO-Binary; SFT; base VLM | Low | high |
| **K: GUI Agent** | VLM+Agent | LiteGUI (Stage 2), TCOD | OS-World; Lite-Bench; ALFWorld; WebShop; ScienceWorld | os-world.github.io | success rate; step-level success curves; stage-separated deltas | MUST report Stage 1 (OPD) vs Stage 2 (GRPO) gains separately; TCOD tests temporal curriculum against error compounding | DigiRL; AgentQ; offline trajectory SFT; base agent | Low | high |
| **L: VLA/Robotics** | VLA | VLA-OPD | LIBERO (4 suites); RoboTwin2.0 | libero-project.github.io; robotwin-platform.github.io | success rate (50+ rollouts); action-token entropy; sample efficiency curve; seen-unseen trade-off | Action-token reverse-KL tests OPD in embodied setting; seen-unseen trade-off catches catastrophic forgetting | DAgger; offline SFT (OpenVLA-OFT); pure online RL | Very low (simulator) | high |
| **M: Agentic Tool/Code** | LLM+Agent | MAD-OPD, SOD, Skill-SD (candidate), OEC | BFCL-v4; tau2-Bench; VitaBench; LiveCodeBench; AppWorld | N/A | accuracy; step-level divergence; tool-call error recovery rate | MAD-OPD tests multi-teacher debate on tool tasks; SOD tests step-divergence reweighting; both test error cascade mitigation | reward-only agent RL; offline agent SFT (FireAct); base agent | Low | high |
| **N: Diffusion/Flow** | Image gen | D-OPSD, Flow-OPD (candidates) | GenEval; PickScore; FID; FID-CLIP; LPIPS diversity | N/A | FID; CLIP score; PickScore; GenEval compositionality; per-step denoising quality; LPIPS diversity | Tests whether velocity-field teacher supervision improves image quality over DDPO-style reward RL | DDPO; DPOK; Flow-GRPO; SFT distillation; base diffusion model | Low | medium |
| **O: Industrial Systems** | LLM | Qwen3 OPD stage, Nemotron-Cascade2, MiMo-V2-Flash, GLM-5 | Method's own reported benchmarks | N/A | accuracy on reported benchmarks; training FLOP efficiency; wall-clock time | Industrial claims must be stage-scoped; compare OPD stage delta, not full pipeline | Non-OPD stages of same pipeline; comparable open-source models | Varies | medium |

---

## 3. Metric Taxonomy Table

| metric_family | measures | applies_to | example_metrics | failure_modes_caught | limitations |
| --- | --- | --- | --- | --- | --- |
| **MF-1: Accuracy / Task Completion** | End-task performance | All methods | pass@1; accuracy; success rate; ROUGE-L; ANLS; IoU | None OPD-specific — necessary but insufficient | Cannot distinguish OPD from RLVR or offline KD gains; does not test mechanism |
| **MF-2: Calibration** | Gap between model confidence and actual accuracy | All methods | ECE (Expected Calibration Error); Brier score; reliability diagram; confidence-accuracy Pearson r | Overconfidence from forward-KL collapse (CaOPD's "Scaling Law of Miscalibration"); under-confidence from reverse-KL mode-seeking | Bin-size sensitivity for ECE; requires enough samples per bin; does not capture token-level distributional shape |
| **MF-3: Distributional Fidelity** | How well student's distribution matches teacher's | White-box OPD; privileged self-OPD | KL(student‖teacher) on held-out prompts; JSD; entropy correlation; top-k token overlap; NLL on teacher-generated text | Mode collapse (entropy too low); mode covering (entropy too high); top-1 accuracy masking top-k divergence | Requires white-box teacher access; meaningless for black-box OPD; sensitive to teacher quality |
| **MF-4: Exposure-Bias Reduction** | Quality degradation as a function of sequence length or rollout step | All methods (especially on-policy vs offline) | Position-conditional accuracy; quality@length curves; prefix-perturbation robustness; rollout-drift (self-BLEU at position N); step-level success rate curves | Error compounding in long generation; early-error cascade in multi-turn agents; length-conditional accuracy collapse | No standard benchmark exists; requires custom evaluation scripts; proxy metrics only |
| **MF-5: OOD Robustness** | Generalization to unseen domains/prompts | All methods | Cross-domain accuracy delta; adversarial prompt accuracy; format transfer accuracy; visual domain shift (for VLMs) | Overfitting to training distribution; teacher-dependent generalization ceiling; reward hacking on in-distribution prompts | No standard OOD suite for OPD; domain definition is arbitrary; results are benchmark-specific |
| **MF-6: Length / Verbosity Control** | Whether OPD causes length inflation or truncation collapse | OPD+RL hybrids; white-box OPD | Output length ratio (student/teacher); length-conditional accuracy; truncation-collapse rate; StableOPD length metrics | Length inflation (reverse-KL causes verbose outputs); truncation collapse (forward-KL causes premature stopping) | Length is correlated with quality on some benchmarks (AlpacaEval); length-controlled metrics partially address this |
| **MF-7: Teacher Agreement Rate** | How often student agrees with teacher decisions | White-box OPD; privileged self-OPD | Top-1 token agreement; top-k agreement; sequence-level agreement; confidence-conditioned agreement | Low agreement may indicate poor distillation; high agreement may indicate teacher dependence (no independent learning) | 100% agreement is not the goal (student should learn from teacher, not copy); requires teacher access at eval time |
| **MF-8: Credit Assignment Quality** | Whether the model correctly identifies which step/token caused an error | Agentic OPD; OPD+RL hybrids; multi-turn | Step-level error localization accuracy; process reward model agreement; self-correction success rate after identified error | Delayed credit assignment failure; cascading error attribution; credit misassignment to irrelevant steps | Requires ground-truth step-level labels (rare); process reward models are imperfect proxies |
| **MF-9: Inference Cost** | Runtime efficiency (training cost is separate) | All methods | Throughput (tokens/sec); latency (ms/token); quality-per-FLOP; training cost decomposition (teacher serving, rollout generation, KL computation) | OPD should have identical inference cost to SFT/RL (no teacher at inference); training cost is the real concern | Training cost is hard to compare fairly across methods (hardware, implementation, hyperparameter search) |
| **MF-10: Visual Perception vs Reasoning** | Whether OPD improves VLM reasoning without hurting perception (or vice versa) | VLM/MLLM OPD only | Perception-only accuracy (OCRBench, ScreenSpot); reasoning-only accuracy (MathVista, LogicVista); McNemar's test (Perception-R1 approach); hallucination rate; caption fidelity | Perception-R1 shows RLVR "fails to enhance multimodal perception" — OPD with VLM teacher should improve both | Requires clean perception-reasoning separation in benchmarks; some benchmarks conflate both |
| **MF-11: Pass@k Diversity** | Whether OPD collapses sampling diversity | White-box OPD; all reasoning OPD | Pass@1 vs Pass@k ratio; self-BLEU across samples; unique-solution count at Pass@32; SCOPE's "Pass@k Paradox" metric | OPD may improve Pass@1 while degrading Pass@32 (diversity collapse from teacher imitation) | Computationally expensive (requires k samples per prompt); k-dependent; Pass@k may not matter for deployment |

---

## 4. Evaluation Protocols by OPD Family

### Protocol A: White-Box Token-Level OPD
**Target:** GKD, MiniLLM, G-OPD, REOPOLD, vOPD, TIP, AOPD, Rock-Tokens, SOD, TCOD, LiteGUI (Stage 1)

**Required baselines:** (1) Offline KD (SeqKD or same divergence on teacher data), (2) GRPO/RLVR (same student, reward-only), (3) SFT-only, (4) Base model (zero-shot), (5) Teacher model (ceiling)

**Metrics beyond accuracy:** ECE, token-entropy histogram (student vs teacher), pass@k diversity, position-conditional accuracy curve, length ratio

**Exposure-bias test protocol:**
1. Evaluate accuracy at output positions 0–128, 128–256, 256–512, 512–1024
2. OPD should show flatter accuracy curve than offline KD
3. Inject prefix perturbations and measure recovery quality
4. Compare rollout quality at step N in agentic tasks

**Ablation requirements:** (a) rollout source: student-on-policy vs teacher-generated vs mixed, (b) divergence: forward-KL vs reverse-KL vs JSD, (c) token selection: all vs selective (TIP, Rock-Tokens), (d) teacher temperature

### Protocol B: Black-Box Response-Level OPD
**Target:** GAD, OVD, SODA, PRISM

**Additional metrics:** Discriminator accuracy (overfitting detection); sampling-based KL estimate (sample N responses, estimate distribution); response-level agreement rate with teacher

**Key test:** Does removing the discriminator and using only scalar reward match performance? If yes, the distributional signal is not necessary, and the method may be just sophisticated reward shaping.

### Protocol C: Privileged Self-Teacher OPD
**Target:** OPSD, CRISP, OPCD, GUI-SD, Skill-SD, CaOPD, OPSDL, SDFT-continual

**Context internalization test:**
1. Train with privileged context (answers, skills, documents, visual bbox)
2. Evaluate WITHOUT privileged context (student mode)
3. Compare against full-context model (teacher mode) as ceiling
4. Measure what fraction of teacher-mode performance the student achieves
5. IF-Eval-style retention tests for specific internalized knowledge

### Protocol D: OPD+RL Hybrids
**Target:** G-OPD, REOPOLD, VOLD, SOD, KDRL, Skill-SD

**Ablation table (mandatory):**
| Config | OPD component | RL component | What it tests |
|---|---|---|---|
| RL-only | OFF | ON | Reward-only baseline |
| OPD-only | ON | OFF | Pure distillation baseline |
| OPD+RL | ON | ON | Full method |
| OPD+RL (lambda=0) | scaled to 0 | ON | Sensitivity to distillation weight |
| OPD+RL (no reward) | ON | ON but reward=0 | Sensitivity to reward signal |
| Offline KD+RL | OFF (offline KD) | ON | Off-policy distillation + RL |

### Protocol E: Action-Token VLA-OPD
**Target:** VLA-OPD

**Embodied requirements:** (a) 50+ rollouts per task for statistical reliability, (b) report mean ± std, not just mean, (c) success rate curves over training steps (sample efficiency), (d) seen-unseen task split (catastrophic forgetting test), (e) action-token entropy analysis

**Safety evaluation:** Constraint violation rate during on-policy exploration; exploration quality (coverage of state space); sim-to-real transfer gap if applicable

### Protocol F: Diffusion/Flow Borderline OPD
**Target:** D-OPSD, Flow-OPD

**Required metrics:** FID; FID-CLIP; CLIP Score; PickScore; GenEval compositionality; LPIPS diversity (sample diversity); step-efficiency curve (quality vs number of denoising steps)

**Per-step quality test:** Evaluate intermediate denoising states to verify that teacher supervision improves per-step quality, not just final output.

---

## 5. Candidate Updates for `tables/benchmarks.md`

| name | category | modality | domain | official_url | metrics | common_for_vlm_rlvr | common_for_vlm_opd | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiveCodeBench | code_generation | text | code | https://livecodebench.github.io/ | pass@1; pass@k | no | yes | **Preferred code benchmark for OPD.** Rolling new problems resist contamination. SDPO, SOD, MAD-OPD use this. |
| AIME 2025 | math_reasoning | text | competition_math | N/A | accuracy (N=30) | unclear | yes | Less contaminated than AIME 2024. SOD reports 26.13% with 0.6B student. Report with confidence intervals. |
| HumanEval+ | code_generation | text | code | https://github.com/evalplus/evalplus | pass@1 | no | unclear | Extended HumanEval with additional test cases. |
| MBPP+ | code_generation | text | code | https://github.com/evalplus/evalplus | pass@1 | no | unclear | Extended MBPP with additional test cases. |
| ALFWorld | agentic | text_env | household_agent | https://alfworld.github.io/ | success rate | no | yes | TCOD uses this. Multi-turn text agent with error compounding. Simulator-based, low contamination. |
| WebShop | agentic | text_env | e_commerce_agent | https://webshop-pnlp.github.io/ | success rate; reward | no | yes | TCOD uses this. E-commerce web agent task. |
| ScienceWorld | agentic | text_env | science_agent | https://sciworld.apps.allenai.org/ | success rate | no | yes | TCOD uses this. Science experiment agent. |
| BFCL-v4 | agentic | text | function_calling | https://gorilla.cs.berkeley.edu/leaderboard.html | accuracy (8 subtasks) | no | yes | MAD-OPD uses this. Berkeley Function-Calling Leaderboard v4. |
| tau2-Bench | agentic | text | customer_service | N/A | success rate | no | yes | MAD-OPD uses this. Multi-turn customer service (retail, airline, telecom). |
| VitaBench | agentic | text | life_service | N/A | success rate | no | yes | MAD-OPD uses this. Real-world life-service agent with 66 tools. |
| AppWorld | agentic | text | app_interaction | N/A | accuracy; completion | no | yes | Skill-SD uses this. Multi-turn app interaction agent. |
| ScreenSpot-Pro | gui_grounding | image_text | gui | N/A | click accuracy | yes | yes | LiteGUI and GUI-SD primary benchmark. Updated from ScreenSpot. |
| Lite-Bench | gui_agent | image_text_action | desktop_agent | N/A | success rate; step accuracy | no | yes | LiteGUI evaluation benchmark. |
| VisuLogic | logic_reasoning | image_text | visual_logic | N/A | accuracy | unclear | yes | Uni-OPD uses this. Visual logic reasoning. |
| DynaMath | math_reasoning | image_text | dynamic_visual_math | N/A | accuracy | unclear | yes | Uni-OPD and VOLD use this. Dynamic visual math variations. |
| WeMath | math_reasoning | image_text | visual_math | N/A | accuracy | unclear | yes | Uni-OPD and VOLD use this. Visual math benchmark. |
| InfoVQA | document | image_text | infographic_vqa | https://www.docvqa.org/datasets/infographicvqa | ANLS | yes | yes | Uni-OPD uses this. Infographic document VQA. |
| AI2D | document | image_text | science_diagram | https://allenai.org/data/diagrams | accuracy | yes | yes | Uni-OPD uses this. Science diagram understanding. |
| Charades-STA | video_grounding | video_text | temporal_grounding | N/A | IoU; mAP | no | yes | Video-OPD primary benchmark. |
| ActivityNet | video_grounding | video_text | temporal_grounding | http://activity-net.org/ | IoU; mAP | no | yes | Video-OPD evaluation benchmark. |
| QVHighlights | video_grounding | video_text | temporal_grounding | N/A | IoU; mAP | no | yes | Video-OPD evaluation benchmark. |
| BIG Bench Audio | speech_reasoning | speech_text | logical_reasoning | N/A | accuracy | no | yes | X-OPD uses this. Audio logical reasoning. |
| VoiceBench | speech_instruction | speech_text | instruction_following | N/A | accuracy | no | yes | X-OPD uses this. Speech instruction following. |
| GenEval | image_generation | text_to_image | compositionality | N/A | compositionality score | no | unclear | Flow-OPD uses this. Text-to-image compositionality. |
| OmniMedVQA | medical_vqa | image_text | medical | N/A | accuracy | no | unclear | KEPO uses this. 8 imaging modalities. |

---

## 6. Gap Map

### 6.1 Missing Benchmarks / Evaluation Methods

| Gap | Description | Impact | Recommended Action |
| --- | --- | --- | --- |
| **No exposure-bias benchmark** | No benchmark explicitly measures error recovery from self-generated mistakes. Multi-turn agentic benchmarks (ALFWorld, WebShop) are the closest proxy. | Cannot directly verify OPD's primary theoretical advantage | Design synthetic error-injection protocol: force student into known error states, measure recovery quality under OPD vs offline KD vs RLVR |
| **ECE almost never reported** | CaOPD is the only OPD paper measuring calibration. All other papers report accuracy only. | Cannot verify whether OPD preserves or distorts teacher uncertainty | Make ECE a mandatory evaluation metric for all OPD papers |
| **No standard OOD suite for OPD** | Cross-domain transfer is tested incidentally (VOLD, X-Reasoner, KEPO) but never systematically. | Cannot verify OPD generalizes better than offline KD to unseen domains | Propose held-out domain evaluation: train on subset of domains, test on held-out domains |
| **No token-level distributional metrics reported** | No OPD paper reports token-level entropy preservation, KL(student‖teacher) on held-out data, or top-k agreement as primary metrics. | Cannot verify OPD actually produces better distributional fidelity than alternatives | Add MF-3 (Distributional Fidelity) as mandatory evaluation for white-box OPD |
| **Pass@k diversity rarely measured** | SCOPE identifies "Pass@k Paradox" (pass@1 up, pass@32 down) but most papers ignore it. | Mode collapse from teacher imitation is undetected | Report pass@1 AND pass@32 (or pass@64) for reasoning tasks |
| **No perception-reasoning decomposition standard** | Perception-R1 uses McNemar's test but no standard protocol exists. | Cannot verify whether VLM OPD improves perception AND reasoning or just reasoning | Adopt Perception-R1's McNemar approach as standard for VLM OPD |
| **Stage separation not enforced** | LiteGUI, VOLD, KDRL, and other hybrid methods do not always report stage-separated ablations. | Cannot attribute gains to OPD vs RL vs SFT stages | Require ablation table (Protocol D) for all multi-stage methods |
| **No diffusion/flow OPD distributional metrics** | D-OPSD and Flow-OPD have no metric for per-step denoising distributional quality. | Cannot verify velocity-field supervision provides distributional benefit over DDPO-style reward | Design per-step FID or per-step CLIP score at intermediate denoising stages |

### 6.2 Contamination Warnings

| Benchmark | Risk | Issue | Recommendation |
| --- | --- | --- | --- |
| GSM8K | Very high | In pretraining data of all major LLMs; saturated (>90% for 7B+) | Do not use as primary OPD evidence |
| AIME 2024 | High | Only N=30; widely used; confidence intervals are ±10% | Report with bootstrapped CIs; supplement with AIME 2025 |
| MATH (Hendrycks) | High | Present in many pretraining corpora | Use sparingly; prefer LiveCodeBench |
| MMMU-Pro | Medium | Private test available; widely used | Use private test when available |
| AlpacaEval 2.0 | Medium | Length bias in judge; GPT-4 judge may favor GPT-4-trained models | Use length-controlled variant only |

### 6.3 Minimum Baselines by OPD Family

| OPD Family | Offline KD Baseline | Reward-Only RL Baseline | SFT Baseline | Additional Required |
| --- | --- | --- | --- | --- |
| White-box token-level | SeqKD (teacher-generated data + same divergence) | GRPO (same student, reward-only) | SFT (same data, no distillation) | Teacher model (ceiling) |
| Black-box response-level | Offline response-level KD | GRPO/RLHF (scalar reward) | SFT | DPO (static preferences) |
| Privileged self-teacher | SFT on context-conditioned data | GRPO | SFT | Full-context model (teacher mode) |
| OPD+RL hybrids | Offline KD + RL | RL-only (remove KD term) | SFT | OPD-only (remove RL term) |
| Action-token VLA | Offline SFT (OpenVLA-OFT) | Pure online RL | SFT on demonstrations | DAgger (hard action labels) |
| Agentic tool/browser | Offline trajectory SFT (FireAct, AgentTuning) | Reward-only agent RL | SFT on tool traces | Expert-completed (OEC) |
| Diffusion/flow | SFT distillation (standard step distillation) | DDPO/DPOK (reward-only RL) | Base diffusion model | Teacher (multi-step) |

---

*End of WP8 evaluation methodology.*
