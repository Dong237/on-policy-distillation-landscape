# WP8 Evaluation Methodology Synthesis

Checked: 2026-05-13

Inputs merged:

- `assets/research/wp8-evaluation-methodology-chatgpt.md`
- `assets/research/wp8-evaluation-methodology-claude.md`
- `assets/research/wp8-evaluation-methodology-gemini.md`
- Current repo tables and WP7 synthesis notes.

Scope: WP8 did not discover or relabel methods. It designed evaluation coverage for the current strict, borderline, partial, adjacent, and not-OPD landscape.

## Memo

All three WP8 agents converged on the same principle: OPD evaluation cannot be an accuracy leaderboard. Accuracy is necessary, but it does not identify the mechanism. A strict OPD claim should be evaluated on:

1. Capability: accuracy, pass rate, success rate, win rate, grounding score, or image quality.
2. On-policy robustness: rollout drift, self-error recovery, long-horizon failure recovery, OOD prompt slices, and step/horizon-conditioned success.
3. Distributional transfer: teacher-student KL/JS, entropy preservation, calibration, top-k support retention, preference/judge margin stability, or action/field fidelity.

The central baseline structure is:

| OPD family | minimum controls |
|---|---|
| White-box token/logit OPD | SFT; offline KD on teacher data; reward-only RLVR where applicable; same-student base model; teacher ceiling. |
| Black-box/API response OPD | SFT on teacher outputs; reward-only RL/RLAIF; DPO/ORPO-style static preference tuning; multi-judge or discriminator ablations. |
| Privileged self-OPD | no-privilege baseline; full-privilege ceiling; offline self-distillation on privileged traces; reward-only baseline when possible. |
| OPD+RL hybrids | OPD-only; RL-only; OPD+RL; offline KD+RL; reward/KL weight sweeps. |
| Action-token/VLA OPD | behavior cloning; DAgger or expert-correction imitation; reward-only RL; horizon-conditioned success and action KL. |
| Diffusion/flow borderline OPD | progressive/consistency distillation; DDPO/DPOK/reward-only diffusion RL; teacher score/velocity-field fidelity; diversity checks. |

Method rows should not be promoted because a benchmark is popular. Conversely, weak static benchmarks can still be useful when bundled with calibration, recovery, and distributional diagnostics.

## Benchmark Bundles

| bundle | modality | target_methods | benchmark_names | official_urls | metrics | why_it_tests_opd | adjacent_baselines | risk | priority |
|---|---|---|---|---|---|---|---|---|---|
| Text reasoning plus instruction robustness | text | Core strict LLM OPD and OPD+RL hybrids | MATH-500; GPQA; IFEval; LiveBench; LiveCodeBench | https://huggingface.co/datasets/HuggingFaceH4/MATH-500; https://github.com/idavidrein/gpqa; https://github.com/google-research/google-research/tree/master/instruction_following_eval; https://livebench.ai/; https://livecodebench.github.io/ | exact/pass rate; pass@k; instruction pass rate; token budget; calibration | Pairs reward-friendly math/code with instruction and live/OOD tasks where reward-only RLVR is weaker. | SFT; offline KD; GRPO/RLVR-only; DPO/IPO/SimPO | MATH/GPQA public; LiveBench and LiveCodeBench reduce but do not remove contamination. | high |
| Rollout drift and self-error recovery | text; agent; multimodal | All methods claiming student-state benefits | Perturb-and-recover overlays on MATH-500; LiveCodeBench; IFEval; OSWorld; WebArena; SWE-bench Verified; LIBERO | Uses official suites above plus https://os-world.github.io/; https://webarena.dev/; https://www.swebench.com/; https://libero-project.github.io/ | recovery after first error; regression after correction; replan success; teacher intervention count | Directly tests the exposure-bias claim: can the trained student recover from its own states? | reward-only self-correction RL; SFT on corrected traces; DAgger-style corrections | No canonical benchmark; publish seeds and perturbation policy. | high |
| Long-context and context-transfer OPD | text | OPSDL-style and privileged-context methods | RULER; LongBench; BABILong | https://github.com/NVIDIA/RULER; https://github.com/THUDM/LongBench; https://github.com/booydar/babilong | accuracy by length; position-conditioned score; privileged-teacher KL; refusal calibration | Tests context drift, but only becomes OPD-specific when rerollout/corrupted-prefix protocols are added. | long-context SFT; retrieval; offline context distillation | Synthetic tasks can be brittle; public suites are contamination-prone. | high |
| Calibration and teacher uncertainty | text; image_text; action | Entropy-aware OPD, CaOPD, Uni-OPD, GUI-SD, VLA-OPD | HELM; GPQA/MMLU-Pro/MathVista overlays; method-held-out rollouts | https://crfm.stanford.edu/helm/; https://github.com/idavidrein/gpqa; https://github.com/TIGER-AI-Lab/MMLU-Pro; https://mathvista.github.io/ | ECE; Brier; NLL; entropy gap; JS/KL; top-k mass; risk-coverage | OPD is a distributional claim, so teacher uncertainty should survive on student-visited states. | hard-label SFT; offline KD; RLVR-only; post-hoc calibration | Requires logit/probability access or careful sampling proxies. | high |
| Black-box/API response alignment | text; image_text | GAD/OVD/SODA/ORPO-Distill/PAD-like partial methods | AlpacaEval 2; MT-Bench; RewardBench; Arena-Hard-style suites | https://github.com/tatsu-lab/alpaca_eval; https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge; https://github.com/allenai/reward-bench; https://github.com/lmarena/arena-hard-auto | length-controlled win rate; judge agreement; margin calibration; judge-swap stability | Uses response-level evidence without pretending token KL is available. | PRISM adjacent; RLHF/RLAIF; DPO/ORPO; teacher-generated SFT | Judge bias, length bias, and private APIs make this auxiliary evidence only. | medium |
| OPD+RL reward-conflict bundle | text; code; image_text | G-OPD, REOPOLD, SOD, KDRL/SCOPE borderline, VOLD, MiMo/Nemotron stages | MATH-500; LiveCodeBench; GPQA; IFEval; MathVista; MMMU-Pro | https://huggingface.co/datasets/HuggingFaceH4/MATH-500; https://livecodebench.github.io/; https://github.com/idavidrein/gpqa; https://github.com/google-research/google-research/tree/master/instruction_following_eval; https://mathvista.github.io/; https://mmmu-benchmark.github.io/ | final accuracy; reward score; teacher KL; reward-KL conflict rate; length inflation | Separates dense teacher signal from scalar reward via OPD-only/RL-only/hybrid ablations. | GRPO/PPO/RLVR-only; offline KD; SFT; DPO/IPO/SimPO | Accuracy-only tables can hide reward hacking. | high |
| Industrial and speculative systems | text | DistillSpec, Fast OPD, Lightning OPD partial, Qwen/Nemotron/MiMo stage claims | Spec-Bench; MT-Bench; AlpacaEval; LiveBench; vendor workloads | https://github.com/hemingkx/Spec-Bench; https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge; https://github.com/tatsu-lab/alpaca_eval; https://livebench.ai/ | acceptance rate; latency; tokens/s; quality parity; teacher calls/token | Tests whether OPD improves quality-compute tradeoffs, not just final score. | untrained draft; offline KD draft; standard speculative decoding | Hardware and private workloads are hard to compare. | medium |
| VLM visual reasoning and perception | image_text | VOLD, Uni-OPD, KEPO borderline | MathVista; MMMU-Pro; MATH-Vision; LogicVista; MathVerse; OCRBench; ChartQA; DocVQA; HallusionBench; POPE | https://mathvista.github.io/; https://mmmu-benchmark.github.io/; https://mathllm.github.io/mathvision/; https://logicvista.github.io/; https://github.com/ZrrSkywalker/MathVerse; https://github.com/Yuliang-Liu/MultimodalOCR; https://github.com/vis-nlp/ChartQA; https://www.docvqa.org/; https://github.com/tianyi-lab/HallusionBench; https://github.com/AoiDragon/POPE | accuracy; ANLS; hallucination rate; perception/reasoning split; calibration | Detects text-teacher visual mismatch and separates reasoning gains from perception regressions. | PRISM adjacent response alignment; VLM-R1/R1-VL/Perception-R1; LLaVA-KD/LLAVADI; visual SFT | Public VLM benchmarks are often contaminated; hidden tests reduce reproducibility. | high |
| Video temporal grounding | video_text | Video-OPD and video RLVR controls | ActivityNet Captions; QVHighlights; Charades-STA; VideoMME | http://activity-net.org/challenges/2017/captioning.html; https://github.com/jayleicn/moment_detr; https://github.com/jiyanggao/TALL; https://github.com/BradyFU/Video-MME | mIoU; Recall@k; boundary error; temporal grounding F1 | Video-OPD claims need temporal localization and drift metrics, not only answer accuracy. | DeepVideo-R1; offline video SFT; video-CoT distillation | Video availability and annotation noise matter. | high |
| Speech and cross-modal OPD | audio_text | X-OPD and future speech OPD | AudioBench; BigBenchAudio; AIR-Bench; Dynamic-SUPERB | https://github.com/AudioLLMs/AudioBench; https://github.com/google/big-bench-audio; https://github.com/OFA-Sys/AIR-Bench; https://github.com/dynamic-superb/dynamic-superb | speech QA accuracy; WER/CER; text-vs-speech transfer gap; calibration | Checks whether a text teacher improves speech-model reasoning without collapsing acoustic information. | ASR-then-LLM; speech SFT; text-only KD; speech RLVR | ASR shortcuts can inflate reasoning claims. | medium |
| VLA and embodied action OPD | vision_language_action | VLA-OPD and action-token OPD | LIBERO; RoboTwin2.0; CALVIN; RoboCasa; SimplerEnv | https://libero-project.github.io/; https://robotwin-platform.github.io/; https://github.com/mees/calvin; https://robocasa.ai/; https://simpler-env.github.io/ | task success; horizon-conditioned success; action KL/NLL; safety violations; expert-query budget | Directly tests compounding action-state shift on visited states. | behavior cloning; DAgger; offline imitation; reward-only RL | Sim-to-real gap and seed sensitivity are high. | high |
| GUI grounding and GUI agents | screenshot_text; image_text_action | GUI-SD, LiteGUI, TCOD/Skill-SD/OpenClaw slices | ScreenSpot; ScreenSpot-Pro; OSWorld; WebArena; Mind2Web | https://github.com/OSU-NLP-Group/SeeClick; https://github.com/OS-Copilot/ScreenSpot-Pro; https://os-world.github.io/; https://webarena.dev/; https://github.com/OSU-NLP-Group/Mind2Web | click accuracy; task success; wrong-click recovery; coordinate-token KL; invalid action rate | Separates single-step coordinate OPD from long-horizon GUI agency. | SeeClick offline; VLM-R1 GUI RLVR; WebRL; Mind2Web SFT | Static screenshots and desktop/browser environments should not be collapsed. | high |
| Agentic tool/browser/code tasks | text_action; browser; code | MAD-OPD, SOD, TCOD, OpenClaw component | SWE-bench Verified; WebArena; AppWorld; ALFWorld; WebShop; ScienceWorld; tau-bench; BFCL | https://www.swebench.com/; https://webarena.dev/; https://appworld.dev/; https://github.com/alfworld/alfworld; https://webshop-pnlp.github.io/; https://github.com/allenai/scienceworld; https://github.com/sierra-research/tau-bench; https://gorilla.cs.berkeley.edu/leaderboard.html | task success; patch acceptance; tool-call correctness; failed-tool recovery; step-level KL | Agent OPD lives or dies on visited-state supervision and long-horizon recovery. | OEC adjacent/DAgger control; ReTool; Search-R1; WebRL; offline trajectory SFT; reward-only agent RL | Pass/fail tests are sparse and live tasks are brittle. | high |
| Diffusion and flow borderline image generation | image | D-OPSD, Flow-OPD | GenEval; T2I-CompBench; HEIM; HPSv2 | https://github.com/djghosh13/geneval; https://github.com/Karine-Huang/T2I-CompBench; https://crfm.stanford.edu/helm/heim/latest/; https://github.com/tgxs002/HPSv2 | prompt compositionality; CLIP/FID/HPS; diversity; teacher velocity/score-field MSE | Reward or image quality alone cannot establish OPD; trajectory/field fidelity is required. | progressive distillation; consistency distillation; DDPO/DPOK; image-preference DPO | Auto-evaluators miss mode collapse and reward overoptimization. | medium |

## Metric Taxonomy

| metric_family | measures | applies_to | examples | failure_modes_caught | limitations |
|---|---|---|---|---|---|
| Task success and capability | Whether the model solves the task. | all | accuracy; pass@k; success rate; win rate; ANLS; mIoU | catastrophic failure; gross regression | Cannot distinguish OPD from RLVR, SFT, or offline KD. |
| Distributional fidelity | Whether the student matches teacher distributions on student rollouts. | white-box OPD; action OPD; diffusion/flow analogs | KL; JS; top-k overlap; entropy correlation; action KL; velocity-field MSE | mode collapse; teacher-distribution drift | Requires teacher probabilities or method-specific proxies. |
| Calibration and uncertainty | Whether confidence tracks correctness and teacher uncertainty. | all, strongest for white-box | ECE; Brier; NLL; reliability diagrams; selective risk-coverage | overconfidence; hallucination; entropy collapse | Binning and API confidence extraction are fragile. |
| Exposure-bias reduction | Whether performance degrades less on self-generated states. | text; agents; GUI; VLA | corrupted-prefix recovery; first-error recovery; horizon-conditioned success; quality vs length | compounding errors; brittle self-correction | No single canonical public benchmark exists. |
| OOD and live robustness | Generalization beyond training distribution. | all | LiveBench; LiveCodeBench; held-out domains; temporal splits | benchmark overfit; narrow teacher mimicry | Attribution to OPD requires matched baselines. |
| Length and verbosity control | Whether OPD changes output length pathologies. | text; black-box; OPD+RL | length ratio; length-controlled win rate; truncation rate | reward/judge length hacking; truncation collapse | Length is task-dependent. |
| Credit assignment | Whether dense supervision improves step/token/action attribution. | reasoning; agents; VLA; diffusion | step-error localization; failed-action recovery; reward-KL conflict rate | sparse-reward shortcuts; delayed credit errors | Needs step labels or careful environment instrumentation. |
| Perception vs reasoning | Whether VLM OPD helps reasoning without degrading visual grounding. | VLM/MLLM/GUI | OCRBench; DocVQA; ChartQA; MathVerse; ScreenSpot; POPE | visual neglect; text-teacher mismatch; hallucination | Perception/reasoning are not always separable. |
| Systems and teacher cost | Whether the method is worth online supervision. | industrial; speculative; all teacher-scored OPD | teacher calls/token; GPU-hours; TTFT; latency; acceptance rate | uneconomical OPD loops; hidden serving costs | Hardware-sensitive and often underreported. |
| Diversity and coverage | Whether distillation collapses the support. | reverse-KL OPD; diffusion; reasoning | pass@k-pass@1 gap; self-BLEU; LPIPS diversity; teacher support recall | sampling diversity loss; mode collapse | Must be paired with capability metrics. |

## Table Updates Merged

`tables/benchmarks.md` was expanded beyond the initial VLM-heavy seed table to cover WP8 evaluation bundles. Merged rows include:

- Text/reasoning/instruction: LiveBench, LiveCodeBench, IFEval, GPQA, MATH-500, MMLU-Pro, RULER, LongBench, BABILong, HELM, RewardBench, AlpacaEval 2, MT-Bench, Spec-Bench.
- VLM/perception: MathVerse, InfoVQA, AI2D, HallusionBench, POPE, MMHal-Bench, ScreenSpot-Pro.
- Agent/tool/code/GUI: WebArena, Mind2Web, SWE-bench Verified, AppWorld, ALFWorld, WebShop, ScienceWorld, tau-bench, Berkeley Function Calling Leaderboard.
- Video/audio: ActivityNet Captions, QVHighlights, Charades-STA, VideoMME, AudioBench, BigBenchAudio, AIR-Bench, Dynamic-SUPERB.
- Robotics/image generation: CALVIN, RoboCasa, SimplerEnv, GenEval, T2I-CompBench, HEIM, HPSv2.

Existing rows were updated conservatively:

- MathVista notes now say to pair it with perception and calibration slices.
- ScreenSpot notes now warn to keep ScreenSpot variants separate.
- OSWorld now records 369 tasks and is tagged as relevant to VLM/GUI OPD evaluation.
- LIBERO now records the spatial/object/goal/long suite split and 130-task scope.

Rows not added in this pass: Lite-Bench, DeepPlanning, DPG-Bench, VitaBench, tau2-Bench, and several newer or name-collided benchmarks where the WP8 files did not supply a stable primary source URL.

## Gap Map

| gap | current state | recommended repo posture |
|---|---|---|
| Canonical exposure-bias benchmark | No official benchmark directly measures self-generated-state recovery for OPD. | Treat recovery as an overlay protocol on official suites; record perturbation policy, seeds, and oracle. |
| Teacher uncertainty transfer | Most method papers report accuracy only. | Require ECE/Brier/NLL plus teacher-student entropy/KL/JS where teacher distributions exist. |
| OPD vs RLVR separation | Many reasoning/VLM gains can be explained by reward-only RLVR. | Require OPD-only, RL-only, OPD+RL, offline KD, and SFT ablations for hybrids. |
| Black-box/API OPD | No source-verified strict black-box/API seed exists in the repo. | Evaluate as response-level alignment with judge robustness and margin calibration. |
| Privileged self-OPD leakage | Privileged context can leak answers or visual annotations. | Require no-privilege, full-privilege, wrong-privilege, and delayed-privilege ablations. |
| Long-context OPD | Long-context suites test retrieval/context length more than self-generated recovery. | Add corrupted-prefix and position-conditioned rerollout protocols. |
| Visual teacher mismatch | Text teachers may confidently supervise visually wrong student states. | Pair VLM reasoning with OCR/document/grounding/hallucination slices and teacher-disagreement filters. |
| Video temporal grounding | Existing seed table had generic video benchmarks only. | Use temporal-grounding rows and report boundary errors, not just QA accuracy. |
| Speech/cross-modal OPD | Speech OPD is new and sparse. | Track ASR shortcut controls and text-teacher upper bounds. |
| VLA/action OPD | Success rate can hide unsafe or brittle trajectories. | Add action KL, recovery after perturbed prefix, safety violations, and expert-query budget. |
| Diffusion/flow OPD | Image quality rewards do not prove distillation. | Require teacher field/trajectory fidelity plus diversity and reward-overoptimization checks. |
| Private/live/rolling benchmarks | They reduce contamination but hurt reproducibility. | Always record split, date, evaluator version, judge model, prompt template, seed, and privacy status. |

## Next Step

Superseded by [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md). The post-WP8 verifier pass has been run and merged; future large strict/borderline batches should receive the same WP9 treatment before table changes are merged.
