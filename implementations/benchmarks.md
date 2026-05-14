# Benchmarks

Benchmarks are tracked in `tables/benchmarks.md`.

## VLM benchmark groups

- Math and visual reasoning: MathVista, MATH-Vision, MMMU-Pro, LogicVista.
- Perception, OCR, documents, and hallucination: OCRBench, DocVQA, ChartQA, InfoVQA, AI2D, MathVerse, HallusionBench, POPE.
- GUI grounding and agents: ScreenSpot, ScreenSpot-Pro, OSWorld, WebArena, Mind2Web.
- Video: SEED-Bench, ActivityNet Captions, QVHighlights, Charades-STA, VideoMME.
- Speech/audio: AudioBench, BigBenchAudio, AIR-Bench, Dynamic-SUPERB.
- Robotics/VLA: LIBERO, RoboTwin2.0, CALVIN, RoboCasa, SimplerEnv.
- Diffusion/flow image generation: GenEval, T2I-CompBench, HEIM, HPSv2.
- Text, code, and agentic controls: LiveBench, LiveCodeBench, IFEval, GPQA, MATH-500, SWE-bench Verified, AppWorld, ALFWorld, WebShop, ScienceWorld, tau-bench, Berkeley Function Calling Leaderboard.

## Evaluation rule

Record benchmark metrics, split protocol, evaluator version, and date. Do not compare papers across private-test and public-test results without noting the split.

Benchmark accuracy alone is not OPD evidence. Pair capability scores with rollout/recovery diagnostics, calibration or distributional-fidelity metrics, and matched adjacent baselines such as offline KD, SFT, reward-only RLVR, DPO/RLHF, behavior cloning, or diffusion reward tuning as appropriate.
