# Industrial Reports

Industrial reports are tracked because they often reveal post-training patterns, but they should not be upgraded to strict OPD without direct evidence.

## Inclusion criteria

- Technical report, model card, or official blog.
- Describes training or post-training for LLMs, VLMs, MLLMs, or VLA systems.
- Helps distinguish OPD from RLHF, RLVR, preference optimization, or offline distillation.

## Current VLM conclusion

No surveyed mainstream VLM/MLLM report currently provides enough public evidence for `strict_opd`. Qwen2.5-VL, Qwen3-VL, InternVL3/3.5, Gemma 3, Gemini 2.5, MiniCPM-V, and LLaVA-RLHF are tracked as `adjacent`, `not_opd`, or `unclear` depending on the disclosed training signal.

## LLM and Systems Conclusion

The strict industrial evidence is stage-specific. Qwen3, Nemotron-Cascade2, MiMo-V2-Flash, DistillSpec, and Fast OPD should be read as strict only for the disclosed OPD/MOPD/draft/prefix stages. Qwen3 remains medium confidence/source-weak because the public report is terse on exact rollout refresh and loss mechanics. Speculative KD and Lightning OPD are useful systems methods but remain partial. DeepSeek-R1 distilled checkpoints and Minitron-style pruning KD are false positives for strict OPD.

See `tables/industrial_reports.md`.
