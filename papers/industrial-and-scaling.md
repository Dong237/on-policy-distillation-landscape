# Industrial and Scaling

This page tracks industrial OPD evidence, scaling behavior, and systems optimizations.

## Current seed evidence

- Qwen3 has a documented OPD stage for lightweight LLMs; do not label the whole training pipeline as OPD. The stage is strict, with medium confidence because the public report is terse on exact loss and rollout refresh.
- Nemotron-Cascade 2 has documented multi-domain OPD inside a broader cascade RL system; sampled-token distillation advantage and domain teachers support a strict stage label.
- MiMo-V2-Flash Stage 3 MOPD is source-verified strict OPD; do not label the whole pipeline strict.
- DistillSpec is tracked as systems-oriented strict OPD for speculative-decoding draft alignment in the draft-generated configuration.
- Fast OPD is strict over current student-generated prefixes and is best treated as prefix-scoring systems evidence.

## Below Strict

- Gemma 2 post-training remains borderline: primary evidence says teacher distillation on the student's distribution, but rollout and objective details are terse.
- Speculative KD is partial because rejected student tokens are replaced by teacher tokens.
- Lightning OPD is partial because teacher log-probs are cached over reference/SFT rollouts.
- Gemma 3, DeepSeek-R1 distilled checkpoints, Minitron-style pruning KD, InternVL3/3.5, Qwen-VL/Qwen3-VL, MiniCPM-V, Gemini 2.5, and LLaVA-RLHF remain adjacent, unclear, or not OPD without direct C1/C2/C3 evidence.

## Frameworks

- TRL GKDTrainer and NeMo RL provide strict-capable LLM OPD paths when configured with current student generations and teacher-logit supervision.
- verl and OpenRLHF are strong RL/RLVR infrastructure, but their official default recipes do not establish strict OPD.
- Tinker / Thinking Machines is useful as an operational OPD recipe and reference implementation, not as a normal paper row.
- KDFlow is systems evidence for cross-tokenizer teacher serving; strictness depends on the configured recipe.

## Boundary

Industrial reports often mention distillation, online RL, and synthetic data separately. Do not classify them as strict OPD unless the report connects student on-policy trajectories to teacher-style supervision and the objective.

See `tables/industrial_reports.md`. Audit provenance is archived under [assets/research](../assets/research/).
