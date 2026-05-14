# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Created the taxonomy-first repository scaffold.
- Added strict OPD, borderline, partial, adjacent, and not-OPD classification rules.
- Added seed Markdown table schemas and initial evidence-backed entries.
- Added validation scripts for Markdown table headers, enum values, URLs, and strict OPD constraints.
- Added VLM/MLLM classification guidance that separates strict OPD from RLVR and offline KD.
- Reorganized the repository around the Tencent OPD survey backbone: feedback signal, teacher access, and loss granularity.
- Added survey-derived table fields for feedback route, teacher access regime, loss granularity, divergence/objective, on-policy strength, and compute-cost notes.
- Added method-family paper pages for foundations, white-box OPD, black-box/API OPD, teacher-free OPD, reasoning hybrids, industrial scaling, agentic OPD, and multimodal frontiers.
- Expanded white-box distributional OPD coverage, including strict rows for vOPD, TIP, AOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, and Rock Tokens; added partial rows for StableOPD, DSKD v2, HPD, PAD, ADPA, and daDPO.
- Expanded black-box/API coverage and kept the bucket conservative: GAD, OVD, SODA, and ORPO-Distill are partial; PRISM is adjacent.
- Expanded teacher-free/self-play coverage with strict SDFT 2026 plus source-verified OPSD, CRISP/OPSDC, SDPO, OPCD, and OPSDL labels.
- Expanded reasoning and OPD+RL coverage while keeping reward-only or scalarized teacher-signal methods below strict.
- Added industrial and systems rows for Qwen3, Nemotron-Cascade2, MiMo-V2-Flash, DistillSpec, Fast OPD, frameworks, and serving-cost patterns.
- Expanded agentic and interactive coverage with SOD, VLA-OPD, GUI-SD, LiteGUI, TCOD, Skill-SD, RLSD, DGPO, RISE, OEC, and related boundary rows.
- Expanded multimodal frontier coverage with VOLD, Video-OPD, X-OPD, Uni-OPD, VLA-OPD, GUI-SD, LiteGUI, KEPO, D-OPSD, Flow-OPD, GTR-Turbo, and industrial VLM false positives.
- Expanded benchmark coverage across text, VLM, GUI/agent, video, audio, robotics, and diffusion/flow evaluation, with OPD-specific diagnostic requirements.
- Finalized verifier cleanup: PRISM and OEC are adjacent; Qwen3 OPD remains stage-scoped, source-weak, and medium confidence; LiteGUI, VOLD, MiMo-V2-Flash, and Nemotron-Cascade2 keep stage/component caveats.
- Moved historical research prompts, synthesis memos, and controller handoff files into `assets/research/` so the root repo reads as the final landscape.
