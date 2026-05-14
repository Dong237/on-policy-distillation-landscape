# Adjacent but Not OPD

Adjacent work belongs here when it helps understand OPD boundaries.

## Include

- VLM RLVR / GRPO methods with reward-only supervision.
- Offline MLLM KD.
- Preference optimization and RLHF reports.
- Objective innovations that can be plugged into KD but do not themselves show student on-policy rollouts.
- Cross-tokenizer KD methods that align teacher and student distributions only on static or teacher-forced data.
- Program/tool distillation without student on-policy training states.
- Industrial model reports with incomplete OPD evidence.
- DeepSeek-R1 distilled checkpoints and Gemma2 pretraining KD as canonical false positives.

## Rule

If a method has student rollouts but no distillation-style teacher/discriminator/reference supervision, it is adjacent. If it has distillation but no student rollouts, it is not OPD.

See `tables/adjacent_work.md` for verified boundary entries.

The white-box audit added CSD, AMiD, BiLD, RSKD, OKD, EGAD, ToDi, HPD, ULD, MultiLevelOT, DWA-KD, DynSDPB, and byte-level cross-tokenizer KD to the adjacent ledger. These are useful for objective or tokenizer design, but they should not be promoted unless a source shows C1/C2/C3 on student-generated states.

The agentic audit added DGPO and RISE as agentic boundaries: DGPO uses offline cold-start KD plus teacher-guided PPO shaping, while RISE uses reward-weighted regression/SFT on improved responses. Neither is strict OPD under the repo definition.

The multimodal candidate audit added GTR-Turbo as a VLM-agent boundary: reverse KL is computed on student thought tokens but collapsed into scalar PPO reward shaping. Qwen3-VL also stays adjacent for multimodal OPD because its disclosed OPD stage is text-only backbone distillation, while multimodal stages use reward-based RL/SAPO.
