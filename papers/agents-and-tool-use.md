# Agents and Tool Use

This page tracks OPD and adjacent work for multi-step agents, GUI tasks, browser/web tasks, tool use, and code agents.

## Inclusion criteria

- Agent policy generates multi-step trajectories during training.
- Teacher, expert, debate, discriminator, or reference feedback is applied to those trajectories.

## Exclusion criteria

- Offline trajectory tuning from synthetic or teacher-generated traces.
- Reward-only agent RL without distillation signal.
- Evaluation-only agent reflection.

## Seed note

MAD-OPD is `strict_opd` after line-audit evidence for multi-teacher token supervision on student on-policy states. OPCD is also strict for context distillation, while OEL remains partial as a whole loop with a strict consolidation substage. VLM/GUI/tool-use OPD remains mostly open; many current systems are adjacent RL or offline trajectory tuning.

SOD is strict as a tool-integrated reasoning hybrid, VLA-OPD is strict action-token OPD for robotics, GUI-SD is strict only for GUI grounding, and LiteGUI is strict for its guided OPD/GKD GUI stage. TCOD, Skill-SD, and OpenClaw-RL OPD component remain borderline, RLSD is partial, and OEC is adjacent because it trains on expert-generated continuations with rejection-sampled SFT/NLL. SCoRe, WebRL, ReTool, Search-R1, Agent-R1, Reflexion, DGPO, RISE, T3-Agent, Visual Program Distillation, SWE-agent, and SAD stay outside strict OPD. GLM-5 cross-stage OPD remains held out pending section-level primary-source evidence.
