# Agentic and Interactive OPD

Agentic OPD covers multi-turn, stateful environments: code agents, browser agents, GUI agents, tool-use agents, database/API agents, and long-horizon interactive systems.

## Current Status

- MAD-OPD / OPAD is strict: debate teachers supervise student on-policy actions or tokens with confidence-weighted JSD/RKL.
- SOD is strict for its step-wise OPD teacher-logit term on tool/reasoning trajectories, with GRPO as an auxiliary reward objective.
- OPCD is strict privileged-context self-distillation and acts as the core primitive behind several agentic variants.
- OEL remains partial at method level because only the OPCD-style consolidation substage is strict.
- VLA-OPD is strict action-token OPD for robotic/VLA trajectories.
- GUI-SD is strict for GUI grounding, but not for long-horizon browser or GUI control.
- LiteGUI is strict only for its guided OPD/GKD GUI stage; the later GRPO stage is separable.
- TCOD, Skill-SD, and OpenClaw-RL OPD component are tracked as `borderline_strict` because each has a real same-rollout supervision structure but method-level caveats.
- OEC is adjacent after the latest verifier pass: it starts from student-reached states but trains on expert-generated continuations with rejection-sampled SFT/NLL.
- RLSD is `partial_opd`: privileged self-distillation affects token-level update magnitude, while RLVR reward determines direction.

## Boundary examples

- SCoRe-style pure RL is adjacent unless the source shows teacher-style supervision consumed by the update.
- Browser/search/tool/code agent RL such as WebRL, Search-R1, ReTool, Agent-R1, and VTool-R1 is adjacent when the signal is outcome reward, ORM reward, or pass/fail tests.
- Reflexion-style verbal memory with no weight update is not OPD.
- DGPO, RISE, T3-Agent, Visual Program Distillation, SAD, and other synthetic, teacher-generated, reward-weighted, or PPO-guided trajectory tuning are adjacent/offline unless the trained student receives direct distillation supervision on its current trajectory.

## Held Out

GLM-5 cross-stage OPD is not yet merged. Public evidence is plausible but too terse; promote only after a primary-source section shows teacher specification, rollout freshness, and the loss form.

## Key issues

- Delayed credit assignment.
- Non-stationary environments.
- Tool-call schema supervision.
- Safety constraints during on-policy exploration.
