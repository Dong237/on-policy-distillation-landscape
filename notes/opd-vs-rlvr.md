# OPD vs RLVR

RLVR trains against verifiable rewards, such as answer correctness, program execution, environment success, or grounding metrics.

OPD transfers a teacher, expert, discriminator, privileged-context, or reference signal onto student-generated trajectories.

## Classification rule

- Reward-only GRPO/RLVR: `adjacent`.
- OPD + RLVR hybrid: strict only when the distillation loss is computed on student rollouts.
- Process rewards or verifiers are not enough by themselves; record them as `reward_only` unless the paper frames them as distillation-style supervision.

## VLM caution

Most VLM-R1-style papers belong here as adjacent work. They are valuable baselines for OPD, but they do not define strict OPD unless a teacher distribution or discriminator signal supervises the student rollout.

