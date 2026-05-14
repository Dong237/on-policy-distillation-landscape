# Long-CoT Drift

Long chain-of-thought generation increases the gap between offline teacher traces and student inference-time behavior.

## OPD relevance

OPD directly targets this gap by training on student-generated trajectories. However, long rollouts also increase teacher scoring cost and compound early mistakes.

## Track

- Rollout length.
- Whether feedback is token-level, step-level, or sequence-level.
- Whether the method reweights hard or informative states.
- Whether incorrect trajectories receive useful supervision or are discarded.

