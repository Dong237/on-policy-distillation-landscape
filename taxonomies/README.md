# Taxonomies

This directory defines how entries are classified in On Policy Distillation Landscape.

The primary taxonomy follows the survey backbone:

1. **Feedback signal:** what kind of feedback supervises student on-policy trajectories.
2. **Teacher access:** what access the student has to teacher or feedback source.
3. **Loss granularity:** whether supervision is token-level, sequence-level, or hybrid/adaptive.

Practical axes such as rollout source, rollout freshness, teacher kind, training stage, modality, and adjacent-work boundaries are secondary in the survey but required for implementation and verification.

## Primary Axes

- [Feedback signal](by-feedback-signal.md)
- [Teacher access](by-teacher-access.md)
- [Loss granularity](by-loss-granularity.md)

## Practical Axes

- [Strict OPD definition](strict-opd-definition.md)
- [Rollout source](by-rollout-source.md)
- [Training stage](by-training-stage.md)
- [Modality](by-modality.md)
- [Loss objective](by-loss-objective.md)
- [Adjacent work](adjacent-work.md)

## Default Classification Posture

When evidence is incomplete, classify conservatively. A method should not be upgraded to `strict_opd` because it mentions "online", "RL", "distillation", or "teacher" in isolation. The evidence must connect those pieces into the same training loop: student rollouts, teacher-style supervision, and objective using that supervision.
