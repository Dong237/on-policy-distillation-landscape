# OPD vs RLHF

RLHF optimizes a policy using human or learned preference rewards. It can involve on-policy rollouts, but reward optimization alone is not distillation.

## Classification rule

- RLHF with reward-only feedback: `adjacent`.
- RLHF plus teacher KL on student rollouts: possible `strict_opd` or `partial_opd`, depending on evidence.
- Industrial reports that mention RLHF but not teacher-on-rollout supervision: `adjacent` or `unclear`.

