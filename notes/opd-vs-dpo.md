# OPD vs DPO

DPO usually optimizes a model on static preference pairs. It is not OPD unless the preference data is produced from current student rollouts and used as feedback in the training loop.

## Classification rule

- Static preference optimization: `adjacent`.
- Offline KD plus DPO: `partial_opd` only if an on-policy feedback stage exists; otherwise `not_opd` or `adjacent`.
- Online preference feedback on student rollouts: `partial_opd` unless the signal is explicitly distillation-style.

