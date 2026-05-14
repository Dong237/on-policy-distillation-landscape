# On-Policy Self-Distillation

On-policy self-distillation uses the same model, a previous checkpoint, or privileged context to supervise the current policy on its own generated states.

## Inclusion criteria

- The current student generates the training trajectory.
- The feedback source is the same model under privileged context, a previous checkpoint, or an internal teacher view.
- The update is distillation-style rather than reward-only.

## Exclusion criteria

- Self-training on filtered generated data only.
- Self-rewarding RL without a distillation target.
- Test-time self-reflection without training updates.

## TODO

- Add verified OPSD/self-distillation entries after source audit.

