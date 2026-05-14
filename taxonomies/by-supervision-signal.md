# Taxonomy by Supervision Signal

The supervision signal describes what the student receives on its own generated states.

| Signal | Description | OPD status |
|---|---|---|
| `token_kl` | Token-level KL from teacher to student. | Strong OPD evidence. |
| `reverse_kl` | Reverse KL or mode-seeking KL on student samples. | Strong OPD evidence. |
| `sequence_feedback` | Sequence-level teacher or expert feedback. | Can be OPD if applied online. |
| `discriminator_feedback` | Adversarial or discriminator signal against target distribution. | Borderline or strict. |
| `privileged_context` | Same model or teacher sees extra context unavailable to student. | Strict if on-policy. |
| `black_box_response_adversarial` | Response-level black-box OPD alignment. | Borderline strict. |
| `hybrid_kl_reward` | Teacher KL plus RL/RLVR reward. | Strict if KL is on student rollouts. |
| `action_token_supervision` | Expert action tokens supervise student trajectories. | Strict for VLA/robotics if on-policy. |
| `reward_only` | Verifiable reward, environment reward, or outcome score only. | Adjacent, not strict OPD. |
| `preference_only` | Static or online preference without distillation target. | Adjacent or partial. |
| `offline_kd` | Teacher signal on fixed dataset. | Not OPD. |
| `synthetic_data` | Teacher generates data before training. | Not OPD by itself. |

## Rule

`strict_opd` rows must not use `reward_only`, `preference_only`, `offline_kd`, `synthetic_data`, or `none` as their supervision signal.

