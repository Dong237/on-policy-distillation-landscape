# Taxonomy by Rollout Source

Rollout source is the most important field for OPD classification.

| Value | Meaning |
|---|---|
| `student_on_policy` | Current student/policy generates the training states. Required for `strict_opd`. |
| `mixed_policy` | Training states mix student, teacher, replay, or offline data. Usually `partial_opd`. |
| `off_policy` | Training states come from fixed data or another policy. Not strict OPD. |
| `teacher_generated` | Teacher creates data before student training. Not OPD by itself. |
| `not_applicable` | No rollout-based training loop. |
| `unclear` | Source evidence does not specify the rollout source. |

## Rollout freshness

`rollout_source` says who generated the states. `rollout_freshness` says how fresh those states are relative to the student being updated.

| Value | Meaning | Typical strictness impact |
|---|---|---|
| `current_policy` | The currently updated student/policy generates the supervised states. | Required for final `strict_opd` unless a paper uses a more specific fresh value. |
| `per_iteration` | Rollouts are regenerated every iteration or update cycle. | Strong OPD evidence. |
| `per_epoch` | Rollouts are refreshed by epoch or large batch. | Usually strict or borderline depending on staleness. |
| `mixed_freshness` | Mixes fresh student rollouts with replay, teacher data, or static data. | Usually `partial_opd` or `borderline_strict`. |
| `replay_buffer_stale` | Student-generated states are reused after the policy has moved. | Usually not final strict without a clear correction. |
| `previous_policy` | Older checkpoint or previous policy generated the states. | Teacher-free or self-play candidate, often partial. |
| `precomputed_sft_rollouts` | Rollouts are frozen before the OPD update. | Usually `partial_opd`, not strict. |
| `static_dataset` | Fixed corpus, teacher traces, or offline data. | `not_opd` unless paired with separate on-policy stage. |
| `not_applicable` | No rollout loop. | Not strict OPD. |
| `unclear` | Source does not specify freshness. | Needs review. |

## Rule

If `rollout_source` is not `student_on_policy`, the row cannot be `strict_opd`.

If `rollout_freshness` is stale, mixed, precomputed, static, or unclear, do not use final `strict_opd` until a verifier confirms that the objective still supervises current student states.

## Verifier Conflict Handling

When verifiers disagree, prefer the more conservative rollout label:

- cached student-generated outputs: `mixed_policy` plus `replay_buffer_stale`;
- student prefix plus teacher completion: `mixed_policy` plus `mixed_freshness`;
- SFT rollouts scored offline: `off_policy` plus `precomputed_sft_rollouts`;
- privileged teacher trajectories: `teacher_generated` or `mixed_policy` unless the unprivileged student produced the exact supervised tokens;
- deployment or experience traces later consolidated by distillation: `mixed_policy` until the consolidation stage is isolated.
