# OPD Failure Modes

## Common failure modes

- Teacher is unreliable on student-visited states.
- Teacher and student have different reasoning styles.
- Teacher uncertainty is not calibrated.
- Student explores uninformative or impossible states.
- Long-CoT rollouts drift away from useful reasoning.
- OPD loss conflicts with RLVR reward.
- Online teacher scoring dominates compute cost.

## VLM-specific failure modes

- Perception error is mistaken for reasoning error.
- Text-only teacher gives confident but visually ungrounded feedback.
- Visual grounding and language reasoning require different teachers.
- Video and GUI tasks compound state mismatch across steps.

