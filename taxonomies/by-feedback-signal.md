# Taxonomy by Feedback Signal

Feedback signal is the first survey axis. It describes what kind of information supervises the student on its own generated trajectories.

| Route | Meaning | Typical label |
|---|---|---|
| `logit_based` | Teacher probabilities, logits, or log-probs supervise student samples. | Often `strict_opd` when on-policy. |
| `outcome_based` | Scalar reward, verifier score, or preference over student samples. | Adjacent unless converted into teacher-style supervision consumed by the loss. |
| `self_play` | Student improves using self-play, previous checkpoints, or contrastive self-generated signals. | Strict or partial depending on the loop. |
| `hybrid_logit_reward` | Dense teacher feedback plus sparse reward/RL objective. | Strict if teacher feedback is on student rollouts and the objective consumes it. |
| `adversarial_black_box` | Discriminator or teacher-as-judge feedback aligns student outputs with teacher outputs. | Usually borderline or partial until the objective is shown to consume non-scalar supervision. |
| `preference_based` | Preferred/dispreferred samples guide learning. | Partial or adjacent unless online and distillation-like. |
| `action_level` | Expert or teacher supervises student actions or action tokens. | Strict for VLA/agents if on-policy. |
| `offline_kd` | Teacher feedback on fixed data. | Not OPD. |

## Rule

Reward-only training can be on-policy without being OPD. Record it as `outcome_based` and usually `adjacent` unless the source explicitly makes it a distillation target.

## Outcome-feedback split

Survey triage found that many false positives come from collapsing all outcome feedback into OPD. Use this split when classifying rows:

| Feedback subtype | OPD status | Examples |
|---|---|---|
| Teacher-derived dense reward or logit-derived reward | Can be OPD | G-OPD-style or REOPOLD-style methods where teacher distributions become dense supervision. |
| Verbal, adversarial, or discriminator teacher feedback | Partial, borderline, or adjacent by default | GAD/PRISM-style methods after objective-level audit; do not mark strict from response-level scores or scalar policy-gradient rewards alone. |
| Pure verifier or rule reward | Adjacent | GRPO/RLVR systems such as VLM-R1 unless a teacher-style distillation loss is present. |
| Static preference pairs | Adjacent | DPO/ORPO/SimPO-style training unless pairs are generated and supervised online. |
| Offline teacher traces | Not OPD | SFT or SeqKD on teacher-generated CoT. |

## Boundary Rule

If the feedback is response-level, verbal-score, or discriminator-only, treat it as `partial_opd` or `borderline_strict` unless the primary source shows a dense teacher-style objective on the same student rollout. A method can still be valuable for black-box OPD, but strict OPD requires more than an on-policy reward signal.

## White-Box Rule

Do not downgrade a method merely because it writes the white-box objective in reinforcement-learning notation. If the signal is a dense teacher logit/log-prob ratio on the student's own sampled tokens, it is still `logit_based` or `hybrid_logit_reward`, not reward-only RLVR.

Conversely, do not upgrade an objective-only paper to OPD just because its loss could be used inside an OPD loop. ToDi, CSD, ULD, sparse-logit KD, and optimal-transport KD remain adjacent until the source shows current student rollouts plus teacher supervision on those states.
