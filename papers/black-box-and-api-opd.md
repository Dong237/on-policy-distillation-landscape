# Black-Box and API OPD

Black-box OPD covers settings where the teacher does not expose logits or weights. Feedback may come from API responses, rankings, verbal scores, or discriminators.

Current status: after discovery and line audit, this bucket has **no source-verified `strict_opd` seed**. The closest methods are useful, but they usually consume the black-box signal as response-level reward, score, or preference supervision.

## Include

- GAD-style adversarial OPD.
- Teacher-as-referee or curriculum loops only when they supervise student-generated states directly.
- On-policy verbal distillation.
- Preference or ranking methods only when they supervise student on-policy samples.

## Exclude

- Offline teacher-generated datasets.
- Static DPO pairs.
- Reward-only RLVR with no teacher/discriminator/reference supervision.
- SFT on teacher corrections or teacher-generated traces.
- Rejection sampling or Best-of-N pipelines without same-rollout teacher supervision.

## Current Evidence Map

The black-box line audit resolved the primary queue, and the latest verifier later tightened PRISM's label. GAD, PRISM, and OVD do supervise current rollouts, but the consumed signal is response-level reward, advantage, or verbal-score filtering. SODA and ORPO-Distill are preference-distillation neighbors with static or mixed-policy student negatives.

| Method | Current label | Why |
|---|---|---|
| GAD | `partial_opd` | Student responses are on-policy and a discriminator compares them to teacher outputs, but the training signal is response-level GRPO reward rather than dense teacher distribution matching. |
| OVD | `partial_opd` | A black-box teacher gives verbal or ordinal scores on student rollouts; the signal is scalar/filter-like, not token-level OPD. |
| PRISM | `adjacent` | Uses black-box response-level adversarial alignment between SFT and RLVR for MLLMs, but the latest verifier downgrades it because the MoE discriminator score is consumed as scalar reward/advantage rather than a distillation loss. |
| SODA | `partial_opd` | Semi-on-policy black-box preference distillation; rollout freshness fails strict OPD because q0 negatives are a one-time static snapshot. |
| ORPO-Distill | `partial_opd` | Mixed teacher-positive/student-negative ORPO is useful cross-architecture distillation, but the objective is preference optimization rather than same-rollout distribution matching. |

## Adjacent and False Positives

| Method family | Why it is not strict OPD by default |
|---|---|
| Critique or correction distillation | The student output may be used as context, but the target is often teacher-generated corrected text trained by SFT. |
| On-policy DPO or discriminator-guided DPO | Student samples can be fresh, but the consumed signal is a preference label rather than teacher-style supervision. |
| LLM-as-judge RL | Judge feedback is usually a scalar reward consumed by PPO/GRPO. |
| Best-of-N reward-model distillation | The target distribution comes from selection under a reward model, not teacher supervision on exact student rollouts. |

## Secondary Spot-Check Resolved

The secondary spot-check found no strict black-box/API OPD seed. GAKD, ADPA, daDPO, PAD, and CTPD are white-box redirects because they require teacher distributions, logits, or token probabilities. RLKD and GAR are reward-RL neighbors because their teacher or discriminator signal is consumed as scalar or slice-level reward. RLTF, FCP, and ALT are verbal-feedback neighbors, but their objectives compile feedback into self-distilled generations, feedback prediction, or conditional SFT rather than dense teacher-style supervision on the exact rollout.

The white-box redirect spot-check is resolved: PAD, ADPA, and daDPO are partial method-level white-box rows; GAKD and CTPD remain adjacent. Audit provenance is archived under [assets/research](../assets/research/).
