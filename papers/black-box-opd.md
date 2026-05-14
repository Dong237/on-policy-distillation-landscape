# Black-Box OPD

Black-box OPD uses teacher responses, discriminators, or response-level feedback without direct teacher logits.

This page is a compact method note. The canonical table rows live in [tables/opd_papers.md](../tables/opd_papers.md), and the black-box audit provenance is archived under [assets/research](../assets/research/).

## Inclusion criteria

- Student generates on-policy samples.
- A black-box teacher, discriminator, or response-level critic gives corrective feedback on those samples.
- The training objective is explicitly distillation or distribution alignment, not reward-only RL.

## Borderline handling

Black-box response-level OPD should usually be marked `partial_opd`, `borderline_strict`, or `adjacent` unless the paper gives strong evidence that the feedback is a non-scalar distillation target on exact student rollouts. If the policy consumes the signal only as scalar reward or policy-gradient advantage, use `adjacent`.

## Current entries

### PRISM / Beyond SFT-to-RL

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2604.28123
- **Project:** https://xiao4579.github.io/PRISM/
- **Modality:** MLLM
- **Rollout source:** student_on_policy
- **Signal:** black-box response-level discriminator reward
- **Strictness:** adjacent
- **Why it matters:** Inserts OPD-inspired distribution alignment between SFT and RLVR for Qwen3-VL, but the latest verifier pass downgrades it because the discriminator score is consumed as scalar reward/advantage rather than a distillation target.

### GAD

- **Year:** 2025
- **Link:** https://arxiv.org/abs/2511.10643
- **Modality:** LLM
- **Rollout source:** student_on_policy
- **Signal:** discriminator feedback
- **Strictness:** partial_opd
- **Why it matters:** A close black-box adversarial candidate, but it remains below strict because the discriminator signal is consumed as response-level GRPO reward.

### OVD

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2601.21968
- **Modality:** LLM
- **Rollout source:** student_on_policy
- **Signal:** verbal/API teacher scores
- **Strictness:** partial_opd
- **Why it matters:** Useful for black-box teacher feedback, but ordinal/verbal scoring is not dense distillation supervision by default.

### SODA

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2604.03873
- **Modality:** LLM
- **Rollout source:** mixed_policy
- **Signal:** preference supervision
- **Strictness:** partial_opd
- **Why it matters:** Semi-on-policy black-box preference distillation; needs line audit for rollout freshness.

### ORPO-Distill

- **Year:** 2025
- **Link:** https://arxiv.org/abs/2509.25100
- **Modality:** LLM
- **Rollout source:** mixed_policy
- **Signal:** teacher-positive/student-negative ORPO preference pair
- **Strictness:** partial_opd
- **Why it matters:** A practical cross-architecture black-box distillation route, but preference optimization is not strict OPD.

## Black-Box Takeaway

The black-box audit did not find a strict black-box/API OPD seed. Future promotion requires line-level evidence that a method uses non-scalar teacher-style supervision on exact current student rollouts.
