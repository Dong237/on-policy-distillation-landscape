# Contributing to On Policy Distillation Landscape

Thank you for contributing.

This repository is taxonomy-first. Please do not add papers as plain links without classification and source evidence.

## Before submitting

Classify the work as one of:

- `strict_opd`
- `borderline_strict`
- `partial_opd`
- `adjacent`
- `not_opd`
- `unclear`

## Strict OPD criteria

A method is `strict_opd` only if:

1. **C1:** The current student generates trajectories, completions, rationales, actions, or intermediate states during training.
2. **C2:** Teacher, expert, discriminator, privileged-context, reference-model, previous-checkpoint, or equivalent feedback is applied to those exact student-generated trajectories.
3. **C3:** The training objective consumes that supervision signal.

If a method uses reward-only GRPO/RLVR without teacher/discriminator/reference supervision on student rollouts, mark it `adjacent`.

## Required fields

Every paper or system contribution should include:

- Title
- Link
- Year
- Code link, if available
- Modality
- Feedback route
- Teacher access regime
- Loss granularity
- Divergence or objective
- On-policy strength
- Rollout source
- Rollout freshness
- Teacher kind
- Strictness evidence using `C1 ...; C2 ...; C3 ...` for strict or borderline rows
- Conflict status if agents disagree or verifier is pending
- Training stage
- Strictness judgment
- Evidence URL
- Compute-cost notes, if relevant
- One-sentence classification rationale
- Limitations or uncertainty

## Placement guide

- Use `papers/foundations-and-theory.md` for theory, survey, exposure-bias, and f-divergence foundations.
- Use `papers/white-box-distributional-opd.md` for logit/log-prob teacher methods.
- Use `papers/black-box-and-api-opd.md` for response-only, API, verbal, preference, or discriminator methods.
- Use `papers/self-play-and-teacher-free-opd.md` for self-play, previous-checkpoint, or privileged-context methods.
- Use `papers/reasoning-and-opd-rl-hybrids.md` for long-CoT, math, code, and KD+RL methods.
- Use `papers/industrial-and-scaling.md` for official model reports and systems/scaling methods.
- Use `papers/agentic-and-interactive-opd.md` for agentic, tool-use, and environment-interaction methods.
- Use `papers/multimodal-frontiers.md` for VLM/MLLM/VLA/speech/video frontier methods.
- Use `papers/adjacent-but-not-opd.md` for relevant non-OPD work.
- Use `tables/industrial_reports.md` for technical reports where training evidence is not enough for strict OPD.

## Avoid

Please avoid adding:

- Plain SFT papers without on-policy student rollouts.
- Generic RLHF/RLVR papers without distillation-style feedback.
- Offline KD papers without student-generated training states.
- Blog posts without technical substance.
- Strict OPD claims based only on "online RL" or "distillation" appearing separately.

## Validation

Run this before submitting:

```bash
python3 scripts/validate_entries.py
```

The validator rejects malformed Markdown table headers, wrong column counts, invalid enum values, missing evidence URLs, strict/borderline rows without C1/C2/C3 evidence, and strict OPD rows that do not satisfy the repository strictness rule.
