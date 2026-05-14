# VLM / MLLM / Multimodal OPD

This page is intentionally conservative. Most VLM post-training papers are currently RLVR, RLHF, preference optimization, or offline KD, not strict OPD.

## Classification rule

A VLM/MLLM method is `strict_opd` only if the VLM/MLLM student generates its own multimodal trajectories during training and receives distillation-style teacher, expert, discriminator, privileged-context, or reference supervision on those trajectories.

Reward-only GRPO/RLVR is `adjacent`. Offline MLLM KD is `not_opd`.

## Strict and near-strict seed entries

### VOLD: Reasoning Transfer from LLMs to VLMs via On-Policy Distillation

- **Year:** 2025
- **Link:** https://arxiv.org/abs/2510.23497
- **OpenReview:** https://openreview.net/forum?id=lkv7sOGtfk
- **Modality:** VLM / image-text reasoning
- **Teacher access:** text-only teacher
- **Rollout source:** student_on_policy
- **Signal:** hybrid_kl_reward
- **Strictness:** strict_opd
- **Evidence:** The paper combines GRPO with on-policy distillation so student reasoning traces are guided by a teacher during online training.

### Video-OPD: Efficient Post-Training of MLLMs for Temporal Video Grounding

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2602.02994
- **Modality:** video-language
- **Teacher access:** frontier teacher
- **Rollout source:** student_on_policy
- **Signal:** reverse_kl
- **Strictness:** strict_opd
- **Evidence:** Current-policy video grounding trajectories receive dense token-level teacher supervision.

### Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2605.03677
- **Modality:** LLM + MLLM
- **Teacher access:** stronger LLM/MLLM, single or multi-teacher
- **Rollout source:** student_on_policy
- **Signal:** token_kl
- **Strictness:** strict_opd
- **Evidence:** Line audit found current-policy student trajectories, teacher token guidance, and reverse-KL / margin-calibrated objective evidence.

### PRISM / Beyond SFT-to-RL

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2604.28123
- **Project:** https://xiao4579.github.io/PRISM/
- **Modality:** MLLM
- **Teacher access:** black-box response-level discriminator
- **Rollout source:** student_on_policy
- **Signal:** response-level discriminator reward
- **Strictness:** adjacent
- **Evidence:** The latest verifier confirms current-policy response rollouts and MoE discriminator supervision, but the policy consumes scalar reward/advantage feedback rather than token/logit distillation.

## VLA / embodied extension

### VLA-OPD

- **Year:** 2026
- **Link:** https://arxiv.org/abs/2603.26666
- **Modality:** VLA / robotics
- **Rollout source:** student_on_policy
- **Signal:** action_token_supervision
- **Strictness:** strict_opd
- **Evidence:** Line audit confirms student VLA rollouts, expert teacher action logits on the same visited states, and reverse-KL/action-token objective.

VLA methods are tracked separately from ordinary image-text VLM reasoning because the action space, environment state, and evaluation protocol differ.

## Adjacent VLM RLVR

The following families should not be marked strict unless new evidence shows teacher/discriminator/reference supervision on student rollouts:

- VLM-R1
- R1-VL / StepGRPO
- Vision-R1
- OpenVLThinker / OpenVLThinkerV2
- Perception-R1
- VLAA-Thinker / SFT or RL?
- DeepVideo-R1
- VTool-R1
- ManipLVM-R1
- LaViPlan

## Not OPD VLM / MLLM KD

- LLaVA-KD: offline MLLM KD.
- LLAVADI: offline KD study.
- LLaVA-MoD: progressive KD plus preference optimization; not strict without on-policy teacher supervision evidence.
- Visual Program Distillation: offline program/tool distillation.
- Mini-InternVL: offline vision encoder distillation.

## Industrial reports

Qwen2.5-VL, Qwen3-VL, InternVL3/3.5, Gemma 3, Gemini 2.5, MiniCPM-V, and LLaVA-RLHF should be tracked as industrial or adjacent systems unless their public reports explicitly connect teacher feedback to student on-policy trajectories.
