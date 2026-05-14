# Foundations

Foundational OPD papers define the training pattern used throughout this repository.

## Inclusion criteria

- Student generates training outputs or states.
- Teacher or stronger model supervises those student-generated outputs.
- The work introduces a reusable objective, algorithm, or theoretical framing.

## Seed entries

### On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

- **Year:** 2023 / ICLR 2024
- **Link:** https://arxiv.org/abs/2306.13649
- **Family:** Generalized Knowledge Distillation (GKD)
- **Rollout source:** student_on_policy
- **Signal:** teacher feedback on self-generated outputs
- **Strictness:** strict_opd
- **Why it matters:** Establishes the core exposure-bias argument for OPD in autoregressive language models.

### MiniLLM: On-Policy Distillation of Large Language Models

- **Year:** 2023 / ICLR 2024
- **Link:** https://arxiv.org/abs/2306.08543
- **Family:** Reverse-KL OPD
- **Rollout source:** student_on_policy
- **Signal:** reverse KL from teacher to student on on-policy generation
- **Strictness:** strict_opd
- **Why it matters:** Popularizes reverse-KL OPD for compressing LLMs.

