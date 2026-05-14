# Frameworks

Framework rows should be tracked by **module or recipe**, not only by repository.

For example, TRL VLM GRPO and TRL GKD are separate rows because one supports VLM reward-only training while the other supports text GKD-style distillation.

## Current framework picture

| Framework | Module / recipe | VLM support | OPD support | Classification note |
|---|---|---|---|---|
| TRL | GRPOTrainer VLM recipe | yes | no/unclear | VLM GRPO support, but reward-only unless custom teacher signal is added. |
| TRL | GKDTrainer | no official VLM example | yes for language models | Useful foundation for text OPD. |
| verl | VLM RL recipes | yes | no by default | Distributed RL framework; no official strict OPD teacher-logit loop found in the systems audit. |
| OpenRLHF | RLHF/RLVR/KD stack | yes | no by default | Supports RLHF/RLVR and KD infrastructure, but no audited same-rollout strict OPD recipe. |
| NeMo RL | OPD example + VLM GRPO | yes | yes for LLM OPD example | VLM OPD status remains unclear. |
| Tinker / Thinking Machines | OPD recipe | no official VLM example | yes for language models | Operational sample-then-teacher-logprob reference recipe. |
| KDFlow | Cross-tokenizer KD serving | unclear | configurable | Systems framework; strictness depends on configured current-rollout teacher scoring. |
| VLMEvalKit | Evaluation suite | yes | not training | Evaluation harness only. |

See `tables/frameworks.md`.
