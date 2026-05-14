# OPD + RL / RLVR Hybrids

Hybrid methods combine a distillation loss with reinforcement learning or verifiable rewards.

## Inclusion criteria

- OPD loss is computed on student-generated states.
- RL/RLVR reward is part of the same training stage or pipeline.
- The paper reports both signals clearly enough to classify.

## Exclusion criteria

- Reward-only GRPO/RLVR.
- SFT followed by RLVR with no teacher-on-rollout feedback.

## Seed entries

### VOLD

- **Link:** https://arxiv.org/abs/2510.23497
- **Modality:** VLM
- **Signal:** on-policy teacher guidance plus GRPO
- **Strictness:** strict_opd

### Video-OPD

- **Link:** https://arxiv.org/abs/2602.02994
- **Modality:** video-language
- **Signal:** dense teacher supervision plus reverse KL on current-policy TVG trajectories
- **Strictness:** strict_opd

