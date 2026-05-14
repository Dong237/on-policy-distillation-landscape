# Multimodal Frontiers

Multimodal OPD extends the same definition to visual, video, speech, action, and GUI states. It is a frontier section, not the repo's backbone.

## Strict or near-strict seeds

| Work | Modality | Classification note |
|---|---|---|
| [VOLD](https://arxiv.org/abs/2510.23497) | VLM | Text teacher supervises VLM student rollouts; strict seed. |
| [Video-OPD](https://arxiv.org/abs/2602.02994) | Video-language | Dense teacher supervision on current-policy temporal grounding trajectories. |
| [X-OPD](https://arxiv.org/abs/2603.24596) | Speech / LLM | Cross-modal strict seed with text teacher feedback on speech-model rollouts. |
| [Uni-OPD](https://arxiv.org/abs/2605.03677) | LLM / MLLM | Source-verified strict after line audit. |
| [VLA-OPD](https://arxiv.org/abs/2603.26666) | VLA / robotics | Source-verified strict action-token OPD after line audit. |
| [GUI-SD](https://arxiv.org/abs/2605.00642) | GUI grounding | Source-verified strict with high confidence and a grounding-only caveat. |
| [LiteGUI](https://arxiv.org/abs/2605.07505) | GUI agent | Strict for the guided OPD/GKD stage; later GRPO is separate. |
| [KEPO](https://arxiv.org/abs/2602.00400) | Medical VLM | Borderline quality-gated OPD; teacher divergence is gated and not fully specified. |
| [D-OPSD](https://arxiv.org/abs/2605.05204) | Text-to-image diffusion | Borderline privileged self-distillation on denoising rollouts with velocity-field MSE. |
| [Flow-OPD](https://arxiv.org/abs/2605.08063) | Text-to-image flow matching | Borderline multi-teacher flow OPD; KL-derived velocity reward is consumed through PPO-style optimization. |

## Adjacent Boundary

PRISM and GTR-Turbo are adjacent. PRISM's MoE discriminator score is consumed as scalar policy-gradient reward rather than a distillation loss, while GTR-Turbo's KL variant collapses reverse KL into scalar PPO reward shaping and its SFT variant uses teacher-generated thoughts.

## False positives to map

Reward-only VLM-R1-style RLVR, offline MLLM KD, synthetic visual instruction data, and tool/program distillation should stay adjacent or not OPD unless source evidence shows the full OPD loop.
