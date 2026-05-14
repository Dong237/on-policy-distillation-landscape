# Taxonomy by Loss Granularity

Loss granularity is the third survey axis. It describes where the feedback is applied.

| Granularity | Description | Main tradeoff |
|---|---|---|
| `token` | Feedback at individual token/action-token positions. | Dense and stable, but can be biased for sequence-level goals. |
| `sequence` | Feedback over complete generated trajectories. | Better matches global quality, but higher variance. |
| `hybrid` | Combines token-level and sequence/outcome feedback. | Useful for OPD+RL and long-CoT reasoning. |
| `token_or_sequence` | Method can operate at either level or paper is ambiguous. | Needs source-specific audit. |
| `action` | Feedback over action tokens or state-action pairs. | Relevant for VLA and agents. |
| `outcome` | Sparse final score or preference. | Usually adjacent unless combined with distillation. |
| `relation` | Relation/feature/affinity-level supervision. | Usually offline KD unless on-policy states are used. |

## Bias-Variance Note

Token-level OPD often has lower variance but can overfit local token choices. Sequence-level OPD better reflects full trajectory quality but can be noisy. Hybrid and adaptive objectives are a major open direction.

