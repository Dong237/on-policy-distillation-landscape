# WP7 Candidate Line Audit Synthesis

Checked: 2026-05-13

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) supersedes the PRISM label and records it as `adjacent`.

This note merges:

- `assets/research/wp7-candidate-line-audit-chatgpt.md`
- `assets/research/wp7-candidate-line-audit-claude.md`
- `assets/research/wp7-candidate-line-audit-gemini.md`

## Merge Decision

The WP7 candidate line audit resolved the multimodal borderline queue conservatively.

| Method | Repo action | Reason |
|---|---|---|
| KEPO | Add `borderline_strict` to `tables/opd_papers.md` and `tables/vlm_opd_papers.md`. | Student medical VQA trajectories receive a quality-gated teacher-divergence term, but the exact divergence/logit mechanics are referenced rather than fully defined and only reward-passing trajectories receive distillation. |
| D-OPSD | Add `borderline_strict` to main and VLM tables. | Student denoising rollouts are scored by an EMA privileged self-teacher with target-image context, but the consumed signal is velocity-field MSE rather than a proven reverse-KL objective. |
| Flow-OPD | Add `borderline_strict` to main and VLM tables. | Student flow trajectories receive multi-teacher velocity supervision with a local KL-to-L2 derivation, but the update consumes the signal through PPO-style dense reward plus L2 regression. |
| GTR-Turbo | Add adjacent row. | The KL variant averages reverse KL into a scalar auxiliary PPO reward, while the SFT variant uses teacher-generated thoughts. |

## Metadata Updates

| Method | Update |
|---|---|
| Video-OPD | Teacher pinned to Qwen3-VL-32B-GRPO and objective described as reverse-KL-equivalent token reward. |
| X-OPD | Teacher/student pinned to Qwen3-A3B-Instruct and Qwen3-Omni-A3B-Instruct; source-verified token log-probability supervision. |
| Uni-OPD | Margin mask and margin shift metadata recorded in the objective notes. |
| PRISM | Superseded by post-WP8 WP9: now `adjacent`; response-level MoE discriminator reward is not strict token/logit OPD. |
| Qwen3-VL | Adjacent/industrial metadata updated: text-only OPD for the LLM backbone is confirmed, but visual-language OPD is not disclosed. |

## Boundary Notes

- Quality-gated OPD is not full strict method-level OPD when only reward-passing trajectories receive the teacher-divergence term.
- Diffusion and flow velocity-field supervision is a real same-rollout distillation boundary, but not clean token/logit OPD. Keep `borderline_strict` until the repo adopts a dedicated diffusion/flow OPD taxonomy.
- KL-derived scalar PPO rewards are adjacent or borderline at most unless the objective directly optimizes a distillation loss on the same rollout.
- Text-only OPD inside a multimodal model report does not prove multimodal OPD.

## Next Router State

WP7 broad and candidate passes are complete. Proceed to WP8 evaluation methodology: design modality-aware evaluation bundles that compare strict OPD against adjacent RLVR/offline KD baselines without mixing labels.
