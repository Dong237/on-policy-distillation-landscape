# WP9 Post-WP8 Verifier Synthesis

Checked: 2026-05-13

Inputs:

- `assets/research/wp9-post-wp8-verifier-chatgpt.md`
- `assets/research/wp9-post-wp8-verifier-claude.md`
- `assets/research/wp9-post-wp8-verifier-gemini.md`

This pass rechecked strict and borderline rows changed by WP6, WP7, and WP8. It did not discover new papers or evaluate benchmark quality. The merge rule remains conservative: when verifiers disagree, the table keeps the lower strictness label unless primary-source evidence clearly supports all C1/C2/C3 conditions.

## Merge Decision

| Method | Decision | Reason |
|---|---|---|
| GUI-SD | Keep `strict_opd`. | All verifiers confirm on-policy coordinate tokens, privileged visual self-teacher target distributions on the same trajectory, and weighted reverse-KL. |
| LiteGUI | Keep stage-scoped `strict_opd`. | Guided OPD/GKD is strict; later GRPO remains separate and not strict by itself. |
| TCOD | Keep `borderline_strict`. | Strict-like F2B KL exists, but B2F teacher-prefix structure and replay/freshness caveats keep the whole method below clean strict. |
| Skill-SD | Keep `borderline_strict`. | Importance-weighted reverse-KL self-distillation exists, but it is a small component inside a GRPO-heavy hybrid. |
| OEC | Downgrade to `adjacent`. | Expert completes from student-reached states, but the expert does not score student tokens; training uses rejection sampling plus SFT/NLL on expert continuations. |
| OpenClaw-RL OPD component | Keep `borderline_strict`. | Hint-conditioned self-teacher token gaps are real, but the signal is consumed as policy-gradient advantage mixed with scalar PRM reward. |
| VLA-OPD | Keep `strict_opd`. | Expert action-token supervision is applied to student-visited environment states and consumed through reverse-KL/action-token objective. |
| MAD-OPD / OPAD | Keep `strict_opd`. | Multi-teacher debate force-decodes student on-policy tokens and consumes token divergence/JSD/RKL supervision. |
| SOD | Keep `strict_opd`. | Step-wise teacher-logit OPD is a primary signal; GRPO is auxiliary. |
| OPCD | Keep `strict_opd`. | Context-conditioned privileged teacher supervises the same context-free student rollouts with reverse KL. |
| VOLD | Keep stage-scoped `strict_opd`. | Strict for Stage 2 unified RL plus OPD; note that the OPD training signal is text-only and visual transfer is indirect. |
| Video-OPD | Keep `strict_opd`. | Current video-grounding rollouts receive Qwen3-VL-32B-GRPO token-level reverse-KL-equivalent supervision. |
| X-OPD | Keep `strict_opd`. | Speech/audiotext student rollouts receive token-level text-teacher feedback. |
| Uni-OPD | Keep `strict_opd`. | Student trajectories receive token-level teacher guidance with margin mask/shift calibration. |
| PRISM | Downgrade to `adjacent`. | The MoE discriminator produces response-level scalar rewards that are converted to policy-gradient advantages; no token/logit distillation objective is consumed. |
| KEPO | Keep `borderline_strict`. | Quality-gated teacher divergence exists, but exact divergence mechanics are weak and only reward-passing trajectories receive the distillation term. |
| D-OPSD | Keep `borderline_strict`. | Same-rollout privileged diffusion self-distillation exists, but the objective is velocity-field MSE rather than token/logit KL. |
| Flow-OPD | Keep `borderline_strict`. | Multi-teacher flow supervision exists, but it is consumed through PPO-style dense reward plus L2/velocity matching. |
| Qwen3-VL multimodal RL/SAPO | Keep `adjacent`. | Public evidence scopes OPD to text/backbone distillation; visual-language same-rollout OPD is not disclosed. |
| Qwen3 OPD stage | Keep stage-scoped `strict_opd` with medium confidence. | Primary source states on-policy student sequences, teacher logits, and KL minimization, but the public report is still terse on implementation details. |
| MiMo-V2-Flash MOPD | Keep stage-scoped `strict_opd`. | MOPD stage has current student samples and domain-teacher token/log-ratio supervision; full pipeline is multi-stage. |
| Nemotron-Cascade2 MOPD | Keep stage-scoped `strict_opd`. | Multi-domain OPD component has dense token-level distillation advantages on student samples. |
| OPSD / CRISP / SDFT continual | Keep `strict_opd`. | Privileged self-teacher or demonstration-conditioned self-teacher scores student rollouts with per-token divergence objectives. |
| KDRL / SCOPE / Gemma2 post-training | Keep `borderline_strict`. | Each has either hybrid/subcomponent-only OPD or terse public evidence that prevents a clean strict method-level label. |

## Downgrades

| Method | New label | Why |
|---|---|---|
| PRISM | `adjacent` | The primary source describes policy rollouts scored by a MoE discriminator, then policy-gradient optimization with discriminator rewards. This is useful black-box response alignment, but it fails strict C3 under the repo rule because the consumed signal is scalar reward/advantage, not distributional distillation. |
| OEC | `adjacent` | OEC fixes covariate shift by starting from student rollouts and switching to an expert, but the expert generates continuations and the student is trained with rejection-sampled SFT/NLL. This is DAgger-style imitation, not OPD. |

## Stage Caveats Preserved

- `litegui-2026`: strict only for Guided OPD/GKD.
- `vold-2025`: strict for Stage 2 OPD, with a text-only teacher/training caveat.
- `qwen3-opd-2025`: strict only for the strong-to-weak on-policy distillation phase; keep medium confidence because the technical report is concise.
- `mimo-v2-flash-2026`: strict only for Stage 3 MOPD.
- `nemotron-cascade2-2026`: strict only for the MOPD component inside Cascade RL.

## Table Edits Merged

- `tables/opd_papers.md`: downgraded `prism-2026` and `oec-2025` to `adjacent` with `wp9_downgraded`.
- `tables/vlm_opd_papers.md`: downgraded the mirrored `prism-2026` row to `adjacent`.
- `tables/adjacent_work.md`: added PRISM and OEC as explicit adjacent/false-positive boundary rows.
- Current narrative docs were updated so WP9 is marked complete and PRISM/OEC are no longer listed as borderline current rows.

## Next Router State

WP0 through post-WP8 WP9 are now merged. No immediate external verifier is needed for the current queue.

The next useful work is local consistency and release-readiness cleanup: make sure current tables, README, taxonomy files, and reading paths agree; then decide whether to launch a new research package for one of the remaining gaps, such as diffusion/flow OPD taxonomy, black-box response-level OPD boundaries, or long-video/document/GUI frontier discovery.
