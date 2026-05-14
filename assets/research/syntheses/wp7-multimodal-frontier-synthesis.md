# WP7 Multimodal Frontier Synthesis

Checked: 2026-05-13

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) keeps the strict multimodal core but downgrades PRISM to `adjacent` because its MoE discriminator signal is consumed as scalar policy-gradient reward rather than token/logit distillation.

This note merges:

- `assets/research/wp7-multimodal-frontier-chatgpt.md`
- `assets/research/wp7-multimodal-frontier-claude.md`
- `assets/research/wp7-multimodal-frontier-gemini.md`

## Core Finding

The multimodal strict OPD core is real but still small. All three audits agree that strict multimodal OPD requires the same C1/C2/C3 loop as text OPD, with the supervised state now being a visual-language response, temporal grounding trace, speech response, action trajectory, GUI coordinate/action, or multimodal rollout.

Most multimodal frontier methods remain adjacent: they use GRPO/RLVR, visual or perception rewards, RLAIF/DPO, offline MLLM KD, teacher-generated visual program traces, or industrial post-training reports without a line-visible teacher-on-rollout distillation objective.

## Merge Decision

| Method | Repo action | Reason |
|---|---|---|
| VOLD | Keep `strict_opd`. | Stage-2 VLM student rollouts receive text-teacher token/logit supervision plus GRPO. |
| Video-OPD | Keep `strict_opd`. | Current-policy video grounding trajectories receive dense teacher supervision through reverse-KL style OPD. |
| X-OPD | Keep `strict_opd`. | Speech/audiotext student rollouts receive token-level cross-modal teacher feedback. |
| Uni-OPD | Keep `strict_opd`. | Student trajectories across LLM/MLLM settings receive teacher token guidance with margin-calibrated OPD. |
| VLA-OPD | Keep `strict_opd`. | Expert VLA action-token supervision is applied to student-visited states. |
| GUI-SD | Upgrade to source-verified `strict_opd` with high confidence. | WP7 audits found enough C1/C2/C3 evidence for GUI grounding: student coordinate tokens, privileged visual self-teacher on the same trajectory, and weighted reverse KL. |
| LiteGUI | Keep stage-scoped `strict_opd`. | Stage-1 guided OPD/GKD is strict; Stage-2 GRPO is separable and not strict. |
| PRISM | Downgraded to `adjacent` by post-WP8 WP9. | Current multimodal responses are scored by a response-level MoE discriminator, but the signal is consumed as scalar policy-gradient reward/advantage rather than dense token/logit teacher supervision. |
| VLM-R1, Perception-R1, R1-VL/StepGRPO, Vision-R1, DeepVideo-R1, VTool-R1, OpenVLThinker | Keep adjacent. | On-policy multimodal rollouts exist, but consumed feedback is reward/verifier/preference or offline SFT rather than distillation-style teacher supervision. |
| LLaVA-KD, LLAVADI, Visual Program Distillation | Keep not-OPD/offline. | Teacher or tool traces supervise fixed/offline data; C1 fails. |
| Qwen3-VL, InternVL3.5, Gemma 3, Gemini, MiniCPM-V | Keep industrial adjacent/held out. | Public evidence does not expose strict multimodal OPD. Qwen3-VL may disclose text-only OPD inside an MLLM report, but not visual-language same-rollout OPD. |

## Candidate Queue Resolution

The follow-up line audit is synthesized in [wp7-candidate-line-audit-synthesis.md](wp7-candidate-line-audit-synthesis.md).

| Method | Final repo action | Reason |
|---|---|---|
| KEPO | Added `borderline_strict`. | Quality-gated teacher divergence exists, but strictness is limited to reward-passing trajectories and exact D/logit mechanics are not fully defined. |
| D-OPSD | Added `borderline_strict`. | Same-rollout privileged diffusion self-distillation exists, but the objective is velocity-field MSE without formal reverse-KL proof. |
| Flow-OPD | Added `borderline_strict`. | Multi-teacher flow supervision exists, but it is consumed through PPO-style dense reward plus L2 regression. |
| GTR-Turbo | Added adjacent. | Reverse KL is collapsed into scalar PPO reward shaping; SFT variant uses teacher-generated thoughts. |

## Source Conflicts And Holds

- Video-OPD teacher identity was pinned by candidate line audit to Qwen3-VL-32B-GRPO.
- X-OPD teacher access and token-level loss were source-verified.
- Uni-OPD margin mask and margin shift were source-verified.
- PRISM is adjacent after post-WP8 WP9 unless a future source shows non-scalar same-rollout distillation supervision.
- GUI-SD is strict only for single-step GUI grounding, not long-horizon GUI or browser control.
- LiteGUI is strict only for Stage 1 guided OPD/GKD, not Stage 2 GRPO.

## Gap Map

| Gap | Current status | Needed next |
|---|---|---|
| Medical VLM OPD | KEPO is borderline. | Recheck if code clarifies token-level teacher divergence. |
| Image generation OPD | D-OPSD and Flow-OPD are borderline. | Decide whether diffusion/flow velocity-field losses belong in a dedicated OPD taxonomy. |
| Long-video OPD | Video-OPD covers temporal grounding, not general long-video QA or summarization. | Search future long-video OPD separately. |
| Document/chart OPD | Uni-OPD covers document/chart benchmarks, but no dedicated document OPD method is confirmed. | Keep as benchmark coverage, not a separate method family yet. |
| Industrial multimodal OPD | No strict public VLM industrial row. | Keep Qwen3-VL, InternVL, Gemma 3, Gemini, and MiniCPM-V held out until a public training-loop source appears. |
| Evaluation | Strict OPD should be compared against RLVR/offline KD baselines without mixing labels. | WP8 should build evaluation bundles by modality and label type. |

## Next Router State

WP7 broad and candidate passes are complete. Proceed to WP8 evaluation methodology.
