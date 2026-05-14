# WP0 Survey Synthesis

Last checked: 2026-05-11

Post-WP9 status note: this WP0 note is historical survey triage. Later audits resolved several queue items: Uni-OPD, VLA-OPD, and MAD-OPD are strict; PRISM is adjacent after the post-WP8 WP9 verifier.

This memo merges the independent WP0 research outputs under `assets/research/`. It is a curator note, not a primary source. Use it to decide what to verify next; do not cite it as evidence for paper rows.

## Consensus

The Tencent survey is a good organizing backbone, but it is intentionally broad. This repo should keep the survey axes while adding stricter implementation and evidence controls.

Strict OPD requires all three conditions:

- **C1: Student rollout.** The current student or policy generates the trajectory, completion, rationale, action, tool call, or intermediate state used for training.
- **C2: Teacher-style supervision.** A teacher, expert, discriminator, reference model, previous checkpoint, or privileged-context model supervises that exact student-generated state.
- **C3: Objective consumption.** The training objective directly consumes that supervision signal. Reward-only optimization is not enough.

The missing axes from the survey are now repo-level requirements:

- **Rollout provenance and freshness:** current-policy, per-iteration, per-epoch, replay-buffer-stale, previous-policy, precomputed-on-SFT-rollouts, or static dataset.
- **Teacher kind:** larger LLM, multimodal teacher, text teacher for cross-modal transfer, expert policy, discriminator, previous checkpoint, privileged self, multi-teacher, reward-only verifier, or none.
- **Signal consumption:** whether the loss actually consumes teacher-style supervision on the rollout.

## High-Confidence Seeds

These entries have enough WP0 agreement to appear as initial seed rows, with primary-source URLs still recorded per row:

| Bucket | Seeds |
|---|---|
| Foundational white-box OPD | GKD pure on-policy variant, MiniLLM |
| Systems and divergence OPD | DistillSpec, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD |
| Teacher-free / privileged self OPD | OPSD, OPSDC / CRISP |
| Industrial OPD | Qwen3 documented OPD stage, Nemotron-Cascade 2 multi-domain OPD |
| Cross-modal and multimodal frontier | X-OPD speech, VOLD, Video-OPD |

These seeds are not a claim that every training stage in a paper or report is OPD. For industrial systems, only the disclosed OPD stage should be labeled strict.

## Conflict Queue

These candidates should not be labeled final `strict_opd` until a WP9 verifier confirms C1/C2/C3 from primary sources:

| Candidate | Conservative status | Why |
|---|---|---|
| DistiLLM / DistiLLM-2 | `partial_opd` or `borderline_strict` | Student-generated outputs are mixed with adaptive or stale replay, so rollout freshness decides the label. |
| GAD | `borderline_strict` pending audit | Black-box adversarial signal can be OPD, but response-level objective needs source-level verification. |
| Lion / OVD | `partial_opd` or `borderline_strict` | Need to separate black-box teacher supervision from reward or prompt curation. |
| PACED | `partial_opd` or conditional strict | It is a curriculum/weighting framework; strictness depends on the underlying distillation track. |
| AdaSwitch / BOND / Lightning OPD / DASD / DDT | `partial_opd`, `adjacent`, or `unclear` | Each has a rollout freshness or signal-consumption ambiguity. |
| MiMo-V2-Flash / Gemma2 post-training | `unclear` until official audit | Industrial claims may be real but need isolated evidence for the OPD stage. |
| Uni-OPD / PRISM / VLA-OPD / MAD-OPD | historical `borderline_strict` queue | Later audits resolved this queue: Uni-OPD, VLA-OPD, and MAD-OPD are strict; PRISM is adjacent. |

## Default Downgrades

These patterns must not enter strict OPD tables unless new primary evidence proves C1/C2/C3:

| Pattern or work | Default label | Missing condition |
|---|---|---|
| DeepSeek-R1 distilled checkpoints | `not_opd` | SFT from teacher traces; no student on-policy distillation stage. |
| Gemma2 pretraining KD | `not_opd` | Soft-target KD over a corpus is not student-rollout OPD. |
| SeqKD, SFT on teacher CoT, synthetic teacher data | `not_opd` | Missing C1. |
| ToDi, AKL, TAID, PromptKD, ATKD, AlignDistil, f-DISTILL, TED | `adjacent` or `not_opd` | Useful objectives or weighting ideas, but original formulations are off-policy. |
| SPIN / DAIL | `adjacent` unless re-audited | Self-play or imitation signal does not necessarily provide teacher-style supervision on current rollouts. |
| SCoRe / reward-only GRPO, RLVR, RLHF | `adjacent` | Missing C2/C3 distillation-style supervision. |
| VLM-R1, Perception-R1, DeepVideo-R1, VTool-R1, ManipLVM-R1, LaViPlan | `adjacent` | On-policy reward or verifier optimization, not OPD. |
| LLaVA-KD, LLAVADI, VPD | `not_opd` | Offline multimodal distillation or program/tool-generated data. |

## Qwen Research File Policy

`assets/research/wp0-survey-qwen.md` is useful as a broad clue map but too weak to act as an evidence source by itself. It includes secondary sources, unsupported industrial extrapolation, and broad OPD labels that conflict with the stricter repo definition.

Use it only to create backlog items. A row from that file can merge only after a verifier finds primary evidence and fills the canonical table fields.

## WP9 Prompt

Superseded: the first WP9 and post-WP8 WP9 passes are complete. Use the prompt in `notes/agent-prompts.md` only for future new strict/borderline batches. Discovery agents can be broad; verifier agents must be skeptical and should downgrade any row missing C1, C2, or C3 evidence.
