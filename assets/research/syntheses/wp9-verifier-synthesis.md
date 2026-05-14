# WP9 Verifier Synthesis

Last checked: 2026-05-12

Post-WP8 update: [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) supersedes PRISM and OEC current labels: both are now `adjacent`.

This note merges four independent WP9 verifier outputs:

- `assets/research/wp9-verifier-chatgpt.md`
- `assets/research/wp9-verifier-claude-code.md`
- `assets/research/wp9-verifier-claude.md`
- `assets/research/wp9-verifier-qwen.md`

The merge policy is conservative. A majority vote does not promote a method to `strict_opd` by itself. A method must still pass the C1/C2/C3 test with primary-source evidence, and disagreements are preserved in `conflict_status`.

Update: the follow-up line-level audits are summarized in `wp9-line-audit-synthesis.md`. That later note resolves the original `human_spotcheck_needed` queue.

## Merge Rules

Strict OPD still requires all three:

- **C1: student rollout.** The current student or policy generates the training trajectory.
- **C2: same-rollout teacher-style supervision.** A teacher, expert, discriminator, reference model, previous checkpoint, or privileged-context model supervises that exact student trajectory.
- **C3: objective consumption.** The training objective consumes the teacher-style signal, not only a scalar reward.

WP9 adds these interpretation rules:

- Token/logit KL, teacher log-probs, privileged self-teacher KL, and action-token VLA teacher supervision can satisfy C3.
- OPD+RL hybrids can be strict only for the component where dense teacher-style supervision is consumed on current student rollouts.
- Response-level discriminator, verbal-score, or reward-model feedback is usually `partial_opd` or `borderline_strict`, not locked `strict_opd`, unless the source shows a direct non-scalar distillation objective.
- Stage-specific industrial claims must be scoped to the qualifying phase; a full training pipeline should not inherit a strict label from one phase.
- The Thinking Machines Lab OPD blog is useful as an operational definition/background source, but it is not a normal paper row.

## WP9 Status Buckets

| wp9_status | Meaning | Merge action |
|---|---|---|
| `safe_strict` | Verifiers agree or primary evidence is already strong. | Keep strict; record phase/configuration caveat if needed. |
| `promote_to_strict` | Several verifiers found strict evidence, but the repo had it as borderline. | Promote only with `wp9_conflict` or `human_spotcheck_needed` until line-level quotes are extracted. |
| `keep_borderline` | OPD-like, but signal type, rollout purity, or objective is still ambiguous. | Keep `borderline_strict`; do not use as clean seed. |
| `downgrade` | Missing one or more C1/C2/C3 conditions under the repo definition. | Move to `partial_opd`, `adjacent`, or `not_opd`. |
| `human_spotcheck` | Verifiers conflict or source evidence is too abstract/new. | Queue for line-level primary-source audit. |
| `false_positive` | Hallucination, name collision, or non-OPD method. | Keep only in adjacent/false-positive ledger. |

## Safe Strict Seeds

These are safe to keep as strict OPD seeds, with any scope caveat recorded in `notes`:

| Method | WP9 status | Caveat |
|---|---|---|
| MiniLLM | safe_strict | Canonical reverse-KL OPD. |
| GKD, pure on-policy variant | safe_strict | Strict only for the `lambda=1` or pure student-sampled setting; mixed fixed-data variants are not strict. |
| DistillSpec | safe_strict | Strict inside speculative-drafter training; keep separate from capability post-training. |
| Entropy-Aware OPD | safe_strict | Needs line-level audit only for detailed objective variants. |
| Veto | safe_strict | Strict target-reformulation OPD. |
| Fast OPD | safe_strict | Prefix truncation is an efficiency variant, not a strictness failure. |
| OPSD | safe_strict | Privileged self-teacher satisfies C2 when it supervises current student rollouts. |
| CRISP / OPSDC | safe_strict | Same-model privileged/concise path is the teacher role. |
| Qwen3 OPD phase | safe_strict_with_phase_caveat | Strict only for the documented OPD phase, not the whole Qwen3 pipeline. |
| Nemotron-Cascade2 MOPD | safe_strict_with_phase_caveat | Strict for the MOPD component inside the cascade. |
| Video-OPD | safe_strict | Current-policy video trajectories plus dense teacher supervision. |
| X-OPD | safe_strict | Cross-modal speech OPD seed. |

## Promoted With Conflict Notes

These candidates are promoted to `strict_opd` in the tables because WP9 plus a primary-source spot-check found strong C1/C2/C3 evidence, but they still carry a line-level audit flag because they are new and conflict-prone.

| Method | Prior repo label | New repo label | Conflict note |
|---|---|---|---|
| Uni-OPD | borderline_strict | strict_opd | arXiv abstract explicitly mentions student-generated states, aggregated token-level guidance, and margin calibration; full objective lines still need extraction. |
| VLA-OPD | borderline_strict | strict_opd | Line audit confirms student VLA rollouts, expert teacher action logits on visited states, and reverse-KL/action-token objective. |
| MAD-OPD | borderline_strict | strict_opd | PDF spot-check shows student trajectories, multi-teacher debate over the on-policy state, token-level supervision, and JSD/RKL objective; still new and should be audited line by line. |

## Keep Borderline Or Partial

| Method | Conservative label | Reason |
|---|---|---|
| PRISM | adjacent after post-WP8 update | Response-level black-box/adversarial supervision is consumed as scalar reward/advantage, not classic token/logit KL. |
| G-OPD | strict_opd with human spot-check | Likely strict under OPD formulation, but generalized reward-extrapolation framing needs objective-level quotes. |
| REOPOLD | strict_opd with human spot-check | Likely strict, but reward-as-log-ratio wording sits on the RL/KD boundary. |
| KDRL | borderline_strict | KD-RKL component may be strict; the full method is a KD+RL hybrid. |
| SDPO | borderline_strict | Verifiers disagree; needs primary exact quotes for self-teacher signal and objective. |
| OPCD | borderline_strict | Strong privileged-context candidate; keep below strict until line-level audit. |
| OEL | partial_opd | Broader online experience loop; OPCD-like consolidation substage may be strict. |
| SCOPE | partial_opd | Real expansion is Signal-Calibrated On-Policy Distillation Enhancement; dual path means not every branch has teacher supervision. |
| PACED | partial_opd | Curriculum/weighting and OPD details need line-level confirmation. |
| MiMo-V2-Flash | partial_opd | arXiv exists, but OPD-stage evidence needs extraction from the report. |
| Gemma2 post-training | partial_opd | Official wording suggests teacher on student distribution, but exact rollout/update recipe is too terse. |

## Downgrades

| Method | Conservative label | Missing condition |
|---|---|---|
| DistiLLM | partial_opd | Mixed or stale student-generated outputs weaken C1 freshness. |
| DistiLLM-2 | partial_opd | Mixed teacher/student response treatment; exact same-rollout supervision is not clean. |
| GAD | partial_opd | Discriminator signal is consumed mainly as reward/GRPO-style feedback. |
| OVD | partial_opd | Verbal score feedback is scalar or ordinal rather than dense teacher distribution. |
| Speculative KD | partial_opd | Interleaved student/teacher trajectory weakens pure student rollout. |
| AdaSwitch | partial_opd | Student starts the sequence, teacher finishes; rollout is hybrid. |
| Lightning OPD | partial_opd | Offline or precomputed SFT rollouts fail current-policy freshness. |
| GATES | partial_opd | Tutor/privileged traces appear to drive supervision rather than the exact student rollout. |
| HDPO | partial_opd | Distillation arm may use privileged teacher rollouts rather than exact student rollouts. |
| RLKD | adjacent | Teacher structure becomes scalar structural reward, so C3 fails. |
| BOND | adjacent | Reward-model best-of-N distribution matching, not teacher-on-rollout distillation. |
| LUFFY | adjacent | Explicit mixed/off-policy guidance plus RLVR. |
| Lion | adjacent | Teacher-generated responses for instruction tuning; C1 fails. |
| DASD | adjacent | Teacher completion or sequence distillation, not same-rollout OPD. |
| DDT | not_opd | No teacher-style distillation signal. |

## False Positives And Name Collisions

| Name | Status | Repo handling |
|---|---|---|
| DistillDirect | name collision / unverified OPD | Keep only as adjacent false positive; do not use as OPD seed. |
| DBKD | real acronym collision | Decision-based or dual-branch KD variants are not OPD. |
| TED as OPD | name collision | TED variants found by verifiers are not strict OPD under this repo's definition. |
| SCOPE as "Self-play Contrastive On-Policy Evaluation" | fabricated expansion | Correct expansion is Signal-Calibrated On-Policy Distillation Enhancement. |

## Human Spot-Check Queue

The line-level audits resolved the original queue:

- resolved strict: `Uni-OPD`, `MAD-OPD`, `VOLD`, `MiMo-V2-Flash MOPD`, `G-OPD`, `REOPOLD`, `SDPO`, `OPCD`, `VLA-OPD`;
- resolved borderline or partial: `Gemma2 post-training`, `PACED`, `OEL`, `SCOPE`, `KDRL`, `Lightning OPD`, `DistiLLM-2`; PRISM is adjacent after the post-WP8 verifier.

For each method, extract exact evidence lines for:

- `C1 quote`
- `C2 quote`
- `C3 quote`
- `rollout_freshness`
- `final_label`
- `reason_to_downgrade`

No new method should be promoted from any future queue without those extracted lines.
