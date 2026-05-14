# WP3: Teacher-Free, Self-Play, Privileged-Context, Previous-Checkpoint, and Self-Distillation OPD Audit

**Auditor:** ChatGPT  
**Date:** 2026-05-13  
**Source policy:** Primary sources only: arXiv papers, PMLR proceedings, OpenReview/project pages when authored by the paper authors. Secondary summaries were not used as evidence.

## 1. Memo

The strict WP3 core is real but narrower than the broad "self-play" label suggests. The high-confidence strict cases are the ones where a current student rollout is evaluated by a same-weight privileged/conditioned self-teacher and the loss consumes that signal as token/logit distillation. Under C1/C2/C3, **OPSD**, **CRISP/OPSDC**, **SDPO**, **OPCD**, and **SDFT 2026** are strict. **OEL** contains a strict OPCD-style consolidation substage, but the whole OEL loop also includes deployment collection and experience extraction, so the paper-level row should stay `partial_opd` unless the table tracks substages.

The main false-positive pattern is self-play preference learning. **SPIN** and related DPO/SPO/IPO-style loops use previous-model outputs, self-generated rejected samples, human or teacher positives, or discriminator-created preference pairs. These can be on-policy or self-play in a loose sense, but they generally fail strict OPD because the objective is sequence-level preference discrimination rather than distillation-style supervision on the exact current rollout. Static or semi-static DPO/ORPO/IPO variants should be `adjacent` or at most `partial_opd` unless a source explicitly adds token/logit teacher supervision on current student rollouts.

The second false-positive pattern is privileged self-distillation where the teacher generates a different trajectory. **HDPO** and **GATES** both use same-model privileged teachers and distillation-like objectives, but their primary supervised traces are privileged/tutor rollouts, not the student's exact current rollout. GATES has an on-policy auxiliary mode, but its own paper states the off-policy trajectory loss is the primary driver; HDPO's distillation arm targets separately generated privileged cliff-prompt rollouts. These are best kept as `partial_opd`, not strict.

The third boundary is bundled methods. **Privileged Information Distillation** includes a strict-looking OPSD/RKL submethod, but the paper-level contribution also includes a broader pi-Distill joint teacher-student objective. Conservatively, track the paper as `partial_opd` unless the row is explicitly scoped to the OPSD variant. Similarly, **OEL** should remain method-level partial with a strict consolidation substage.

## 2. Evidence Ledger

| method | primary source | exact source location | C1 current student rollout | C2 same-rollout teacher substitute | C3 distillation-style objective | teacher substitute | objective family | rollout freshness | final label | reason to downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OPSD / Self-Distilled Reasoner | https://arxiv.org/abs/2601.18734 | Abstract; Sec. 1; Sec. 3.2 | Student policy samples its own reasoning trajectories during training. | Same LLM acts as teacher when conditioned on privileged ground-truth or reasoning information and evaluates the student's rollout prefixes. | Per-token divergence between privileged teacher and student distributions over the student rollout. | privileged self-teacher | token KL / divergence | current_policy | strict_opd | None. Meets C1/C2/C3 directly. | high |
| CRISP / OPSDC | https://arxiv.org/abs/2603.05433 | Abstract; Sec. 1; Sec. 3 | Student generates reasoning rollouts without the conciseness instruction. | Same model with a conciseness instruction provides teacher logits for those rollout states. | Per-token reverse KL distills concise behavior back into the base prompt behavior. | concise self-teacher | reverse KL | current_policy with periodic refresh | strict_opd | None. Previous/self checkpoint refresh is allowed by C2. | high |
| SDPO / Reinforcement Learning via Self-Distillation | https://arxiv.org/abs/2601.20802 | Abstract; Sec. 1; Sec. 2; Appendix A | Current policy samples attempts in verifiable or rich-feedback environments. | The current model re-evaluates the same attempt conditioned on environment feedback. | Logit-level/token-level self-distillation transfers feedback-conditioned next-token predictions into the policy. | feedback-conditioned self-teacher | KL/JSD/top-k logit distillation | current_policy | strict_opd | RL framing is not a downgrade because the learning signal is dense self-teacher logits, not only scalar reward. | high |
| OPCD | https://arxiv.org/abs/2602.12275 | Abstract; Sec. 3; Algorithm 1 | Student samples responses without the privileged context. | Context-conditioned teacher evaluates the same student response prefixes with the context prepended. | Token-level reverse KL between student and context-conditioned teacher. | context-conditioned teacher or self-teacher | reverse KL | current_policy | strict_opd | None. Clean privileged-context OPD. | high |
| OEL | https://arxiv.org/abs/2603.16856 | Abstract; Sec. 3.2; Algorithm 1 | Consolidation stage samples single-turn responses from the current server-side model on deployment-derived prefixes. | Frozen same-model teacher conditioned on extracted experiential knowledge evaluates those sampled prefixes. | Token-level reverse KL consolidates experiential context into parameters. | experiential-context self-teacher | OPCD-style reverse KL | iterative current_policy per consolidation round | partial_opd | The consolidation substage is strict, but OEL as a whole includes trajectory collection and knowledge extraction. | high |
| HDPO | https://arxiv.org/abs/2603.23871 | Abstract; Sec. 1; Sec. 3 | GRPO branch uses current rollouts; the distillation branch is triggered by cliff prompts. | Privileged same-model teacher is conditioned on ground truth. | JSD/token-level distillation is applied to filtered privileged rollouts. | privileged self-teacher | GRPO + JSD | mixed_policy | partial_opd | Distillation happens on separately generated privileged rollouts, not the exact failed student rollout. | high |
| GATES | https://arxiv.org/abs/2602.20574 | Abstract; Sec. 1; Sec. 3.2-3.3 | Primary training uses tutor-generated document-grounded trajectories; optional on-policy mode samples student trajectories. | Same model as tutor has privileged document context; consensus gates reliability. | Off-policy NLL/trajectory distillation is primary; on-policy token scoring is auxiliary. | privileged tutor/self-teacher with consensus gate | consensus-gated trajectory distillation plus auxiliary on-policy scoring | mixed_policy | partial_opd | C1 is weak at method level because tutor rollouts drive the main supervised trajectory loss. | high |
| Privileged Information Distillation | https://arxiv.org/abs/2602.04942 | Abstract; method summary in primary arXiv page | The OPSD variant uses student policy behavior; pi-Distill is broader joint teacher-student training. | PI-conditioned same model serves as teacher for the unconditioned student. | OPSD variant uses RL with reverse-KL penalty to the PI-conditioned teacher. | privileged-information self-teacher | pi-Distill plus OPSD/RKL | mixed_by_submethod | partial_opd | Paper bundles strict-looking OPSD with broader pi-Distill; split submethod if adding a strict row. | medium |
| SDFT 2026 / Self-Distillation Enables Continual Learning | https://arxiv.org/abs/2601.19897 | Abstract; official project page https://self-distillation.github.io/SDFT | Student produces on-policy trajectories while learning from demonstrations. | Same model conditioned on demonstrations acts as teacher. | Token-level reverse-KL self-distillation preserves prior capabilities while acquiring the demonstrated skill. | demo-conditioned self-teacher | reverse KL | current_policy | strict_opd | None, if this refers to the 2026 Shenfeld et al. paper. | high |
| SDFT 2024 / Self-Distillation Bridges Distribution Gap | https://arxiv.org/abs/2402.13669 | Abstract | Uses a model-generated distilled dataset before fine-tuning. | No teacher substitute supervises exact current student rollouts. | Standard fine-tuning on distilled data, not same-rollout distillation. | self-generated offline data | SFT | static_self_dataset | not_opd | Acronym collision: this SDFT is offline/self-generated-data fine-tuning. | high |
| SPIN | https://proceedings.mlr.press/v235/chen24j.html | PMLR abstract; arXiv 2401.01335 | Generates training data from previous iterations, not the current updated student at the supervised state. | Human-annotated data and prior self-generated responses define a discrimination game, not teacher scoring on the current rollout. | IPM/logistic self-play objective, not token/logit distillation. | previous checkpoint plus human data | self-play preference/logistic | previous_policy | adjacent | Fails strict C1/C2/C3 as OPD, despite being important teacher-free self-play. | high |
| D2PO | https://arxiv.org/abs/2405.01511 | Algorithm/objective sections | Can use current policy rollouts to form candidate pairs. | Discriminator scores pairs or labels preferences. | DPO preference loss consumes binary pair labels, not a distillation target. | discriminator/preference source | DPO | current_policy pairs | adjacent | C3 fails strict: preference optimization is not OPD by default. | high |
| SODA | https://arxiv.org/abs/2604.03873 | Sec. 2; Algorithm 1 | Uses base-student snapshot responses rather than evolving current policy rollouts. | Black-box teacher supplies preferred responses. | DPO consumes teacher-positive/base-student-negative preference pairs. | black-box teacher | DPO | static_base_snapshot | partial_opd | Semi-on-policy at most; freshness and C3 are weak. | high |
| daDPO | https://arxiv.org/abs/2506.15717 | Method/objective sections | Student responses appear in constructed preference data, not as live current rollouts. | Teacher distribution terms appear in the DPO-style objective. | Static or dataset-level DPO with distribution-aware terms. | white-box teacher distribution | DPO plus teacher-distribution term | static_dataset | partial_opd | Has distributional teacher evidence but fails clean current-rollout OPD. | medium |
| ORPO-Distill | https://arxiv.org/abs/2509.25100 | Method/objective sections | Student traces are rejected-side data, often mixed-policy. | Teacher generates independent chosen traces rather than supervising the exact student trace. | ORPO preference objective, not same-rollout distillation. | black-box teacher text | ORPO | mixed_policy | adjacent | C2 and C3 fail strict. | medium |

## 3. Candidate Rows / Table Deltas

### Rows safe to keep strict or update as strict

| id | action | target label | source | table rationale |
|---|---|---|---|---|
| opsd-2026 | keep; raise confidence to high/source_verified | strict_opd | https://arxiv.org/abs/2601.18734 | C1 student rollouts; C2 privileged self-teacher; C3 per-token divergence. |
| opsdc-crisp-2026 | keep; normalize authors to Sang et al.; raise confidence | strict_opd | https://arxiv.org/abs/2603.05433 | C1 student reasoning rollouts; C2 concise self-teacher logits; C3 reverse KL. |
| sdpo-2026 | keep; normalize title to "Reinforcement Learning via Self-Distillation" and authors to Hübotter et al. | strict_opd | https://arxiv.org/abs/2601.20802 | C1 current-policy attempts; C2 feedback-conditioned self-teacher on same attempt; C3 logit-level distillation. |
| opcd-2026 | keep; normalize authors to Ye et al. | strict_opd | https://arxiv.org/abs/2602.12275 | Clean context-conditioned OPD: student samples without context, teacher evaluates with context, reverse KL trains student. |
| sdft-continual-2026 | add to `tables/opd_papers.md` | strict_opd | https://arxiv.org/abs/2601.19897 | On-policy self-distillation fine-tuning with demonstration-conditioned self-teacher and reverse KL. |

Suggested compact row fields for the new strict SDFT row:

| id | title | year | authors | paper_url | method_family | teacher_access_regime | loss_granularity | divergence_or_objective | on_policy_strength | opd_strictness | teacher_kind | rollout_source | rollout_freshness | supervision_signal | loss_objective | strictness_evidence | confidence | notes |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdft-continual-2026 | Self-Distillation Enables Continual Learning | 2026 | Shenfeld et al. | https://arxiv.org/abs/2601.19897 | self_distillation_opd | teacher_free | token | reverse_kl | strong | strict_opd | demo_conditioned_self_teacher | student_on_policy | current_policy | demonstration_conditioned_self_logits | reverse_kl | C1 on-policy trajectories; C2 demo-conditioned self-teacher supervises those trajectories; C3 token-level reverse-KL objective consumes that signal. | high | Disambiguate from SDFT 2024, which is not OPD. |

### Rows to keep or change below strict

| id | action | target label | source | rationale |
|---|---|---|---|---|
| oel-2026 | keep method-level `partial_opd`; optionally add substage note | partial_opd | https://arxiv.org/abs/2603.16856 | OPCD consolidation is strict, but the full method includes deployment collection and knowledge extraction. |
| gates-2026 | keep `partial_opd`; fix title to "GATES: Self-Distillation under Privileged Context with Consensus Gating" | partial_opd | https://arxiv.org/abs/2602.20574 | Main loss distills tutor trajectories; on-policy mode is auxiliary and not the method-level driver. |
| hdpo-2026 | keep `partial_opd`; fix title to "HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation" | partial_opd | https://arxiv.org/abs/2603.23871 | Distillation uses privileged rollouts generated separately from failed current rollouts. |
| privileged-information-distillation-2026 | add method-level row or split submethod | partial_opd, or strict_opd only for OPSD subrow | https://arxiv.org/abs/2602.04942 | Paper bundles pi-Distill and OPSD; conservative table should not mark the whole paper strict. |
| spin-2024 | add to `tables/adjacent_work.md` | adjacent | https://proceedings.mlr.press/v235/chen24j.html | Previous-iteration self-play preference/discrimination, not same-rollout distillation. |
| sdft-2024 | add to `tables/adjacent_work.md` if not present | not_opd | https://arxiv.org/abs/2402.13669 | Self-generated static dataset plus fine-tuning; acronym collision with SDFT 2026. |

Suggested adjacent rows:

| id | title | year | link | category | why_adjacent | why_not_strict_opd | evidence_url | confidence | last_checked | notes |
|---|---|---:|---|---|---|---|---|---|---|---|
| spin-2024 | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models | 2024 | https://proceedings.mlr.press/v235/chen24j.html | self_play_preference_finetuning | Canonical teacher-free self-play method using previous-iteration self-generated responses. | Prior checkpoint negatives and human positives define a sequence-level discrimination game; no teacher substitute supervises exact current rollout with a distillation loss. | https://proceedings.mlr.press/v235/chen24j.html | high | 2026-05-13 | Boundary case for self-play vs strict OPD. |
| sdft-2024 | Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning | 2024 | https://arxiv.org/abs/2402.13669 | self_distilled_dataset_finetuning | Uses self-generated data to reduce fine-tuning distribution gap. | Static distilled dataset and SFT-style training; no same-rollout teacher supervision. | https://arxiv.org/abs/2402.13669 | high | 2026-05-13 | Acronym collision with SDFT 2026. |

## 4. Adjacent / False-Positive Ledger

| method family | representative sources | conservative label | why it is not strict OPD |
|---|---|---|---|
| Previous-checkpoint self-play preference learning | SPIN: https://proceedings.mlr.press/v235/chen24j.html | adjacent | Previous checkpoint creates negatives and the objective distinguishes human vs self responses; it does not distill teacher logits/signals on current rollouts. |
| DPO/SPO/IPO variants with current-policy samples | D2PO: https://arxiv.org/abs/2405.01511 | adjacent | Even when rollouts are current, C3 is a preference objective over chosen/rejected pairs, not a distillation-style same-rollout target. |
| Semi-on-policy black-box DPO | SODA: https://arxiv.org/abs/2604.03873 | partial_opd | Uses a base student snapshot and teacher responses in DPO pairs; rollout freshness is stale and the loss is preference optimization. |
| Distribution-aware or teacher-augmented DPO | daDPO: https://arxiv.org/abs/2506.15717 | partial_opd | Teacher distributions appear, but training remains DPO-style over constructed/static pairs rather than live same-rollout OPD. |
| Teacher-chosen vs student-rejected ORPO/DPO distillation | ORPO-Distill: https://arxiv.org/abs/2509.25100 | adjacent | Teacher chosen trace is independently generated; the teacher does not supervise the exact student rollout. |
| Privileged tutor trajectory distillation | GATES: https://arxiv.org/abs/2602.20574 | partial_opd | The primary supervised traces are privileged tutor rollouts, although an auxiliary on-policy scoring mode exists. |
| Privileged cliff-prompt distillation | HDPO: https://arxiv.org/abs/2603.23871 | partial_opd | The distillation target is generated with ground-truth context after a failed rollout group; it is not mounted on the failed student rollouts themselves. |
| Offline self-distillation fine-tuning | SDFT 2024: https://arxiv.org/abs/2402.13669 | not_opd | The self signal becomes a static distilled dataset for SFT. |

## 5. Gap Map

| gap | affected methods | current recommendation | needed evidence |
|---|---|---|---|
| Paper-level vs submethod-level labels | OEL, Privileged Information Distillation, GATES, HDPO | Keep whole methods below strict unless the row name scopes to the strict substage. | Algorithm-level proof that the listed row's main objective uses current student rollouts and same-rollout teacher supervision. |
| Self-play DPO/SPO/IPO family audit | SPIN, D2PO, SODA, daDPO, ORPO-Distill, DistillDirect-like variants | Treat as adjacent/partial by default. | A primary source showing token/logit teacher or self-teacher supervision on the exact current rollout, not only pairwise preference labels. |
| Previous-checkpoint OPD | SPIN-like and previous-checkpoint self-distillation methods | Previous checkpoint is allowed by C2, but only if it supervises current rollouts with distillation-style loss. | Current-policy rollout collection plus previous-checkpoint logits/scores consumed by a distillation objective. |
| Verifier/self-play opponent as teacher substitute | verifier-guided self-play, opponent/judge methods | Reward-only verifier feedback is adjacent; dense verifier distillation may be borderline. | Non-scalar distillation signal mounted on the same rollout, with objective consumption beyond reward/advantage. |
| GATES on-policy branch | GATES | Keep whole method partial; consider a subrow only if needed. | Quantitative and algorithmic evidence that the on-policy branch alone is a central method variant and satisfies C1/C2/C3. |
| SDFT acronym collision | SDFT 2024 vs SDFT 2026 | Add both with disambiguating IDs. | None; the current primary sources already establish different methods. |
| Author/title normalization | CRISP/OPSDC, SDPO, GATES, HDPO, OPCD/OEL | Normalize table names and authors during table cleanup. | Bibliographic cleanup from arXiv/PMLR pages. |

## 6. Classification Summary

| label | methods |
|---|---|
| strict_opd | OPSD, CRISP/OPSDC, SDPO, OPCD, SDFT 2026 |
| partial_opd | OEL as whole method, GATES, HDPO, Privileged Information Distillation as whole paper, SODA, daDPO |
| adjacent | SPIN, D2PO, ORPO-Distill, generic static DPO/SPO/IPO self-play variants |
| not_opd | SDFT 2024 |
| unclear | Any self-play SPO/IPO variant without a checked primary source showing C1/C2/C3 |

Bottom line: WP3 should be split into two shelves. The first shelf is **strict privileged-context self-distillation**: current student rollouts plus same-model privileged logits plus token-level KL/JSD. The second shelf is **adjacent teacher-free self-play/preference learning**: useful and often influential, but not strict OPD unless the primary source shows same-rollout distillation.
