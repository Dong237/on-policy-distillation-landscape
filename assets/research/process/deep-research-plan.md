# Deep Research Plan

Use this plan when assigning external LLMs or professional research agents such as Claude, ChatGPT, Gemini, Perplexity, Elicit, Semantic Scholar agents, or local agents.

The point is not to get quick paper summaries. The point is to build a verifier-ready research map: memo, evidence ledger, candidate Markdown rows, and gap map.

## Research Contract

Every work package must return:

1. **Research memo:** what was searched, what was found, what was ruled out, and what changed the landscape.
2. **Evidence ledger:** source-by-source notes with links, method evidence, and classification rationale.
3. **Candidate Markdown rows:** merge-ready rows only after evidence has been collected.
4. **Gap map:** unresolved questions, missing code, ambiguous claims, and recommended next searches.

## Global Rules

- Use primary sources first: arXiv, OpenReview, official project pages, official GitHub repositories, official model cards, official blogs, and technical reports.
- Treat the Tencent survey as the organizing starting point, not as ground truth.
- `strict_opd` requires all three: student on-policy trajectories, teacher-style supervision on those trajectories, and an objective using that supervision.
- Reward-only GRPO/RLVR/RLHF is not strict OPD.
- Offline KD, teacher-generated data, and SFT on teacher traces are not strict OPD.
- If evidence is incomplete, use `partial_opd`, `adjacent`, or `unclear`.
- Always record disconfirming evidence.

## WP0: Survey Replication and Citation Audit

### Mission

Reconstruct the survey's map independently. Verify every major method family and important citation before the repo accepts the survey's taxonomy or claims.

### Must answer

- Which methods does the survey classify under logit-based, outcome-based, and self-play feedback?
- Which methods are white-box, black-box/API, teacher-free, or multi-teacher?
- Which methods are token-level, sequence-level, hybrid, or adaptive?
- Which industrial claims in the survey are directly supported by official sources?
- Which cited papers are only adjacent, not strict OPD?
- Which references are unavailable, ambiguous, withdrawn, or unreproducible?

### Deliverables

- Survey outline with source-backed method families.
- Citation ledger with one row per cited method.
- List of claims that need human verification.
- Proposed edits to repo tables, without merging them directly.

## WP1: White-Box OPD and Divergence Methods

### Mission

Deeply audit white-box OPD where teacher logits, log-probs, or distributions supervise student-generated trajectories.

### Current status

WP1 discovery and candidate line audit have been run and synthesized in [wp1-whitebox-synthesis.md](wp1-whitebox-synthesis.md) and [wp1-candidate-line-audit-synthesis.md](wp1-candidate-line-audit-synthesis.md). The main table now includes promoted strict WP1 candidates and conservative partial/adjacent labels for mixed methods.

### Seeds

GKD, MiniLLM, DistiLLM, adaptive KL, entropy-aware OPD, sampled-token OPD failure analyses, top-K/local-support matching, cross-tokenizer logit transfer if on-policy.

### Must answer

- What divergence is used: forward KL, reverse KL, skewed KL, JSD, f-divergence, adaptive objective?
- Is the supervision token-level, sequence-level, or hybrid?
- Does the teacher score true student-generated prefixes?
- How does each method manage bias-variance and compute cost?
- Which codebases reproduce the objective?

## WP2: Black-Box/API OPD

### Mission

Map OPD when teacher logits are unavailable.

### Seeds

GAD, Lion, On-policy Verbal Distillation, preference/adversarial OPD, cross-architecture black-box distillation.

### Must answer

- Does the teacher/API supervise student rollouts or only generate data?
- Is feedback adversarial, verbal, preference-based, or ranking-based?
- Is the signal distillation-style or just reward optimization?
- What API cost, instability, or discriminator overfitting risks exist?

## WP3: Teacher-Free and Self-Play OPD

### Mission

Audit OPD-like methods without an external teacher.

### Seeds

SPIN, OPSD, SDPO, HDPO, privileged-context self-distillation, previous-checkpoint supervision, self-play and contrastive loops.

### Must answer

- What replaces the teacher?
- Does the current policy generate the supervised samples?
- Is the method self-training, self-play, DPO, or true OPD?
- When does teacher-free OPD behave like RL or preference optimization?

## WP4: Reasoning and OPD + RL Hybrids

### Mission

Study OPD for math, code, long-CoT, and reasoning-heavy domains, especially where dense teacher feedback is combined with sparse rewards.

### Current status

WP4 has been run and synthesized in [wp4-reasoning-opd-rl-synthesis.md](wp4-reasoning-opd-rl-synthesis.md). The merged result confirms the existing conservative labels: dense teacher log-probs/log-ratios on student rollouts can be strict even inside RL notation; scalar reward-only RLVR remains adjacent.

### Seeds

RLKD, G-OPD, RLAD, KDRL, Fast OPD, PACED, alternating KD/RL loops, unified KD+RL frameworks.

### Must answer

- Which methods combine teacher supervision and RL in the same loop?
- Which only use reward-only RLVR?
- When does OPD stabilize RL, and when does RL explore beyond the teacher?
- How do methods avoid forgetting, mode collapse, and long-CoT drift?

## WP5: Industrial and Systems Audit

### Mission

Verify industrial claims and systems-level OPD support.

### Current status

WP5 has been run and synthesized in [wp5-industrial-systems-synthesis.md](wp5-industrial-systems-synthesis.md). The merged result keeps industrial claims stage-scoped, downgrades Qwen3 confidence to medium pending exact mechanics, confirms NeMo RL and TRL GKDTrainer as strict-capable LLM framework paths, and keeps verl/OpenRLHF default recipes outside strict OPD.

### Seeds

Qwen3, Gemma, MiMo, Nemotron-Cascade, Speculative KD, DistillSpec, cross-tokenizer ULD, Minitron, multi-teacher routing, latency hiding, online teacher serving.

### Must answer

- Which official reports explicitly disclose on-policy teacher supervision?
- Which reports mention distillation and RL separately but not OPD?
- What are the compute bottlenecks?
- What systems tricks reduce teacher forward-pass cost?
- Which claims are too vague to classify?

## WP6: Agentic OPD

### Mission

Map OPD in multi-turn, stateful, tool-using, and environment-interacting agents.

### Current status

The WP6 priority pass and candidate line audit have both been merged. See [wp6-agentic-opd-synthesis.md](wp6-agentic-opd-synthesis.md), [wp6-agentic-candidate-line-audit-synthesis.md](wp6-agentic-candidate-line-audit-synthesis.md), and [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md). The merge confirms strict agentic seeds for MAD-OPD/OPAD, SOD, OPCD, VLA-OPD, GUI-SD, and the LiteGUI guided-OPD stage; keeps OEL and RLSD partial; keeps TCOD, Skill-SD, and OpenClaw-RL OPD component borderline; downgrades OEC to adjacent; and expands the adjacent ledger for reward-only agent RL, offline tool imitation, DAgger-style expert continuation, DGPO, RISE, and SAD. GLM-5 cross-stage OPD remains held out pending section-level primary-source evidence.

### Seeds

SCoRe, OEL, privileged information distillation, tool-call-level supervision, environment-state feedback, browser/code/GUI agents.

### Must answer

- Does the student generate environment trajectories?
- Does the teacher correct the student's actual visited state?
- Is feedback token-level, tool-call-level, trajectory-level, or earliest-error correction?
- How are safety constraints handled during on-policy exploration?

## WP7: Multimodal Frontier

### Mission

Extend the LLM OPD map into VLM, MLLM, video, speech, VLA, GUI, and multimodal agents without confusing RLVR/offline KD with OPD.

### Current status

The WP7 broad audit and candidate line audit have been run and synthesized in [wp7-multimodal-frontier-synthesis.md](wp7-multimodal-frontier-synthesis.md), [wp7-candidate-line-audit-synthesis.md](wp7-candidate-line-audit-synthesis.md), and [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md). The merge confirms the existing strict multimodal core, upgrades GUI-SD to high-confidence/source-verified for GUI grounding, downgrades PRISM to adjacent response-level adversarial alignment, adds KEPO/D-OPSD/Flow-OPD as borderline multimodal candidates, classifies GTR-Turbo as adjacent, and keeps reward-only VLM RLVR/offline MLLM KD/industrial reports outside strict OPD.

### Seeds

VOLD, Video-OPD, X-OPD speech, VLA-OPD, GUI-SD, LiteGUI, PRISM, Uni-OPD, VLM-R1-style RLVR, Perception-R1, LLaVA-KD, LLAVADI, VPD, Qwen-VL, InternVL, Gemma 3, Gemini, MiniCPM-V.

### Must answer

- Is there strict multimodal OPD beyond known seeds?
- Does the teacher supervise student multimodal/action trajectories or only generate cold-start data?
- Which domains remain empty: document, chart, GUI, medical, long video, embodied?
- What modality-specific failure modes need taxonomy support?

## WP8: Evaluation Methodology

### Mission

Design evaluation coverage beyond static benchmark accuracy.

### Current status

WP8 has been run and synthesized in [wp8-evaluation-methodology-synthesis.md](wp8-evaluation-methodology-synthesis.md). The merge expands `tables/benchmarks.md` across text, VLM, GUI/agent, video, audio, robotics, and diffusion/flow evaluation, and adds the rule that benchmark accuracy must be paired with rollout/recovery, calibration, distributional-fidelity, or teacher-uncertainty diagnostics.

### Must answer

- Which benchmarks measure exposure-bias reduction rather than only accuracy?
- How should OOD prompts, adversarial variants, calibration, hallucination, and teacher uncertainty preservation be measured?
- Which benchmarks test perception versus reasoning for VLM OPD?
- Which benchmarks are contaminated, private, or unsuitable for OPD claims?
- What evaluation bundles should this repo recommend for LLM, VLM, VLA, and agent OPD?

## WP9: Red-Team Verification

### Mission

Downgrade false positives. This agent should be skeptical and should not discover new papers.

### Input

Rows marked `strict_opd` or `borderline_strict`.

### Checks

1. Does the current student/policy generate training trajectories?
2. Is teacher/expert/discriminator/reference/privileged-context supervision applied to those trajectories?
3. Does the objective use that supervision?
4. Is the evidence direct or inferred from vague wording?
5. Is a simpler adjacent label more accurate: RLVR, RLHF, DPO, offline KD, synthetic data, self-training, imitation learning?

### Deliverables

Return a verifier memo and a table:

`id | current_label | verified_label | pass_or_fail | evidence_for_rollout | evidence_for_teacher_signal | evidence_for_objective | missing_evidence | downgrade_reason | verifier_notes`

## Recommended Execution Order

1. Run WP0 first with two independent agents.
2. Run WP1 after WP0 to stabilize the white-box baseline.
3. Run WP9 on every strict/borderline row from WP0/WP1.
4. Run WP2-WP4 for black-box, self-play, and OPD+RL. Current pass is merged.
5. Run WP5 only with official sources open; industrial claims are easy to overstate. Current pass is merged.
6. Run WP6 for agentic and interactive OPD before broad multimodal expansion. Current pass is merged.
7. Run WP7 broad audit before WP8. Current broad and candidate passes are merged.
8. Run WP8 to define modality-aware evaluation bundles. Current pass is merged.
9. Run WP9 again before merging any future large batch. The post-WP8 WP9 verifier is complete; the next recommended work is local consistency/release cleanup or a new gap-focused research package.
