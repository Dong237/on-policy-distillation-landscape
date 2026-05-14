# WP7: Multimodal Frontier — Deep Research Memo

**Agent:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Scope:** VLM, MLLM, video, speech, VLA, GUI, and multimodal-agent OPD methods.

---

## 1. Concise Memo

### 1.1 Strict OPD Seeds (7 confirmed, 0 new promotions)

All 7 existing strict multimodal seeds are **confirmed with no classification changes**:

| Seed | Modality | Status | Key Update |
|---|---|---|---|
| **VOLD** | VLM (visual reasoning) | strict_opd, v2 | ICLR 2026 withdrawn; evaluates 8 benchmarks (not 4) |
| **Video-OPD** | Video-language grounding | strict_opd, v1 | Teacher is Qwen3-VL-32B (GRPO post-trained), not generic "frontier" |
| **X-OPD** | Speech/LLM cross-modal | strict_opd, v2 | Submitted to Interspeech 2026; identifies "knowledge gap" (too-large teacher hurts) |
| **Uni-OPD** | LLM + MLLM (broadest) | strict_opd, v1 | 10 MLLM benchmarks across math/logic/document; confirmed broadest multimodal OPD |
| **VLA-OPD** | VLA (robotic manipulation) | strict_opd, v1 | Reverse-KL on action tokens; catastrophic forgetting analysis |
| **GUI-SD** | GUI grounding (VLM) | strict_opd (**upgraded** from medium → high confidence) | Pure weighted reverse-KL, no GRPO; identifies "Distillation-to-SFT Collapse" |
| **LiteGUI** | GUI agent (VLM) | strict_opd (WP6-verified) | GKD Stage 1 strict; Stage 2 GRPO is separate |

### 1.2 Existing Borderline (1 confirmed)

| Seed | Modality | Status |
|---|---|---|
| **PRISM** | MLLM reasoning | borderline_strict (response-level MoE discriminator, not token-level KL) |

### 1.3 New Borderline Candidates (4 found, all need line audit)

| Method | arXiv | Modality | Why Borderline | Needs |
|---|---|---|---|---|
| **KEPO** | 2602.00400 | Medical VLM | Quality-gated OPD; C1/C2 clear, C3 needs exact loss verification | Line audit of distillation loss |
| **D-OPSD** | 2605.05204 | Text-to-image diffusion | On-policy self-distillation; C3 uses MSE on velocity fields (reverse-KL equivalent in SDE) | Verify MSE↔KL equivalence claim |
| **Flow-OPD** | 2605.08063 | Text-to-image flow matching | Multi-teacher dense velocity supervision; reverse-KL equivalence formally established | Verify multi-teacher mechanism |
| **GTR-Turbo** | 2512.13043 | VLM agent (visual envs) | Merged-checkpoint teacher provides KL on thought tokens; PPO on action tokens | Verify thought-token KL is direct loss, not reward |

### 1.4 Adjacent / Not-OPD (confirmed, no changes)

**Reward-only VLM RLVR (adjacent):** VLM-R1, Perception-R1, R1-VL/StepGRPO, Vision-R1, DeepVideo-R1, VTool-R1, ManipLVM-R1, LaViPlan, VLAA-Thinker

**Offline MLLM KD (not_opd):** LLaVA-KD, LLAVADI, Visual Program Distillation, Switch-KD, TE-VLM

**Cross-modal transfer / mixed pipeline (adjacent):** OpenVLThinker

**Industrial (no multimodal OPD disclosed):** Qwen3-VL, InternVL3.5, Gemma 3, Gemini, MiniCPM-V, Cambrian-1, Molmo, Pixtral

### 1.5 New Adjacent (found in search)

| Method | arXiv | Modality | Why Adjacent |
|---|---|---|---|
| **Switch-KD** | 2604.14629 | VLM | Offline distillation, no student rollouts |
| **Dual-Process Image Gen** | 2506.01955 | Image gen + VLM | VLM provides scalar rating reward, not distributional signal |
| **OPD for AV Planning** | 2604.07944 | LLM (driving) | Text-only GKD on language-tokenized trajectories, not truly multimodal |

---

## 2. Evidence Ledger

### 2.1 Strict Seeds (Verified)

| method | source_url | exact_section | modality | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence | suggested_repo_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOLD | https://arxiv.org/abs/2510.23497 | Stage 2, KL loss term | VLM | Qwen2.5-VL-3B generates on-policy reasoning traces | Qwen3-8B text-only teacher provides token-level KL on same traces | L_VOLD = L_GRPO + beta * E[sum D_KL(pi_teacher \|\| pi_student)]; reverse-KL term is distributional | white_box (text LLM logits on VLM student rollouts) | token | kl_plus_grpo | current_policy | strict_opd | none | high | Update benchmarks to 8 (add MMStar, MathVerse, DynaMath, WeMath) |
| Video-OPD | https://arxiv.org/abs/2602.02994 | main method | MLLM (video) | Qwen3-VL-8B generates current-policy video grounding trajectories | Qwen3-VL-32B (GRPO post-trained) provides dense token-level supervision via r_t = -(log pi_student - log pi_teacher) | Reverse-KL with TVDF (Teacher-Validated Disagreement Focusing); pure distillation | white_box (multimodal teacher logits) | token | reverse_kl | current_policy | strict_opd | none | high | Clarify teacher as "Qwen3-VL-32B (GRPO post-trained)" not generic "frontier" |
| X-OPD | https://arxiv.org/abs/2603.24596 | main method | Speech/LLM | Qwen3-Omni-A3B generates rollouts from speech and text inputs | Qwen3-A3B or Qwen3-A22B text teacher provides token-level scoring on student trajectories | Two-component: in-modal loss + cross-modal loss, combined with KL for dynamic credit assignment | white_box (text LLM logits on speech student rollouts) | token | cross_modal_kl | current_policy | strict_opd | none | high | Note "knowledge gap" finding (too-large teacher hurts); submitted Interspeech 2026 |
| Uni-OPD | https://arxiv.org/abs/2605.03677 | main method | LLM + MLLM | Qwen3-VL-2B/4B generate on-policy trajectories with data balancing | Domain-specific teachers (Qwen3-VL-4B post-trained) provide token-level KL guidance | Token-level KL with outcome-guided margin calibration (margin mask + margin shift) | white_box (domain teacher logits) | token | token_kl_with_margin_calibration | current_policy | strict_opd | none | high | Confirmed broadest multimodal OPD (10 MLLM benchmarks: MathVision, DynaMath, WeMath, LogicVista, VisuLogic, AI2D, ChartQA, DocVQA, InfoVQA) |
| VLA-OPD | https://arxiv.org/abs/2603.26666 | main method | VLA | Student VLA (from OpenVLA-OFT) generates trajectories in LIBERO/RoboTwin2.0 | Frozen expert teacher (SimpleVLA-RL) provides action logits at every student-visited state | Reverse-KL on action tokens; formal comparison of FKL vs HardCE vs RKL | white_box (expert VLA action logits) | action-token | reverse_kl | current_policy | strict_opd | none | high | No changes needed |
| GUI-SD | https://arxiv.org/abs/2605.00642 | S2, S4.1-4.2, Eq. 1-6 | VLM (GUI grounding) | "the student generates an on-policy trajectory under x"; L(theta) = E_{y ~ P_S}[...]; single rollout per input | "the teacher, conditioned on both (x, r), produces step-wise target distributions along the same trajectory"; P_T(y_t) := pi_theta(y_t \| x, r, y_{<t}); teacher sees red bbox + Gaussian soft mask | Eq. 3: L = E_{y ~ P_S}[(1/\|y\|) sum_t w(t) D_KL(P_S(y_t) \|\| P_T(y_t))]; weighted reverse-KL with positional + entropy gating. **No GRPO.** | privileged_self (same weights + visual bbox/mask + textual hint) | token (coordinate digit tokens) | weighted_reverse_kl | current_policy | strict_opd | none; **upgraded** from medium to high confidence | high | Upgrade confidence to high; note "Distillation-to-SFT Collapse" finding |
| LiteGUI | https://arxiv.org/abs/2605.07505 | S3.2, Eq. 10-12, A.5 | VLM (GUI agent) | "the student always generates the response that is used for distillation" (A.5); lambda=1.0 (100% on-policy) | "the teacher distribution is conditioned on pi_T(.\|x_t, g_t)"; Qwen3-VL-32B with oracle ground-truth actions; Most-Matched-GT dynamically selects closest ground truth to student's output | Eq. 12: reverse-KL (GKD-style); Stage 1 only. Stage 2 GRPO is separate. | white_box + privileged_context (oracle ground-truth valid actions) | token | reverse_kl (GKD) | current_policy | strict_opd (Stage 1) | Stage 2 GRPO is reward-only RL, cleanly separated | high | Already in VLM table; no changes needed |

### 2.2 Borderline Strict (Existing)

| method | source_url | exact_section | modality | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence | suggested_repo_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRISM | https://arxiv.org/abs/2604.28123 | main method | MLLM | current-policy responses are sampled | MoE discriminator supervises those response rollouts | adversarial objective consumes response-level discriminator signal + GRPO advantages, not token logits | black_box (MoE discriminator) | sequence | adversarial_distribution_alignment | current_policy | borderline_strict | response-level MoE discriminator is not dense token/logit supervision; signal consumed as reward/advantages | high | No changes needed |

### 2.3 New Borderline Candidates (Need Line Audit)

| method | source_url | exact_section | modality | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence | suggested_repo_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KEPO | https://arxiv.org/abs/2602.00400 | not verified | Medical VLM | "on-policy trajectories"; "rejectively sample reward-positive on-policy trajectories" | "quality-gated on-policy distillation objective that selectively applies dense teacher guidance only to high-quality trajectories" | described as KD + preference optimization; exact loss equation not verified | white_box (Qwen3-VL-32B teacher logits) | likely token | kd_plus_preference | likely current_policy | borderline_strict (candidate) | C3 exact loss unverified; quality-gating introduces selectivity; may mix KD with preference optimization | medium | Add to candidate queue for WP9 line audit; first medical VLM OPD |
| D-OPSD | https://arxiv.org/abs/2605.05204 | Eq. 7 | Text-to-image diffusion | "minimizes the two predicted distributions over the student's own roll-outs" | same model as teacher conditioned on multimodal features (text + target image) supervises student conditioned on text-only | L = E[1/K sum \|\|u_k^s - sg(u_k^t)\|\|_2^2]; MSE on velocity fields; authors claim reverse-KL equivalence in SDE framework | privileged_self (same model + target image as privileged context) | step-level (denoising steps) | mse_velocity_field (reverse_kl_equivalent) | current_policy | borderline_strict (candidate) | C3 uses MSE on velocity fields, not explicit KL; reverse-KL equivalence claimed but is domain-specific (SDE framework) | medium | Add to candidate queue; extends OPD to diffusion models; new modality axis |
| Flow-OPD | https://arxiv.org/abs/2605.08063 | main method | Text-to-image flow matching | on-policy sampling in "three-step orchestration" | multi-teacher dense velocity field supervision on student denoising paths | "Reverse KL in SDE framework translates to time-weighted L2 distance between vector fields"; r_t = -w(t) \|\|v_student - v_teacher\|\|^2 | white_box (multi-teacher velocity fields) | step-level (denoising steps) | time_weighted_l2_velocity (reverse_kl_equivalent) | current_policy | borderline_strict (candidate) | Same MSE↔KL question as D-OPSD; multi-teacher mechanism needs verification | medium | Add to candidate queue; first multi-teacher OPD for flow matching |
| GTR-Turbo | https://arxiv.org/abs/2512.13043 | main method | VLM agent (visual envs) | VLM student generates actions via RL training | merged-checkpoint teacher provides negative KL divergence as "thought reward" via soft logit distillation | KL-based logit distillation for thought tokens; PPO for action tokens; "soft logit distillation imposes a more relaxed constraint" | privileged_self (merged RL checkpoints via TIES merging) | token (thought tokens only) | kl_plus_ppo_hybrid | current_policy | borderline_strict (candidate) | KL may be consumed as reward ("thought reward") rather than direct loss; action tokens use PPO reward only; method-level is hybrid | medium | Add to candidate queue; verify whether thought-token KL is direct loss or reward-shaped |

### 2.4 Adjacent / Not-OPD

| method | source_url | modality | final_label | reason_to_downgrade | confidence | suggested_repo_action |
| --- | --- | --- | --- | --- | --- | --- |
| VLM-R1 | https://arxiv.org/abs/2504.07615 | VLM | adjacent | Reward-only GRPO; no teacher distribution on rollouts | high | Already tracked |
| Perception-R1 | https://arxiv.org/abs/2506.07218 | MLLM | adjacent | Perception reward for GRPO; no teacher distillation | high | Already tracked |
| R1-VL / StepGRPO | https://arxiv.org/abs/2503.12937 | VLM | adjacent | Step-wise rule rewards; still reward-only | medium | Already tracked |
| Vision-R1 | https://arxiv.org/abs/2503.06749 | VLM | adjacent | CoT cold start + GRPO; reward-only RL | medium | Already tracked |
| DeepVideo-R1 | https://arxiv.org/abs/2506.07464 | Video | adjacent | R1-style training for video; no teacher distillation | medium | Already tracked |
| VTool-R1 | https://arxiv.org/abs/2505.19255 | VLM | adjacent | Outcome/tool reward only; no teacher distillation | medium | Already tracked |
| OpenVLThinker | https://arxiv.org/abs/2503.17352 | VLM | adjacent | Cross-modal transfer via offline SFT; online stage is reward-only | medium | Already tracked |
| ManipLVM-R1 | https://arxiv.org/abs/2505.16517 | Embodied VLM | adjacent | Task outcome rewards; no teacher distillation | medium | Already tracked |
| LaViPlan | https://arxiv.org/abs/2507.12911 | Driving VLM | adjacent | Planning reward; not distillation-style teacher supervision | medium | Already tracked |
| VLAA-Thinker | https://arxiv.org/abs/2504.11468 | VLM | adjacent | SFT/RL tradeoff study; no teacher distillation on rollouts | high | Already tracked |
| LLaVA-KD | https://arxiv.org/abs/2410.16236 | MLLM | not_opd | Offline KD on fixed data; C1 fails | high | Already tracked |
| LLAVADI | https://arxiv.org/abs/2407.19409 | MLLM | not_opd | Offline systematic MLLM KD study | high | Already tracked |
| Visual Program Distillation | https://arxiv.org/abs/2312.03052 | VLM | not_opd | Teacher/tool pipeline generates offline data; C1 fails | high | Already tracked |
| Switch-KD | https://arxiv.org/abs/2604.14629 | VLM | not_opd | Offline distillation; novel architecture but no student rollouts | high | Add to adjacent table |
| TE-VLM | OpenReview (withdrawn) | VLM | not_opd | Offline CLIP embedding distillation; withdrawn from ICLR 2026 | medium | Do not add |
| Dual-Process Image Gen | https://arxiv.org/abs/2506.01955 | Image gen | adjacent | VLM provides scalar rating reward, not distributional signal | high | Add to adjacent table |
| OPD for AV Planning | https://arxiv.org/abs/2604.07944 | LLM (driving) | adjacent | Text-only GKD on language-tokenized trajectories; not truly multimodal | high | Note but do not add to VLM table (text-only domain application) |
| Qwen3-VL | https://arxiv.org/abs/2511.21631 | MLLM | adjacent (industrial) | No strict OPD training loop disclosed | medium | Already tracked |
| InternVL3.5 | https://internvl.github.io/blog/2025-08-26-InternVL-3.5/ | MLLM | adjacent (industrial) | Cascade RL without teacher distillation on rollouts | medium | Already tracked |
| Gemma 3 | https://arxiv.org/abs/2503.19786 | MLLM | adjacent (industrial) | Off-policy KD for multimodal; GKD-style OPD for text only | medium | Note: text OPD exists but multimodal component is off-policy KD |
| Gemini 2.5 | official docs | MLLM | adjacent (industrial) | Off-policy k-sparse distribution distillation; "online distillation" may be iterative offline | low | Insufficient evidence for any multimodal OPD |
| MiniCPM-V | various releases | MLLM | not_opd (industrial) | No OPD disclosed in any release; uses RLAIF-V | medium | No action |
| Cambrian-1 | https://arxiv.org/abs/2406.16860 | MLLM | not_opd (industrial) | Explicitly avoids proprietary distillation; two-stage SFT | high | No action |
| Molmo / Molmo 2 | https://arxiv.org/abs/2409.17146 | MLLM | not_opd (industrial) | "Built without distilling from proprietary systems"; two-stage SFT only | high | No action |
| Pixtral | Mistral docs | MLLM | not_opd (industrial) | Cascade distillation is off-policy only (prune-distill-repeat) | medium | No action |

---

## 3. Candidate Table-Row Suggestions

### 3.1 Rows to Add to `tables/vlm_opd_papers.md` (only after line audit)

#### KEPO (borderline_strict candidate)
```
id: kepo-2026
title: KEPO: Knowledge-Enhanced Preference Optimization for Medical VLM
year: 2026
paper_url: https://arxiv.org/abs/2602.00400
modality: VLM
domain: medical_visual_reasoning
opd_strictness: borderline_strict (candidate, needs line audit)
teacher_kind: larger_vlm (Qwen3-VL-32B)
student_model: Qwen3-VL-2B
rollout_source: student_on_policy
supervision_granularity: token (likely)
loss_objective: kd_plus_preference_optimization
strictness_evidence: C1 on-policy trajectories; C2 quality-gated dense teacher guidance; C3 exact loss unverified
benchmarks: OmniMedVQA (8 imaging modalities)
confidence: medium
notes: First medical VLM OPD. Quality-gating is novel. Needs line audit for C3 exact loss.
vision_task: medical_visual_reasoning
visual_input_type: medical_images (MRI, CT, X-Ray, US, Dermoscopy, Fundus, OCT, Microscopy)
```

#### D-OPSD (borderline_strict candidate)
```
id: d-opsd-2026
title: D-OPSD: On-Policy Self-Distillation for Step-Distilled Diffusion Models
year: 2026
paper_url: https://arxiv.org/abs/2605.05204
code_url: https://github.com/vvvvvjdy/D-OPSD
modality: Text-to-image (diffusion)
domain: image_generation
opd_strictness: borderline_strict (candidate, needs line audit)
teacher_kind: privileged_self (same model + target image)
rollout_source: student_on_policy
supervision_granularity: step (denoising steps)
loss_objective: mse_velocity_field (reverse_kl_equivalent_in_sde)
strictness_evidence: C1 student rollouts in denoising; C2 privileged self-teacher (text+image vs text-only); C3 MSE on velocity fields
confidence: medium
notes: Extends OPD to diffusion models. MSE↔KL equivalence is domain-specific. New modality axis.
vision_task: text_to_image_generation
```

#### Flow-OPD (borderline_strict candidate)
```
id: flow-opd-2026
title: Flow-OPD: On-Policy Distillation for Flow Matching Models
year: 2026
paper_url: https://arxiv.org/abs/2605.08063
code_url: https://github.com/CostaliyA/Flow-OPD
project_url: https://costaliya.github.io/Flow-OPD/
modality: Text-to-image (flow matching)
domain: image_generation
opd_strictness: borderline_strict (candidate, needs line audit)
teacher_kind: multi_teacher (GenEval, OCR, DeQA, PickScore specialized teachers)
rollout_source: student_on_policy
supervision_granularity: step (denoising steps)
loss_objective: time_weighted_l2_velocity (reverse_kl_equivalent)
strictness_evidence: C1 on-policy denoising; C2 multi-teacher dense velocity supervision; C3 reverse-KL equivalence established
confidence: medium
notes: First multi-teacher OPD for flow matching. SD3.5 Medium base.
vision_task: text_to_image_generation
```

### 3.2 Updates to Existing Rows

| Row | Field | Old Value | New Value | Reason |
|---|---|---|---|---|
| gui-sd-2026 | confidence | medium | high | GUI-SD line audit confirms all C1/C2/C3 with exact quotes |
| gui-sd-2026 | verification_status | needs_review | verified | Line audit complete |
| gui-sd-2026 | conflict_status | human_spotcheck_needed | source_verified | Line audit confirms strict |
| vold-2025 | benchmarks | MMMU-Pro; MathVision; MathVista; LogicVista | MMMU-Pro; MathVision; MathVista; LogicVista; MMStar; MathVerse; DynaMath; WeMath | 8 benchmarks confirmed |
| video-opd-2026 | teacher_model | frontier teacher | Qwen3-VL-32B (GRPO post-trained) | Verified specific teacher model |

---

## 4. Gap Map

### 4.1 Empty Multimodal Domains (No Strict OPD Found)

| Domain | Status | Closest Method | Gap |
|---|---|---|---|
| **Medical VLM** | No strict (KEPO is borderline candidate) | KEPO (2602.00400) | Quality-gated OPD on OmniMedVQA; C3 needs verification |
| **Document/chart VLM** | Covered by Uni-OPD (AI2D, ChartQA, DocVQA, InfoVQA) | Uni-OPD | Uni-OPD benchmarks include documents; no dedicated document-OPD method |
| **Text-to-image generation** | No strict (D-OPSD, Flow-OPD are borderline) | D-OPSD, Flow-OPD | MSE↔KL equivalence in SDE framework needs assessment for OPD taxonomy |
| **Long video understanding** | Empty | Video-OPD (temporal grounding only) | Video-OPD handles grounding, not long-video QA/summarization |
| **3D/point-cloud VLMs** | Empty | None found | No OPD for 3D vision-language models |
| **Multimodal dialogue** | Empty | None found | Multi-turn VLM conversation with on-policy teacher distillation is unstudied |
| **Audio-visual / video+speech** | Empty | X-OPD (speech-only) | No joint audio-visual OPD |

### 4.2 Multimodal-Specific Failure Modes (from sources)

| Failure Mode | Identified By | Description |
|---|---|---|
| **Distillation-to-SFT Collapse** | GUI-SD | Giving teacher the answer as text collapses distillation to SFT (teacher entropy → 0.17). Visual privilege preserves soft distributions (entropy → 0.50). |
| **Knowledge gap** | X-OPD | Too-large teacher (A22B vs A3B) can hinder alignment, unlike "bigger is better" assumption. |
| **Sparse reward in long-horizon video** | Video-OPD | Temporal grounding over long videos suffers from poor credit assignment with sparse rewards. |
| **Multi-rollout overhead from long visual contexts** | Video-OPD | Video inputs create expensive multi-rollout costs. |
| **Text-visual teacher mismatch** | VOLD | Text-only teacher on VLM student creates modality mismatch risk. |
| **Vision token mismatch** | NOT identified by any paper | Open gap: no paper discusses how visual token representations affect OPD alignment. |
| **Visual hallucination under OPD** | NOT identified by any paper | Open gap: whether OPD exacerbates or mitigates visual hallucination is unstudied. |

### 4.3 Methods Needing Line-Level Audit (WP9 Queue)

| Method | Priority | What to Verify |
|---|---|---|
| **KEPO** | High | C3 exact loss equation; verify quality-gating mechanism; is it KL + preference, or reward-weighted KD? |
| **D-OPSD** | Medium | Verify MSE↔reverse-KL equivalence proof; confirm student generates denoising rollouts |
| **Flow-OPD** | Medium | Verify multi-teacher mechanism; confirm reverse-KL equivalence; check if rewards are scalar or distributional |
| **GTR-Turbo** | Medium | Verify thought-token KL is a direct loss term (not reward-shaped via PPO); confirm merged-checkpoint teacher mechanism |

### 4.4 Industrial OPD Status

| Organization | Text OPD | Multimodal OPD | Evidence |
|---|---|---|---|
| **Qwen (Alibaba)** | Strict (Qwen3 OPD stage) | Not disclosed | Qwen3-VL report does not mention OPD |
| **Google** | Strict (Gemma 2 GKD) | Off-policy KD only | Gemma 3 uses 256-logit teacher for pre-training (off-policy) |
| **NVIDIA** | Strict (Nemotron-Cascade2 MOPD) | Not disclosed | No multimodal MOPD variant |
| **Xiaomi** | Strict (MiMo-V2-Flash MOPD) | Not disclosed | No multimodal extension |
| **Zhipu AI** | Stage-strict (GLM-5) | Not disclosed | Cross-stage OPD for text reasoning/general, not multimodal |
| **OpenGVLab** | N/A | Not disclosed | InternVL3.5 uses cascade RL, not OPD |
| **OpenBMB** | N/A | Not disclosed | MiniCPM-V uses RLAIF-V |
| **AI2** | N/A | Explicitly avoided | Molmo built "without distilling from proprietary systems" |
| **Mistral** | N/A | Off-policy cascade only | Pixtral uses iterative prune-distill (off-policy) |

### 4.5 Modality Coverage Matrix

| Modality | Strict Seeds | Borderline | Adjacent | Not OPD |
|---|---|---|---|---|
| **VLM (image+text reasoning)** | VOLD, Uni-OPD | KEPO (candidate) | VLM-R1, Vision-R1, R1-VL, OpenVLThinker, VLAA-Thinker | — |
| **Video-language** | Video-OPD | — | DeepVideo-R1 | — |
| **Speech/audio-language** | X-OPD | — | — | — |
| **VLA/embodied** | VLA-OPD | — | ManipLVM-R1, LaViPlan | — |
| **GUI grounding** | GUI-SD | — | — | — |
| **GUI agent (multi-step)** | LiteGUI | — | — | — |
| **MLLM (multi-domain)** | Uni-OPD | PRISM | Perception-R1, VTool-R1 | LLaVA-KD, LLAVADI, VPD |
| **Text-to-image diffusion** | — | D-OPSD, Flow-OPD (candidates) | Dual-Process | — |
| **VLM agent (visual envs)** | — | GTR-Turbo (candidate) | — | — |
| **Medical VLM** | — | KEPO (candidate) | — | — |
| **Document/chart** | Uni-OPD (subsumes) | — | — | — |
| **Industrial multimodal** | — | — | Qwen3-VL, InternVL3.5, Gemma 3, Gemini, MiniCPM-V | Cambrian-1, Molmo, Pixtral |

---

## 5. Summary Statistics

| Category | Count |
|---|---|
| Strict OPD (verified) | 7 (VOLD, Video-OPD, X-OPD, Uni-OPD, VLA-OPD, GUI-SD, LiteGUI) |
| Borderline strict (verified) | 1 (PRISM) |
| Borderline strict (new candidates, need line audit) | 4 (KEPO, D-OPSD, Flow-OPD, GTR-Turbo) |
| Adjacent (reward-only / industrial) | 16 |
| Not OPD (offline KD / no training) | 8 |
| Industrial (no multimodal OPD disclosed) | 8 organizations checked |
| **Total multimodal methods audited** | 36 |

---

*End of WP7 multimodal frontier memo.*
