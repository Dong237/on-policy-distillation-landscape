# WP6: Agentic and Interactive OPD — Deep Research Memo

**Agent:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Scope:** Multi-turn, tool-using, code, browser/GUI, environment-interacting, and trajectory-level methods where the student policy generates states and a teacher/expert/privileged model supervises those exact states.

---

## 1. Executive Summary

WP6 audited 60+ methods spanning self-correction, code/SWE agents, browser/GUI agents, tool-use distillation, multi-turn agent training, environment-feedback methods, and multi-agent debate. The agentic OPD landscape is **sparse but rapidly emerging** (all strict seeds are from 2026).

**Strict OPD seeds found: 6** (3 already tracked, 3 new)
- Already tracked: MAD-OPD, SOD, OPCD
- New candidates: **TCOD**, **LiteGUI**, **Skill-SD**

**Borderline strict: 3** (1 already tracked, 2 new)
- Already tracked: OEL (consolidation substage strict)
- New: **OEC** (DAgger-style for SWE agents), **OpenClaw-RL HG-OPD** (hindsight-guided self-teacher)

**Key correction:** SAD (Structured Agent Distillation, arXiv 2505.13820) was classified strict in a legacy survey but is **not_opd** — it uses teacher-forced training on pre-collected teacher trajectories (offline KD).

**Dominant false-positive patterns:**
1. Reward-only RL on agent trajectories (SCoRe, DigiRL, AgentQ, CodeRL, RLTF, SWE-RL, DeepSWE, etc.)
2. SFT on teacher-generated agent traces (Toolformer, Gorilla, ToolLLM, FireAct, CogAgent, AgentTuning, etc.)
3. DPO/preference optimization on trajectory pairs (ETO, IPR, AlphaLLM-CPL)
4. Inference-time self-correction with no training (Reflexion, Self-Refine)

---

## 2. Research Memo

### 2.1 What Was Searched

Six parallel research agents investigated:
1. Self-correction methods: SCoRe, Reflexion, Self-Refine, RISE, GLoRE, SimpleTIR
2. Code/SWE agent training: CodeRL, RLTF, StepCoder, CodeAct, OpenHands, SWE-Gym, SWE-RL, DeepSWE, SWE-smith, Agentless, SAD, SOD, DGPO, Chain-of-Agents, SWE-Master, GLM-5
3. Browser/GUI agent training: WebArena, BrowserGym, CogAgent, OS-Atlas, OSWorld, WebGPT, DigiRL, AgentQ, AgentTrek, ScreenAgent, ShowUI, Agent-FLAN, AgentTuning, UI-TARS, Mem-W, TCOD, LiteGUI
4. Tool-use and multi-turn distillation: Toolformer, Gorilla, ToolLLM, FireAct, VTool-R1, SAD, Skill-SD, OpenClaw-RL, DGPO, ODIA, AgentArk, agent distillation papers, ETO, credit assignment methods
5. Environment-feedback and RL agents: DAgger, OEC, Hybrid RL+IL, AgentQ, RLEF, LLM4Teach, ETO, IPR, pi-Distill, SDPO, RLSD, HDPO, AlphaLLM-CPL, ILF, SCoRe, HINT, DEPS
6. Multi-agent debate and existing seeds: MAD-OPD, SOD, OPCD, OEL verification; Latent Agents/IMAD, AgentArk, D&R, ProductResearch, Skill-SD, RLSD, SD-Zero, cascade routing

### 2.2 What Was Found — Strict OPD

#### Already tracked (confirmed):

**MAD-OPD** (arXiv 2605.01347) — Multi-teacher debate provides token-level JSD/RKL supervision on student on-policy states. Agentic benchmarks: BFCL-v4, tau2-Bench, VitaBench. Models: Qwen3/3.5 1.7B–14B students, 8B–32B teachers. The OPAD variant extends to multi-step agentic tasks by having teachers debate at each decision point. **Confirmed strict_opd.**

**SOD** (arXiv 2605.07725) — Step-wise adaptive reweighting of token-level teacher-logit OPD for tool-integrated reasoning (TIR). Step-divergence ratios attenuate misleading teacher signals in high-divergence regions caused by cascading tool-call errors. Benchmarks: AIME 2024/2025, GPQA-Diamond, LiveCodeBench. GRPO is auxiliary. **Confirmed strict_opd.**

**OPCD** (arXiv 2602.12275) — Privileged-context self-teacher (same model with vs. without context) provides token-level reverse-KL on student-generated trajectories. Applications: experiential knowledge distillation, system prompt distillation. Agentic evaluation limited to text games (Frozen Lake, Sokoban). **Confirmed strict_opd.**

#### New strict candidates:

**TCOD** (arXiv 2604.24005) — Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents. Student generates multi-turn trajectories in ALFWorld, WebShop, ScienceWorld; teacher computes token-level KL on student-generated action sequences at each turn; temporal curriculum progressively expands trajectory depth to mitigate inter-turn error compounding. Results: up to +18 points over vanilla OPD. Student can surpass teacher. Models: Qwen2.5-3B/7B, Qwen3-1.7B/4B students. **Candidate strict_opd. Needs line audit.**

**LiteGUI** (arXiv 2605.07505) — First GKD-style on-policy distillation for GUI agents. Student (Qwen3-VL-2B/3B) generates GUI action sequences; teacher (Qwen3-VL-32B) with oracle-guided privileged context computes token-level log-probabilities on student sequences; reverse-KL distillation objective. Two-stage: Stage 1 GKD + Stage 2 Multi-solution Dual-level GRPO. Benchmarks: ScreenSpot-Pro, OS-World, Lite-Bench. Claims "first attempt to apply distillation to the GUI agent domain." **Candidate strict_opd. Needs line audit.**

**Skill-SD** (arXiv 2604.10674) — Skill-Conditioned Self-Distillation for Multi-turn LLM Agents. Student generates multi-turn trajectories under plain prompt; privileged self-teacher (same model conditioned on extracted skills from completed trajectories) re-scores those exact trajectories; importance-weighted reverse-KL loss provides dense token-level supervision. Benchmarks: AppWorld (+14.0%), Sokoban (+10.9%). **Candidate strict_opd. Needs line audit.**

### 2.3 What Was Found — Borderline Strict

**OEC** (arXiv 2512.14895) — On-policy Expert Corrections. Most directly DAgger-inspired LLM agent method. Student generates first K turns of a multi-turn SWE trajectory; expert completes remaining turns from student-visited state. Loss is NLL on expert completions (student turns masked). Structurally very close to OPD but uses hard action labels rather than soft distribution matching. 14% improvement over pure behavioral cloning on SWE-bench. **Borderline_strict.**

**OpenClaw-RL HG-OPD** (arXiv 2603.10165) — Hindsight-Guided OPD component. Student generates live agent trajectories; self-teacher (same model with hint-augmented context from next state) provides token-level log-probability gap as directional supervision. However, full system also combines with Binary RL (scalar PRM rewards), making method-level classification hybrid. **Borderline_strict for HG-OPD component. Needs line audit.**

**GLM-5** (arXiv 2602.15763) — Industrial cross-stage OPD for agentic/coding. Uses "On-Policy Cross-Stage Distillation" to prevent catastrophic forgetting across sequential RL stages. Details are terse — exact loss, teacher specification, and rollout mechanics not detailed. **Borderline_strict (industrial, insufficient detail).**

### 2.4 What Was Found — Partial OPD

**OEL** (arXiv 2603.16856) — Confirmed partial_opd. The OPCD consolidation substage satisfies C1/C2/C3 (student generates, knowledge-conditioned teacher supervises, reverse-KL objective). But full pipeline includes deployment collection (no teacher supervision) and knowledge extraction (LM summarization). Stage-specific labeling prevents strict for the full method.

**DGPO** (arXiv 2508.20324) — Student generates on-policy PPO trajectories; teacher KL penalties applied only on incorrect predictions. But teacher KL is consumed as scalar reward by PPO's advantage estimator, not as a direct distillation loss. The teacher acts as "reward shaper rather than imposing direct parameter-level constraints." **Partial_opd at best, adjacent also defensible.**

**RISE** (arXiv 2407.18219) — Student generates multi-turn self-correction attempts (C1 pass). Distillation variant has a teacher provide correct response TEXT at student-visited states. But C3 fails: objective is reward-weighted SFT (not KL/distribution matching). Paper explicitly disclaims: "this is different from the classic notion of knowledge distillation." **Adjacent.**

### 2.5 What Was Found — Adjacent (Key False Positives)

**Self-correction methods (all adjacent or not_opd):**
The self-correction field has explicitly moved AWAY from teacher dependence. SCoRe (Kumar et al., 2024) uses REINFORCE with binary correctness reward — no teacher logits, no distillation loss. The paper emphasizes independence from teacher supervision as a design goal. Reflexion and Self-Refine are inference-time prompting strategies with no weight updates. GLoRE trains via SFT on paired solutions with no external teacher.

**Code/SWE agent RL (all adjacent):**
The entire SWE-agent training ecosystem uses reward-only RL or rejection fine-tuning:
- CodeRL: REINFORCE + critic (token-level reward estimates, not teacher logits)
- RLTF: RL with compiler execution shaped reward
- StepCoder: PPO with compiler feedback + curriculum
- SWE-RL: RL with similarity-score reward
- DeepSWE: GRPO++ with binary test-pass reward, explicitly "no distillation"
- SWE-smith: GRPO with test-pass reward
- SWE-Gym: Iterative RFT with ORM verifier (YES/NO reward)

**Browser/GUI agent training (all SFT or reward-RL):**
- CogAgent, OS-Atlas, ScreenAgent, ShowUI: SFT on pre-collected GUI demonstrations
- Agent-FLAN, AgentTuning: SFT on agent interaction data
- DigiRL: Advantage-weighted RL with scalar task-completion reward
- AgentQ: MCTS + DPO with self-critique scores
- UI-TARS: Iterative SFT + DPO with human-filtered student trajectories (closest adjacent)
- WebGPT: Imitation learning + reward model rejection sampling

**Tool-use (all offline SFT):**
- Toolformer: Self-supervised data augmentation + SFT (one-shot, no iterative loop)
- Gorilla: SFT on GPT-4-generated API call demonstrations
- ToolLLM: SFT on ChatGPT-generated DFSDT solution paths
- FireAct: SFT on GPT-4-generated ReAct trajectories

### 2.6 What Was Ruled Out

**SAD (Structured Agent Distillation, arXiv 2505.13820):** Two independent agents confirmed this uses teacher-forced alignment on pre-collected teacher demonstrations. The student does NOT generate its own trajectories. This is offline behavioral cloning with span-aware KL, not on-policy distillation. A legacy survey misclassified it as strict. **Corrected to not_opd.**

**Self-Distilled RLVR / RLSD (arXiv 2604.03128):** Two agents reached conflicting conclusions. Agent 5 classified it strict_hybrid; Agent 6 classified it adjacent because "the teacher becomes a magnitude evaluator rather than a distributional target" and the paper argues pure OPSD causes "fatal information asymmetry" and "severe information leakage." The paper explicitly decouples direction (from reward) from magnitude (from teacher). Under strict OPD criteria, this is a reward-directed method where teacher evidence ratios modulate gradient magnitude but do not determine gradient direction. **Conservative classification: partial_opd.** The teacher's per-token evidence ratios are consumed by the objective (satisfying a weak C3), but the primary update direction is reward-based.

**LLM4Teach (arXiv 2311.13373):** Structurally strict OPD (student RL agent generates trajectories, LLM teacher provides soft action distributions, dual KD+RL loss). But this is for traditional RL agents (StarCraft, robotics scale), not LLM text agents. **Strict_opd in MARL domain. Out of core LLM scope but noted as foundational reference.**

**HINT (arXiv 2601.05407):** Strict OPD for cooperative MARL. Same domain limitation as LLM4Teach. **Out of core LLM scope.**

### 2.7 What Changed the Landscape

1. **Agentic OPD is a 2026 phenomenon.** All strict seeds (MAD-OPD, SOD, OPCD, TCOD, LiteGUI, Skill-SD) are from January–May 2026. No 2024 or earlier paper applies strict OPD to agents.

2. **Three distinct teacher patterns emerged for agentic OPD:**
   - **External teacher logits:** MAD-OPD (multi-teacher debate), SOD (step-weighted), TCOD (temporal curriculum), LiteGUI (oracle-guided)
   - **Privileged-context self-teacher:** OPCD, OEL, Skill-SD (skill-conditioned), OpenClaw-RL (hindsight-guided)
   - **DAgger-style expert action labels:** OEC (hard labels, borderline)

3. **Inter-turn error compounding is the central failure mode.** SOD addresses it with step-divergence reweighting; TCOD with temporal curriculum; MAD-OPD with per-step debate. This is the agentic analog of exposure bias in text OPD.

4. **GUI/web agent OPD just began.** LiteGUI (May 2026) claims "first attempt to apply distillation to the GUI agent domain." TCOD (April 2026) is the first multi-turn agent OPD on standard benchmarks (ALFWorld, WebShop, ScienceWorld).

5. **The DAgger→OPD connection is now explicit.** GKD cites DAgger as theoretical motivation. OEC is the most direct modern DAgger variant for LLM agents. The key gap: DAgger uses hard action labels; strict OPD uses soft distribution matching.

---

## 3. Evidence Ledger

### 3.1 Strict OPD Seeds

| method | source_url | exact_section | C1_quote | C2_quote | C3_quote | teacher_access | state_or_action_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAD-OPD | https://arxiv.org/abs/2605.01347 | S3.1, S4.1, S4.3 Eq.7-8 | "y_hat ~ pi_theta(.\|x): the student generates trajectories along its own distribution" | "teachers debate over the student's on-policy state...token-level supervision, with each teacher's contribution weighted by its post-debate confidence" | "L_{mad-opd}(theta) = E[sum_t sum_k w_k * D(p_{T_k} \|\| p_S)]" (Eq.8), JSD for agentic, RKL for code | multi_teacher_debate (K teachers, R=2 debate rounds) | token | jsd_or_reverse_kl | current_policy | strict_opd | none | high |
| SOD | https://arxiv.org/abs/2605.07725 | Eq.9-10 | student generates trajectories tau = (x, y_1, o_1, ..., y_K, o_K, y_{K+1}) where y_k are model-generated responses | teacher logits supervise student-visited states; l_OPD(y_t) = log pi_theta - log pi_teacher | L_OPD^step = E[sum_k w_k * sum_{t in I_k} l_OPD(y_t)]; L = L_GRPO + L_OPD^step | white_box (teacher logits) | token with step-level reweighting | reverse_kl + grpo | current_policy | strict_opd | none; GRPO is auxiliary | high |
| OPCD | https://arxiv.org/abs/2602.12275 | S3 Eq.1 | "let the student model pi_theta generate complete response trajectories y...generated without context c" | "evaluate it using the teacher model pi_teacher, which processes [c;x;y]" | "L(theta) = E[D_KL(pi_theta(.\|x,y_{<t}) \|\| pi_teacher(.\|c,x,y_{<t}))]" (Eq.1), top-256 vocab | privileged_self (same model ± context) | token | reverse_kl | current_policy | strict_opd | none | high |
| TCOD | https://arxiv.org/abs/2604.24005 | main method section | student (Qwen2.5-3B/7B, Qwen3-1.7B/4B) generates multi-turn agent trajectories in ALFWorld/WebShop/ScienceWorld | teacher (Qwen2.5-7B GRPO-tuned, Qwen3-30B-A3B) computes token-level KL on student-generated action sequences at each turn | D_KL(pi_teacher \|\| pi_student) on student on-policy states with temporal curriculum scheduling | white_box (teacher logits) | token with temporal curriculum | forward_kl | current_policy | strict_opd (candidate) | needs line audit to confirm rollout freshness | high |
| LiteGUI | https://arxiv.org/abs/2605.07505 | GKD stage description | "fully on-policy: the student always generates the response that is used for distillation" | teacher (Qwen3-VL-32B) with oracle reference trajectories as privileged context computes token-level log-probabilities on student sequences | reverse-KL-style objective between student and teacher token distributions on student-generated GUI actions, following GKD (Agarwal et al., 2024) | white_box + privileged_context (oracle-guided) | token | reverse_kl (GKD-style) | current_policy | strict_opd (candidate) | needs line audit; Stage 2 GRPO is separate | high |
| Skill-SD | https://arxiv.org/abs/2604.10674 | SDL loss definition | "Sample {tau_i} ~ pi_{theta_old}^stu(.\|x)" — trajectories from old student under plain prompt | "the same token sequence is re-scored under the student prompt and the skill-augmented teacher prompt" | L_SDL(theta) = (1/N) sum_i sum_t rho_{i,t}^on (e^{-l_{i,t}} - 1 + l_{i,t}); importance-weighted reverse-KL | privileged_self (skill-conditioned same model) | token | importance_weighted_reverse_kl | per_iteration (theta_old) | strict_opd (candidate) | skills extracted from completed trajectories (not current rollout) may introduce staleness; needs line audit | medium-high |

### 3.2 Borderline Strict

| method | source_url | exact_section | C1_quote | C2_quote | C3_quote | teacher_access | state_or_action_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OEL | https://arxiv.org/abs/2603.16856 | S3.2 Eq.2 | "The student pi_theta generates a response y conditioned only on x" (consolidation substage) | "optimized to match the knowledge-conditioned output of a teacher pi_teacher through token-level reverse KL" | L(theta) = E[D_KL(pi_theta \|\| pi_teacher)] identical to OPCD | privileged_self (experiential knowledge context) | token (consolidation only) | reverse_kl | mixed_freshness (deployment + consolidation loop) | partial_opd (consolidation substage strict) | full pipeline includes non-OPD deployment collection and knowledge extraction stages | high |
| OEC | https://arxiv.org/abs/2512.14895 | main method | student generates first K turns of multi-turn SWE trajectory | expert completes remaining turns from student-visited state | NLL on expert-completed portion (student turns masked) | expert_policy (external LLM) | action (hard labels, not distributions) | cross_entropy_on_expert_actions | current_policy | borderline_strict | C3 uses hard action labels (SFT) rather than soft distribution matching; structurally DAgger, not classic OPD | high |
| OpenClaw-RL HG-OPD | https://arxiv.org/abs/2603.10165 | HG-OPD component | student agent generates live trajectories | self-teacher (same model with hint from next state) provides token-level log-probability gap | per-token log-probability gap provides directional guidance | privileged_self (hindsight hint) | token | log_prob_gap_advantage | current_policy | borderline_strict (HG-OPD component) | full system combines HG-OPD with Binary RL (scalar PRM rewards); method-level is hybrid | medium |
| GLM-5 cross-stage OPD | https://arxiv.org/abs/2602.15763 | post-training pipeline description | claimed on-policy student training across RL stages | claimed teacher distillation bridging stages | claimed on-policy distillation objective | unclear (industrial) | unclear | unclear | unclear | borderline_strict (industrial) | exact loss, teacher, rollout mechanics not detailed; similar to Qwen3 OPD stage | medium |

### 3.3 Partial OPD

| method | source_url | C1 | C2 | C3 | final_label | reason_to_downgrade | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RLSD / Self-Distilled RLVR | https://arxiv.org/abs/2604.03128 | PASS (student rollouts) | PASS (privileged self-teacher evidence ratios) | WEAK (teacher determines magnitude only; direction from reward) | partial_opd | paper explicitly decouples direction (reward) from magnitude (distillation); "fatal information asymmetry" argument against pure OPSD | medium |
| DGPO | https://arxiv.org/abs/2508.20324 | PASS (PPO student rollouts) | PARTIAL (KL penalty only on incorrect predictions) | FAIL strict (KL consumed as scalar reward by PPO advantage estimator) | partial_opd / adjacent | teacher KL is reward-shaped RL, not direct distillation loss | medium |
| RISE (distillation variant) | https://arxiv.org/abs/2407.18219 | PASS (multi-turn student rollouts) | PARTIAL (teacher provides response TEXT, not logits) | FAIL (reward-weighted SFT, not KL) | adjacent | paper explicitly disclaims "different from classic knowledge distillation" | medium |

---

## 4. Candidate Table Rows / Updates

### 4.1 New Rows for `tables/opd_papers.md`

#### TCOD (strict_opd candidate)
```
id: tcod-2026
title: TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents
year: 2026
paper_url: https://arxiv.org/abs/2604.24005
modality: LLM;Agent
domain: multi_turn_agent
method_family: temporal_curriculum_opd
teacher_access_regime: white_box
loss_granularity: token
opd_strictness: strict_opd (candidate, needs line audit)
teacher_kind: larger_llm
rollout_source: student_on_policy
rollout_freshness: current_policy
supervision_granularity: token
loss_objective: forward_kl_with_temporal_curriculum
strictness_evidence: C1 student generates multi-turn agent trajectories; C2 teacher computes token-level KL on student actions; C3 KL objective with temporal curriculum scheduling
benchmarks: ALFWorld; WebShop; ScienceWorld
confidence: high
```

#### LiteGUI (strict_opd candidate)
```
id: litegui-2026
title: LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning
year: 2026
paper_url: https://arxiv.org/abs/2605.07505
modality: VLM;Agent
domain: gui_agent
method_family: gkd_gui_opd
teacher_access_regime: white_box
loss_granularity: token
opd_strictness: strict_opd (candidate, needs line audit)
teacher_kind: larger_vlm (oracle-guided privileged context)
rollout_source: student_on_policy
rollout_freshness: current_policy
supervision_granularity: token
loss_objective: reverse_kl_gkd
strictness_evidence: C1 student generates GUI actions from current policy; C2 Qwen3-VL-32B teacher with oracle context scores student sequences; C3 reverse-KL distillation
benchmarks: ScreenSpot-Pro; OS-World; Lite-Bench
confidence: high
```

#### Skill-SD (strict_opd candidate)
```
id: skill-sd-2026
title: Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents
year: 2026
paper_url: https://arxiv.org/abs/2604.10674
modality: LLM;Agent
domain: multi_turn_agent
method_family: privileged_self_opd
teacher_access_regime: teacher_free
loss_granularity: token
opd_strictness: strict_opd (candidate, needs line audit)
teacher_kind: privileged_self (skill-conditioned)
rollout_source: student_on_policy
rollout_freshness: per_iteration
supervision_granularity: token
loss_objective: importance_weighted_reverse_kl
strictness_evidence: C1 student generates trajectories under plain prompt; C2 skill-conditioned self-teacher re-scores same tokens; C3 importance-weighted reverse-KL loss
benchmarks: AppWorld; Sokoban
confidence: medium-high
```

### 4.2 New Rows for Borderline/Partial Tracking

#### OEC (borderline_strict)
```
id: oec-2025
title: Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections
year: 2025
paper_url: https://arxiv.org/abs/2512.14895
modality: LLM;Agent
domain: swe_agent
opd_strictness: borderline_strict
strictness_evidence: C1 student generates first K turns; C2 expert completes from student state; C3 NLL on expert completions (hard labels, not distribution matching)
confidence: high
notes: Most directly DAgger-inspired LLM agent method. Borderline because SFT on expert completions rather than teacher logit KL.
```

#### TCOD already covers WebShop/ALFWorld. OEL and OPCD already tracked.

### 4.3 Updates to Existing Rows

No changes needed for MAD-OPD, SOD, OPCD, OEL — all confirmed at current classifications.

---

## 5. Adjacent / False-Positive Ledger

### 5.1 Reward-Only RL Agent Methods (adjacent)

| method | year | url | category | why_adjacent | missing_condition | confidence |
| --- | --- | --- | --- | --- | --- | --- |
| SCoRe (Kumar) | 2024 | https://arxiv.org/abs/2409.12917 | agent_self_correction_rl | REINFORCE with binary correctness reward; KL against own base (self-regularization, not teacher); paper explicitly emphasizes independence from teacher supervision | C2 (no teacher), C3 (reward-only) | high |
| DigiRL | 2024 | https://arxiv.org/abs/2406.11896 | gui_agent_rl | Two-stage offline-to-online RL with scalar task-completion reward from VLM evaluator; explicitly no teacher distillation | C2 (VLM evaluator is reward, not teacher), C3 (reward-only) | high |
| AgentQ | 2024 | https://arxiv.org/abs/2408.07199 | web_agent_mcts_dpo | MCTS + off-policy DPO with self-critique scores; no external teacher provides distributions | C2 (self-critique is reward), C3 (DPO preference, not distillation) | high |
| CodeRL | 2022 | https://arxiv.org/abs/2207.01780 | code_agent_rl | REINFORCE + critic providing token-level reward ESTIMATES (not teacher logits); reward from unit tests | C2 (critic reward, not teacher), C3 (RL reward) | high |
| RLTF | 2023 | https://arxiv.org/abs/2307.04349 | code_execution_rl | Multi-granularity feedback from compiler/tests (error-location reward shaping); no teacher model | C2 (execution feedback, not teacher), C3 (shaped reward) | high |
| StepCoder | 2024 | https://arxiv.org/abs/2402.01391 | code_step_rl | PPO with compiler feedback reward + curriculum from canonical solutions; FGO masks unexecuted code | C2 (compiler reward), C3 (PPO reward) | high |
| SWE-RL | 2025 | https://arxiv.org/abs/2502.18449 | swe_agent_rl | RL with scalar similarity-score reward between ground-truth and LLM-generated solutions | C2 (scalar reward), C3 (RL) | high |
| DeepSWE | 2025 | https://www.together.ai/blog/deepswe | swe_agent_rl | GRPO++ with binary test-pass reward; explicitly "no distillation from closed models" | C2 (test reward), C3 (GRPO reward) | high |
| SWE-smith | 2025 | github.com/SWE-bench/SWE-smith | swe_agent_rl | GRPO with test-pass reward via SkyRL | C2 (test reward), C3 (GRPO) | high |
| SWE-Gym | 2024 | https://arxiv.org/abs/2412.21139 | swe_agent_rft | Iterative RFT with ORM verifier (YES/NO success prediction); teacher traces for initial SFT only | C2 (ORM reward), C3 (RFT/NLL on filtered traces) | high |
| ETO | 2024 | https://arxiv.org/abs/2403.02502 | agent_trajectory_dpo | Student explores and fails; expert provides separate trajectories; DPO on failure-success pairs | C2 (expert generates independent trajectories, not scoring student states), C3 (DPO preference) | high |
| IPR | 2024 | https://aclanthology.org/2024.emnlp-main.93 | agent_step_dpo | Step-level reward scoring + contrastive DPO + SFT on student-explored actions | C2 (reward scorer, not teacher distributions), C3 (DPO preference) | high |
| AlphaLLM-CPL | 2024 | https://arxiv.org/abs/2410.06508 | mcts_preference_learning | MCTS stepwise evaluation + DPO; self-generated search, no external teacher distributions | C2 (self-search scores), C3 (DPO) | high |
| WebGPT | 2021 | https://arxiv.org/abs/2112.09332 | web_agent_rlhf | Imitation learning + reward model rejection sampling; no teacher token-level distillation | C2 (reward model), C3 (rejection sampling) | high |
| RLEF | 2024 | https://arxiv.org/abs/2410.02089 | code_execution_rl | PPO with binary test pass/fail reward from execution feedback | C2 (execution reward), C3 (PPO) | high |
| RISE | 2024 | https://arxiv.org/abs/2407.18219 | agent_self_correction | Reward-weighted SFT; teacher provides text responses not logits; paper disclaims "different from classic KD" | C3 (reward-weighted SFT, not KL) | medium |
| GLoRE | 2024 | https://arxiv.org/abs/2402.10963 | self_correction_sft | SFT on paired correct/incorrect student solutions with learned SORM; no external teacher | C2 (no teacher), C3 (SFT cross-entropy) | high |
| UI-TARS | 2025 | https://arxiv.org/abs/2501.12326 | gui_agent_iterative_dpo | Iterative SFT + DPO with human-filtered student trajectories; no teacher model | C2 (human annotators, not teacher model), C3 (DPO preference) | high |
| ILF | 2024 | https://arxiv.org/abs/2303.16749 | language_feedback_sft | Student outputs → human feedback → refinement model → filter → SFT on refined outputs | C3 (SFT on refined outputs, not distribution matching) | high |

### 5.2 Offline SFT / No Training (not_opd)

| method | year | url | category | why_not_opd | confidence |
| --- | --- | --- | --- | --- | --- |
| SAD | 2025 | https://arxiv.org/abs/2505.13820 | offline_agent_kd | Teacher-forced training on pre-collected teacher demonstrations; student does NOT generate trajectories. CORRECTION from legacy classification. | high |
| Toolformer | 2023 | https://arxiv.org/abs/2302.04761 | offline_tool_sft | Self-supervised one-shot data augmentation + SFT; no iterative on-policy loop | high |
| Gorilla | 2023 | https://arxiv.org/abs/2305.15334 | offline_api_sft | SFT on GPT-4-generated API call demonstrations | high |
| ToolLLM | 2023 | https://arxiv.org/abs/2307.16789 | offline_tool_sft | SFT on ChatGPT-generated DFSDT solution paths | high |
| FireAct | 2023 | https://arxiv.org/abs/2310.05915 | offline_agent_sft | SFT on GPT-4-generated ReAct trajectories | high |
| CogAgent | 2023 | https://arxiv.org/abs/2312.08914 | offline_gui_sft | Pre-training + SFT on pre-collected GUI screenshots/demonstrations | high |
| OS-Atlas | 2024 | https://arxiv.org/abs/2410.23218 | offline_gui_sft | GUI grounding pre-training + SFT on static data | high |
| ScreenAgent | 2024 | https://arxiv.org/abs/2402.07945 | offline_gui_sft | SFT on pre-collected screenshot-action pairs | high |
| ShowUI | 2024 | https://arxiv.org/abs/2411.17465 | offline_gui_sft | SFT on 256K curated GUI instruction-following data | high |
| Agent-FLAN | 2024 | https://arxiv.org/abs/2403.12881 | offline_agent_sft | Offline SFT with capability-decomposed data balancing | high |
| AgentTuning | 2023 | https://arxiv.org/abs/2310.12823 | offline_agent_sft | Instruction tuning on curated agent trajectories | high |
| CodeAct | 2024 | https://arxiv.org/abs/2402.01030 | offline_code_sft | SFT on GPT-4/Claude-generated instruction-tuning data | high |
| SWE-Master | 2026 | https://arxiv.org/abs/2602.03411 | offline_swe_sft | SFT on filtered teacher-generated trajectories | high |
| Chain-of-Agents | 2025 | https://arxiv.org/abs/2508.13167 | offline_agent_sft_plus_rl | SFT on multi-agent teacher traces (Phase 1) + reward-only RL (Phase 2) | high |
| Reflexion | 2023 | https://arxiv.org/abs/2303.11366 | inference_only | Inference-time in-context learning; no weight updates at all | high |
| Self-Refine | 2023 | https://arxiv.org/abs/2303.17651 | inference_only | Inference-time iterative refinement with frozen model; no training | high |
| AgentDistill | 2025 | https://arxiv.org/abs/2506.14728 | training_free | Training-free MCP-based knowledge transfer at inference | high |
| Agentless | 2024 | https://arxiv.org/abs/2407.01489 | inference_only | Inference-time pipeline (localize → repair → validate); no training | high |
| WebArena | 2023 | https://arxiv.org/abs/2307.13854 | benchmark_only | Evaluation environment, no training method | high |
| BrowserGym | 2024 | https://arxiv.org/abs/2412.05467 | benchmark_only | Evaluation ecosystem, no training method | high |
| OSWorld | 2024 | https://arxiv.org/abs/2404.07972 | benchmark_only | Evaluation platform, no training method | high |
| Latent Agents/IMAD | 2026 | https://arxiv.org/abs/2604.24881 | offline_sft_plus_rl | SFT on teacher-generated debate transcripts + scalar-reward RL | high |
| AgentArk | 2026 | https://arxiv.org/abs/2602.03955 | offline_debate_distillation | Multi-agent debate generates data; SFT + PRM-guided GRPO | high |
| D&R | 2025 | https://arxiv.org/abs/2506.03541 | debate_preference_optimization | Tree-DPO on debate-informed preference labels | high |
| ProductResearch | 2026 | https://arxiv.org/abs/2602.23716 | offline_synthetic_trajectory_sft | Alibaba multi-agent synthetic trajectory distillation via SFT | high |
| ODIA | 2025 | https://arxiv.org/abs/2507.08877 | offline_function_call_kd | Production routing + offline KD for function calling | high |
| Web Agent Distill (Lu & Reddy) | 2026 | https://arxiv.org/abs/2604.07776 | offline_web_agent_sft | SFT on Gemini-3-Pro-generated trajectories | high |
| Mem-W | 2026 | https://arxiv.org/abs/2605.09317 | offline_context_distillation_plus_rl | Context distillation on offline demonstrations + reward-only RL | medium-high |

---

## 6. Gap Map

### 6.1 Empty Agent Domains (No Strict OPD Found)

| Domain | Status | Closest Method | Gap |
| --- | --- | --- | --- |
| **SWE/code-repair agents** | No strict OPD | OEC (borderline_strict, DAgger-style SFT) | No method applies teacher logit distillation on student-generated code patches. Entire SWE ecosystem is reward-only RL. |
| **Database/SQL agents** | Empty | None found | No papers investigated OPD for database interaction agents |
| **Long-horizon planning agents** | Empty | None found | DEPS/Plan-and-Solve are prompting only; no OPD for planning-heavy agents |
| **Conversational/dialogue agents** | Empty | None found | Multi-turn dialogue distillation with on-policy student turns is unstudied |
| **API/function-calling agents** | Mostly covered by MAD-OPD | MAD-OPD (BFCL-v4) | Only MAD-OPD evaluates on function-calling benchmarks |
| **Embodied/robotics agents (non-VLA)** | Out of scope (WP7) | VLA-OPD, LLM4Teach, HINT | VLA-OPD covers action-token OPD; LLM4Teach/HINT cover MARL but not LLM-scale |

### 6.2 Methods Needing Line-Level Audit

| Method | Priority | What to Verify |
| --- | --- | --- |
| **TCOD** | High | Confirm rollout freshness is current-policy at each curriculum stage; verify KL is token-level not trajectory-level |
| **LiteGUI** | High | Confirm GKD loop is fully on-policy; verify oracle-guided context setup; confirm Stage 1 and Stage 2 are separable |
| **Skill-SD** | High | Confirm skills are extracted from student's own completed trajectories (not external data); verify importance weights; check rollout staleness from theta_old |
| **OpenClaw-RL HG-OPD** | Medium | Verify whether token-level log-prob gap functions as distillation loss (KL-like) or RL advantage; separate HG-OPD from Binary RL component |
| **GLM-5 cross-stage OPD** | Low | Industrial report is terse; unlikely to yield strict evidence without code release |
| **RLSD** | Medium | Resolve conflict: is per-token evidence ratio consumption a distillation-style objective or reward-shaping? |

### 6.3 Structural Gaps in Agentic OPD

1. **Safety during on-policy exploration:** No agentic OPD paper systematically addresses safety constraints during student exploration. SOD and TCOD attenuate misleading teacher signals but do not handle harmful actions.

2. **Credit assignment across tool calls:** SOD's step-divergence reweighting is the only attempt. Delayed credit assignment in long-horizon agent tasks (e.g., 20+ turn trajectories) remains open.

3. **Environment reset cost:** All strict methods assume cheap environment resets. SWE-bench tasks (code compilation, test execution) and real-world web tasks (irreversible state changes) pose systems challenges unaddressed by current OPD methods.

4. **Teacher-student architecture mismatch:** LiteGUI uses a VLM teacher for a VLM student; MAD-OPD uses LLM teachers for LLM students. Cross-architecture agentic OPD (e.g., VLM teacher for text-only agent student) is unstudied.

5. **Standard benchmark coverage:** Only TCOD evaluates on classic agent benchmarks (ALFWorld, WebShop, ScienceWorld). MAD-OPD uses newer benchmarks (BFCL-v4, tau2-Bench, VitaBench). No OPD paper evaluates on AgentBench, InterCode, or ToolBench.

6. **Multi-agent to single-agent OPD:** AgentArk and Latent Agents distill multi-agent intelligence into single agents via offline SFT. No method applies on-policy distillation for multi-agent → single-agent compression.

### 6.4 Foundational References (Out of Core LLM Scope)

| Method | Domain | OPD Relevance |
| --- | --- | --- |
| DAgger (2011) | Imitation learning | Structural ancestor of OPD for agents; GKD cites it. Hard action labels vs. soft distributions is the key gap. |
| LLM4Teach (2024) | MARL | Strict OPD in MARL: LLM teacher provides soft action distributions to RL agent. Not LLM-scale. |
| HINT (2026) | Cooperative MARL | Interactive distillation from centralized teacher to decentralized agents. Not LLM-scale. |
| Hybrid RL+IL (2025) | Theoretical | Proves hybrid RL+imitation objectives contain an OPD component (dense gradient). Framework, not empirical. |

---

## 7. Summary Statistics

| Category | Count | Methods |
| --- | --- | --- |
| Strict OPD (confirmed) | 3 | MAD-OPD, SOD, OPCD |
| Strict OPD (new candidates) | 3 | TCOD, LiteGUI, Skill-SD |
| Borderline strict | 4 | OEL (substage), OEC, OpenClaw-RL HG-OPD, GLM-5 |
| Partial OPD | 3 | RLSD, DGPO, RISE (distillation variant) |
| Adjacent (reward-only RL) | 16 | SCoRe, DigiRL, AgentQ, CodeRL, RLTF, StepCoder, SWE-RL, DeepSWE, SWE-smith, SWE-Gym, ETO, IPR, AlphaLLM-CPL, WebGPT, RLEF, UI-TARS |
| Adjacent (other) | 2 | GLoRE, ILF |
| Not OPD (offline SFT) | 16 | SAD, Toolformer, Gorilla, ToolLLM, FireAct, CogAgent, OS-Atlas, ScreenAgent, ShowUI, Agent-FLAN, AgentTuning, CodeAct, SWE-Master, Chain-of-Agents, Web Agent Distill, Mem-W |
| Not OPD (no training) | 7 | Reflexion, Self-Refine, AgentDistill, Agentless, WebArena, BrowserGym, OSWorld |
| Not OPD (debate/synthesis offline) | 4 | Latent Agents, AgentArk, D&R, ProductResearch, ODIA |
| Foundational (out of LLM scope) | 4 | DAgger, LLM4Teach, HINT, Hybrid RL+IL |
| **Total methods audited** | **62** | |

---

## 8. Agentic OPD Taxonomy

### 8.1 By Teacher Pattern

| Pattern | Methods | Signal Type |
| --- | --- | --- |
| External teacher logits (white-box) | MAD-OPD, SOD, TCOD, LiteGUI | Token-level KL on student-generated actions |
| Privileged-context self-teacher | OPCD, OEL, Skill-SD, OpenClaw-RL | Same model ± context provides token-level supervision |
| DAgger-style expert action labels | OEC | Hard action labels at student-visited states |
| Multi-teacher debate | MAD-OPD | Confidence-weighted debate over student states |

### 8.2 By Agent Domain

| Domain | Strict | Borderline | Adjacent |
| --- | --- | --- | --- |
| Multi-turn tool-use | SOD, MAD-OPD | OEL | VTool-R1 |
| GUI/web navigation | LiteGUI, TCOD | — | DigiRL, AgentQ, UI-TARS |
| Code/SWE repair | — | OEC | SWE-RL, DeepSWE, CodeRL, RLTF |
| Context internalization | OPCD | — | — |
| Multi-turn agent | Skill-SD | OpenClaw-RL | ETO, IPR |
| Function calling | MAD-OPD | — | — |
| Self-correction | — | — | SCoRe, RISE, GLoRE |

### 8.3 By Error-Compounding Mitigation

| Strategy | Method | How |
| --- | --- | --- |
| Step-divergence reweighting | SOD | Attenuate teacher signal in high-divergence regions |
| Temporal curriculum | TCOD | Progressive trajectory depth expansion |
| Per-step debate | MAD-OPD (OPAD) | Teachers debate at each decision point |
| Skill conditioning | Skill-SD | Privileged skills reduce trajectory variance |
| DAgger truncation | OEC | Expert takes over from student-visited state |

---

*End of WP6 research memo.*
