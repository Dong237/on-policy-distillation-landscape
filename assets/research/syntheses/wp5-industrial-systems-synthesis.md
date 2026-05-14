# WP5 Industrial And Systems OPD Synthesis

Checked: 2026-05-13

This note merges:

- `assets/research/wp5-industrial-systems-chatgpt.md`
- `assets/research/wp5-industrial-systems-claude.md`
- `assets/research/wp5-industrial-systems-gemini.md`

## Core Finding

WP5 confirms that industrial OPD claims should be stage-scoped. The strongest public evidence is for specific OPD/MOPD/draft-model stages, not for whole post-training pipelines. Industrial reports often combine offline distillation, SFT, synthetic data, RLVR, DPO, and one OPD stage; the whole system should not inherit the strict label from that stage.

The systems picture is also clearer: strict OPD remains expensive because a teacher or target model must score student-generated tokens. Public cost mitigations include sampled-token log-probs, prefix-only teacher scoring, cached teacher log-probs, draft-model f-divergences, interleaved speculative verification, multi-teacher routing, sequence packing, FP8, and distributed rollout/teacher-serving backends.

## Merge Decision

| Method or system | Final repo action | Reason |
|---|---|---|
| Qwen3 OPD stage | Keep `strict_opd` stage label, downgrade confidence to medium. | Agents agree the public report supports a distinct on-policy logit-distillation phase, but the exact loss, rollout refresh, and serving mechanics are terse. |
| Nemotron-Cascade2 MOPD | Keep `strict_opd` stage label. | Public evidence supports sampled student responses, selected domain teachers, and dense token-level distillation advantage. |
| MiMo-V2-Flash MOPD | Keep/add stage-scoped `strict_opd`. | MOPD uses domain teachers and token log-ratio advantages on student samples; the full MiMo pipeline remains multi-stage. |
| DistillSpec | Keep/add `strict_opd` systems row. | Strict for draft-generated/on-policy drafter training with target-model distributional supervision. |
| Speculative KD | Keep/add `partial_opd`. | Student proposes tokens, but rejected tokens are teacher-replaced, creating mixed-policy trajectories. |
| Lightning OPD | Keep/add `partial_opd`. | Cached teacher log-probs are useful systems evidence, but rollouts are precomputed from reference/SFT policy. |
| Fast OPD | Keep/add `strict_opd` systems row with prefix caveat. | Teacher scores current student-generated reasoning prefixes, not necessarily full trajectories. |
| Gemma 2 post-training | Add/keep `borderline_strict`. | The report says teacher distillation on the student's distribution and cites OPD-style methods, but omits the formal objective. |
| Gemma 3 | Keep adjacent. | KD and RL are disclosed, but no current student rollout plus same-state teacher supervision is public. |
| DeepSeek-R1 distilled checkpoints | Keep/add `not_opd`. | SFT on fixed teacher-generated traces, not OPD. |
| Minitron / pruning KD | Add `not_opd`. | Pruning plus offline KD/retraining does not disclose current student rollouts. |
| TRL GKDTrainer | Keep `strict_opd_support=yes` for LLM. | Official docs expose the GKD loop when configured as fully on-policy. |
| NeMo RL OPD | Upgrade confidence to high. | Official repo documents an OPD example; VLM OPD remains unclear. |
| verl | Resolve `strict_opd_support=no`. | Strong RL/RLVR infrastructure, but no official built-in teacher-logit OPD loop found. |
| OpenRLHF | Resolve `strict_opd_support=no`. | Supports RLHF/RLVR/KD infrastructure, but no audited same-rollout strict OPD recipe. |
| Tinker / Thinking Machines OPD recipe | Add framework-style reference, not a paper row. | Canonical sample-then-teacher-logprob recipe; useful operational definition. |
| KDFlow | Add systems framework row as `unclear` for strict OPD. | Cross-tokenizer serving machinery can support teacher dataflow, but strictness depends on configuration. |

## Evidence Ledger

| method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 OPD stage | https://arxiv.org/abs/2505.09388 | post-training section | "on-policy knowledge transfer" | "teacher logits" | "On-policy Distillation" | larger Qwen teacher logits | reported OPD-vs-direct-RL GPU-hour comparison | token logits/KL | stated on-policy, exact refresh not exposed | strict_opd_stage | strict only for the disclosed OPD phase; public mechanics are terse | medium |
| Nemotron-Cascade2 MOPD | https://arxiv.org/abs/2603.19220 | Multi-Domain OPD section | "sample a response" | "domain teacher" | "distillation advantage" | multi-domain teacher checkpoints | sampled-token log-prob differences; truncated importance weighting; NeMo RL backend | token | current student samples | strict_opd_stage | cascade RL stages are not globally OPD | high |
| MiMo-V2-Flash MOPD | https://arxiv.org/abs/2601.02780 | Sec. 4.1; Sec. 4.4 | "samples from its own evolving distribution" | "specialized teachers" | "surrogate loss" | domain-specialized teachers | SGLang inference, Megatron-LM training, FP8, sampled-token log-ratio advantage | token plus optional reward | current student samples | strict_opd_stage | strict only for MOPD stage | high |
| DistillSpec | https://arxiv.org/abs/2310.08461 | Sec. 4; Theorem 4.1 | "draft model" on-policy data | "target model" distributions | "on-policy KD loss" | target-model logits | speculative-decoding acceptance alignment; task-specific f-divergences | token | draft-generated data | strict_opd | strict for draft-generated configuration | high |
| Speculative KD | https://arxiv.org/abs/2410.11325 | Algorithm 1 | "student proposes" | "teacher model evaluates" | "minimize ... D" | teacher logits and top-k acceptance | interleaved student-propose / teacher-replace | token | mixed student/teacher trajectory | partial_opd | teacher replacement weakens exact student-rollout C1 | high |
| Fast OPD | https://arxiv.org/abs/2602.15260 | prefix training section | "student-generated outputs" | "teacher model" | "reverse-KL loss" | white-box teacher on prefixes | early-stop prefix generation and prefix schedule | prefix-token | current student prefixes | strict_opd_prefix | strict only on supervised prefixes | high |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | Sec. 3.2 | "SFT rollouts" | "precompute ... log-probabilities" | "dense per-token" | cached teacher log-probs | removes live teacher server; precomputes teacher log-probs once | token | precomputed reference/SFT rollouts | partial_opd | fails strict current-policy freshness | high |
| Gemma 2 post-training | https://arxiv.org/abs/2408.00118 | Sec. 4 | "student's distribution" | "larger model" | "probabilities" | larger teacher probabilities | internal post-training details not public | token implied | unclear | borderline_strict | exact rollout loop and loss are not public | medium |
| Gemma 3 | https://arxiv.org/abs/2503.19786 | training report | not found | "knowledge distillation" | sampled-logit/cross-entropy KD | teacher sampled logits | sampled 256 logits per token; RL phase separate | token KD | not found | adjacent | KD is not tied to current student rollouts | high |
| DeepSeek-R1 distilled checkpoints | https://arxiv.org/abs/2501.12948 | distillation section | not found | "800k data" | "fine-tuned" | teacher-generated traces | SFT-style checkpoint distillation | sequence traces | static supervised data | not_opd | SFT on fixed teacher traces | high |
| Minitron / pruning KD | https://arxiv.org/abs/2407.14679 | compression reports | not found | "pruning and distillation" | "distillation-based retraining" | teacher model for recovery KD | structured pruning plus offline recovery training | token/logit KD | static recovery data | not_opd | compression KD is not current-rollout OPD | high |
| TRL GKDTrainer | https://huggingface.co/docs/trl/gkd_trainer | official docs | "self-generated output sequences" | "teacher model" | "token-specific feedback" | local teacher model | trainer-level generation; configurable on-policy setting | token | current when fully on-policy | strict_framework_support_text | framework support, not method evidence | high |
| NeMo RL OPD | https://github.com/NVIDIA-NeMo/RL | official README/example | "on-policy sequences" | "larger teacher" | "aligns logits" | local teacher model | Megatron, Ray, vLLM/SGLang-style distributed serving support | token | current in example | strict_framework_support_text | VLM OPD recipe still unclear | high |
| verl | https://github.com/verl-project/verl | official README | rollout generation | not found | not found | none by default | distributed PPO/GRPO/DAPO infrastructure | reward/RL unless customized | current for RL rollouts | adjacent_framework_rl | no built-in strict OPD teacher-logit loop found | high |
| OpenRLHF | https://github.com/OpenRLHF/OpenRLHF | official README | RL rollouts | not found | not found | reward/reference/KD infra | Ray, DeepSpeed, vLLM, hybrid engine | reward/RL or offline KD unless customized | current or async for RL | adjacent_framework_rl | no audited same-rollout strict OPD recipe found | high |

## Systems Patterns

| Pattern | Where it appears | Strictness impact |
|---|---|---|
| Sampled-token teacher log-probs | REOPOLD, MiMo-V2-Flash, Nemotron-Cascade2 | Can remain strict if evaluated on current student tokens. |
| Prefix-only teacher scoring | Fast OPD | Strict for prefixes; do not imply full-trajectory OPD. |
| Cached teacher log-probs | Lightning OPD | Useful systems approximation, but strict C1 fails. |
| Draft-model f-divergence | DistillSpec | Strict when the draft generates the supervised data. |
| Interleaved teacher replacement | Speculative KD | Partial because supervised trajectory is mixed-policy. |
| Multi-teacher routing | MiMo-V2-Flash, Nemotron-Cascade2 | Strict stage possible, but serving/routing costs remain underreported. |
| Cross-tokenizer serving | SimCT, KDFlow, DSKD-style mechanisms | Strict only if teacher distributions score current student-generated text states. |

## Framework Boundary

Framework support should be tracked at module or recipe level. A repository that supports rollouts and KD separately is not automatically strict OPD. The official recipe must show student-generated samples, teacher scoring on those samples, and a distillation loss consuming that signal.

## Remaining Gaps

| Gap | Current action | Needed next |
|---|---|---|
| Qwen3 exact OPD mechanics | Keep strict stage label with medium confidence. | Monitor official Qwen docs/code for exact loss and rollout refresh details. |
| Multi-teacher serving cost | Keep stage strict but avoid throughput claims. | Audit released MiMo/Nemotron configs if teacher-routing scripts appear. |
| Gemma 2 post-training | Keep borderline. | Upgrade only if Google publishes objective/rollout details. |
| Gemma 3 and VLM industrial reports | Keep adjacent/unclear. | Require direct C1/C2/C3 evidence before promoting. |
| NeMo/TRL VLM OPD | Keep strict only for LLM examples. | Look for official VLM-GKD or VLM-OPD recipes later. |
| verl/OpenRLHF custom OPD | Mark default support as no/adjacent. | Add strict rows only for specific recipes with teacher-logit dataflow. |

## Next Router State

WP5 is complete for the current pass. Later WP6-WP7 passes have been merged separately; current routing state is tracked in [deep-research-plan.md](deep-research-plan.md).
