# Agent Prompts

This file is a short index. Use [deep-research-plan.md](deep-research-plan.md) for the full prompts.

The older "fill this table" prompt style is intentionally avoided. Deep research agents should first produce a research memo, evidence ledger, candidate rows, and gap map. Structured Markdown rows come after source discovery and reasoning, not before it.

## Recommended agent roles

Run these as independent agents rather than one monolithic prompt:

1. **Survey replication researcher:** Work Package 0 in `deep-research-plan.md`.
2. **White-box OPD researcher:** Work Package 1.
3. **Black-box/API OPD researcher:** Work Package 2.
4. **Teacher-free/self-play researcher:** Work Package 3.
5. **Reasoning and OPD+RL researcher:** Work Package 4.
6. **Industrial and systems researcher:** Work Package 5.
7. **Agentic OPD researcher:** Work Package 6.
8. **Multimodal frontier researcher:** Work Package 7.
9. **Evaluation methodology researcher:** Work Package 8.
10. **Red-team verifier:** Work Package 9.

## Coordination rule

Do not let the same agent both discover and verify its own strict OPD claims. Discovery agents should be expansive; verifier agents should be skeptical.

## Merge rule

Only merge rows that include source links, classification rationale, confidence, and unresolved questions. If two agents disagree, keep the more conservative label until a verifier resolves it.

## Completed WP2 black-box/API OPD discovery prompt

This discovery prompt was used for `notes/wp2-blackbox-synthesis.md`. The follow-up WP2 line audit and secondary spot-check are also complete. Keep it as a template, but do not rerun WP2 without new candidate evidence.

```text
You are a deep research agent for the “On Policy Distillation Landscape” repository.

Task: map WP2 — black-box/API on-policy distillation for LLMs and adjacent response-level methods. Do not produce a shallow paper list. Build a verifier-ready research memo with evidence.

Core definition:
Strict OPD requires all three:
C1. The current student/policy generates the training rollout, response, trajectory, or prefix.
C2. A black-box teacher/API, discriminator, preference judge, verbal-feedback model, reference model, or teacher-derived process supervises that exact student-generated state.
C3. The training objective consumes that supervision as a distillation-style signal, not merely as reward-only RLVR or static preference optimization.

Scope:
- Black-box/API teacher methods where token logits are unavailable.
- Adversarial or discriminator-based distillation on student rollouts.
- Verbal feedback, critique, ranking, preference, or response-level supervision from a teacher/API.
- Cross-architecture and cross-tokenizer black-box distillation if on-policy.
- Methods that call themselves OPD but may really be RLVR, RLHF, DPO, rejection sampling, or synthetic data.

Seeds to audit, but do not stop there:
GAD, Lion, OVD, PRISM, ORPO-Distill, SuperCorrect, black-box preference/adversarial OPD, API teacher-as-judge distillation, response-level discriminator OPD, verbal reward or critique distillation.

Also search for false positives:
methods that use student rollouts plus scalar rewards only; methods that train on teacher-generated data; SFT on API traces; DPO/IPO/ORPO/SimPO on static preference pairs; rejection sampling; reward-model-only GRPO; synthetic data pipelines that are not same-rollout distillation.

Use primary sources where possible:
- arXiv PDF or HTML
- OpenReview paper PDF
- official technical reports
- official project pages
- official GitHub/docs
- official model cards or blogs

Do not rely on SEO pages, random blog posts, media summaries, or social posts for final classification.

Deliverables:
1. Memo:
   - What you searched.
   - Method families found.
   - What is actually black-box OPD versus adjacent.
   - Key differences from white-box OPD.
   - Practical costs and failure modes: API cost, response inflation, discriminator overfitting, reward hacking, teacher drift, calibration, prompt sensitivity.
2. Evidence ledger:
   One row per method with:
   method | primary_source_url | source_status | C1 evidence | C2 evidence | C3 evidence | teacher_access | supervision_granularity | objective_family | rollout_freshness | suggested_label | confidence | downgrade_reason
3. Candidate Markdown rows:
   Only for methods with enough evidence to enter `tables/opd_papers.md`; include all fields needed by the repo schema if possible.
4. Adjacent/false-positive ledger:
   method | link | why_adjacent | missing_condition | confidence
5. Gap map:
   Open questions requiring line-level audit.

Classification rules:
- Response-level discriminator/adversarial feedback is `borderline_strict` at best unless the objective clearly consumes non-scalar teacher-style supervision on the exact student rollout.
- Verbal critique or API score is usually `partial_opd` or `adjacent` unless converted into a distillation objective.
- Reward-only GRPO/RLVR is not OPD even if on-policy.
- Static DPO/preference optimization is not OPD unless pairs are generated and supervised online.
- Teacher-generated SFT traces are offline KD/SFT, not OPD.
- If evidence is missing, write “not found” and downgrade conservatively.

Return a concise memo first, then tables. Do not invent missing evidence.
```

## Completed WP2 black-box/API line-level audit prompt

This prompt was used to resolve the WP2 primary queue. It is narrow by design: do not ask agents to rediscover the whole field. Keep it as a template for future black-box candidates.

```text
You are a line-level primary-source auditor for the “On Policy Distillation Landscape” repository.

Task: audit only the WP2 black-box/API candidate queue. Do not write a broad survey. Do not add new papers unless an assigned method has a name collision that must be resolved.

Strict OPD requires all three:
C1. The current student/policy generates the training rollout, response, trajectory, or prefix.
C2. A black-box teacher/API, discriminator, preference judge, verbal-feedback model, reference model, or teacher-derived process supervises that exact student-generated state.
C3. The training objective consumes that supervision as a distillation-style signal, not merely as reward-only RLVR, scalar judge reward, or static preference optimization.

Audit these candidates:
GAD, PRISM, OVD, SODA, ORPO-Distill.

Secondary spot-check only if time remains:
GAKD, ADPA, PAD, RLTF, FCP, ALT, daDPO, CTPD, RLKD, GAR.

Use only primary sources:
- arXiv PDF or arXiv HTML
- OpenReview paper PDF
- official project page
- official GitHub/docs when the paper is ambiguous

For each method output exactly these fields in a Markdown table:
method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence

Decision rules:
- Response-level discriminator/adversarial feedback is `borderline_strict` at best unless the objective clearly consumes non-scalar teacher-style supervision on exact student rollouts.
- Verbal critique, verbal score, API score, or LLM-as-judge feedback is usually `partial_opd` or `adjacent` unless converted into a distillation objective.
- Preference losses such as DPO/ORPO are not strict OPD unless the source proves online same-rollout teacher supervision and a distillation-style objective beyond static preference fitting.
- Reward-only PPO/GRPO/RLVR is not OPD even if the rollout is on-policy.
- Teacher-generated SFT traces are offline KD/SFT, not OPD.
- If a quote is unavailable, write “not found” and downgrade conservatively.
- Keep quotes short and locatable. Do not invent missing evidence.

Return a concise memo first, then the table.
```

## Completed WP1 white-box redirect spot-check prompt

This prompt was used after the WP2 secondary cleanup redirected PAD, ADPA, GAKD, daDPO, and CTPD into WP1. It is synthesized in `notes/wp1-whitebox-redirect-spotcheck-synthesis.md`. Do not rerun this queue unless a new primary source changes rollout freshness or objective evidence.

```text
You are a line-level primary-source auditor for the “On Policy Distillation Landscape” repository.

Task: audit the WP1 white-box redirect queue from WP2 secondary cleanup.

Output:
assets/research/wp1-whitebox-redirect-spotcheck-chatgpt.md

Priority:
PAD, ADPA, GAKD, daDPO, then CTPD.

Use primary sources only. For each method, extract short locatable C1/C2/C3 quotes and classify conservatively under the repo strict OPD definition.

Focus:
- PAD: confirm whether student-sampled responses are current-policy/on-policy during training; teacher scoring via token probabilities; JSD preference-distribution loss.
- ADPA: confirm student-generated state source, teacher/reference probability access, and whether the advantage-guided objective is strict OPD, borderline, or partial.
- GAKD: resolve whether reverse-KL/adversarial losses are computed on student rollouts or teacher-generated/corpus sequences.
- daDPO: check white-box teacher distribution use, sampled teacher/student pairs, and off-policy/static preference constraints.
- CTPD: check cross-tokenizer teacher log-prob projection, static preference data, and whether any current student rollout loop exists.

Do not reopen WP2 black-box primary queue: GAD, PRISM, OVD, SODA, ORPO-Distill.

Deliver:
concise memo + Markdown table matching assets/research/wp2-line-audit-chatgpt.md columns:
method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence

Classification rules:
- strict_opd requires current student rollout, teacher supervision on that same rollout, and a richer-than-reward distillation objective.
- White-box teacher logits/probabilities are allowed in WP1, but static preference optimization, offline teacher traces, teacher-generated data, or scalar reward/RL objectives are not strict.
- If only a subcomponent is strict but the full method mixes off-policy/static data, classify the method-level row conservatively as partial or borderline and explain.
```

## Completed WP3 teacher-free/self-play prompt

This prompt was used to resolve the WP3 teacher-free/self-play batch. It is synthesized in `notes/wp3-teacher-free-selfplay-synthesis.md`. Do not rerun this queue unless new self-play or privileged-context methods are added.

```text
Audit WP3: teacher-free, self-play, privileged-context, previous-checkpoint, and self-distillation OPD.

Use primary sources only. Build a memo, evidence ledger, candidate rows, adjacent ledger, and gap map.

Strict OPD requires:
C1 current student rollout;
C2 supervision from previous checkpoint, privileged self-view, reference model, self-play opponent, verifier, or other teacher substitute on that same rollout;
C3 an objective consuming that signal as distillation-style supervision, not only reward, SFT, static DPO, or self-training.

Prioritize:
SPIN, OPSD, OPSDC/CRISP, SDPO, OPCD, OEL, HDPO, GATES, Privileged Information Distillation, SDFT, self-play DPO/SPO/IPO variants that are often mislabeled OPD.

Return conservative labels: strict_opd, borderline_strict, partial_opd, adjacent, not_opd, unclear.
```

## Completed WP4 reasoning and OPD+RL prompt

This prompt was used to resolve the WP4 reasoning and OPD+RL batch. It is synthesized in `notes/wp4-reasoning-opd-rl-synthesis.md`. Do not rerun this queue unless new reasoning KD/RL methods are added.

```text
You are a deep research agent for the “On Policy Distillation Landscape” repository.

Task: audit WP4 — reasoning and OPD+RL hybrids for math, code, long-CoT, and reasoning-heavy LLM training.

Use primary sources only:
- arXiv PDF or HTML
- OpenReview paper PDF
- official project page, technical report, or GitHub/docs when the paper is ambiguous

Strict OPD requires all three:
C1. The current student/policy generates the training rollout, reasoning trace, response, or prefix.
C2. A teacher, reference model, previous checkpoint, privileged-context model, discriminator, or other teacher-style source supervises that exact student-generated state.
C3. The objective consumes that signal as distillation-style supervision, not only scalar reward, verifier reward, static DPO, or SFT on teacher traces.

Prioritize:
G-OPD, REOPOLD, Fast OPD, PACED, KDRL, RLKD, SCOPE, Lightning OPD, HDPO, SOD, MiMo-V2-Flash MOPD, Nemotron-Cascade2, Qwen3 OPD stage, any alternating KD/RL or KL+GRPO methods not already resolved.

Focus questions:
- Which methods combine dense teacher token/log-prob supervision with GRPO/PPO/RLVR?
- Which only use reward-only RLVR despite on-policy rollouts?
- Which have strict subcomponents but partial method-level recipes?
- How do methods handle long-CoT drift, mode collapse, forgetting, length inflation, and teacher over-imitation?
- What compute or serving cost is introduced by teacher scoring during RL?

Deliver:
1. Concise memo.
2. Evidence ledger with:
   method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence
3. Candidate Markdown rows only for source-supported table changes.
4. Adjacent/false-positive ledger for reward-only RLVR, static DPO, and SFT-on-trace methods.
5. Gap map with unresolved methods needing line-level audit.

Classification rules:
- Dense teacher logits/log-probs, privileged self-teacher KL, or teacher-derived token log-ratio on current student rollouts can satisfy C3 even if written in RL notation.
- Reward-only GRPO/PPO/RLVR is adjacent, not strict OPD.
- If a strict KD/RKL arm is mixed with reward-only RL or off-policy traces, classify the method-level row conservatively as partial or borderline.
- Static preference optimization and SFT on teacher reasoning traces are not strict OPD.
- Do not invent missing evidence; write “not found” and downgrade conservatively.
```

## Completed WP5 industrial and systems prompt

This prompt was used to resolve the WP5 industrial and systems batch. It is synthesized in `notes/wp5-industrial-systems-synthesis.md`. Do not rerun this queue unless new official industrial reports, training recipes, or framework docs appear.

```text
You are a deep research agent for the “On Policy Distillation Landscape” repository.

Task: audit WP5 — industrial and systems OPD claims. Verify official evidence for on-policy teacher supervision, multi-teacher routing, speculative/draft-model OPD, cross-tokenizer/logit serving, and training-system support.

Output one file:
assets/research/wp5-industrial-systems-[agent-name].md

Use only official or primary sources:
- official technical reports
- official blogs and model cards
- official GitHub repositories, docs, release notes, or examples
- arXiv PDF/HTML from the model or framework authors

Do not rely on media posts, SEO pages, social posts, or secondary summaries for final labels.

Strict OPD requires all three:
C1. Current student/policy/draft model generates the rollout, sequence, prefix, token, or action state used for training.
C2. A teacher, reference model, target model, privileged model, domain teacher, or multi-teacher router supervises that exact student-generated state.
C3. The objective consumes that supervision as distillation-style signal beyond scalar reward.

Prioritize:
Qwen3 OPD stage, Nemotron-Cascade2, MiMo-V2-Flash MOPD, Gemma2 post-training, DistillSpec, Speculative KD, Lightning OPD, Fast OPD, Minitron/Nemotron pruning plus distillation, Qwen and DeepSeek distilled checkpoints, and framework support in NeMo RL, verl, OpenRLHF, and TRL GKDTrainer.

Also scan official reports for false positives:
Qwen-VL/Qwen3-VL, InternVL, Gemma/Gemma 3, MiniCPM, DeepSeek-R1 distilled models, RL-only GRPO/RLVR systems, synthetic-data distillation, SFT-on-traces, static DPO, and pruning-only pipelines.

Focus questions:
- Which official reports expose exact C1/C2/C3 evidence?
- Which claims are stage-specific strict OPD versus whole-pipeline overclaims?
- Which are offline KD/SFT, reward-only RL, synthetic data, or static preference tuning?
- What systems tricks reduce teacher forward-pass cost: cached log-probs, sampled-token log-probs, prefix scoring, multi-teacher routing, batching, speculative draft alignment, cross-tokenizer projection, or latency hiding?
- Which frameworks can implement strict OPD out of the box, and which only support adjacent RLVR/KD?

Deliver:
1. Concise memo.
2. Evidence ledger:
   method/system | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | systems_mechanism | supervision_granularity | rollout_freshness | final_label | reason_to_downgrade | confidence
3. Candidate table rows/updates.
4. Adjacent/false-positive ledger.
5. Gap map.

Classification rules:
- White-box teacher logits/log-probs, target-model probabilities, draft-model acceptance distributions, and multi-teacher token advantages can satisfy C2/C3 only when evaluated on current student or draft-generated states.
- A full industrial pipeline should not be labeled strict just because one stage is strict; label the stage and downgrade the method-level pipeline.
- Cached or precomputed teacher log-probs are useful systems evidence but usually fail strict current-policy freshness.
- SFT on teacher traces, distilled checkpoints released after offline KD, and reward-only GRPO/RLVR are not strict OPD.
- Framework support is not method evidence unless the docs/examples show the required rollout and teacher-supervision loop.
- If evidence is missing, write “not found” and downgrade conservatively.
```

## Completed WP6 agentic and interactive OPD prompt

This prompt was used to resolve the WP6 priority batch. It is synthesized in `notes/wp6-agentic-opd-synthesis.md`. Do not rerun the broad WP6 prompt unless a major new agentic OPD family appears.

```text
You are a deep research agent for the “On Policy Distillation Landscape” repository.

Task: audit WP6 — agentic and interactive OPD. Focus on multi-turn, tool-using, code, browser/GUI, environment-interacting, and trajectory-level methods where the student policy visits states and a teacher/expert/privileged model may supervise those exact states.

Output one file:
assets/research/wp6-agentic-opd-[agent-name].md

Use primary sources only:
- arXiv PDF/HTML
- OpenReview paper PDF
- official project pages
- official GitHub/docs
- official model cards or technical reports

Do not rely on media posts, social posts, or secondary summaries for final labels.

Strict OPD requires all three:
C1. The current student/policy/agent generates the tool calls, actions, messages, code edits, browser/GUI steps, environment trajectory, or intermediate state used for training.
C2. A teacher, expert policy, verifier-with-distillation-signal, privileged-context model, previous checkpoint, discriminator, or human/expert demonstrator supervises that exact visited state.
C3. The objective consumes that supervision as distillation-style signal beyond scalar reward, success/failure reward, static preference, or SFT on traces.

Prioritize:
MAD-OPD, OEL, OPCD, SOD, VLA-OPD, SCoRe, Reflexion-style self-correction, SWE-agent/code-agent training reports, tool-call correction methods, browser/GUI agent training, environment-feedback methods, multi-turn web agents, and any method claiming online experience learning, agent distillation, or tool-use distillation.

Also scan false positives:
offline behavior cloning from expert traces, SFT on tool-use trajectories, rejection sampling, reward-only RL agents, RLHF/RLAIF for tools, static DPO on agent traces, self-correction datasets, and verifier-only success rewards.

Focus questions:
- Does the student agent generate the trajectory being supervised?
- Does the teacher/expert correct or score the exact visited state, tool call, action token, or earliest-error state?
- Is the consumed signal token/action-level distillation, trajectory-level correction, discriminator feedback, privileged-context KL, or only scalar reward?
- Are corrections online/current-policy, per-iteration, replay-buffer stale, or static?
- How are safety constraints handled during on-policy exploration?
- What is the systems cost: environment resets, tool execution, teacher calls, rollout replay, and delayed credit assignment?

Deliver:
1. Concise memo.
2. Evidence ledger:
   method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | state_or_action_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence
3. Candidate table rows/updates.
4. Adjacent/false-positive ledger.
5. Gap map.

Classification rules:
- Tool/action-token teacher supervision on current student trajectories can satisfy strict OPD.
- Privileged-context self-teacher supervision can satisfy strict OPD only when it scores the same student-generated trajectory/state.
- Reward-only success metrics, unit-test pass/fail, verifier reward, environment reward, or scalar judge reward are adjacent, not strict.
- SFT or behavior cloning on expert/tool traces is offline imitation, not OPD.
- Static DPO/preference optimization on agent traces is not strict OPD unless the source proves online same-rollout teacher supervision and a distillation-style objective.
- If only one substage is strict, classify the method-level row conservatively as partial or borderline and explain.
- If evidence is missing, write “not found” and downgrade conservatively.
```

## Completed WP6 agentic candidate line-audit prompt

This prompt has been run and merged. See [wp6-agentic-candidate-line-audit-synthesis.md](wp6-agentic-candidate-line-audit-synthesis.md).

## Completed WP7 multimodal frontier audit prompt

This prompt has been run and merged. See [wp7-multimodal-frontier-synthesis.md](wp7-multimodal-frontier-synthesis.md).

## Completed WP7 candidate line-audit prompt

This prompt has been run and merged. See [wp7-candidate-line-audit-synthesis.md](wp7-candidate-line-audit-synthesis.md).

## Completed WP8 evaluation methodology prompt

This prompt has been run and merged. See [wp8-evaluation-methodology-synthesis.md](wp8-evaluation-methodology-synthesis.md). The original prompt is kept below as a reusable evaluation-audit template.

```text
You are an evaluation-methodology auditor for the “On Policy Distillation Landscape” repository.

Task: run WP8 evaluation methodology. Do not discover new OPD papers and do not relitigate strictness labels. Use the repo's current tables and notes as the method set, then design benchmark and metric bundles that can fairly compare strict OPD, borderline/partial OPD, adjacent RLVR/RLHF/DPO, and offline KD.

Output one file:
assets/research/wp8-evaluation-methodology-[agent-name].md

Primary repo inputs:
- tables/opd_papers.md
- tables/vlm_opd_papers.md
- tables/benchmarks.md
- tables/adjacent_work.md
- notes/wp7-candidate-line-audit-synthesis.md
- taxonomies/strict-opd-definition.md

Use primary sources for benchmarks where possible:
- official benchmark papers
- official leaderboard or dataset pages
- official project/GitHub docs

Cover these evaluation domains:
LLM reasoning and instruction following; long-context; calibration/uncertainty; black-box/API response-level alignment; teacher-free/self-play; OPD+RL hybrids; industrial/system efficiency; VLM/MLLM visual reasoning; video temporal grounding; speech/cross-modal OPD; VLA/robotics; GUI grounding/GUI agents; agentic tool/browser/code tasks; diffusion/flow image generation borderline methods.

Must answer:
- Which benchmark bundles best separate OPD gains from reward-only RLVR gains?
- Which benchmarks actually test exposure-bias reduction or recovery from self-generated errors?
- Which metrics capture teacher uncertainty preservation, calibration, entropy, and distributional fidelity?
- Which benchmark families are contaminated, private, brittle, or too accuracy-only for OPD claims?
- How should evaluation differ for token/logit OPD, response-level black-box OPD, privileged self-OPD, action-token OPD, and diffusion/flow borderline OPD?
- What adjacent baselines should accompany each strict OPD family?
- What benchmark rows should be added or updated in tables/benchmarks.md?

Deliver:
1. Concise memo with recommended evaluation philosophy.
2. Benchmark bundle table:
   bundle | modality | target_methods | benchmark_names | official_urls | metrics | why_it_tests_opd | adjacent_baselines | contamination_or_private_risk | priority
3. Metric taxonomy table:
   metric_family | measures | applies_to | example_metrics | failure_modes_caught | limitations
4. Candidate updates for tables/benchmarks.md using the existing columns where possible.
5. Gap map for missing benchmarks or evaluation methods.

Decision rules:
- Do not promote or downgrade methods; classify evaluation needs only.
- Do not use benchmark popularity as evidence of OPD relevance.
- Prefer benchmark bundles that include OPD-specific stress tests: rollout drift, OOD prompts, long-horizon credit assignment, calibration/entropy, teacher disagreement, and self-generated error recovery.
- Keep strict OPD methods and adjacent controls in separate columns.
- If a benchmark source is unavailable, write "not found" and do not suggest a table row for it.
```

## Completed WP9 post-WP8 verifier prompt

This prompt has been run and merged. See [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md). Keep the prompt as a reusable template for future large-batch verifier passes. Current outcomes from that pass supersede the audit set below: PRISM and OEC are now `adjacent`.

```text
You are a red-team verifier for the “On Policy Distillation Landscape” repository.

Task: run a post-WP8 verifier pass. Do not discover new papers. Do not evaluate benchmark quality. Independently audit whether the assigned strict_opd and borderline_strict rows still satisfy the repo C1/C2/C3 definition after the WP6/WP7/WP8 merges.

Output one file:
assets/research/wp9-post-wp8-verifier-[agent-name].md

Primary repo inputs:
- tables/opd_papers.md
- tables/vlm_opd_papers.md
- tables/adjacent_work.md
- notes/wp6-agentic-opd-synthesis.md
- notes/wp6-agentic-candidate-line-audit-synthesis.md
- notes/wp7-multimodal-frontier-synthesis.md
- notes/wp7-candidate-line-audit-synthesis.md
- notes/wp8-evaluation-methodology-synthesis.md
- taxonomies/strict-opd-definition.md

Priority audit set:
GUI-SD, LiteGUI, TCOD, Skill-SD, OEC, OpenClaw-RL OPD component, VLA-OPD, MAD-OPD/OPAD, SOD, OPCD, VOLD, Video-OPD, X-OPD, Uni-OPD, PRISM, KEPO, D-OPSD, Flow-OPD, Qwen3-VL/Qwen3-Omni-related rows, Qwen3 OPD stage, MiMo-V2-Flash, Nemotron-Cascade2, and any row currently marked strict_opd or borderline_strict with verification_status source_verified after 2026-05-13 edits.

Use primary sources only for final evidence:
- arXiv PDF or HTML
- OpenReview paper PDF
- official technical report
- official project/GitHub/docs when the paper is ambiguous
- official model cards/blogs for industrial rows

For each method output exactly this table:
method | current_label | verified_label | primary_source_url | exact_section | C1 quote | C2 quote | C3 quote | rollout_freshness | teacher_access | supervision_granularity | objective_family | pass_or_fail | downgrade_reason | confidence

Rules:
- Strict OPD requires current student rollout, teacher/expert/discriminator/reference/privileged supervision on that same rollout/state, and objective consumption of a richer-than-scalar-reward distillation signal.
- Action-token, coordinate-token, privileged-context, previous-checkpoint, or teacher-field supervision can satisfy C2 only if it supervises the student-visited state being trained.
- Reward-only RLVR, unit-test pass/fail, environment reward, verifier scalar reward, static DPO/preference optimization, offline teacher traces, teacher-generated data, or behavior cloning are not strict.
- If only a substage or subcomponent is strict, keep the method-level row borderline/partial unless the table already scopes the row to that stage/component.
- If source evidence is not locatable, write “not found” and downgrade conservatively.
- Keep quotes short and locatable.
- Do not invent missing evidence or infer C1/C2/C3 from benchmark results.

Return a concise memo first, then the table, then a short list of recommended table edits.
```

## WP9 red-team verifier prompt

Use this prompt after a discovery batch produces `strict_opd` or `borderline_strict` candidates.

```text
You are a red-team verifier for the “On Policy Distillation Landscape” repository.

Task: independently audit whether each candidate is strict_opd, borderline_strict, partial_opd, adjacent, not_opd, or unclear.

Strict OPD requires all three:
C1. The student/current policy generates the training rollout.
C2. A teacher/expert/discriminator/reference/privileged-context model supervises that exact student rollout.
C3. The training objective consumes that supervision signal, not just a reward-only scalar.

Use primary sources only where possible: arXiv, OpenReview, official technical reports, official blogs, official GitHub/docs. Do not rely on SEO pages, LinkedIn summaries, random media, or secondary blog posts for final classification.

Audit these candidates:
MiniLLM, GKD, DistiLLM, DistiLLM-2, GAD, Lion, OVD, DistillSpec, Speculative KD, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD, PACED, AdaSwitch, BOND, Lightning OPD, DASD, DDT, OPSD, OPSDC/CRISP, GATES, SDPO, OPCD, OEL, HDPO, SDFT, Privileged Information Distillation, SCOPE, RLKD, KDRL, LUFFY, Qwen3 OPD stage, Gemma2 KD/post-training, MiMo-V2-Flash, Nemotron-Cascade2, Thinking Machines Lab On-Policy Distillation blog, VOLD, Video-OPD, X-OPD, Uni-OPD, PRISM, VLA-OPD.

Also check whether these are hallucinated, name-collided, or not OPD:
DistillDirect, DBKD, TED as OPD, SCOPE as “Self-play Contrastive On-Policy Evaluation”.

For each item output:
- method
- primary_source_url
- source_status: primary_verified / secondary_only / not_found
- rollout_source_evidence
- rollout_freshness_evidence
- teacher_kind
- teacher_signal_evidence
- loss_objective_evidence
- strictness_verdict
- confidence: high / medium / low
- downgrade_reason if not strict
- suggested Markdown table row fields
- notes on ambiguity or conflicts

Return a memo first, then a table. Do not invent missing evidence.
```

## WP9 line-level primary-source audit prompt

Use this after `notes/wp9-verifier-synthesis.md`. Do not send another broad survey prompt. This pass is only for methods already marked `human_spotcheck_needed` or `wp9_conflict`.

```text
You are a line-level primary-source auditor for the “On Policy Distillation Landscape” repository.

Task: open the original PDF/HTML/official report for each assigned method and extract exact evidence for the C1/C2/C3 OPD test. Do not write a broad survey. Do not introduce new candidate papers unless the assigned paper has a clear name collision that must be resolved.

Strict OPD requires:
C1. The student/current policy generates the training rollout.
C2. A teacher/expert/discriminator/reference/privileged-context model supervises that exact student rollout.
C3. The training objective consumes that supervision signal, not just a reward-only scalar.

Priority methods:
Uni-OPD, MAD-OPD, VOLD, MiMo-V2-Flash, Gemma2 post-training, G-OPD, REOPOLD, PACED, SDPO, OPCD, OEL, SCOPE, KDRL, Lightning OPD, DistiLLM-2.

Use only primary sources for final evidence:
- arXiv PDF or HTML
- OpenReview paper PDF
- official technical report
- official project/GitHub/docs when the paper is unavailable

Output exactly these fields in a Markdown table:
method | source_url | exact_section | C1 quote | C2 quote | C3 quote | rollout_freshness | final_label | reason_to_downgrade | confidence

Rules:
- Quotes should be short but exact enough to locate the source line.
- If a quote is unavailable, write “not found” and downgrade conservatively.
- If the method has multiple stages, label only the stage that satisfies the evidence.
- If a method is strict only for a subcomponent, state that subcomponent in `exact_section` and `reason_to_downgrade`.
- If a response-level discriminator or scalar/verbal reward is the only supervision, default to `partial_opd` or `adjacent`, not `strict_opd`.
- Do not infer C1/C2/C3 from the title alone.
- Do not invent missing evidence.

Return a short memo first, then the table.
```

## Completed WP1 white-box candidate line-audit prompt

This prompt was used after `notes/wp1-whitebox-synthesis.md` to close the WP1 candidate queue. Keep it as a template, but do not rerun it unless new white-box candidates are added.

Important: this was not the older 15-method spot-check queue for Uni-OPD, MAD-OPD, VOLD, MiMo-V2-Flash, Gemma2, G-OPD, REOPOLD, PACED, SDPO, OPCD, OEL, SCOPE, KDRL, Lightning OPD, and DistiLLM-2. That batch is synthesized in `notes/wp1-line-audit-synthesis.md`; the candidate queue below is synthesized in `notes/wp1-candidate-line-audit-synthesis.md`.

```text
You are a line-level primary-source auditor for the “On Policy Distillation Landscape” repository.

Task: audit only the WP1 white-box OPD candidate queue. Do not write a broad survey. Do not add new papers unless an assigned method has a name collision that must be resolved.

Strict OPD requires all three:
C1. The student/current policy generates the training rollout or prefix being supervised.
C2. A teacher, reference model, previous checkpoint, privileged self-view, or projected teacher distribution supervises that exact student-generated state.
C3. The training objective consumes that supervision signal as a distillation-style objective, not just a scalar reward.

Audit these candidates:
vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens, DSKD v2, SelecTKD, ToDi, HPD.

Use only primary sources:
- arXiv PDF or arXiv HTML
- OpenReview paper PDF
- official project page
- official GitHub code or docs when the paper is ambiguous

For each method output exactly these fields in a Markdown table:
method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | rollout_freshness | objective_family | final_label | reason_to_downgrade | confidence

Decision rules:
- For vOPD/AOPD/G-OPD-style methods, RL notation is acceptable only if the reward or advantage is computed from teacher logits or log-probs on student-generated tokens.
- For TIP/Rock Tokens/SOD, token selection or weighting does not prove OPD by itself; verify the base rollout and teacher-logit objective.
- For SimCT/DSKD, cross-tokenizer alignment can satisfy C2 only if the teacher distribution is evaluated on student-generated text states after alignment.
- For OPSDL, privileged self-teacher can satisfy C2 only if it supervises the same generated tokens from the long-context student.
- For DP-OPD, privacy noise does not affect strictness; verify the underlying OPD loop.
- For SelecTKD/ToDi/HPD, downgrade unless the source explicitly shows current student rollouts plus same-rollout teacher supervision.
- If any quote is unavailable, write “not found” and downgrade conservatively.
- Keep quotes short and locatable. Do not invent missing evidence.

Return a concise memo first, then the table.
```

## Completed narrow audit prompt

This prompt was used to close the VLA-OPD and PRISM spot-check queue. Keep it as a template for future narrow audits.

Supersession note: for PRISM itself, [wp9-post-wp8-verifier-synthesis.md](wp9-post-wp8-verifier-synthesis.md) applies the stricter C3 rule and finalizes the current label as `adjacent`.

```text
You are a line-level primary-source auditor for the “On Policy Distillation Landscape” repository.

Task: audit only VLA-OPD and PRISM. Do not write a survey. Do not add new papers.

Strict OPD requires:
C1. The student/current policy generates the training rollout.
C2. A teacher/expert/discriminator/reference/privileged-context model supervises that exact student rollout.
C3. The training objective consumes that supervision signal, not just a reward-only scalar.

Sources to inspect:
- VLA-OPD: https://arxiv.org/abs/2603.26666
- PRISM: https://arxiv.org/abs/2604.28123

For each method output exactly:
method | source_url | exact_section | C1 quote | C2 quote | C3 quote | supervision_granularity | final_label | reason_to_downgrade | confidence

Decision rules:
- VLA action-token supervision can satisfy strict OPD if the student generates the action trajectory and an expert teacher supervises those same action tokens with a KL/distillation objective.
- PRISM-style methods should remain below strict unless the paper shows non-scalar same-rollout distillation supervision. A response-level discriminator, verbal score, or policy-gradient reward should be `borderline_strict`, `partial_opd`, or `adjacent`, not `strict_opd`.
- If a quote is unavailable, write “not found” and downgrade conservatively.
- Do not infer from the title alone.
- Do not invent missing evidence.

Return a short memo first, then the table.
```
