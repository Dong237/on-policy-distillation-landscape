# Compute and Systems

OPD is system-heavy because the teacher often needs to score student rollouts online.

## Track

- Teacher serving cost.
- Rollout generation cost.
- Logit/log-prob storage.
- Distributed training backend.
- Async rollout support.
- Multi-teacher serving.
- Cross-tokenizer overhead.
- VLM-specific image/video preprocessing cost.

## Industrial Patterns

- Sampled-token teacher log-probs can reduce bandwidth while staying strict when scored on current student tokens.
- Prefix-only teacher scoring is strict only over the supervised prefix window.
- Cached teacher log-probs remove live teacher serving but usually fail strict current-policy freshness.
- Multi-teacher routing can be strict when domain teachers score current student samples, but public reports rarely isolate routing latency.
- Cross-tokenizer serving frameworks are systems enablers; strictness depends on whether the recipe scores current student-generated text states.

## Failure modes

- Teacher bottleneck dominates training throughput.
- Online generation creates non-deterministic datasets.
- VLM teacher scoring requires expensive visual context.
- Long-CoT rollouts increase memory and latency.
