# Open Problems

- Robust teacher reliability on student-visited states.
- OPD scaling laws across student size, teacher size, and rollout budget.
- VLM perception versus reasoning disentanglement.
- Text-only teacher mismatch for visual reasoning.
- Token-level versus sequence-level supervision.
- Token selection during OPD: entropy gates, importance scores, rock-token freezing, and step-level weighting.
- Variance reduction for reverse-KL OPD without changing the teacher signal.
- Systematic divergence scheduling across training phase, token type, and task domain.
- Cross-tokenizer OPD (SimCT is the first source-verified strict method in this repo; DSKD v2 is partial at method level; most others remain offline or adjacent).
- Renyi, alpha-divergence, and Bregman-style OPD objectives are not yet mapped in published LLM OPD work.
- Black-box teacher limitations.
- Multi-teacher conflict resolution.
- OPD + RLVR stability.
- Long-CoT drift.
- Compute and serving overhead.
- Agent-level and GUI-level OPD.
- Video and temporal grounding OPD beyond narrow TVG settings.
