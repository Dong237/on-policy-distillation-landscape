# Adjacent Work

Adjacent work is valuable because it defines the boundary of OPD.

## Include

- RLVR or GRPO methods that optimize verifiable rewards without teacher distillation.
- Offline KD methods for LLMs, VLMs, or MLLMs.
- DPO, RLHF, or preference optimization reports.
- Synthetic-data distillation.
- Tool-use or agent trajectory tuning without student on-policy feedback.
- Industrial reports with incomplete OPD evidence.

## Exclude from Strict OPD

A method should remain adjacent when:

- Student rollouts exist, but feedback is reward-only.
- A teacher exists, but only generated the static dataset.
- The method has online RL, but no teacher/discriminator/reference supervision.
- The method has distillation, but no student on-policy training states.

## Why this matters

VLM and MLLM research currently has many strong RLVR systems and offline KD systems. Treating them as strict OPD would make the landscape misleading and would hide the actual gap: robust multimodal OPD is still underdeveloped.

