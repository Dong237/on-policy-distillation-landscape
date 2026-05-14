# Taxonomy by Loss Objective

OPD methods differ in how teacher feedback is converted into an optimization target.

| Objective | Notes |
|---|---|
| Forward KL | Mode-covering; may overfit low-probability teacher regions. |
| Reverse KL | Mode-seeking; common in MiniLLM-style and VLA-OPD-style formulations. |
| Jensen-Shannon divergence | Sometimes used for stability in multi-agent or agentic settings. |
| Cross-entropy | Can be used for hard teacher labels, but needs on-policy student samples to count as OPD. |
| Discriminator loss | Used in black-box or adversarial OPD variants. |
| Hybrid KL + reward | OPD plus RLVR/RL objective; strict only when the KL/distillation term is on student rollouts. |
| Process reward | Adjacent unless the process signal acts as a teacher distribution or distillation target. |

## Recording guidance

Use `loss_objective` for the mathematical objective and `supervision_signal` for the source of feedback. Do not infer one from the other.

## White-Box Objective Map

White-box audits show that OPD is no longer only a forward-KL versus reverse-KL question. Record the precise objective family because it affects stability, compute, and whether a method should be split into strict and partial substages.

| Objective family | OPD status rule | Examples |
|---|---|---|
| Generalized KL / JSD on student states | Strict when the sampled states are current student rollouts. | GKD pure on-policy, DistillSpec draft-generated variant. |
| Sequence or token reverse KL | Strict when teacher log-probs are evaluated on student samples. | MiniLLM, Fast OPD, REOPOLD. |
| Skew KL / skew reverse KL | Usually partial if paired with replay, stale, or batched student outputs. | DistiLLM, DistiLLM-2. |
| Entropy-gated or adaptive KL | Strict only when the gate or bridge target is computed on student-visited states. | Entropy-Aware OPD, Veto, CaOPD candidates. |
| Dense teacher log-ratio policy objective | Can be strict despite RL notation if the reward is derived from teacher logits/log-probs on student rollouts. | G-OPD, REOPOLD, KDRL subcomponent. |
| Offline cached log-ratio objective | Partial or adjacent when rollouts are frozen from a reference/SFT policy. | Lightning OPD. |
| Interleaved token KL | Partial when teacher replaces or completes part of the sequence. | Speculative KD, AdaSwitch. |
| Token selection or reweighting | Does not change strictness by itself; strictness follows the underlying rollout and teacher signal. | TIP, Rock Tokens, SOD, SelecTKD. |
| Cross-tokenizer aligned KL | Strict only when aligned teacher supervision is applied to student-generated text states. | SimCT is source-verified strict; DSKD v2 is partial at method level; ULD and MultiLevelOT stay adjacent by default. |
