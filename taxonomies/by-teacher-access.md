# Taxonomy by Teacher Access

Teacher access is the second survey axis. It determines what information the student can obtain from the teacher or feedback source.

| Category | Description | Typical strictness |
|---|---|---|
| `white_box` | Teacher logits, log-probs, or token distributions are available. | Usually strict if applied to student rollouts. |
| `black_box` | Teacher returns responses, rankings, verbal scores, or API-level feedback. | Borderline or strict depending on objective. |
| `teacher_free` | No external teacher; student uses self-play, previous checkpoints, or privileged context. | Strict or partial depending on feedback construction. |
| `multi_teacher` | Multiple teachers provide routed, weighted, or debated supervision. | Usually strict if applied to student rollouts. |
| `expert_policy` | Expert policy supervises actions or action tokens. | Strict for VLA/agent settings if on-policy. |
| `no_teacher` | Reward-only or environment-only signal. | Usually adjacent, not OPD. |

Verifier guidance clarifies that `teacher_free` does not mean "no teacher-like signal." Strict teacher-free OPD still needs a concrete privileged self, previous checkpoint, or context-conditioned teacher path. Pure self-play, scalar verifier reward, or environment feedback alone is not enough.

For `expert_policy`, action-token VLA supervision can be strict OPD when the student generates the action trajectory and the expert teacher supervises those same action tokens.

## Teacher kind

`teacher_access_regime` describes access. `teacher_kind` normalizes what is actually supervising the student:

| Value | Meaning |
|---|---|
| `larger_llm` | A stronger text LLM provides logits, log-probs, labels, or feedback. |
| `larger_mllm` | A stronger multimodal teacher supervises MLLM states. |
| `text_teacher_llm` | Text teacher supervises cross-modal or VLM reasoning traces. |
| `multimodal_teacher` | Teacher consumes visual/video/multimodal input directly. |
| `expert_policy` | Expert action policy supervises VLA or agent actions. |
| `discriminator` | Learned discriminator or adversarial judge provides feedback. |
| `reference_model` | Reference/target model defines a distributional target. |
| `previous_checkpoint` | Older self checkpoint acts as teacher. |
| `privileged_self` | Same model under privileged context, answer, memory, or alternate view acts as teacher. |
| `multi_teacher` | Multiple teachers are routed, debated, ensembled, or domain-selected. |
| `reward_model_or_verifier` | Scalar reward/verifier only; usually adjacent unless converted into distillation supervision. |
| `none` | No teacher-like source. |
| `not_disclosed` | Industrial report does not disclose enough. |

## VLM-specific issues

VLM teachers may have stronger perception but weaker reasoning, or vice versa. Record teacher modality and failure mode explicitly in `tables/vlm_opd_papers.md`.
