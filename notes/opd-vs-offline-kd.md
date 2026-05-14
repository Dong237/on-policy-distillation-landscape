# OPD vs Offline KD

Offline KD transfers knowledge from a teacher on a fixed dataset. The dataset may contain teacher logits, labels, rationales, or synthetic traces, but the student does not create the training states.

OPD requires teacher supervision on student-generated states.

## Examples

- LLaVA-KD: offline MLLM KD, `not_opd`.
- LLAVADI: offline MLLM KD study, `not_opd`.
- Mini-InternVL vision encoder distillation: offline compression, `not_opd`.
- GKD / MiniLLM: OPD because the student generates outputs and the teacher evaluates them.

