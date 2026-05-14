# Taxonomy by Training Stage

| Stage | Description | Classification caution |
|---|---|---|
| Cold-start | SFT or alignment before online training. | Usually not OPD by itself. |
| Post-SFT | Training after instruction or reasoning SFT. | Can be OPD if student rollouts are supervised. |
| Inside RL/RLVR | OPD loss is combined with RL objective. | Strict only if a distillation signal exists. |
| Post-RL | Distillation after RL-trained teacher or checkpoint. | Depends on rollout source. |
| Compression | Smaller model learns from a larger one. | Offline compression is not OPD. |
| Continual learning | Student adapts over time. | Strict only with on-policy supervised updates. |

## VLM note

Many VLM papers use SFT followed by GRPO/RLVR. That is not OPD unless the online stage includes teacher/discriminator/reference feedback on student-generated multimodal outputs.

