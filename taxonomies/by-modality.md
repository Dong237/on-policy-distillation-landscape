# Taxonomy by Modality

| Modality | Scope | Current OPD maturity |
|---|---|---|
| LLM | Text-only language models. | Strongest OPD evidence and foundations. |
| VLM / MLLM | Image-text or multimodal language models. | Emerging; many adjacent RLVR works. |
| Video-language | Video understanding and grounding. | Early; Video-OPD is a strict seed. |
| VLA / embodied | Vision-language-action robotics and control. | Emerging extension; action-level OPD. |
| Agent / tool-use | Multi-step environments, browser, GUI, code agents. | Early; distinguish offline trajectory tuning from OPD. |
| Speculative decoding | Draft-target model distillation for inference. | Task-specific; often not OPD unless drafter rollouts are supervised. |

## VLM failure modes to track

- Visual perception error.
- Visual grounding error.
- Text-only teacher mismatch.
- Perception versus reasoning entanglement.
- Temporal grounding drift.
- GUI/action-state mismatch.

