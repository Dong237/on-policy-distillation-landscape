# Reproduction Recipes

Recipes should describe how to reproduce a training loop or evaluation result.

## Recipe template

```markdown
### Recipe name

- **Method:** paper/system
- **Framework:** TRL / verl / OpenRLHF / NeMo RL / custom
- **Models:** student and teacher
- **Data:** training and validation data
- **Rollout source:** student_on_policy / mixed_policy / off_policy
- **Teacher signal:** logits / log-probs / discriminator / reward
- **Loss:** KL / reverse KL / hybrid
- **Evaluation:** benchmarks and metrics
- **Compute:** GPUs, memory, runtime, serving requirements
- **Known gaps:** missing code, missing weights, unclear preprocessing
```

## TODO

- Add text GKD recipe.
- Add MiniLLM reverse-KL recipe.
- Add VOLD-style VLM OPD recipe after code availability or enough implementation detail exists.
- Add VLM GRPO baseline recipe as adjacent comparison.

