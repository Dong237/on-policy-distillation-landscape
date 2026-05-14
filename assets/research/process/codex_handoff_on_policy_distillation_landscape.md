# Codex Handoff: `on-policy-distillation-landscape`

## 1. Project context

We are creating a GitHub repository named:

```text
on-policy-distillation-landscape
```

The repository should be positioned as a **taxonomy-first, implementation-aware research map** of **On-Policy Distillation (OPD)** for **LLMs, VLMs, and MLLMs**.

This should **not** be just another flat paper list.  The value of this repo should be that it helps researchers understand the whole landscape:

- What truly counts as OPD?
- What is only partial OPD?
- What is adjacent but not OPD?
- How do OPD methods differ by teacher access, supervision signal, rollout source, modality, loss, and training stage?
- Which methods are reproducible?
- Which frameworks support OPD?
- What are the key open problems in LLM and VLM/MLLM OPD?

The repo should be useful both as:

1. a **research map** for understanding the OPD field, and  
2. an **implementation tracker** for researchers who want to reproduce or build OPD systems.

---

## 2. Core positioning

Use this as the core identity:

```text
On Policy Distillation Landscape is a taxonomy-first research map of On-Policy Distillation for LLMs, VLMs, and MLLMs.
```

Longer positioning:

```text
A structured landscape of On-Policy Distillation: definitions, papers, methods, implementations, and open problems for LLMs and VLMs.
```

The repo should focus on:

- LLM reasoning OPD
- VLM / MLLM OPD
- White-box OPD
- Black-box OPD
- On-policy self-distillation
- OPD + RL / RLVR hybrids
- Multi-teacher OPD
- Industrial post-training reports
- Frameworks and implementation recipes
- Failure modes and open problems

---

## 3. Definition of OPD

Use this strict definition throughout the repo:

> A method is **Strict OPD** only if:
>
> 1. the student or policy generates its own trajectories, completions, or intermediate states during training; and
> 2. a teacher, verifier, discriminator, privileged-context model, reference model, or other feedback source provides supervision on those student-generated trajectories; and
> 3. the objective consumes that supervision as distillation-style signal, not just scalar reward, static preference, or SFT on traces.

Methods that satisfy only part of the definition should be marked as:

```text
Partial OPD
```

Methods that are relevant but do not use on-policy student rollouts should be placed under:

```text
Adjacent Work
```

---

## 4. What should not count as strict OPD

By default, the following should **not** be classified as strict OPD:

- Plain SFT on teacher-generated traces
- Offline knowledge distillation on fixed datasets
- DPO on static preference pairs
- Generic RLHF or RLVR without distillation-style teacher feedback
- Self-training without explicit supervision on student-generated trajectories
- Evaluation-time reranking without training-time on-policy updates
- Generic online training that does not distill from a teacher/verifier/discriminator/reference signal

These can be included only in `Adjacent Work` if they are conceptually important.

---

## 4a. Current repo state as of 2026-05-13

WP0 through post-WP8 WP9 are merged. WP6 priority and candidate line audits are complete: LiteGUI has a strict stage-scoped row, TCOD/Skill-SD/OpenClaw-RL OPD component are borderline, OEC is now adjacent after post-WP8 verification, RLSD is partial, DGPO/RISE are adjacent, and GLM-5 cross-stage OPD is held out. WP7 confirms the strict multimodal core, upgrades GUI-SD to high-confidence/source-verified for GUI grounding, adds KEPO/D-OPSD/Flow-OPD as borderline multimodal rows, and keeps GTR-Turbo adjacent. WP8 adds evaluation-methodology coverage across text, VLM, GUI/agent, video, audio, robotics, and diffusion/flow. The post-WP8 WP9 verifier keeps the strict frontier rows stage-scoped, downgrades PRISM and OEC to adjacent, keeps Qwen3 OPD as source-weak stage-scoped strict, and leaves no immediate verifier queue open.

---

## 5. Desired repository structure

Create the repo with this structure:

```text
on-policy-distillation-landscape/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── CHANGELOG.md
├── CITATION.cff
│
├── assets/
│   ├── figures/
│   │   ├── opd-taxonomy.png
│   │   ├── opd-training-pipeline.png
│   │   └── llm-vs-vlm-opd.png
│   └── badges/
│
├── taxonomies/
│   ├── README.md
│   ├── strict-opd-definition.md
│   ├── by-teacher-access.md
│   ├── by-supervision-signal.md
│   ├── by-loss-objective.md
│   ├── by-training-stage.md
│   ├── by-modality.md
│   ├── by-rollout-source.md
│   └── adjacent-work.md
│
├── papers/
│   ├── README.md
│   ├── surveys-and-position-papers.md
│   ├── foundations.md
│   ├── white-box-opd.md
│   ├── black-box-opd.md
│   ├── on-policy-self-distillation.md
│   ├── opd-rl-hybrids.md
│   ├── multi-teacher-opd.md
│   ├── llm-reasoning.md
│   ├── vlm-mlm-multimodal.md
│   ├── agents-and-tool-use.md
│   ├── speculative-decoding.md
│   ├── industrial-reports.md
│   └── adjacent-but-not-opd.md
│
├── implementations/
│   ├── README.md
│   ├── frameworks.md
│   ├── codebases.md
│   ├── reproduction-recipes.md
│   ├── datasets.md
│   ├── benchmarks.md
│   ├── evaluation.md
│   ├── compute-and-systems.md
│   └── cross-tokenizer-distillation.md
│
├── tables/
│   ├── opd_papers.md
│   ├── vlm_opd_papers.md
│   ├── frameworks.md
│   ├── industrial_reports.md
│   ├── benchmarks.md
│   └── adjacent_work.md
│
├── notes/
│   ├── README.md
│   ├── opd-vs-sft.md
│   ├── opd-vs-offline-kd.md
│   ├── opd-vs-dpo.md
│   ├── opd-vs-rlhf.md
│   ├── opd-vs-rlvr.md
│   ├── opd-failure-modes.md
│   ├── llm-vs-vlm-opd.md
│   ├── teacher-reliability.md
│   ├── long-cot-drift.md
│   ├── entropy-and-uncertainty.md
│   └── open-problems.md
│
├── reading-paths/
│   ├── beginner.md
│   ├── llm-reasoning.md
│   ├── vlm-mlm.md
│   ├── black-box-opd.md
│   ├── implementation.md
│   └── industrial-opd.md
│
└── scripts/
    ├── arxiv_search.py
    ├── github_topic_search.py
    ├── paper_schema.py
    ├── validate_entries.py
    └── update_readme.py
```

---

## 6. README.md target structure

The top-level README should read like a mini landscape paper, not a simple link dump.

Use this structure:

```markdown
# On Policy Distillation Landscape

A taxonomy-first research map of **On-Policy Distillation (OPD)** for **LLMs, VLMs, and MLLMs**.

## Why this repo exists

## What is On-Policy Distillation?

## What counts as OPD?

## What does not count as OPD?

## Landscape at a glance

## Taxonomy

## Reading paths

## Papers

### Surveys and position papers
### Foundations
### White-box OPD
### Black-box OPD
### On-policy self-distillation
### OPD + RL / RLVR hybrids
### Multi-teacher OPD
### LLM reasoning
### VLM / MLLM / multimodal OPD
### Agents and tool use
### Speculative decoding
### Industrial reports

## Implementations and frameworks

## Benchmarks and evaluation

## Open problems

## Contributing

## Citation
```

---

## 7. Suggested README opening

Use or adapt this opening:

```markdown
# On Policy Distillation Landscape

**On Policy Distillation Landscape** is a taxonomy-first research map of **On-Policy Distillation (OPD)** for **LLMs, VLMs, and MLLMs**.

The goal is not only to collect papers, but to clarify what counts as OPD, how OPD methods differ, which methods are reproducible, and where the field is going.

We focus on:

- LLM reasoning OPD
- VLM / MLLM OPD
- White-box and black-box OPD
- On-policy self-distillation
- OPD + RL / RLVR hybrids
- Multi-teacher OPD
- Industrial post-training reports
- Frameworks and implementation recipes
- Failure modes and open problems
```

---

## 8. README landscape table

Include a table like this near the top of the README:

```markdown
## Landscape at a glance

| Family | Teacher Access | Signal | Typical Use Case | Strict OPD? |
|---|---|---|---|---|
| White-box OPD | Logits / log-probs | Token-level KL | LLM reasoning, compression | Usually yes |
| Black-box OPD | Responses only | Sequence / discriminator / reward | Closed teacher APIs | Often yes |
| OPSD | Same model + privileged context | Token / sequence supervision | Self-improvement, reasoning | Usually yes |
| OPD + RLVR | Teacher + verifier / reward | Hybrid KL + reward | Math/code reasoning | Depends |
| Multi-teacher OPD | Multiple teachers | Token / sequence / reward | Capability consolidation | Usually yes |
| VLM / MLLM OPD | Text or multimodal teacher | Hybrid | Visual reasoning, grounding | Emerging |
| Speculative OPD | Draft/target models | Token-level guidance | Decoding acceleration | Task-specific |
```

---

## 9. Paper entry schema

Every paper entry should include structured metadata. Use this schema:

```yaml
title:
year:
authors:
link:
code:
modality:
  - LLM
  - VLM
  - MLLM
  - Agent
  - Speculative Decoding
method_family:
  - White-box OPD
  - Black-box OPD
  - OPSD
  - OPD-RL hybrid
  - Multi-teacher OPD
teacher_access:
  - full logits
  - top-k logits
  - log-probs
  - sequence outputs
  - verifier
  - discriminator
  - same model with privileged context
supervision_granularity:
  - token
  - sequence
  - hybrid
rollout_source:
  - student on-policy
  - mixed-policy
  - off-policy
  - unclear
training_stage:
  - cold-start
  - post-SFT
  - inside RL/RLVR
  - post-RL
  - compression
  - continual learning
strictness:
  - strict OPD
  - partial OPD
  - adjacent
benchmarks:
summary:
why_it_matters:
limitations:
```

In Markdown files, each paper can be represented more compactly as:

```markdown
### Paper Title

- **Year:** 2026
- **Link:** ...
- **Code:** ...
- **Modality:** LLM / VLM / MLLM
- **Family:** White-box OPD / Black-box OPD / OPSD / OPD-RL hybrid
- **Teacher access:** ...
- **Signal:** ...
- **Rollout source:** ...
- **Strictness:** Strict OPD / Partial OPD / Adjacent
- **Summary:** ...
- **Why it matters:** ...
- **Limitations:** ...
```

---

## 10. Markdown table schemas

Historical bootstrap note: this section records the original scaffold-era schema request. It is not the current table contract. Current tables use the expanded validated schemas in `tables/*.md` and `scripts/paper_schema.py`; validate with `python3 scripts/validate_entries.py`.

Original scaffold schema examples:

### `tables/opd_papers.md`

```text
title,year,authors,link,code,modality,method_family,teacher_access,supervision_signal,supervision_granularity,rollout_source,training_stage,strictness,benchmarks,framework,summary,limitations
```

### `tables/vlm_opd_papers.md`

```text
title,year,link,backbone,teacher_type,vision_task,supervision_signal,uses_rlvr,benchmarks,strictness,code,notes
```

### `tables/frameworks.md`

```text
name,link,supports_llm,supports_vlm,teacher_logits,black_box_teacher,multi_teacher,cross_tokenizer,distributed_training,notes
```

### `tables/industrial_reports.md`

```text
name,organization,year,link,model_family,uses_opd,opd_type,teacher_access,training_stage,notes
```

### `tables/benchmarks.md`

```text
name,modality,task_type,link,used_by,notes
```

### `tables/adjacent_work.md`

```text
title,year,link,category,why_adjacent,why_not_strict_opd,notes
```

---

## 11. Suggested taxonomy content

### `taxonomies/strict-opd-definition.md`

Should include:

- strict OPD definition
- partial OPD definition
- adjacent work definition
- examples of what counts
- examples of what does not count
- checklist for classification

Checklist:

```markdown
## Strictness checklist

A method is strict OPD only if all are true:

- [ ] Student generates trajectories during training.
- [ ] Supervision is applied to those student-generated trajectories.
- [ ] Feedback source is teacher, verifier, discriminator, privileged model, reference model, or previous checkpoint.
- [ ] The training update depends on the on-policy student samples.

If one or more are unclear, mark as Partial OPD or Unclear.
```

### `taxonomies/by-teacher-access.md`

Categories:

- full logits
- top-k logits
- log-probs
- sequence outputs only
- verifier / reward model
- discriminator
- same model with privileged context
- previous checkpoint
- multi-teacher

### `taxonomies/by-supervision-signal.md`

Categories:

- token-level KL
- sequence-level CE
- reward / verifier score
- preference signal
- discriminator loss
- verbal feedback
- hybrid token + outcome signal

### `taxonomies/by-modality.md`

Categories:

- text-only LLM
- VLM / MLLM
- video-language model
- agent / tool-use
- speculative decoding

---

## 12. Suggested notes

The `notes/` directory should contain short explainer documents.

Important files:

### `notes/opd-vs-sft.md`

Explain that SFT usually imitates fixed teacher traces, while OPD supervises student-generated trajectories.

### `notes/opd-vs-offline-kd.md`

Explain that offline KD uses a fixed dataset, while OPD uses student rollouts.

### `notes/opd-vs-dpo.md`

Explain that DPO usually trains on static preference pairs, while OPD gives feedback on student-generated trajectories.

### `notes/opd-vs-rlvr.md`

Explain that RLVR optimizes rewards from verifiable outcomes, while OPD transfers teacher/verifier/discriminator signal on student trajectories. OPD + RLVR hybrids combine dense teacher supervision with outcome rewards.

### `notes/llm-vs-vlm-opd.md`

Explain that VLM/MLLM OPD has additional failure modes:

- visual perception error
- visual grounding error
- text-only teacher mismatch
- multimodal context mismatch
- perception vs reasoning disentanglement
- teacher reliability under visual input

### `notes/open-problems.md`

Include:

- Teacher reliability on student-visited states
- Teacher-student thinking-pattern mismatch
- Entropy and uncertainty calibration
- Long-CoT drift
- Token-level vs sequence-level supervision
- Cross-tokenizer OPD
- Black-box teacher limitations
- Multi-teacher conflict resolution
- VLM perception vs reasoning errors
- OPD + RLVR stability
- Compute and serving overhead

---

## 13. CONTRIBUTING.md requirements

`CONTRIBUTING.md` should emphasize that every contribution must include classification.

Use this structure:

```markdown
# Contributing to On Policy Distillation Landscape

Thank you for contributing.

This repo is taxonomy-first. Please do not add papers as plain links without classification.

## Before submitting

Check whether the work is:

- Strict OPD
- Partial OPD
- Adjacent work
- Not relevant

## Strict OPD criteria

A method is strict OPD only if:

1. The student generates trajectories or completions during training.
2. Teacher/verifier/discriminator/reference feedback is applied to those student-generated trajectories.

## Required fields

Every PR should include:

- Paper title
- Link
- Year
- Code link, if available
- Modality
- Teacher access
- Supervision signal
- Rollout source
- Training stage
- Strictness judgment
- One-sentence summary
- Why it matters
- Limitations

## Placement guide

- Use `papers/white-box-opd.md` for logit/log-prob teacher methods.
- Use `papers/black-box-opd.md` for response-only teacher methods.
- Use `papers/on-policy-self-distillation.md` for same-model or privileged-context methods.
- Use `papers/vlm-mlm-multimodal.md` for VLM/MLLM methods.
- Use `papers/adjacent-but-not-opd.md` for relevant but non-OPD methods.

## Avoid

Please avoid adding:

- Plain SFT papers without on-policy student rollouts
- Generic RLHF/RLVR papers without distillation signal
- Offline KD papers without student-generated training states
- Blog posts without technical substance
```

---

## 14. Initial GitHub issues to create

Create these as starter issues:

```text
[Taxonomy] Define strict OPD vs partial OPD
[Survey] Add foundational OPD / GKD / MiniLLM-style papers
[Survey] Add 2024–2026 LLM reasoning OPD papers
[Survey] Add VLM / MLLM OPD papers
[Survey] Add black-box OPD methods
[Survey] Add OPD + RLVR hybrid methods
[Survey] Add multi-teacher OPD methods
[Implementation] Track frameworks supporting OPD
[Table] Build Markdown table schema for all entries
[Note] Write OPD vs RLVR explainer
[Note] Write VLM-specific OPD failure modes
[Automation] Add validation script for paper metadata
```

---

## 15. Historical bootstrap: what Codex was originally asked to do

This section is retained for provenance only. The initial repository scaffold and WP0 through post-WP8 WP9 are now merged. The current controller state is Section 4a: no immediate verifier queue is open; next work is local consistency/release cleanup or a new gap-focused research package.

Original bootstrap request:

### Step 1: Create directory structure

Create all directories and placeholder files listed in Section 5.

### Step 2: Write `README.md`

Generate a strong first version of `README.md` using Sections 6, 7, and 8.

The README should include:

- Project title
- Tagline
- Why this repo exists
- Strict OPD definition
- What does not count as OPD
- Landscape at a glance table
- Links to taxonomy pages
- Links to paper category pages
- Links to implementation pages
- Open problems list
- Entry schema
- Contributing pointer

### Step 3: Write taxonomy files

Generate initial content for:

- `taxonomies/README.md`
- `taxonomies/strict-opd-definition.md`
- `taxonomies/by-teacher-access.md`
- `taxonomies/by-supervision-signal.md`
- `taxonomies/by-loss-objective.md`
- `taxonomies/by-training-stage.md`
- `taxonomies/by-modality.md`
- `taxonomies/by-rollout-source.md`
- `taxonomies/adjacent-work.md`

### Step 4: Write paper category placeholders

For each file in `papers/`, create:

- short definition of the category
- inclusion criteria
- exclusion criteria
- placeholder section for entries
- expected entry format

### Step 5: Write implementation pages

For each file in `implementations/`, create initial sections and placeholder tables.

### Step 6: Create Markdown tables

Create all Markdown table files in `tables/` with headers only.

### Step 7: Write notes

Create short first versions of:

- `notes/opd-vs-sft.md`
- `notes/opd-vs-offline-kd.md`
- `notes/opd-vs-dpo.md`
- `notes/opd-vs-rlhf.md`
- `notes/opd-vs-rlvr.md`
- `notes/opd-failure-modes.md`
- `notes/llm-vs-vlm-opd.md`
- `notes/teacher-reliability.md`
- `notes/long-cot-drift.md`
- `notes/entropy-and-uncertainty.md`
- `notes/open-problems.md`

### Step 8: Write reading paths

Create short guided reading path placeholders:

- `reading-paths/beginner.md`
- `reading-paths/llm-reasoning.md`
- `reading-paths/vlm-mlm.md`
- `reading-paths/black-box-opd.md`
- `reading-paths/implementation.md`
- `reading-paths/industrial-opd.md`

Each should include:

- audience
- goals
- suggested paper categories to read
- TODO placeholder for paper links

### Step 9: Add scripts

Create lightweight Python scripts:

#### `scripts/paper_schema.py`

Define allowed enum values for:

- modality
- method family
- teacher access
- supervision granularity
- rollout source
- training stage
- strictness

#### `scripts/validate_entries.py`

Initially validate that Markdown table files have required headers.

#### `scripts/update_readme.py`

Placeholder script for future automatic README generation from Markdown tables.

#### `scripts/arxiv_search.py`

Placeholder script for future arXiv search.

#### `scripts/github_topic_search.py`

Placeholder script for future GitHub topic search.

### Step 10: Add metadata files

Create:

- `LICENSE`: MIT License unless otherwise specified.
- `CHANGELOG.md`: initial unreleased section.
- `CITATION.cff`: basic citation metadata with placeholder author info.
- `CONTRIBUTING.md`: use Section 13.

---

## 16. Historical bootstrap acceptance criteria

The first Codex pass is complete. The checklist below is retained as historical acceptance criteria for the initial scaffold, not as current work remaining.

- [ ] The repo has the full directory structure.
- [ ] `README.md` clearly communicates the taxonomy-first positioning.
- [ ] Strict OPD, Partial OPD, and Adjacent Work are defined.
- [ ] Paper category files exist and contain inclusion/exclusion criteria.
- [ ] Taxonomy files exist and are internally consistent.
- [ ] Markdown table files exist with correct headers.
- [ ] Notes and reading path files exist.
- [ ] `CONTRIBUTING.md` requires classification metadata.
- [ ] Basic validation script checks Markdown table headers.
- [ ] No fake paper entries are added.
- [ ] Placeholder text is clearly marked as TODO where evidence-backed entries are still needed.

---

## 17. Important instruction for Codex

Do **not** invent paper entries.

The first implementation pass should focus on:

- repository structure,
- README,
- taxonomy pages,
- schemas,
- contribution rules,
- placeholder sections,
- validation scripts.

Actual paper population should happen in a separate research pass with source verification.
