# Industrial OPD Reading Path

## Audience

Readers tracking training recipes from model reports, model cards, and official blogs.

## Goals

- Separate disclosed OPD from generic post-training.
- Track where reports provide insufficient evidence.

## Path

1. Read [Industrial and scaling](../papers/industrial-and-scaling.md).
2. Inspect `tables/industrial_reports.md`.
3. Read [OPD vs RLHF](../notes/opd-vs-rlhf.md) and [OPD vs RLVR](../notes/opd-vs-rlvr.md).
4. Only upgrade a report to `strict_opd` when it clearly connects teacher feedback to student on-policy trajectories.
