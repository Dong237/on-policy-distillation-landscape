# WP1 Candidate Line-Audit Synthesis

Checked: 2026-05-12

This note merges:

- `assets/research/wp1-candidate-line-audit-chatgpt.md`
- `assets/research/wp1-candidate-line-audit-claude.md`
- `assets/research/wp1-candidate-line-audit-gemini.md`

## Scope

This was the correct WP1 candidate queue:

vOPD, TIP, AOPD, StableOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens, DSKD v2, SelecTKD, ToDi, and HPD.

## Merge Decision

The repo now promotes nine candidates to `strict_opd`, keeps three as `partial_opd`, and leaves two as adjacent/not strict.

| Bucket | Methods | Repo action |
|---|---|---|
| Promote to `strict_opd` | vOPD, TIP, AOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, Rock Tokens | Add to `tables/opd_papers.md` as source-verified strict rows. |
| Keep method-level `partial_opd` | StableOPD, DSKD v2, HPD | Add or keep in `tables/opd_papers.md` as partial rows with method-level caveats. |
| Keep adjacent | SelecTKD, ToDi | Keep in `tables/adjacent_work.md`; do not promote without explicit current-rollout evidence. |

## Conservative Choices

| Method | Audits | Conservative decision |
|---|---|---|
| StableOPD | Two audits called it strict; one called it partial because of golden/off-policy mixture. | `partial_opd` at method level. A strict OPD core exists, but the proposed stabilization recipe mixes rollout sources. |
| CaOPD | Two audits called it strict; one called it borderline because calibration changes the target. | `strict_opd` with medium confidence. Privileged self-teacher supervision on student rollouts still satisfies C1/C2/C3. |
| DP-OPD | Two audits called it strict; one called it partial because mixed lambda settings exist. | `strict_opd` for the on-policy GKD-style loop, with caveat that mixed lambda variants should be split if tracked later. |
| DSKD v2 | All audits say strict only in an optional on-policy mode. | `partial_opd` at method level. Split an on-policy-mode row later only with experiment-level evidence. |
| SelecTKD | Audits agree no clean same-rollout dense teacher objective is shown in experiments. | `adjacent`. Propose-and-verify token selection is useful but not enough for strict OPD. |
| ToDi | Audits agree the method is a fixed-data divergence innovation. | `adjacent`. |
| HPD | Audits agree it is off-policy-first with approximate on-policy next-token sampling. | `partial_opd`, not strict. |

## What Changed In The Landscape

WP1 now has a verified second wave beyond GKD/MiniLLM:

- variance-reduced reverse-KL OPD: vOPD;
- token selection and masking: TIP, Rock Tokens;
- asymmetric or step-weighted OPD: AOPD, SOD;
- privileged self-teacher OPD: CaOPD, OPSDL;
- privacy-preserving OPD: DP-OPD;
- strict cross-tokenizer OPD: SimCT;
- partial cross-tokenizer or hybrid methods: DSKD v2, HPD.

## Remaining Gaps

- Normalize authors and code status for the new 2026 preprints.
- Decide later whether the repo wants substage rows such as `stableopd-opd-core` or `dskd-on-policy-mode`.
- Verify implementation availability for SimCT, vOPD, TIP, AOPD, CaOPD, SOD, OPSDL, DP-OPD, and Rock Tokens.
- Extend framework tracking to see which libraries can implement these methods without custom trainers.
