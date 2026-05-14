# WP1 White-Box Redirect Spot-Check Synthesis

Checked: 2026-05-13

This note reconciles the WP1 redirect audits produced after the WP2 secondary cleanup:

- `assets/research/wp2-secondary-spotcheck-chatgpt.md`
- `assets/research/wp1-whitebox-redirect-spotcheck-chatgpt.md`
- `assets/research/wp1-whitebox-redirect-spotcheck-claude.md`
- `assets/research/wp1-whitebox-redirect-spotcheck-gemini.md`

## Scope

The redirected methods were PAD, ADPA, GAKD, daDPO, and CTPD. They should not be treated as unresolved WP2 black-box/API candidates because all five require white-box teacher distributions, teacher logits, token probabilities, or projected teacher log-probs.

The WP2 primary queue remains closed: GAD, PRISM, OVD, SODA, and ORPO-Distill should not be reopened unless their source evidence changes.

## Merge Decision

The repo uses the conservative method-level merge rule: if a strict-looking subcomponent is mixed with static data, stale checkpoints, offline preference pairs, or optional modes, the method-level row stays below `strict_opd`.

| Method | Final repo action | Reason |
|---|---|---|
| PAD | `partial_opd` method-level row in `tables/opd_papers.md` | Student response sets and teacher log-probability rewards are OPD-like, and iterative PAD is the strongest redirect candidate, but default sampling is one-shot and the iterative variant is only epoch-level fresh. Split a borderline iterative-PAD variant only if the repo later tracks variant-level rows. |
| ADPA | `partial_opd` row in `tables/opd_papers.md` | White-box DPO-teacher/reference advantages provide rich token/state supervision, but the main loop uses frozen or precomputed states and offline preference-alignment machinery. |
| daDPO | `partial_opd` row in `tables/opd_papers.md` | Teacher and student responses are sampled to construct preference pairs, and teacher distributions enter the DPO-style loss, but the update is static/off-policy rather than current-rollout supervision. |
| GAKD | Adjacent ledger row in `tables/adjacent_work.md` | Reverse-KL and adversarial losses are computed on corpus or teacher-generated sequences through forward passes; no current student autoregressive rollout loop was verified. |
| CTPD | Adjacent ledger row in `tables/adjacent_work.md` | Cross-tokenizer teacher log-prob projection is useful, but training uses static preference data with no current student rollout loop. |

## Conflict Resolution

The raw ChatGPT redirect artifact labels PAD as `strict_opd` and ADPA as `borderline_strict`. Claude and Gemini were more conservative, especially on rollout freshness. The merged repo follows the stricter reading because the table label describes the full method row, not the most favorable variant or subcomponent.

The practical outcome is:

- no new clean `strict_opd` seed from the redirect queue;
- PAD, ADPA, and daDPO are useful white-box partial rows;
- GAKD and CTPD stay outside the OPD table as adjacent methods;
- WP2 secondary cleanup is resolved and should not be rerouted.

## Next Router State

The repo is ready to move past WP1/WP2 cleanup. The next external-agent request should start a new work package rather than re-audit this queue.
