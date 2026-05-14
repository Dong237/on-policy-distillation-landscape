# White-Box OPD

White-box OPD assumes access to teacher logits, log-probs, or token-level probabilities.

## Inclusion criteria

- Student generates training states.
- Teacher distribution is evaluated on those states.
- Loss uses token-level or sequence-level distribution matching.

## Exclusion criteria

- Offline KD on fixed teacher data.
- Reward-only RLVR.
- Black-box teacher APIs without probabilities.

## Source Audit Result

Detailed provenance is archived under [assets/research](../assets/research/).

White-box OPD rows should pass all three checks:

- C1: current student or policy rollout supplies the supervised text state.
- C2: teacher logits, probabilities, log-probs, entropy, or transformed white-box distributions are evaluated on that exact state.
- C3: a token-level, sequence-level, or dense log-ratio objective consumes the teacher-style signal.

Strict white-box examples include GKD pure on-policy variants, MiniLLM, Entropy-Aware OPD, G-OPD, REOPOLD, Veto, Fast OPD, and the draft-generated variant of DistillSpec.

The candidate line audit promotes vOPD, TIP, AOPD, CaOPD, SOD, SimCT, OPSDL, DP-OPD, and Rock Tokens as strict rows. StableOPD, DSKD v2, and HPD remain partial at method level. SelecTKD and ToDi remain adjacent objective or selection methods, not strict OPD.

## White-Box Redirect Spot-Check

Detailed redirect-audit provenance is archived under [assets/research](../assets/research/).

The secondary cleanup routed PAD, ADPA, GAKD, daDPO, and CTPD into the white-box bucket because they require teacher logits, token probabilities, or teacher/reference distributions. The redirect audits found no new clean `strict_opd` seed. PAD, ADPA, and daDPO are tracked as partial method-level rows because they use white-box teacher distribution information but weaken rollout freshness through one-shot, frozen, or static preference data. GAKD and CTPD stay adjacent: GAKD uses corpus or teacher-generated sequences rather than student autoregressive rollouts, and CTPD projects teacher log-probs over static preference pairs.
