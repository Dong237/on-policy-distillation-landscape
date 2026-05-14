# WP1 — White-Box OPD & Divergence/Objective Methods (LLM only)

**Scope narrowing applied:** text-only LLMs; white-box teacher (logits / log-probs / distributions) OR objective/divergence innovations that slot into an OPD loop. VLM, VLA, pure RLVR, offline KD, static DPO, black-box API methods, and full industrial pipelines are excluded except as contrast rows.

**Evidence basis:** primary arXiv sources already line-audited in `assets/research/wp9-line-audit-claude.md` and `assets/research/wp0/wp0-survey-claude.md`. No new claims are fabricated; weak evidence is flagged and conservatively downgraded.

---

## 1. Research memo — method families & objective taxonomy

WP1's white-box universe decomposes along four axes (freshness axis added to the survey's three):

### A. Divergence family

- **Forward KL** (mode-covering): GKD (λ configurable), DistillSpec (FKL ablation), Lightning OPD forward variant, PACED forward-KL track.
- **Reverse KL** (mode-seeking): MiniLLM (seq-level REINFORCE RKL), GKD (λ-mixed), Fast OPD, REOPOLD (relaxed/clipped RKL), PACED reverse-KL track, TML blog operational default.
- **Symmetric / JSD / TVD**: GKD JSD variant, DistillSpec JSD/TVD ablation.
- **Skew-KL family**: DistiLLM (SKL + SRKL with α skew), DistiLLM-2 (contrastive SKL on teacher branch + SRKL on student branch).
- **Adaptive / entropy-gated KL**: Entropy-Aware OPD (gates FKL↔RKL by teacher token entropy).
- **Bridged / geometric target**: Veto (geometric PoE bridge target between teacher and student before KL).
- **KL-constrained RL view of OPD**: G-OPD (teacher log-ratio as reward, α≥1 extrapolation), REOPOLD (clipped log-ratio), KDRL (RKL + GRPO, hybrid).

### B. Teacher access & kind (white-box only here)

- Separate larger frozen LLM: GKD, MiniLLM, DistiLLM, DistiLLM-2, Entropy-Aware OPD, Fast OPD, G-OPD, REOPOLD, Veto, PACED, Lightning OPD.
- Target LM as teacher (systems OPD): DistillSpec (draft↔target), Speculative KD (interleaved target verification).
- Cross-tokenizer white-box: DSKD (dual-space projection), GOLD (TRL; JSD+ULD hybrid), ULD (adjacent — not inherently on-policy).
- Privileged-context self / older checkpoint counts as white-box teacher source under repo rules but is out-of-scope for WP1 (handled in WP3/WP4).

### C. Rollout freshness (discriminator the survey omits)

- **Per-iteration current-policy**: GKD (λ=1), MiniLLM, Entropy-Aware OPD, Fast OPD, G-OPD, REOPOLD, Veto, DistillSpec (on-policy ablation).
- **Per-epoch batched / replay-stale**: DistiLLM (author-declared "adaptive off-policy"), DistiLLM-2 (author-declared batch, θ_{e−1}).
- **Interleaved student+teacher tokens**: Speculative KD (accept/reject mixes teacher tokens into rollout).
- **Precomputed on SFT/reference rollouts**: Lightning OPD (frozen π_ref rollouts + cached teacher log-probs).
- **Split-track**: PACED (forward-KL stage uses teacher-generated y_T; reverse-KL stage uses student y_S).

### D. Objective families (taxonomy)

- Classical token/sequence KL: GKD, MiniLLM, DistillSpec.
- Streamlined/contrastive skew-KL: DistiLLM, DistiLLM-2.
- Entropy/position-adaptive KL: Entropy-Aware OPD (loss-innovation cousins ToDi/AKL listed only as contrast).
- Teacher-log-ratio-as-reward: G-OPD, REOPOLD (white-box RL view of OPD).
- Bridged target: Veto (β-interpolated logit target).
- Prefix-restricted RKL: Fast OPD.
- Curriculum / pass-rate-weighted KL: PACED.
- Systems OPD (speculative decoding): DistillSpec (tailored f-div), Speculative KD (hybrid-trace token-KL).
- Offline approximation to OPD: Lightning OPD (cached teacher log-probs advantage PG).
- Cross-tokenizer bridging: DSKD (dual-space KL with ETA alignment), GOLD (JSD+ULD over on-policy student).

---

## 2. Evidence ledger (primary sources only)

| id | primary_source | teacher_kind | rollout | objective | verdict |
|---|---|---|---|---|---|
| GKD | https://arxiv.org/abs/2306.13649 | larger frozen LLM | current-policy (λ=1) / mix | generalized KL (FKL/RKL/JSD, λ mixture) | strict_opd at λ=1 |
| MiniLLM | https://arxiv.org/abs/2306.08543 | larger frozen LLM | student, per-iteration | reverse KL via REINFORCE on rollouts | strict_opd |
| DistiLLM | https://arxiv.org/abs/2402.03898 | larger frozen LLM | adaptive replay (SGO+TGO, per-epoch) | skew-FKL + skew-RKL | partial_opd (authors: "adaptive off-policy") |
| DistiLLM-2 | https://arxiv.org/abs/2503.07067 | larger frozen LLM | SGO from θ_{e−1} + TGO | contrastive SKL on teacher branch + SRKL on student branch | partial_opd (authors: "batch…rather than on-policy") |
| Entropy-Aware OPD | https://arxiv.org/abs/2603.07079 | stronger reasoning LLM | current-policy | entropy-gated FKL↔RKL per token | strict_opd (preprint, medium conf.) |
| G-OPD | https://arxiv.org/abs/2602.12125 | larger frozen LLM | current-policy y∼π_θ | KL-constrained RL with teacher log-ratio reward (α≥1) | strict_opd (line-audited) |
| REOPOLD | https://arxiv.org/abs/2603.11137 | larger frozen LLM | current-policy (old-policy alg.1) | clipped teacher-student log-ratio, top-(1-ρ) masking, relaxed RKL | strict_opd (line-audited) |
| Veto | https://arxiv.org/abs/2601.07155 | larger frozen LLM | current-policy | KL to geometric/β-bridge target between teacher and student | strict_opd (WP0 consensus, medium conf.) |
| Fast OPD | https://arxiv.org/abs/2602.15260 | stronger reasoning LLM | current-policy, prefix-only | reverse KL on reasoning prefixes | strict_opd (medium conf.; FLOP claims unaudited) |
| PACED | https://arxiv.org/abs/2603.11178 | frozen Qwen3-14B + self-teacher | split: teacher y_T (fwd-KL) + student y_S (rev-KL) | pass-rate β-weighted FKL or RKL | partial_opd overall; reverse-KL substage strict |
| DistillSpec | https://arxiv.org/abs/2310.08461 | target LM | draft/student (on-policy ablation crucial) | tailored f-divergence (FKL/RKL/JSD/TVD) | strict_opd (on-policy ablation) |
| Speculative KD | https://arxiv.org/abs/2410.11325 | target LM | interleaved student+target verified tokens | token-level KL on hybrid trace | partial_opd / borderline (acceptance-rate dependent) |
| Lightning OPD | https://arxiv.org/abs/2604.13010 | larger frozen LLM | precomputed on π_ref rollouts | advantage-weighted PG with cached teacher log-probs | partial_opd (authors self-identify as offline approx.) |
| DSKD (v1/v2) | https://arxiv.org/abs/2406.17328 / https://arxiv.org/abs/2504.11426 | cross-tokenizer projected teacher | dataset (v1) / on-policy mode (v2) | FKL/RKL/JSD/AKL/Skew-KL via dual-space projector | not_opd (v1) / borderline (v2) |
| GOLD (TRL) | HF TRL docs (no arXiv paper) | cross-tokenizer teacher | student on-policy | JSD + ULD hybrid | **gap** — secondary source only |
| ULD | https://arxiv.org/abs/2402.12030 | cross-tokenizer teacher | dataset | Wasserstein/ULD on logit sequences | adjacent (loss only; not inherently on-policy) |

**Contrast rows** (not WP1 merge candidates; listed only to discipline the boundary): ToDi `https://arxiv.org/abs/2505.16297`, AKL `https://arxiv.org/abs/2404.02657`, TAID `https://arxiv.org/abs/2501.16937`, PromptKD `https://arxiv.org/abs/2402.12842`, AlignDistil `https://arxiv.org/abs/2503.02832`, f-DISTILL `https://arxiv.org/abs/2307.15190`, ATKD `https://arxiv.org/abs/2402.11890` — all white-box divergence work but off-policy in their published experiments.

---

## 3. C1/C2/C3 classification (white-box WP1 set)

| method | C1 student rollout | C2 same-rollout teacher | C3 objective consumes non-scalar teacher signal | final |
|---|---|---|---|---|
| GKD (λ=1) | yes | yes (teacher log-probs on student prefixes) | yes (generalized KL) | strict_opd |
| MiniLLM | yes (REINFORCE student samples) | yes (teacher distribution) | yes (reverse KL) | strict_opd |
| DistiLLM | partial (per-epoch replay; not per-iter) | yes | yes (skew-KL) | partial_opd |
| DistiLLM-2 | partial (θ_{e−1} + teacher responses) | yes | yes (contrastive skew-KL) | partial_opd |
| Entropy-Aware OPD | yes | yes | yes (entropy-gated FKL/RKL) | strict_opd |
| G-OPD | yes | yes (log-ratio per token) | yes (KL-constrained RL with teacher ratio) | strict_opd |
| REOPOLD | yes | yes (per-token log-ratio) | yes (clipped/masked RKL) | strict_opd |
| Veto | yes | yes (bridged teacher-student target) | yes (KL to PoE bridge) | strict_opd |
| Fast OPD | yes (prefix only) | yes | yes (reverse KL on prefix) | strict_opd |
| PACED (overall) | split | split | yes where applicable | partial_opd; rev-KL substage strict |
| DistillSpec | yes (draft samples) | yes (target LM on draft tokens) | yes (tailored f-div) | strict_opd |
| Speculative KD | hybrid trace weakens C1 | yes | yes (token KL on mixed seq) | partial_opd |
| Lightning OPD | **fails C1** (precomputed π_ref rollouts) | yes (cached) | yes | partial_opd |
| DSKD v2 | conditional | yes (projected teacher) | yes | borderline |
| GOLD | yes per HF docs | yes (cross-tok projected) | yes (JSD+ULD) | **unclear — no primary paper** |
| ULD | no (dataset) | yes | yes | adjacent |

---

## 4–6. Exact objective, rollout freshness, teacher access (condensed)

| method | exact objective/divergence | rollout freshness | teacher access / kind |
|---|---|---|---|
| GKD | `E_{y∼π_mix}[Σ_t D_f(p_T,p_θ)]`, `f∈{FKL,RKL,JSD}`, λ controls mixture | per-iteration current-policy at λ=1 | white-box; separate larger LLM |
| MiniLLM | seq-level reverse KL via policy-gradient on student rollouts | per-iteration current-policy | white-box; separate larger LLM |
| DistiLLM | Skew-FKL(α) on TGO + Skew-RKL(α) on SGO | SGO replay buffer, per-epoch (declared off-policy) | white-box; separate larger LLM |
| DistiLLM-2 | contrastive: SKL on y_t(teacher) + SRKL on y_s (θ_{e−1}) | per-epoch batched | white-box; separate larger LLM |
| Entropy-Aware OPD | per-token entropy-gated α(e)·FKL + (1−α(e))·RKL | per-iteration current-policy | white-box; stronger reasoning LLM |
| G-OPD | `J = E_{y∼π_θ}[λ·log(π*/π_ref) − D_KL(π_θ‖π_ref)]`, λ≥1 | per-iteration current-policy | white-box; separate larger LLM |
| REOPOLD | `max Σ ρ_{i,t}(θ)·R̂^λ_{i,t}·M_{i,t}`, R=log π_T/π_θ, clipped + dynamic mask | per-iteration current-policy | white-box; separate larger LLM |
| Veto | KL to β-interpolated bridge target Q between teacher & student logits | per-iteration current-policy | white-box; separate larger LLM |
| Fast OPD | reverse KL on teacher reasoning-prefix segment only | per-iteration current-policy (prefix) | white-box; stronger reasoning LLM |
| PACED fwd | `Σ_t D_KL(p_T(·|y_{T,<t})‖p_S)` on teacher seq | teacher tokens | white-box |
| PACED rev | `Σ_t D_KL(p_S(·|y_{S,<t})‖p_T)` on student seq, β pass-rate weighted | per-iteration current-policy | white-box + self-teacher |
| DistillSpec | tailored f-div among FKL/RKL/JSD/TVD (task-specific) | draft-on-policy ablation | white-box; target LM as teacher |
| Speculative KD | token KL on interleaved (student + target-verified) sequence | hybrid, acceptance-rate dependent | white-box; target LM |
| Lightning OPD | advantage-weighted PG with `A_t = log π_T − log π_θ`, teacher precomputed | precomputed on SFT/π_ref rollouts | white-box; cached teacher log-probs |
| DSKD v2 | FKL/RKL/JSD/AKL/Skew-KL in projected dual space with ETA token alignment | supports on-policy mode | white-box cross-tokenizer |
| GOLD | JSD + ULD hybrid over on-policy student (per HF docs) | current-policy claimed | white-box cross-tokenizer |

---

## 7. Suggested Markdown rows

Rows already merged in `tables/opd_papers.md`: `gkd-2024, minillm-2024, distillm-2024, distillm2-2025, entropy-aware-opd-2026, g-opd-2026, reopold-2026, veto-2026, fast-opd-2026, paced-2026, distillspec-2024, speculative-kd-2024, lightning-opd-2026`. WP1 does not propose changes to these — line audit has confirmed them.

Proposed new WP1 rows (cross-tokenizer white-box OPD only; add after WP9 line audit):

```markdown
| dskd-v2-2025 | DSKD: A Dual-Space Framework for General Knowledge Distillation of LLMs (v2) | 2025 | Zhang et al. | arXiv | https://arxiv.org/abs/2504.11426 | https://github.com/songmzhang/DSKD | | LLM | cross_tokenizer | white_box_opd | logit_based | white_box | token | dual_space_kl | partial | borderline_strict | teacher_logits_projected | language_model | larger_llm | smaller_lm | larger_lm | mixed_policy | configurable | teacher_logits_projected | token | fkl_rkl_jsd_skew_in_projected_space | C1 supports on-policy mode (must be verified per-experiment); C2 projected teacher distribution supervises student states; C3 KL in projected dual space consumes the signal. | no | | post_training | instruction_following | custom | released | not_applicable | https://arxiv.org/abs/2504.11426 | arxiv | needs_review | human_spotcheck_needed | medium | Mark strict only for experiments configured with student rollouts. | dual-space projector + teacher forward per step | Cross-tokenizer enabler; disambiguate from DSKD v1 which is not_opd. | 2026-05-12 |
```

No GOLD row proposed until a primary paper is located; see gap map §9. No new rows proposed for GKD/MiniLLM/DistillSpec variants — they are already represented.

---

## 8. False positives & downgrade notes

- **DistiLLM / DistiLLM-2 as strict_opd**: recurring false positive in secondary sources. Authors self-describe the SGO mechanism as "adaptive off-policy" / "batch approach rather than on-policy approach which samples at every training iteration." Keep **partial_opd**. The survey's taxonomy implicitly elevates them; this repo should not.
- **Lightning OPD as strict_opd**: the name misleads. §3.1 states rollouts are fixed to π_ref and teacher log-probs are precomputed. **C1 fails.** Partial only.
- **Speculative KD as strict_opd**: interleaved teacher tokens replace rejected student tokens before KL; rollout distribution is not π_θ. Borderline/partial.
- **ULD as OPD**: it is a loss function for tokenizer mismatch; not inherently on-policy. Keep adjacent until a paper applies ULD to student rollouts.
- **GOLD as an OPD entry**: only HF TRL docs/blog exist. No arXiv paper located. Treat as framework artifact, not an OPD paper row, until a primary paper appears.
- **SCOPE listed as "Self-play Contrastive On-Policy Evaluation"**: fabricated expansion. The real SCOPE (`https://arxiv.org/abs/2604.10688`) is *Signal-Calibrated OPD Enhancement* — already tracked as `borderline_strict`. Do not re-introduce the fabricated gloss in WP1.
- **"DistillDirect"** and **"DBKD"**: no canonical OPD paper located; do not add to white-box WP1 even if seen in secondary lists.
- **ToDi, AKL, TAID, AlignDistil, PromptKD, f-DISTILL, ATKD**: white-box divergence innovations but off-policy in their published experiments. They compose with OPD but are not OPD themselves. Keep `adjacent`.
- **Phi-4-reasoning, DeepSeek-R1 distilled series**: white-box or quasi-white-box teacher data, but trained by SFT on teacher traces → **not_opd**. Listed only as contrast.
- **ImitKD (2020)**: DAgger-style KD precursor to GKD; if added, classify as `borderline_strict` / foundational-mixed — only add if the repo decides to include historical seeds.

---

## 9. Open gaps requiring line-level audit

Ordered by priority for WP9-style token-level audit:

1. **GOLD (HF TRL)** — no primary paper located; docs indicate JSD+ULD hybrid over on-policy student. Need either a TML-equivalent blog citation or a code-level read of the TRL `GOLDTrainer` to confirm C1/C2/C3. Currently out of scope for merge.
2. **DSKD v2 (arXiv:2504.11426)** — "supports on-policy KD" is asserted; need per-experiment confirmation that the reported numbers are actually run with student rollouts. Different tables in the paper may use different modes.
3. **Veto (arXiv:2601.07155)** — confidence medium; author normalization and exact bridge-target algebra (what β schedule, stopgrad placement) need line extraction.
4. **Fast OPD (arXiv:2602.15260)** — FLOP-reduction claim (2–47×) needs controlled verification; author normalization missing.
5. **Entropy-Aware OPD (arXiv:2603.07079)** — preprint only; verify whether the entropy gate uses teacher entropy or student entropy (abstract ambiguous) and whether gating is per-token or per-segment.
6. **G-OPD vs REOPOLD** — both frame OPD as KL-constrained RL with teacher log-ratio. Need side-by-side algebraic verification that G-OPD's extrapolation coefficient α (survey uses λ) and REOPOLD's clipped R̂^λ with dynamic mask M are genuinely distinct, not renamings.
7. **DistillSpec on-policy vs off-policy ablation** — the paper's strict OPD claim rests on the on-policy ablation specifically; need to confirm which reported numbers use on-policy sampling and which use teacher-forced data.
8. **Speculative KD acceptance-rate dependency** — partial vs borderline depends on empirical acceptance rate. Need rate numbers from §4/§5 to decide whether it is closer to strict (high acceptance → close to on-policy) or partial (low acceptance → near teacher-forcing).
9. **PACED forward-KL stage** — confirm whether any reported single-stage configuration runs reverse-KL-only (which would be strictly OPD end-to-end) or whether the recommended recipe always mixes stages.
10. **Lightning OPD** — confirm whether an "online" variant exists in the paper (some ablations may recompute teacher log-probs on current student rollouts); current label is based on the default/advertised setting.
11. **DSKD v1 → v2 continuity** — confirm that v2's on-policy mode uses the same ETA token-alignment as v1, since alignment errors can silently off-policy-ize the loss.
12. **Cross-tokenizer correctness bug in TRL GKDTrainer (Issue #4562)** — noted in the legacy survey; if confirmed, it affects the reproducibility of any cross-tokenizer white-box OPD row relying on TRL defaults.
13. **Fast OPD vs PACED prefix-restriction distinction** — both focus on informative prefixes; need to verify they are not near-duplicates.
14. **ImitKD (2020) vs GKD on-policy mode** — foundational attribution; useful for the repo's historical/theory section but not a WP1 strict row.

---

## Boundary guardrails applied

- All VLM/MLLM/VLA/Speech rows (VOLD, Video-OPD, X-OPD, VLA-OPD, PRISM, Uni-OPD, MAD-OPD) are deferred to WP7.
- All industrial rollouts (Qwen3, Nemotron-Cascade 2, MiMo-V2-Flash, Gemma 2, Thinking Machines blog) are deferred to WP5 even when the OPD stage is white-box.
- All teacher-free / privileged-context self-teacher rows (OPSD, CRISP, GATES, SDPO, OPCD, OEL, SDFT, Priv-Info Distill, HDPO, MTP-SD, DAIL, TMS) are deferred to WP3.
- All black-box / discriminator / verbal / preference rows (GAD, Lion, OVD, ORPO-Distill, SuperCorrect) are deferred to WP2.
- All reward-only / RLVR rows (GRPO, DeepSeek-R1 family, LUFFY, RLKD, KDRL-full-method, VLM-R1) are deferred to WP4 or held as `adjacent`.
- KDRL full method and SCOPE are listed in the repo but belong to WP4 audit (hybrid) — excluded here except as boundary contrast.

---

**Audit date:** 2026-05-12
**Scope:** WP1 white-box OPD; LLM-only; text-only.
**Conservatism:** evidence-weak seeds (Veto, Fast OPD, Entropy-Aware OPD, DSKD v2, GOLD) remain at medium confidence and are flagged for line-level WP9 follow-up rather than merged as strict.
