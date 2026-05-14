# WP1 White-Box Redirect Spot-Check: Primary-Source Audit

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML full text for PAD, ADPA, daDPO, CTPD; OpenReview/arXiv for GAKD
**Methodology:** Five independent research agents fetched primary-source HTML, extracted exact section/equation-level quotes, and assessed C1/C2/C3 against `taxonomies/strict-opd-definition.md`. Results reconciled below.

---

## Scope

These five methods were routed from the WP2 secondary spot-check (`assets/research/wp2-secondary-spotcheck-chatgpt.md`) as white-box redirects requiring WP1 line-level audit:

> PAD, ADPA, GAKD, daDPO, CTPD

The WP2 black-box primary queue (GAD, PRISM, OVD, SODA, ORPO-Distill) is NOT reopened.

## OPD Test

Strict OPD requires all three conditions:

- **C1 (Student rollout):** The current student/policy generates the training trajectory.
- **C2 (Same-rollout supervision):** A teacher supervises that exact student-generated state with white-box token/logit-level signal.
- **C3 (Distillation-style consumption):** The training objective directly consumes that supervision as a distributional signal, not merely as scalar reward or preference label.

---

## Memo

### Central finding

**No method in the WP1 redirect queue qualifies as strict OPD.** PAD comes closest (C2 and C3 cleanly satisfied), but C1 is weakened by single-pass or epoch-level-stale sampling. The remaining four methods fail C1 decisively: ADPA and daDPO pre-sample from frozen student checkpoints, GAKD runs forward passes on corpus sequences with no autoregressive student generation, and CTPD trains on a static external preference dataset.

### PAD: structurally closest to strict OPD, but C1 is epoch-level stale

PAD has the cleanest C2/C3 of the five. The teacher computes its own log-likelihood on student-generated texts (Eq. 7), and the JSD over Plackett-Luce preference distributions (Eq. 11-12) is a genuine distributional loss -- richer than scalar reward, richer than DPO preference labels. However, the default configuration is single-pass (1x4: sample once, train once), making the "student-generated" responses stale from the pre-training checkpoint. The iterative variant (Fig. 3, Sec 4.4: 2x2, 3x4 configurations) re-samples per iteration, but each iteration trains on responses from the *previous* checkpoint, not the continuously-updating policy. This is epoch-level freshness at best, not step-level on-policy. Classification: **borderline_strict** for the iterative variant, **partial_opd** for the default single-pass.

### ADPA: offline advantage-guided distillation, not on-policy

ADPA generates states from a frozen SFT student (Algorithm 1, line 5: `pi_theta'`) and precomputes advantages once before training begins (Sec 4.1: "precompute log pi_dpo(.|s_t) - log pi_ref(.|s_t)"). The paper itself calls this "offline optimization" (Sec 4.4). The advantage-weighted objective (Eq. 10) is distribution-level (full log-ratio vector enters the gradient), satisfying C3. But C1 fails: states are from a frozen snapshot, never refreshed. Appendix E compares state sources but only tests {preferred, dispreferred, teacher, student} -- all from fixed datasets, never "regenerate each epoch." Classification: **partial_opd**.

### GAKD: off-policy white-box KD with no student autoregressive generation

The critical finding is that GAKD's student never generates tokens autoregressively. The reverse KL is computed on training corpus sequences via importance sampling (Corollary 1; Appendix A.1: "no additional autoregressive sampling loop is required"). The adversarial loss also operates on corpus sequences: both teacher and student produce logits via forward passes on the same corpus inputs (Algorithm 1). The discriminator consumes V-dimensional logit vectors per token -- strictly white-box. Despite the "Generative Adversarial" framing, the student is a "generator" only in the sense of producing logits, not generating text. Both C1 conditions fail (reverse KL: corpus sequences; adversarial: corpus sequences). Classification: **not_opd** -- this is offline white-box KD with an adversarial regularizer.

### daDPO: off-policy distribution-aware DPO on static pairs

daDPO samples responses once from the initial student and teacher (Sec 3.2) and constructs a fixed preference dataset C_dpo before training begins. No iterative re-sampling exists. The teacher distribution pi_te enters the gradient directly via two beta-weighted KL terms (Eq. 10), satisfying C3 structurally. But the student responses y^s are from the initial policy, not the current training iterate. The paper uses OpenRLHF with a fixed dataset and no online component. Classification: **adjacent** -- this is off-policy distributional DPO, not OPD.

### CTPD: offline cross-tokenizer preference distillation on static external data

CTPD trains on the UltraFeedback Binarized dataset (Sec 4.1: "over 63k high quality preference pairs"). The student never generates its own trajectories. The teacher's role is as a cross-tokenizer reference model (Sec 3.3: "CTPD adopts the teacher model itself as the reference distribution pi_ref") with span-projected log-probabilities and importance weights from contrastive teacher pairs. The supervision is rich (span-level, non-scalar), satisfying C3 structurally. But C1 and C2 both fail: neither the training data nor the supervised states are student-generated. Classification: **adjacent** -- useful cross-tokenizer distillation technique, but entirely offline.

---

## Detailed Evidence Table

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PAD | https://arxiv.org/abs/2502.14272 | S4.1 Eq.7, S4.2 Eq.9, S4.3 Eq.11-12, S4.4 Fig.3, Limitations | "we directly sample n responses Y_n from the student model pi^stu through repeated sampling" (S4.1) | "we calculate the rewards for both the teacher and student models for each response y_i in Y_n using Equation (7)" (S4.2); r(y\|x) = (1/\|y\|) log p_pi(y\|x) on student texts | "We employ the Jensen-Shannon divergence loss to align the student's and teacher's preference distributions" (S4.3); JSD over Plackett-Luce distributions Eq.12 | white-box token-level log-probabilities | response-set preference distribution via Plackett-Luce | JSD preference-distribution matching (PPD) or ranking-based NLL (VPD) | default: single-pass stale; iterative variant: epoch-level per-iteration re-sampling (Fig.3, 2x2/3x4) | borderline_strict | C1 weakened: default is 1-iteration (stale checkpoint); iterative variant re-samples per iteration but epoch-level, not step-level on-policy; "requires token-level probabilities, which are unavailable in some black-box models" (Limitations) | high |
| ADPA | https://arxiv.org/abs/2502.17927 | S3.3 Eq.9-10, S4.1, Alg.1 line 5, App.E, App.G | "Generate outputs from the SFT student model pi_theta' for the given prompt x to obtain y-hat" (Alg.1 line 5); states from frozen SFT student, not current policy | "precompute log pi_dpo(.\|s_t) - log pi_ref(.\|s_t) for the top 50 probabilities" (S4.1); precomputed once on frozen student states (App.G) | Eq.10: inner product of pi_theta(.\|s_t) with log(pi_dpo/pi_ref) -- full log-ratio distribution enters gradient; "its offline optimization process is more stable" (S4.4) | white-box DPO teacher + reference teacher log-probability distributions | token/state-level distributional advantage (top-k vocabulary) | advantage-weighted policy gradient + SFT | static: frozen SFT student snapshot, precomputed advantages, no regeneration | partial_opd | C1 fails: states from frozen SFT student pi_theta', not current policy; C2 partial: advantages precomputed once, never refreshed; paper self-identifies as "offline" (S4.4) | high |
| GAKD | https://arxiv.org/abs/2501.11153 | S3.1-3.3 Eq.7-9, Alg.1, Corollary 1, App.A.1 Eq.10-11 | C1 FAILS: "All conditional probabilities needed for w(x,y) and l(x,y) are produced by a single forward pass through each model on the full target sequence, so no additional autoregressive sampling loop is required" (App.A.1); reverse KL on corpus C via importance sampling (Corollary 1, Alg.1 line 307) | "the role of the discriminator model is to determine whether a given logits sequence originates from the teacher model T or the student model G" (S3.2) -- on corpus sequences, not student rollouts | Eq.9: min_theta max_phi combined RGAN + importance-weighted reverse KL + NLL; gradient flows through all three terms (Alg.1 line 311: 1/2 L_NLL + alpha/2 L_KL + beta L^G_adv) | white-box teacher logits + learned logit-level discriminator | token logits (V-dimensional vectors per position) | reverse KL (importance-sampled on corpus) + sequence-level adversarial GAN + NLL | off-policy: all sequences from training corpus C; no student autoregressive generation | not_opd | C1 fails on both terms: reverse KL computed on corpus via importance sampling (Corollary 1); adversarial loss also on corpus forward passes (Alg.1); student never generates tokens; "Generative" is a misnomer for the training procedure | high |
| daDPO | https://arxiv.org/abs/2506.15717 | S3.2, S4.1-4.2 Eq.7-10, Theorem 1 Eq.8, Gradient Analysis | C1 FAILS: "for each x_i in X, we sample a response y_i^t ~ pi_te(x_i) from the teacher model and another response y_i^s ~ pi_st(x_i) from the student model" (S3.2) -- sampled once before training | C2 PARTIAL: pi_te(y^s\|x) appears in loss Eq.10, but y^s is stale/pre-sampled from initial pi_st; "the teacher model can be frozen in a gradient-free mode" (S4.2) | Eq.10: DPO loss with pi_te^{beta_2} terms in both preferred and dispreferred log-ratios; gradient includes delta_te = log[pi_te(y^t)/pi_ref(y^t)] - log[pi_te(y^s)/pi_ref(y^s)] (Gradient Analysis, S4.2) | white-box teacher distribution pi_te; pi_ref = pi_dSFT (student SFT), NOT teacher | sequence-level preference pair with distributional KL anchor | distribution-aware DPO with dual KL regularization (to pi_ref and pi_te) | static: responses sampled once before training; no iterative loop | adjacent | C1 fails: y^s from initial pi_st, not current policy; off-policy static DPO; C2 partial: pi_te(y^s) in loss but y^s is stale; paper explicitly says "white-box setting, where the teacher model's distribution pi_te is fully accessible" (S3) | high |
| CTPD | https://arxiv.org/abs/2601.11865 | S3.1-3.4, S4.1, Fig.1 | C1 FAILS: "we utilize the UltraFeedback Binarized dataset...which contains over 63k high quality preference pairs" (S4.1); student never generates trajectories | C2 FAILS: both models score fixed dataset text via aligned spans (S3.2 Def.1); "The SFT student model is then further trained using preference data" (Fig.1) -- preference data is external, not student-generated | "CTPD adopts the teacher model itself as the reference distribution pi_ref in the DPO-style objective" (S3.3); TIS-DPO with span-projected teacher log-probs as pi_ref + importance weights w_i from contrastive teachers (S3.4) | white-box teacher log-probabilities through character-level aligned span projection; contrastive teacher pair (pi+, pi-) for importance weights | aligned-span token/sub-word level via character projection | cross-tokenizer TIS-DPO with teacher-anchored reference | static: fixed UltraFeedback dataset; pre-computed span weights; single epoch training (S4.1) | adjacent | C1 and C2 both fail: static external preference dataset; student never generates; teacher scores dataset text, not student rollouts; rich cross-tokenizer supervision but entirely offline | high |

---

## Cross-validation with prior repo labels

| Method | Prior label (WP2 secondary) | This audit | Change? |
|---|---|---|---|
| PAD | not_wp2 (white-box) | borderline_strict (iterative variant) / partial_opd (default) | **RESOLVED** -- iterative PAD is the strongest WP1 white-box candidate; default single-pass is partial |
| ADPA | not_wp2 (white-box) | partial_opd | **RESOLVED** -- offline advantage-guided distillation; C3 satisfied but C1 fails |
| GAKD | not_wp2 (white-box, C1 uncertain) | not_opd | **DOWNGRADE from borderline_strict candidate** -- student never generates tokens; importance sampling on corpus eliminates autoregressive rollouts entirely |
| daDPO | not_wp2 (white-box, off-policy) | adjacent | **RESOLVED** -- off-policy distributional DPO on static pairs; no on-policy component |
| CTPD | not_wp2 (white-box) | adjacent | **RESOLVED** -- offline cross-tokenizer preference distillation on static external data |

### Key upgrade/downgrade rationale

**GAKD downgrade (borderline_strict candidate -> not_opd):** The prior Gemini analysis flagged GAKD as a "borderline_strict candidate" based on the abstract's claim of "sequence-level adversarial loss and reverse KLD loss" combined with a token-level discriminator. Line-level audit reveals this framing is misleading. The student never generates tokens autoregressively -- all sequences are from the training corpus, processed via forward passes. The importance-sampling trick in Corollary 1 was specifically designed to avoid student rollouts (Appendix A.1: "no additional autoregressive sampling loop is required"). The reviewer's concern about "feasibility proof uses teacher-generated sequences" turns out to describe not just the proof but the actual training procedure. GAKD is standard white-box KD (NLL + reverse KL) with an adversarial logit-level regularizer, all on fixed corpus data.

**PAD upgrade (not_wp2 -> borderline_strict for iterative variant):** PAD's iterative variant (Sec 4.4, Fig. 3) creates a genuine re-sampling loop where each iteration generates fresh student responses from the latest checkpoint. Combined with clean C2 (teacher log-likelihood on student texts) and clean C3 (JSD over preference distributions), this is the strongest white-box OPD candidate in this queue. The main caveat is epoch-level freshness, not step-level.

---

## Reconciliation notes

### Why borderline_strict for PAD (iterative) and not strict_opd

PAD's iterative variant satisfies C2 and C3 cleanly, but C1 is weakened:

1. Default configuration (1x4) is single-pass -- responses from the pre-training checkpoint
2. Even in multi-iteration mode (2x2, 3x4), each iteration uses responses from the *previous* checkpoint, not the continuously-updating policy
3. The paper never claims "on-policy" or "online" training
4. The supervision is over response-set preference rankings, not per-token teacher guidance during generation

The distributional nature of the supervision (JSD over Plackett-Luce, Eq. 12) is genuinely richer than scalar reward, justifying borderline_strict over partial_opd for the iterative variant.

### Why partial_opd for ADPA and not adjacent

ADPA's advantage-weighted objective (Eq. 10) consumes the full teacher log-probability distribution over the vocabulary, not just a scalar. The gradient (Eq. 12) is a policy-gradient-style update weighted by per-token advantages from teacher distributions. This is distribution-level supervision, stronger than preference labels or scalar rewards. The stale-snapshot C1 failure prevents strict or borderline, but C3 quality keeps it above adjacent.

### Why not_opd for GAKD and not partial_opd

GAKD's student never generates tokens during training. The "Generative" in the name describes the GAN framework, not the student's behavior. Without any student generation (autoregressive or otherwise), C1 fails completely -- not partially. This is standard offline KD with an adversarial logit discriminator regularizer.

### Why adjacent for daDPO and CTPD

Both methods operate on static preference datasets with no on-policy component:
- daDPO: responses sampled once from initial student/teacher, fixed before DPO training
- CTPD: external UltraFeedback dataset, pre-computed span weights, single epoch

The teacher distributions enter the loss directly (C3 structurally satisfied), making these more interesting than pure static DPO, but the complete absence of student-generated training data (C1 fail) and same-rollout supervision (C2 fail for CTPD; partial for daDPO) places them in adjacent territory.

---

## Verification methodology

1. **Primary-source fetching:** Five agents fetched arXiv/OpenReview HTML for each paper independently.

2. **Algorithm-level tracing:** For each method, the exact training loop was reconstructed from Algorithm pseudocode. The critical test: does the student generate tokens autoregressively during the training loop, or only via forward passes on existing sequences?

3. **Loss equation tracing:** For each method, the exact equation where the teacher signal enters the student's gradient was identified. The critical test: does the teacher signal appear as (a) a distributional target (KL, JSD, log-prob ratio), (b) a scalar reward, or (c) a preference label?

4. **Conservative rules applied:**
   - Single-pass student sampling from initial checkpoint = C1 partial at best
   - Precomputed teacher advantages = C2 partial
   - Forward-pass logits on corpus sequences, no autoregressive generation = C1 fails completely
   - Static external preference dataset = C1 and C2 both fail
   - Teacher distribution entering loss directly = C3 satisfied even when C1/C2 fail
   - White-box requirement confirmed for all five methods
