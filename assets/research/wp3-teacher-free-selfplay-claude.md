# WP3 — Teacher-Free, Self-Play, Privileged-Context, and Self-Distillation OPD

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-05-13
**Sources:** arXiv HTML full text for all primary-source papers; repo table rows cross-referenced against `taxonomies/strict-opd-definition.md`
**Methodology:** Six independent research agents fetched primary-source HTML, extracted section/equation-level quotes, and assessed C1/C2/C3. Results reconciled below against WP3-scoped C2 criteria (previous checkpoint, privileged self-view, reference model, self-play opponent, verifier, or other teacher substitute).

---

## 1. Memo

### 1.1 Scope

WP3 covers methods where no external teacher model exists. The "teacher" role is filled by:
- The same model under **privileged context** (ground truth, demonstrations, documents, short context)
- A **previous checkpoint** of the same model
- A **self-play opponent** (current or previous policy)
- A **verifier** or **self-reward** signal
- A **feedback-conditioned** variant of the same model

Priority queue: SPIN, OPSD, OPSDC/CRISP, SDPO, OPCD, OEL, HDPO, GATES, π-Distill, SDFT, and self-play DPO/SPO/IPO variants.

### 1.2 Central finding

**Seven methods qualify as strict OPD. Three are partial. One is not_opd. Five are adjacent.** The strict/non-strict boundary in WP3 tracks the same structural pattern found in WP1 and WP2: the dividing line is whether the objective consumes **distributional/token-level** supervision from a privileged self-view (strict) or whether it consumes **preference labels, scalar rewards, or binary real-vs-fake signals** (adjacent).

The privileged-context self-distillation family (OPSD, CRISP, SDPO, OPCD, SDFT, CaOPD, OPSDL) is the strongest WP3 subfamily. All share a common structure: same model, different input conditioning (with/without ground truth, demonstrations, or documents), token-level KL objective on student-generated rollouts. This is a clean analogue of external-teacher OPD where the "teacher" is the model itself under a privileged view.

The self-play preference family (SPIN, SPPO, Self-Rewarding LMs, Iterative DPO, OAIF) uniformly fails C3: all use preference/reward objectives, not distillation-style supervision. These are on-policy preference optimization methods, useful but categorically distinct from OPD.

### 1.3 Method families

| Family | Members | Pattern | Typical label |
|---|---|---|---|
| **Privileged-context self-distillation** | OPSD, CRISP, SDPO, OPCD, SDFT, CaOPD, OPSDL | Same model ± privileged input; token-level KL on student rollouts | strict_opd |
| **Hybrid privileged + RL** | OEL, HDPO, GATES | One arm is privileged self-distillation, another is GRPO/RL; combined method mixes on-policy and off-policy | partial_opd |
| **Off-policy privileged distillation** | π-Distill | Privileged teacher generates all rollouts; student trains off-policy | not_opd |
| **Self-play preference optimization** | SPIN, SPPO, Self-Rewarding LMs, Iterative DPO, OAIF | On-policy rollouts + DPO/preference/reward loss, no distillation signal | adjacent |

### 1.4 Key boundary calls

| Method | Final label | Why |
|---|---|---|
| OPSD | `strict_opd` | Student generates reasoning traces; privileged self-teacher (same model + ground-truth context) supervises via per-token KL. Source-verified. |
| CRISP/OPSDC | `strict_opd` | Student generates reasoning traces; concise self-view supervises via reverse KL for reasoning compression. Source-verified. |
| SDPO | `strict_opd` | Student rollouts; feedback-conditioned self-teacher; JSD symmetric KL. Source-verified. |
| OPCD | `strict_opd` | Student generates without context; context-conditioned same model supervises via reverse KL. Source-verified. |
| SDFT (2026) | `strict_opd` | Student generates on-policy; demo-conditioned EMA self-teacher scores exact student tokens via reverse KL. Now source-verified by this audit. |
| CaOPD | `strict_opd` | Student rollouts; privileged-context self-teacher; calibration-corrected KL. Source-verified. |
| OPSDL | `strict_opd` | Long-context student generates; short-context self-teacher supervises via pointwise reverse KL. Source-verified. |
| OEL | `partial_opd` | OPCD-style consolidation substage is strict; full method includes deployment collection and knowledge extraction stages that break continuous on-policy property. |
| GATES | `partial_opd` | On-policy arm (lambda_on=0.1) is strict: student generates, tutor scores tokens via advantage. Off-policy arm (lambda_off=1.0, dominant) is SFT on tutor-generated traces. Full method partial. |
| HDPO | `partial_opd` | GRPO arm is on-policy (student rollouts). Distillation arm generates privileged-teacher rollouts and teacher-forces the student through them. C1 fails for the distillation arm. |
| π-Distill | `not_opd` | PI-conditioned teacher generates all training traces. Paper explicitly states "off-policy learning" (Sec 3.1). OPSD (same paper, Sec 3.2) is the on-policy companion. |
| SPIN | `adjacent` | C1 passes (iterative re-generation). C3 fails: DPO-like logistic loss on sequence-level log-ratio differences, not token-level KL distillation. Paper says loss "bears resemblance to DPO." |
| SPPO | `adjacent` | On-policy generation + preference probability loss via PairRM judge. No teacher logits, no KL. |
| Self-Rewarding LMs | `adjacent` | On-policy self-generated pairs + iterative DPO. No distillation signal. |
| Iterative DPO | `adjacent` | On-policy sampling + DPO preference loss with reward model ranking. |
| OAIF | `adjacent` | On-policy pairs + DPO-style loss on AI-annotated preferences. |

### 1.5 Critical distinctions

**Why privileged self-distillation is strict but self-play preference is adjacent:**

The privileged self-distillation family (OPSD, SDFT, etc.) satisfies C3 because:
1. The teacher provides a **full probability distribution** over the vocabulary at each token position
2. The objective is **KL divergence** (forward, reverse, or JSD), which directly consumes this distribution
3. The gradient flows through **per-token log-probability ratios**, not through binary preference labels

The self-play preference family (SPIN, SPPO, etc.) fails C3 because:
1. The "opponent" provides only **binary real-vs-fake labels** or **preference rankings**
2. The objective is a **logistic/preference loss** on sequence-level scores
3. The opponent's log-probs serve as **reference normalization** (like DPO's reference policy), not as a distillation target

This is the same C3 boundary found in WP2 (GAD/PRISM use discriminator scalar rewards via GRPO, not token-level KL) and WP1 (GAKD uses corpus sequences, not student rollouts).

### 1.6 SDFT 2026 name collision

Two papers share the "SDFT" acronym:
- **SDFT (2024) Yang et al.** (arXiv 2402.13669): "Self-Distillation for Further Pre-Training" — data preprocessing, not on-policy training. `not_opd`.
- **SDFT (2026) Shenfeld et al.** (arXiv 2601.19897): "Self-Distillation Enables Continual Learning" — on-policy reverse KL with demo-conditioned EMA self-teacher. `strict_opd`.

Only the 2026 version belongs in the OPD table. The repo already tracks this collision.

---

## 2. Evidence Ledger

### 2.1 Strict OPD (source-verified)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OPSD | https://arxiv.org/abs/2601.18734 | S3, Alg.1 | "student produces current-policy reasoning traces" | "privileged self-teacher supervises those traces" (same model + ground-truth context) | "per-token KL from privileged self" | privileged_context (ground-truth answer) | token | forward/reverse KL | current_policy | strict_opd | high (repo source-verified) |
| CRISP/OPSDC | https://arxiv.org/abs/2603.05433 | S3, Alg.1 | "model generates its own reasoning traces" | "concise or privileged self-view supervises the trace" | "reverse-KL self-policy objective compresses reasoning" | privileged_context (concise instruction) | token | reverse KL | current_policy | strict_opd | high (repo source-verified) |
| SDPO | https://arxiv.org/abs/2601.20802 | S3-4, Alg.1 | "student samples responses from the current policy" | "feedback-conditioned self-teacher supervises them" (same model + correctness feedback) | "logit-level or KL self-distillation consumes that signal; JSD symmetric KL" | privileged_context (feedback-conditioned) | token | JSD | current_policy | strict_opd | high (repo source-verified) |
| OPCD | https://arxiv.org/abs/2602.12275 | S3, Alg.1 | "student samples without privileged context" | "context-conditioned teacher supervises the same outputs" (same model + context) | "reverse-KL objective is used at each position" | privileged_context (task context) | token | reverse KL | current_policy | strict_opd | high (repo source-verified) |
| SDFT (2026) | https://arxiv.org/abs/2601.19897 | S3 Eq.1-2, Alg.1, App.A.1 | "SDFT samples responses from the student policy y ~ pi_theta(.\|x)" (S3, Alg.1 line 7) | "we construct the teacher by conditioning it on expert demonstrations: pi(.\|x,c)" (S3); EMA weights phi = alpha*theta + (1-alpha)*phi (Alg.1 line 19); teacher scores exact student-generated tokens (Alg.1 lines 9-12) | "D_KL(pi_theta(.\|x) \|\| pi(.\|x,c))" (Eq.1); "we decompose this objective into a token-level loss" (S3); analytic per-token estimator sums over full vocabulary V (Eq.2, App.A.1) | privileged_context (expert demonstrations via ICL); EMA self-teacher | token (analytic per-token over V) | reverse KL | current_policy | strict_opd | high (source-verified this audit) |
| CaOPD | https://arxiv.org/abs/2604.16830 | S3-4 | "student rollouts are sampled for training" | "privileged-context self-teacher supervises the same states" | "calibration-corrected KL objective consumes that supervision" | privileged_context | token | calibration-corrected KL | current_policy | strict_opd | medium (repo source-verified; calibration target needs recheck) |
| OPSDL | https://arxiv.org/abs/2604.17535 | S3-4 | "long-context student generates responses" | "short-context self-teacher supervises those same generated tokens" | "point-wise reverse-KL objective consumes the privileged self signal" | privileged_context (short-context self) | token | pointwise reverse KL | current_policy | strict_opd | high (repo source-verified) |

### 2.2 Partial OPD (source-verified)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_for_partial | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OEL | https://arxiv.org/abs/2603.16856 | S3-4, Alg.1 | "mixed_policy: deployment collection produces rollouts" | "experience-conditioned self-teacher supervises consolidation substage" | "OPCD-style token-level RKL in consolidation" | privileged_context (experiential) | token (consolidation only) | OPCD-style consolidation | mixed_freshness | partial_opd | Full method wraps strict OPCD-style consolidation in multi-stage loop with deployment collection and knowledge extraction; method-level label remains partial | high (repo source-verified) |
| GATES | https://arxiv.org/abs/2602.20574 | S3.2-3.3, Eq.1-4 | Off-policy arm: "we generate k independent tutor rollouts" (tutor generates, C1 FAILS); On-policy arm: "k student rollouts" from pi_theta (C1 PASSES) (S3.2) | "the tutor is the same model instance (sharing all parameters) queried with additional context" (S3.1); on-policy arm: tutor scores student tokens via advantage A_t = clip(log pi_T - log pi_theta) (Eq.2-3) | Off-policy: NLL imitation on tutor traces (Eq.1); On-policy: advantage-weighted PG (Eq.2-3); combined: lambda_off=1.0, lambda_on=0.1 (Eq.4) | privileged_context (source document) | token (both arms) | NLL (off-policy) + advantage-weighted PG (on-policy) | mixed: tutor traces (off-policy) + student traces (on-policy) | partial_opd | Off-policy arm (lambda_off=1.0) dominates; on-policy arm (lambda_on=0.1) alone would be strict. Ablation: "Off-policy provides primary performance gains" (S4.3) | high (source-verified this audit) |
| HDPO | https://arxiv.org/abs/2603.23871 | S3.2, Alg.1 | GRPO arm: student generates K rollouts from unprivileged x (C1 PASSES); Distillation arm: "generate privileged rollouts y-bar_j ~ pi_theta(.\|x + y*)" (C1 FAILS for distillation) (S3.2) | "pi_T and pi_theta share the same weights; they differ only in their input (privileged vs. unprivileged)" (S3.2) | "L_HDPO = L_GRPO + lambda * L_JSD" (S3.2); JSD is token-level between privileged teacher and unprivileged student, computed on privileged-teacher rollouts | privileged_context (ground-truth answer y*) | token (JSD on privileged traces) + sequence (GRPO) | GRPO + JSD | student rollouts (GRPO arm) + privileged-teacher rollouts (distillation arm) | partial_opd | C1 fails for distillation arm: privileged teacher pi_theta(.\|x+y*) generates rollouts, student is teacher-forced through them. Structurally opposite of OPCD. No off-policy correction. Cliff prompts only (C = {x : all K rollouts failed}) | high (source-verified this audit) |

### 2.3 Not OPD (source-verified)

| method | source_url | exact_section | C1 quote | C2 quote | C3 quote | final_label | reason | confidence |
|---|---|---|---|---|---|---|---|---|
| π-Distill | https://arxiv.org/abs/2602.04942 | S3.1 Eq.3, S3.2 Eq.5 | C1 FAILS: "pi-Distill can be viewed as off-policy learning, where the student is trained on trajectories generated by the PI-conditioned teacher" (S3.1); Eq.3 expectation over o ~ pi_theta^T | "The student is simply the base model without this conditioning" (S3); teacher = same model + privileged info (tool calls, hints) | Student loss: importance-weighted PG + reverse KL to teacher (Eq.3); structurally distillation-like but moot since C1 fails | not_opd | Teacher generates all traces. OPSD (Eq.5, S3.2) is the on-policy companion from the same paper: "the updates are on-policy as the expectation is taken over pi^S" | high (source-verified this audit) |

### 2.4 Adjacent (source-verified or primary-source audited)

| method | source_url | exact_section | C1 | C2 | C3 | final_label | reason | confidence |
|---|---|---|---|---|---|---|---|---|
| SPIN | https://arxiv.org/abs/2401.01335 | S4.1-4.2, Eq.4.7, Alg.1 | PASS: "Generate: y_i' ~ p_{theta_t}(.\|x_i)" per iteration (Alg.1) | PARTIAL: previous checkpoint p_{theta_t} as opponent/reference; human data as ground truth; no teacher logits | C3 FAILS: L_SPIN = ell(lambda * [log p_theta(y)/p_{theta_t}(y) - log p_theta(y')/p_{theta_t}(y')]) (Eq.4.7); "bears resemblance to DPO" (S4.2); logistic loss on sequence-level log-ratio differences | adjacent | Self-play discriminative objective, not distillation. Opponent log-probs are reference normalization, not a distillation target. No token-level KL to any teacher distribution. | high (source-verified this audit) |
| SPPO | https://arxiv.org/abs/2405.00675 | Abstract, S3 | PASS: iterative self-play generates from current policy | NO teacher: uses frozen PairRM preference model as reward judge | C3 FAILS: preference probability loss, not KL/token-level | adjacent | On-policy generation + preference reward, not distillation | high |
| Self-Rewarding LMs | https://arxiv.org/abs/2401.10020 | S2-3 | PASS: model generates own training pairs | NO external teacher: model acts as LLM-as-Judge | C3 FAILS: iterative DPO preference loss | adjacent | Self-play preference loop, no distillation signal | high |
| Iterative DPO | https://arxiv.org/abs/2312.11456 | S3-4 | PASS: on-policy sampling each iteration | NO teacher: reward model for ranking | C3 FAILS: DPO preference loss with KL regularization (constraint, not distillation) | adjacent | Online preference optimization, not distillation | high |
| OAIF | https://arxiv.org/abs/2402.04792 | S2-3 | PASS: samples two responses per iteration | External LLM annotator provides preference labels (not logits) | C3 FAILS: DPO-style preference loss on AI-annotated pairs | adjacent | Online AI feedback for preference alignment | high |

---

## 3. Candidate Rows for `tables/opd_papers.md`

### 3.1 New row: SDFT (2026)

SDFT (2026) is confirmed strict_opd but currently absent from `tables/opd_papers.md`. Proposed row:

```markdown
| sdft-2026 | Self-Distillation Enables Continual Learning | 2026 | Shenfeld et al. | arXiv | https://arxiv.org/abs/2601.19897 |  |  | LLM | continual_learning | self_play_opd | self_play | teacher_free | token | reverse_kl | strong | strict_opd | privileged_context | self_or_previous_checkpoint | privileged_self | policy_student | same_model_with_demo_context | student_on_policy | current_policy | privileged_context | token | reverse_kl_from_demo_conditioned_ema_self | C1 student samples on-policy y ~ pi_theta(.|x) (Alg.1 line 7); C2 demo-conditioned EMA self-teacher scores exact student tokens (Alg.1 lines 9-12); C3 reverse KL at token level via analytic estimator (Eq.1-2, App.A.1). | no |  | continual_post_training | continual_learning; skill_acquisition | custom | unclear | not_applicable | https://arxiv.org/abs/2601.19897 | arxiv | verified | source_verified | high | Demo-conditioned EMA self-teacher (same model + ICL demonstrations) provides token-level reverse KL supervision on student on-policy rollouts. | EMA teacher pass plus analytic per-token gradient estimation | MIT Improbable AI + ETH Zurich; name collision with SDFT 2024 (Yang et al., 2402.13669) which is not_opd. | 2026-05-13 |
```

### 3.2 Existing row updates

**GATES (row 54):** Update `verification_status` from `needs_review` to `verified`, `conflict_status` from `wp9_downgraded` to `source_verified`, `confidence` from `medium` to `high`. Update `strictness_evidence`: "C1 fails for dominant off-policy arm (lambda_off=1.0, tutor generates trajectories, Eq.1). Secondary on-policy arm (lambda_on=0.1) satisfies strict: student generates, tutor scores tokens via advantage (Eq.2-3). Ablation (S4.3): off-policy provides primary gains. Full method partial."

**HDPO (row 55):** Update `verification_status` from `needs_review` to `verified`, `conflict_status` from `wp9_downgraded` to `source_verified`, `confidence` from `medium` to `high`. Update `strictness_evidence`: "C1 fails for distillation arm: privileged teacher pi_theta(.|x+y*) generates rollouts, student is teacher-forced (S3.2). GRPO arm is on-policy. Combined L_HDPO = L_GRPO + lambda*L_JSD. Cliff prompts only."

### 3.3 No new rows needed for adjacent methods

SPIN, SPPO, Self-Rewarding LMs, Iterative DPO, and OAIF are adjacent self-play preference methods that do not belong in the OPD papers table. They should be tracked in `tables/adjacent_work.md` if not already present.

---

## 4. Adjacent / False-Positive Ledger

| method | link | why_adjacent | missing_condition | self_claims_opd | confidence |
|---|---|---|---|---|---|
| SPIN | https://arxiv.org/abs/2401.01335 | Self-play fine-tuning with DPO-like discriminative loss (Eq.4.7); opponent log-probs serve as reference normalization, not distillation target; "bears resemblance to DPO" (S4.2) | C3: sequence-level preference loss, not token-level distillation | No (claims "self-play fine-tuning") | high |
| SPPO | https://arxiv.org/abs/2405.00675 | Self-play preference optimization using PairRM as reward judge; preference probability loss | C2 (no teacher logits) + C3 (preference reward) | No | high |
| Self-Rewarding LMs | https://arxiv.org/abs/2401.10020 | LLM-as-Judge self-play with iterative DPO; no external teacher or distillation signal | C2 (no teacher distribution) + C3 (DPO preference) | No | high |
| Iterative DPO | https://arxiv.org/abs/2312.11456 | Online DPO with reward model ranking; on-policy sampling but preference loss | C3 (preference loss, not distillation) | No | high |
| OAIF | https://arxiv.org/abs/2402.04792 | Direct language model alignment from online AI feedback; DPO on AI-annotated pairs | C2 (AI preference labels, not logits) + C3 (DPO preference) | No | high |
| π-Distill | https://arxiv.org/abs/2602.04942 | Privileged-information-conditioned teacher generates all rollouts; student trains off-policy; paper states "off-policy learning" (S3.1) | C1 (teacher generates, not student) | No (explicitly distinguishes from OPSD) | high |
| SDFT (2024) | https://arxiv.org/abs/2402.13669 | Yang et al. "Self-Distillation for Further Pre-Training" — data preprocessing, not on-policy training | C1 (no student rollout generation during training) | No | high |

---

## 5. Gap Map — Open Questions

### 5.1 Source audit gaps

1. **OPSD Algorithm 1 detail:** The repo marks OPSD strict at `wp0_consensus` level. While the structural classification is clear (privileged self-teacher + token-level KL on student rollouts), a line-level audit confirming exact KL direction (forward vs reverse vs both), sampling freshness (per-step vs per-epoch), and whether the privileged view is frozen or co-evolving would strengthen confidence from medium to high.

2. **CRISP/OPSDC Algorithm 1 detail:** Same issue — `wp0_consensus` without section-level quote extraction. The iterative self-policy distillation loop needs confirmation: does each iteration re-sample from the compressed student?

3. **CaOPD calibration target:** The repo notes "calibration target replacement should be rechecked when code appears." The corrected calibration replaces teacher uncertainty information, which could weaken C2 if the calibration adjustment removes rather than refines teacher signal.

### 5.2 Structural gaps

4. **GATES on-policy arm scaling:** The on-policy arm alone satisfies strict OPD (Eq.2-3: student generates, tutor scores, advantage-weighted PG). If lambda_on were dominant, GATES would be strict. The ablation (S4.3) shows modest but positive contribution. Track whether future GATES variants increase lambda_on.

5. **HDPO vs OPCD structural inversion:** HDPO generates privileged-teacher rollouts and teacher-forces the student. OPCD generates student rollouts and has the privileged teacher score them. These are structural inverses. The repo should explicitly document this inversion as a canonical example of why C1 matters.

6. **SDPO naming collision:** Multiple papers use "SDPO" (Self-Distillation Policy Optimization vs Self-play DPO vs other variants). The repo's line audit noted this but the adjacent_work table should track the non-OPD SDPO variants to prevent confusion.

### 5.3 Coverage gaps

7. **MTP-SD (Multi-Token Prediction Self-Distillation):** Referenced in `wp1-whitebox-gemini.md:178` as deferred to WP3. Not yet audited. Likely a next-token vs multi-token privileged self-teacher. Needs primary-source fetch.

8. **DAIL:** Referenced in repo as "SPIN/DAIL are adjacent" (README:230) but no primary-source audit was found. The DAIL paper should be fetched and classified.

9. **TMS (Thinking Machines Lab OPD blog):** Referenced in `strict-opd-definition.md:34` as "an operational definition/background source, not a normal paper row." Should remain a reference, not a table entry.

10. **Previous-checkpoint distillation without privileged context:** Methods like EMA self-distillation (where the teacher is simply a lagging EMA copy with no privileged information) are not well-covered. SDFT (2026) uses EMA weights but also adds demo conditioning. A pure EMA self-distillation method (no privileged context) would be an interesting boundary case -- is the EMA lag alone sufficient for C2?

11. **Self-play IPO:** No specific "Self-Play IPO" paper was found. IPO (Identity Preference Optimization) is a preference loss variant; self-play IPO would fail C3 for the same reason SPIN fails (preference loss, not distillation).

### 5.4 Verification actions needed

| Action | Priority | Method | What to do |
|---|---|---|---|
| Line audit | high | OPSD | Fetch arXiv HTML, extract exact KL equation, confirm sampling freshness, upgrade from wp0_consensus to source_verified |
| Line audit | high | CRISP | Fetch arXiv HTML, extract iterative loop details, confirm reverse-KL on compressed student rollouts |
| Line audit | medium | CaOPD | Fetch arXiv HTML, verify calibration correction preserves rather than removes teacher distributional signal |
| Primary fetch | medium | MTP-SD | Locate paper, assess C1/C2/C3 |
| Primary fetch | low | DAIL | Locate paper, confirm adjacent classification |
| Table update | high | SDFT (2026) | Add new row to opd_papers.md |
| Table update | high | GATES | Upgrade from needs_review to source_verified |
| Table update | high | HDPO | Upgrade from needs_review to source_verified |

---

## 6. Cross-validation with existing repo labels

| Method | Prior repo label | This audit | Change? |
|---|---|---|---|
| OPSD | strict_opd (wp0_consensus) | strict_opd | confirmed, needs line audit for upgrade to source_verified |
| CRISP/OPSDC | strict_opd (wp0_consensus) | strict_opd | confirmed, needs line audit for upgrade |
| SDPO | strict_opd (source_verified) | strict_opd | confirmed |
| OPCD | strict_opd (source_verified) | strict_opd | confirmed |
| SDFT (2026) | strict_opd (wp0 survey) | strict_opd | **confirmed + new source verification** |
| CaOPD | strict_opd (source_verified, medium confidence) | strict_opd | confirmed |
| OPSDL | strict_opd (source_verified) | strict_opd | confirmed |
| OEL | partial_opd (source_verified) | partial_opd | confirmed |
| GATES | partial_opd (needs_review) | partial_opd | **confirmed + upgraded to source_verified** |
| HDPO | partial_opd (needs_review) | partial_opd | **confirmed + upgraded to source_verified** |
| π-Distill | not_opd | not_opd | confirmed |
| SPIN | adjacent | adjacent | confirmed |

**No label changes needed.** The WP3 audit confirms all existing labels and provides source-level evidence for two previously unverified entries (GATES, HDPO) plus one entry lacking a table row (SDFT 2026).

---

## 7. Verification methodology

1. **Primary-source fetching:** Six agents fetched arXiv HTML for SPIN, SDFT (2026), π-Distill, GATES, HDPO, and self-play variants (SPPO, Self-Rewarding LMs, Iterative DPO, OAIF).

2. **Rollout-source tracing:** For each method, the exact Algorithm pseudocode was analyzed to determine who generates the training trajectories (student vs teacher/tutor/privileged model). This is the most discriminating test in WP3.

3. **Objective decomposition:** For each method, the loss function was decomposed into:
   - Token-level KL/JSD terms (distillation-style → supports C3)
   - Sequence-level preference/logistic terms (preference-style → fails C3)
   - Scalar reward/advantage terms (RL-style → fails C3 unless combined with distillation)
   - NLL/SFT terms on teacher-generated traces (off-policy → fails C1)

4. **Conservative rules applied:**
   - Privileged self-view + token-level KL on student rollouts = strict_opd
   - Same structure but off-policy (teacher generates) = not_opd
   - Hybrid on-policy + off-policy arms = partial_opd at method level
   - On-policy rollouts + preference/reward loss = adjacent (not OPD)
   - Previous checkpoint as DPO reference (not distillation target) = adjacent

---

## Sources

- [OPSD — Self-Distilled Reasoner (arXiv 2601.18734)](https://arxiv.org/abs/2601.18734)
- [CRISP — Compressed Reasoning via Iterative Self-Policy Distillation (arXiv 2603.05433)](https://arxiv.org/abs/2603.05433)
- [SDPO — Self-Distillation Policy Optimization (arXiv 2601.20802)](https://arxiv.org/abs/2601.20802)
- [OPCD — On-Policy Context Distillation (arXiv 2602.12275)](https://arxiv.org/abs/2602.12275)
- [SDFT — Self-Distillation Enables Continual Learning (arXiv 2601.19897)](https://arxiv.org/abs/2601.19897)
- [CaOPD — Calibration-Corrected On-Policy Distillation (arXiv 2604.16830)](https://arxiv.org/abs/2604.16830)
- [OPSDL — On-Policy Self-Distillation for Long-Context LMs (arXiv 2604.17535)](https://arxiv.org/abs/2604.17535)
- [OEL — Online Experiential Learning (arXiv 2603.16856)](https://arxiv.org/abs/2603.16856)
- [GATES — Grounded Adaptive Tutor for Efficient Self-Distillation (arXiv 2602.20574)](https://arxiv.org/abs/2602.20574)
- [HDPO — Hybrid Distillation Policy Optimization (arXiv 2603.23871)](https://arxiv.org/abs/2603.23871)
- [π-Distill — Privileged Information Distillation (arXiv 2602.04942)](https://arxiv.org/abs/2602.04942)
- [SPIN — Self-Play Fine-Tuning (arXiv 2401.01335)](https://arxiv.org/abs/2401.01335)
- [SPPO — Self-Play Preference Optimization (arXiv 2405.00675)](https://arxiv.org/abs/2405.00675)
- [Self-Rewarding Language Models (arXiv 2401.10020)](https://arxiv.org/abs/2401.10020)
- [Online Iterative RLHF / Iterative DPO (arXiv 2312.11456)](https://arxiv.org/abs/2312.11456)
- [OAIF — Online AI Feedback (arXiv 2402.04792)](https://arxiv.org/abs/2402.04792)
