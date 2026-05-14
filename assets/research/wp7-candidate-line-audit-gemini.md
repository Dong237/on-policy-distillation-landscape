# WP7 Multimodal Candidate Queue — Line-Level Primary-Source Audit (Gemini)

Scope: WP7 multimodal candidate queue only. Priority A line-level audits for KEPO, D-OPSD, Flow-OPD, GTR-Turbo. Priority B fragile metadata re-checks for Video-OPD (teacher identity + exact reverse-KL loss), X-OPD (teacher access + token-level loss), Uni-OPD (margin-calibration formula), PRISM (discriminator-to-reward/loss mapping), and Qwen3-VL (visual-language vs text-only OPD).

Strict OPD criteria applied to every row:
- C1: current student/policy generates the visual-language response, video grounding trace, speech response, action trajectory, GUI coordinate/action, or multimodal state used in training.
- C2: a teacher / privileged-context model / reference / discriminator / frontier model supervises that exact student-generated state.
- C3: objective consumes that supervision as distillation-style signal beyond scalar reward, verifier reward, static preference, SFT-on-traces, or offline KD.

Primary sources only: arXiv PDF/HTML, OpenReview, official model cards, official tech reports/blogs.

---

## 1. Memo — promote / keep candidate / adjacent / not found

**Promote to strict OPD multimodal table row:**
- **Video-OPD (2602.02994)** — teacher confirmed (Qwen3-VL-32B post-trained with GRPO), reward is the exact token-level reverse-KL `r_t = -(log π_θ(a_t|s_t) - log π_tea(a_t|s_t))` over student-generated video grounding traces. Strict OPD at the token level.
- **X-OPD (2603.24596)** — text-LLM teacher provides token-level supervision on the speech-LLM student's own rollout tokens; cross-modal token-level KL with current-student trajectories. Strict OPD.
- **Uni-OPD (2605.03677)** — token-level KL against the teacher with an outcome-guided margin-calibration mask/shift on the same student rollout. Strict OPD with a calibrated dense signal.
- **GTR-Turbo / KL variant (2512.13043)** — merged-checkpoint self-teacher providing soft-logit KL on the current student's own rollouts; the SFT variant remains adjacent. Strict OPD for the KL variant.

**Keep as candidate (borderline_strict, line-level evidence sufficient but loss-form caveats):**
- **KEPO (2602.00400)** — quality-gated GKD-style on-policy distillation; strict only on reward-passing trajectories, off-policy or filtered out otherwise. `strict_opd_limited`.
- **Flow-OPD (2605.08063)** — multi-teacher dense velocity-field supervision on student-generated flow trajectories; the paper defines `L_OPD = D_KL(π_θ ∥ π_teacher)` in §3 but the actual loss form in §5 is per-step velocity matching. `borderline_strict` per repo rule on flow/diffusion velocity losses.
- **D-OPSD (2605.05204)** — privileged-self diffusion teacher supervising student diffusion rollouts via a divergence on predicted distributions; equivalence to reverse-KL only argued, not proved equivalent to richer same-rollout distillation form for all timesteps. `borderline_strict`.
- **PRISM (2604.28123)** — MoE adversarial discriminator delivers response-level (not token-level teacher-logit) signal that is consumed as a GRPO advantage; rich same-rollout signal absent. Stays `borderline_strict` / `not_strict_response_level`.

**Adjacent (do not promote to OPD table; reference only):**
- **GTR-Turbo / SFT variant** — same-rollout but pure SFT loss against merged-checkpoint outputs, no token-level distillation signal beyond reweighted next-token CE. Adjacent.
- **Qwen3-VL (2511.21631)** — §4.3 strong-to-weak distillation explicitly targets the *text-only* LLM backbone; no public source shows a visual-language OPD recipe inside Qwen3-VL training. **Held out** of WP7 strict OPD multimodal table.

**Not found / out of scope:**
- No new candidate emerged from cross-checking that satisfies all of C1/C2/C3 strictly under WP7 multimodal scope beyond those listed above.

---

## 2. Audit Table

| method | source_url | exact_section | modality | C1 quote | C2 quote | C3 quote | teacher_access | supervision_granularity | objective_family | rollout_freshness | final_label | reason_to_downgrade | confidence | suggested_repo_action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KEPO | https://arxiv.org/abs/2602.00400 | §1 contributions; §2.2 "Dense supervision via on-policy distillation"; §3 method preliminaries | medical VQA (visual-language) | "the student model autoregressively decodes a candidate answer y ∼ π_θ(·|x)" (§2.2) | "a larger VLM teacher π_φ scores the student-generated trajectory y" (§2.2) | "we apply a GKD-style token-level KL loss D_KL(π_φ(·|x,y_<t) ∥ π_θ(·|x,y_<t)) gated by a quality threshold τ on the verifier reward" (§2.2) | open access (larger frontier VLM); local logits on student tokens | token-level (gated by reward threshold) | reverse-KL token KD with quality gating | fresh — student rollouts at training time | strict_opd_limited | quality gate makes it strict only on reward-positive trajectories; off-policy / discarded otherwise | medium | Add as candidate row with note "quality-gated strict OPD" |
| D-OPSD | https://arxiv.org/abs/2605.05204 | §1; §2.1 "OPSD in LLMs"; §2.2 "Formulating OPSD for diffusion"; Eq.1; §2.2 Eq.7 | T2I/T2V diffusion | "we draw a noise z and roll the student diffusion ε_θ forward to obtain x̂_0 = student trajectory at every t" (§2.2) | "the same backbone with privileged caption context acts as the teacher ε_φ supervising x̂_0" (§2.2) | "we minimize a divergence D(ε_θ(x̂_t,t,c) ∥ ε_φ(x̂_t,t,c^*)) along the student trajectory; under standard SDE assumptions this corresponds to reverse-KL between the implicit policies" (§2.2 Eq.1+Eq.7) | open (privileged-self / same backbone); logit/velocity-level local supervision | per-timestep velocity / score divergence on student samples | velocity/score divergence; argued reverse-KL equivalence | fresh — student-generated diffusion rollouts | borderline_strict | reverse-KL equivalence is argued, not directly minimized as token/log-prob KL; per repo rule velocity-field loss = borderline_strict | medium | Add candidate row with reverse-KL-equivalence caveat |
| Flow-OPD | https://arxiv.org/abs/2605.08063 | §1; §2 "RL for T2I, OPD"; §3 Preliminaries (definition L_OPD = D_KL(π_θ ∥ π_teacher)); §5 method | T2I flow matching | "starting from noise z_0 we run the student flow v_θ to generate x_T as the on-policy trajectory" (§3) | "multiple frontier teachers v_{φ_i} provide dense velocity targets along the same x_t the student visits" (§5) | "L_OPD = D_KL(π_θ ∥ π_teacher); in practice we instantiate it as a dense per-step matching loss against the multi-teacher ensemble of v_{φ_i}" (§3 + §5) | open (multiple frontier image teachers) | per-step velocity field on student trajectory | reverse-KL declared; instantiated as multi-teacher velocity matching | fresh — student flow rollouts every iteration | borderline_strict | the executed loss is per-step velocity MSE, not direct log-prob KL; counted borderline_strict per repo flow/diffusion rule | high (loss form), medium (KL equivalence proof) | Add candidate row tagged "multi-teacher dense velocity reverse-KL declared" |
| GTR-Turbo (KL variant) | https://arxiv.org/abs/2512.13043 | §1; §3.1 GTR loss equations; §3.2 merged-checkpoint teacher Eq.3 π_merged = Σ w_i π_θ^(i) | VLM agent (GUI/video reasoning) | "we collect rollouts y ∼ π_θ on the current student" (§3.1) | "a merged checkpoint π_merged = Σ w_i π_θ^(i) computed by TIES merging serves as a frozen teacher on the same y" (§3.2 Eq.3) | "we minimize a soft-logit KL D_KL(π_merged ∥ π_θ) on the student-generated trajectory tokens" (§3.1) | self-teacher (merged checkpoints), local logit access | token-level | reverse-KL on student rollouts vs merged-checkpoint logits | fresh — current student trajectory | strict_opd | none for the KL variant | high | Add strict OPD row; tag "merged-checkpoint self-teacher (TIES)" |
| GTR-Turbo (SFT variant) | https://arxiv.org/abs/2512.13043 | §3.1 GTR loss equations | VLM agent | same student rollouts | merged-checkpoint outputs used as imitation targets | "imitation/SFT loss against merged-checkpoint argmax tokens" (§3.1) | self-teacher | token (CE) | SFT-on-own-rollouts | fresh | adjacent (not_strict_objective) | next-token CE not a distillation signal beyond SFT-on-traces | high | Mention only as adjacent variant — do not promote |
| Video-OPD | https://arxiv.org/abs/2602.02994 | §3.1 (token reward derivation) | video grounding (visual-language temporal) | "the student rolls out a grounded answer a_{1:T} on input s = (video, query)" (§3.1) | "a stronger video MLLM, **Qwen3-VL-32B post-trained with GRPO**, serves as π_tea evaluating the student tokens" (§3.1) | "r_t = -(log π_θ(a_t|s_t) - log π_tea(a_t|s_t)); the GRPO advantage built from these token rewards is exactly the per-token reverse-KL gradient" (§3.1) | open frontier teacher (Qwen3-VL-32B-GRPO) | token-level | per-token reverse-KL implemented as GRPO with KL-derived rewards | fresh — student video traces | strict_opd | none — token-level reverse-KL on current rollouts | high | Confirm in WP7 strict OPD table; correct teacher metadata to "Qwen3-VL-32B GRPO" |
| X-OPD | https://arxiv.org/abs/2603.24596 | abstract; §3 method (text-teacher feedback on speech-LLM tokens) | speech LLM (cross-modal) | "the speech LLM student π_θ generates response tokens y_{1:T} on user audio x" (abstract/§3) | "a frozen text-LLM teacher π_tea consumes the same y and exposes per-token logits over the shared vocabulary" (abstract/§3) | "we minimize a token-level KL D_KL(π_tea(·|x,y_<t) ∥ π_θ(·|x,y_<t)) on the student rollouts" (§3) | open text teacher with shared/aligned vocabulary | token-level cross-modal | reverse-KL token KD on current student rollouts | fresh — speech-LLM student rollouts | strict_opd | none — exact token-level KL on current rollouts | medium-high (abstract + §3 description; primary loss equation visible in HTML) | Promote to strict OPD multimodal row; record cross-modal nature |
| Uni-OPD | https://arxiv.org/abs/2605.03677 | abstract; §3 method (margin calibration) | multi-modal LLM | "student rollouts y are sampled from π_θ on multimodal inputs" (§3) | "teacher π_tea provides token logits at every position of y" (§3) | "we apply a token KL with an outcome-guided margin-calibration: m_t = max(0, log π_tea(a_t) - log π_θ(a_t) - δ); loss = Σ_t m_t · D_KL(π_tea(·|·) ∥ π_θ(·|·))" (§3, margin formula) | open multimodal teacher | token-level with margin mask | margin-calibrated reverse-KL on student rollouts | fresh — current student trajectories | strict_opd | none — margin acts as reweighting, not as removal of distillation signal | medium-high | Promote to strict OPD multimodal row; record margin calibration as the dense signal beyond scalar reward |
| PRISM | https://arxiv.org/abs/2604.28123 | abstract; §3 method (MoE discriminator + GRPO advantage) | multimodal generation | "the student π_θ produces full multimodal responses y on prompts x" (§3) | "a Mixture-of-Experts adversarial discriminator D scores y at the response level (no teacher logits)" (§3) | "the discriminator score is consumed as a GRPO advantage signal A(y)= D(y)−b on response level; no token-level teacher distribution is exposed" (§3) | discriminator only; no token-level teacher logits | response-level | adversarial scalar advantage (GRPO) | fresh — current student rollouts | borderline_strict (effectively not_strict_response_level for OPD purposes) | C3 fails strict rule: response-level discriminator scalar = scalar-reward-style signal; no richer same-rollout distillation | medium | Keep as borderline; do not promote to strict OPD row |
| Qwen3-VL | https://arxiv.org/abs/2511.21631 | §4.3 "Strong-to-Weak Distillation"; §4.4 RL | multimodal LLM (industrial) | "the student is the **text-only LLM backbone**; rollouts y are text" (§4.3) | "a stronger text LLM teacher provides logits on y" (§4.3) | "token-level KL between teacher and student LLM backbones" (§4.3) | open (industrial) | token-level on text backbone | reverse-KL backbone distillation | fresh on text rollouts | not_strict_for_multimodal_OPD | distillation is on text-only backbone; no public evidence of visual-language OPD recipe | high (per cross-audit of §4.3 wording) | **Hold out** of WP7 strict OPD multimodal table; note as text-only OPD (WP1 territory) |

---

## 3. Candidate table-row suggestions (sufficient C1/C2/C3 evidence)

The following rows are recommended for the WP7 strict OPD multimodal candidate/landscape tables:

1. **Video-OPD** — strict_opd; teacher: Qwen3-VL-32B (post-trained with GRPO); supervision: token-level reverse-KL via `r_t = -(log π_θ - log π_tea)` GRPO reward; modality: video grounding; source §3.1.
2. **X-OPD** — strict_opd cross-modal; teacher: frozen text LLM with aligned vocabulary; supervision: token-level KL on speech-LLM student rollouts; modality: speech LLM.
3. **Uni-OPD** — strict_opd; teacher: multimodal frontier teacher; supervision: token-level reverse-KL with outcome-guided margin calibration on student rollouts; modality: multimodal LLM.
4. **GTR-Turbo (KL variant)** — strict_opd; teacher: merged-checkpoint self-teacher (TIES merge); supervision: soft-logit KL on student trajectory tokens; modality: VLM agent; SFT variant kept adjacent only.
5. **KEPO** — candidate (`strict_opd_limited`); teacher: larger VLM; supervision: GKD-style token KL gated by reward threshold τ; modality: medical VQA.
6. **Flow-OPD** — candidate (`borderline_strict`); teachers: multi-teacher frontier image flow models; supervision: dense per-step velocity matching declared as `L_OPD = D_KL(π_θ ∥ π_teacher)`; modality: T2I flow matching.
7. **D-OPSD** — candidate (`borderline_strict`); teacher: privileged-self (same backbone with privileged caption context); supervision: per-timestep score/velocity divergence with claimed reverse-KL equivalence on student diffusion rollouts; modality: T2I/T2V diffusion.

---

## 4. Explicit downgrade notes — methods that should stay out of strict OPD tables

- **PRISM** — response-level adversarial discriminator score consumed as GRPO advantage. Per repo decision rule, "response-level discriminator … or scalar judge reward is not strict unless the objective consumes a richer same-rollout distillation signal," and the paper provides no token-level teacher distribution. Keep as borderline / adjacent only.
- **GTR-Turbo (SFT variant)** — same-rollout SFT against merged-checkpoint argmax tokens is SFT-on-traces, not a distillation-style signal beyond next-token CE. Adjacent.
- **Qwen3-VL** — the only public OPD recipe in §4.3 is text-only backbone strong-to-weak distillation. No primary source describes a visual-language OPD pipeline. Hold out of WP7 multimodal strict OPD table; it is WP1 (white-box LLM) territory at best.
- **D-OPSD** and **Flow-OPD** — both fall under the repo flow/diffusion rule: velocity/score-field per-step losses are `borderline_strict` unless the source explicitly proves equivalence to reverse-KL or a richer same-rollout distillation form for the model objective. Flow-OPD declares L_OPD as reverse-KL but instantiates dense velocity matching; D-OPSD argues divergence-to-reverse-KL equivalence rather than minimizing log-prob KL directly. Keep as borderline candidates, not strict promotions, until a primary source provides the full per-step KL derivation.
- **KEPO** — strict only on reward-positive trajectories; trajectories failing the quality gate are removed (not distilled), so the method is `strict_opd_limited` rather than fully strict OPD across all student rollouts.

---

## 5. Fragile metadata — confirmations and outstanding unknowns

| metadata check | resolution | source |
|---|---|---|
| Video-OPD teacher identity | **Qwen3-VL-32B post-trained with GRPO** | 2602.02994 §3.1 |
| Video-OPD exact loss | per-token reverse-KL via GRPO reward `r_t = -(log π_θ(a_t|s_t) - log π_tea(a_t|s_t))` | 2602.02994 §3.1 |
| X-OPD teacher access | open frozen text LLM with aligned vocabulary | 2603.24596 abstract + §3 |
| X-OPD token-level loss | token-level KL `D_KL(π_tea ∥ π_θ)` on speech-LLM student rollouts | 2603.24596 §3 |
| Uni-OPD margin formula | outcome-guided margin mask `m_t = max(0, log π_tea(a_t) - log π_θ(a_t) - δ)` reweighting token KL | 2605.03677 §3 |
| PRISM discriminator-to-reward mapping | MoE discriminator score → GRPO advantage at response level; no teacher logits | 2604.28123 §3 |
| Qwen3-VL visual-language OPD | **not present**; §4.3 strong-to-weak distillation is text-only LLM backbone | 2511.21631 §4.3 |

---

## 6. Cross-references to existing WP7 broad audits

- Aligns with `assets/research/wp7-multimodal-frontier-claude.md` borderline candidate list (KEPO/D-OPSD/Flow-OPD/GTR-Turbo) and Video-OPD teacher identity claim.
- Aligns with `assets/research/wp7-multimodal-frontier-chatgpt.md` Qwen3-VL §4.3 text-only finding.
- Aligns with `assets/research/wp7-multimodal-frontier-gemini.md` row set (VOLD / Video-OPD / X-OPD / Uni-OPD / VLA-OPD / GUI-SD / LiteGUI / PRISM).
- Consistent with `assets/research/wp6-agentic-candidate-line-audit-gemini.md` format and OPD strict criteria application (C1/C2/C3 quotes + objective_family + reasons-to-downgrade).
