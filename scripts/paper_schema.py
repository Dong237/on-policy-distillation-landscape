"""Shared schema constants for On Policy Distillation Landscape tables."""

BASE_PAPER_HEADER = [
    "id",
    "title",
    "year",
    "authors",
    "venue_or_status",
    "paper_url",
    "code_url",
    "project_url",
    "modality",
    "domain",
    "method_family",
    "feedback_route",
    "teacher_access_regime",
    "loss_granularity",
    "divergence_or_objective",
    "on_policy_strength",
    "opd_strictness",
    "teacher_access",
    "teacher_type",
    "teacher_kind",
    "student_model",
    "teacher_model",
    "rollout_source",
    "rollout_freshness",
    "supervision_signal",
    "supervision_granularity",
    "loss_objective",
    "strictness_evidence",
    "uses_rlvr",
    "rl_algorithm",
    "training_stage",
    "benchmarks",
    "framework",
    "code_status",
    "weights_status",
    "evidence_url",
    "evidence_type",
    "verification_status",
    "conflict_status",
    "confidence",
    "classification_rationale",
    "compute_cost_notes",
    "notes",
    "last_checked",
]

VLM_PAPER_HEADER = BASE_PAPER_HEADER + [
    "vision_task",
    "visual_input_type",
    "base_vlm",
    "teacher_modality",
    "vlm_failure_mode",
]

FRAMEWORKS_HEADER = [
    "framework_id",
    "module_id",
    "name",
    "repo_url",
    "docs_url",
    "active_status",
    "algorithms",
    "strict_opd_support",
    "vlm_support",
    "video_support",
    "teacher_logits",
    "black_box_teacher",
    "rlvr_support",
    "distributed_backend",
    "rollout_backend",
    "example_script",
    "evidence_url",
    "confidence",
    "notes",
    "last_checked",
]

BENCHMARKS_HEADER = [
    "name",
    "category",
    "modality",
    "domain",
    "official_url",
    "repo_url",
    "dataset_url",
    "leaderboard_url",
    "split_names",
    "num_examples",
    "task_format",
    "metrics",
    "private_test",
    "vlmevalkit_supported",
    "opencompass_supported",
    "common_for_vlm_rlvr",
    "common_for_vlm_opd",
    "notes",
]

INDUSTRIAL_REPORTS_HEADER = [
    "id",
    "name",
    "organization",
    "year",
    "report_url",
    "model_family",
    "modality",
    "training_signal",
    "uses_strict_opd",
    "opd_strictness",
    "evidence_url",
    "classification_rationale",
    "confidence",
    "last_checked",
    "notes",
]

ADJACENT_WORK_HEADER = [
    "id",
    "title",
    "year",
    "link",
    "category",
    "why_adjacent",
    "why_not_strict_opd",
    "evidence_url",
    "confidence",
    "last_checked",
    "notes",
]

EXPECTED_HEADERS = {
    "tables/opd_papers.md": BASE_PAPER_HEADER,
    "tables/vlm_opd_papers.md": VLM_PAPER_HEADER,
    "tables/frameworks.md": FRAMEWORKS_HEADER,
    "tables/benchmarks.md": BENCHMARKS_HEADER,
    "tables/industrial_reports.md": INDUSTRIAL_REPORTS_HEADER,
    "tables/adjacent_work.md": ADJACENT_WORK_HEADER,
}

OPD_STRICTNESS = {
    "strict_opd",
    "borderline_strict",
    "partial_opd",
    "adjacent",
    "not_opd",
    "unclear",
}

FEEDBACK_ROUTE = {
    "logit_based",
    "outcome_based",
    "self_play",
    "hybrid_logit_reward",
    "adversarial_black_box",
    "preference_based",
    "action_level",
    "offline_kd",
    "synthetic_data",
    "none",
    "unclear",
}

TEACHER_ACCESS_REGIME = {
    "white_box",
    "black_box",
    "teacher_free",
    "multi_teacher",
    "expert_policy",
    "no_teacher",
    "not_applicable",
    "unclear",
}

LOSS_GRANULARITY = {
    "token",
    "sequence",
    "hybrid",
    "token_or_sequence",
    "action",
    "outcome",
    "relation",
    "not_applicable",
    "unclear",
}

ON_POLICY_STRENGTH = {
    "strong",
    "borderline",
    "partial",
    "adjacent_on_policy_reward_only",
    "off_policy",
    "none",
    "unclear",
}

ROLLOUT_FRESHNESS = {
    "current_policy",
    "per_iteration",
    "per_epoch",
    "mixed_freshness",
    "replay_buffer_stale",
    "previous_policy",
    "precomputed_sft_rollouts",
    "static_dataset",
    "not_applicable",
    "unclear",
}

ROLLOUT_SOURCE = {
    "student_on_policy",
    "mixed_policy",
    "off_policy",
    "teacher_generated",
    "not_applicable",
    "unclear",
}

TEACHER_KIND = {
    "larger_llm",
    "larger_mllm",
    "text_teacher_llm",
    "multimodal_teacher",
    "expert_policy",
    "discriminator",
    "reference_model",
    "previous_checkpoint",
    "privileged_self",
    "multi_teacher",
    "reward_model_or_verifier",
    "none",
    "not_disclosed",
    "not_applicable",
    "unclear",
}

SUPERVISION_SIGNAL = {
    "token_kl",
    "reverse_kl",
    "sequence_feedback",
    "discriminator_feedback",
    "privileged_context",
    "teacher_logits",
    "teacher_logprobs",
    "black_box_response_adversarial",
    "hybrid_kl_reward",
    "action_token_supervision",
    "reward_only",
    "preference_only",
    "offline_kd",
    "synthetic_data",
    "none",
    "unclear",
}

CONFIDENCE = {"high", "medium", "low"}

VERIFICATION_STATUS = {
    "verified",
    "needs_review",
    "unclear",
}

CONFLICT_STATUS = {
    "source_verified",
    "survey_consensus",
    "survey_conflict",
    "verifier_consensus",
    "verifier_conflict",
    "verifier_downgraded",
    "human_spotcheck_needed",
    "needs_verifier",
    "source_weak",
    "not_applicable",
}

YES_NO_UNCLEAR = {"yes", "no", "unclear", "not_applicable"}
