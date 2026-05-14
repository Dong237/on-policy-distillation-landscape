#!/usr/bin/env python3
"""Validate Markdown table metadata for On Policy Distillation Landscape."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

from paper_schema import (
    CONFIDENCE,
    CONFLICT_STATUS,
    EXPECTED_HEADERS,
    FEEDBACK_ROUTE,
    LOSS_GRANULARITY,
    ON_POLICY_STRENGTH,
    OPD_STRICTNESS,
    ROLLOUT_FRESHNESS,
    ROLLOUT_SOURCE,
    SUPERVISION_SIGNAL,
    TEACHER_ACCESS_REGIME,
    TEACHER_KIND,
    VERIFICATION_STATUS,
    YES_NO_UNCLEAR,
)

ROOT = Path(__file__).resolve().parents[1]
URL_RE = re.compile(r"^https?://")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RATIONALE_KEYWORDS = {
    "teacher",
    "expert",
    "discriminator",
    "privileged",
    "reference",
    "supervision",
    "distillation",
    "logit",
    "kl",
}
NON_DISTILLATION_SIGNALS = {
    "reward_only",
    "preference_only",
    "offline_kd",
    "synthetic_data",
    "none",
}
NON_TEACHER_KINDS = {"none", "not_applicable", "not_disclosed", "unclear", "reward_model_or_verifier"}


class ValidationError(Exception):
    """Raised when a validation issue is found."""


def split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return []
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line[1:-1]:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_start = next((index for index, line in enumerate(lines) if line.strip().startswith("|")), None)
    if table_start is None:
        return [], []

    header = split_markdown_row(lines[table_start])
    if table_start + 1 >= len(lines):
        return header, []

    separator = split_markdown_row(lines[table_start + 1])
    if not is_separator_row(separator):
        return header, []

    rows: list[dict[str, str]] = []
    for line in lines[table_start + 2 :]:
        if not line.strip().startswith("|"):
            break
        values = split_markdown_row(line)
        row = {
            key: (values[position].strip() if position < len(values) else "")
            for position, key in enumerate(header)
        }
        if len(values) != len(header):
            row["__column_count_error"] = f"line has {len(values)} cells, expected {len(header)}"
        rows.append(row)
    return header, rows


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_url(value: str, field: str, row_id: str, errors: list[str], required: bool = False) -> None:
    if not value:
        require(not required, f"{row_id}: missing required URL in {field}", errors)
        return
    require(bool(URL_RE.match(value)), f"{row_id}: {field} is not an http(s) URL: {value}", errors)


def check_date(value: str, field: str, row_id: str, errors: list[str]) -> None:
    require(bool(DATE_RE.match(value)), f"{row_id}: {field} must be YYYY-MM-DD", errors)
    if DATE_RE.match(value):
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(f"{row_id}: invalid date in {field}: {value}")


def check_enum(value: str, allowed: Iterable[str], field: str, row_id: str, errors: list[str]) -> None:
    require(value in set(allowed), f"{row_id}: invalid {field}: {value}", errors)


def row_label(path: str, index: int, row: dict[str, str]) -> str:
    identifier = row.get("id") or row.get("framework_id") or row.get("name") or row.get("title") or f"row_{index}"
    return f"{path}:{identifier}"


def validate_paper_row(path: str, index: int, row: dict[str, str], errors: list[str]) -> None:
    label = row_label(path, index, row)

    check_enum(row["opd_strictness"], OPD_STRICTNESS, "opd_strictness", label, errors)
    check_enum(row["feedback_route"], FEEDBACK_ROUTE, "feedback_route", label, errors)
    check_enum(row["teacher_access_regime"], TEACHER_ACCESS_REGIME, "teacher_access_regime", label, errors)
    check_enum(row["loss_granularity"], LOSS_GRANULARITY, "loss_granularity", label, errors)
    check_enum(row["on_policy_strength"], ON_POLICY_STRENGTH, "on_policy_strength", label, errors)
    check_enum(row["teacher_kind"], TEACHER_KIND, "teacher_kind", label, errors)
    check_enum(row["rollout_source"], ROLLOUT_SOURCE, "rollout_source", label, errors)
    check_enum(row["rollout_freshness"], ROLLOUT_FRESHNESS, "rollout_freshness", label, errors)
    check_enum(row["supervision_signal"], SUPERVISION_SIGNAL, "supervision_signal", label, errors)
    check_enum(row["confidence"], CONFIDENCE, "confidence", label, errors)
    check_enum(row["verification_status"], VERIFICATION_STATUS, "verification_status", label, errors)
    check_enum(row["conflict_status"], CONFLICT_STATUS, "conflict_status", label, errors)
    check_enum(row["uses_rlvr"], YES_NO_UNCLEAR, "uses_rlvr", label, errors)
    check_url(row["paper_url"], "paper_url", label, errors, required=True)
    check_url(row["code_url"], "code_url", label, errors)
    check_url(row["project_url"], "project_url", label, errors)
    check_url(row["evidence_url"], "evidence_url", label, errors, required=True)
    check_date(row["last_checked"], "last_checked", label, errors)

    rationale = row["classification_rationale"].lower()
    strictness_evidence = row["strictness_evidence"].lower()
    require(bool(rationale), f"{label}: classification_rationale is required", errors)

    if row["opd_strictness"] in {"strict_opd", "borderline_strict"}:
        require(
            all(marker in row["strictness_evidence"] for marker in ("C1", "C2", "C3")),
            f"{label}: strict/borderline rows require C1/C2/C3 strictness_evidence",
            errors,
        )

    if row["opd_strictness"] == "strict_opd":
        require(
            row["rollout_source"] == "student_on_policy",
            f"{label}: strict_opd requires rollout_source=student_on_policy",
            errors,
        )
        require(
            row["rollout_freshness"] in {"current_policy", "per_iteration", "per_epoch"},
            f"{label}: strict_opd requires fresh current-policy rollout evidence",
            errors,
        )
        require(
            row["supervision_signal"] not in NON_DISTILLATION_SIGNALS,
            f"{label}: strict_opd cannot use {row['supervision_signal']} as supervision_signal",
            errors,
        )
        require(
            row["teacher_kind"] not in NON_TEACHER_KINDS,
            f"{label}: strict_opd requires a concrete teacher_kind or privileged/self teacher",
            errors,
        )
        require(
            row["on_policy_strength"] == "strong",
            f"{label}: strict_opd requires on_policy_strength=strong",
            errors,
        )
        require(
            bool(row["divergence_or_objective"]),
            f"{label}: strict_opd requires divergence_or_objective",
            errors,
        )
        require(
            any(keyword in rationale for keyword in RATIONALE_KEYWORDS),
            f"{label}: strict_opd rationale must mention teacher/discriminator/privileged/reference supervision",
            errors,
        )

    if row["opd_strictness"] in {"adjacent", "not_opd"}:
        require(
            "missing" in strictness_evidence,
            f"{label}: adjacent/not_opd rows must state the missing OPD condition in strictness_evidence",
            errors,
        )

    if row["opd_strictness"] == "adjacent":
        require(
            row["supervision_signal"] in {"reward_only", "preference_only", "synthetic_data", "unclear", "none"},
            f"{label}: adjacent rows should not look like direct distillation unless marked partial/borderline",
            errors,
        )


def validate_framework_row(path: str, index: int, row: dict[str, str], errors: list[str]) -> None:
    label = row_label(path, index, row)
    for field in ("repo_url", "docs_url", "evidence_url"):
        check_url(row[field], field, label, errors, required=field == "evidence_url")
    for field in ("strict_opd_support", "vlm_support", "video_support", "teacher_logits", "black_box_teacher", "rlvr_support"):
        check_enum(row[field], YES_NO_UNCLEAR, field, label, errors)
    check_enum(row["confidence"], CONFIDENCE, "confidence", label, errors)
    check_date(row["last_checked"], "last_checked", label, errors)


def validate_benchmark_row(path: str, index: int, row: dict[str, str], errors: list[str]) -> None:
    label = row_label(path, index, row)
    for field in ("official_url", "repo_url", "dataset_url", "leaderboard_url"):
        check_url(row[field], field, label, errors)
    for field in ("private_test", "vlmevalkit_supported", "opencompass_supported", "common_for_vlm_rlvr", "common_for_vlm_opd"):
        check_enum(row[field], YES_NO_UNCLEAR, field, label, errors)


def validate_industrial_row(path: str, index: int, row: dict[str, str], errors: list[str]) -> None:
    label = row_label(path, index, row)
    check_url(row["report_url"], "report_url", label, errors, required=True)
    check_url(row["evidence_url"], "evidence_url", label, errors, required=True)
    check_enum(row["uses_strict_opd"], YES_NO_UNCLEAR, "uses_strict_opd", label, errors)
    check_enum(row["opd_strictness"], OPD_STRICTNESS, "opd_strictness", label, errors)
    check_enum(row["confidence"], CONFIDENCE, "confidence", label, errors)
    check_date(row["last_checked"], "last_checked", label, errors)


def validate_adjacent_row(path: str, index: int, row: dict[str, str], errors: list[str]) -> None:
    label = row_label(path, index, row)
    check_url(row["link"], "link", label, errors, required=True)
    check_url(row["evidence_url"], "evidence_url", label, errors, required=True)
    check_enum(row["confidence"], CONFIDENCE, "confidence", label, errors)
    check_date(row["last_checked"], "last_checked", label, errors)
    require(bool(row["why_not_strict_opd"]), f"{label}: why_not_strict_opd is required", errors)


def main() -> int:
    errors: list[str] = []

    for relative_path, expected_header in EXPECTED_HEADERS.items():
        path = ROOT / relative_path
        require(path.exists(), f"missing required table: {relative_path}", errors)
        if not path.exists():
            continue

        header, rows = read_rows(path)
        require(header == expected_header, f"{relative_path}: header mismatch\nexpected: {expected_header}\nactual:   {header}", errors)

        for index, row in enumerate(rows, start=2):
            if "__column_count_error" in row:
                errors.append(f"{row_label(relative_path, index, row)}: {row['__column_count_error']}")
                continue
            if relative_path in {"tables/opd_papers.md", "tables/vlm_opd_papers.md"}:
                validate_paper_row(relative_path, index, row, errors)
            elif relative_path == "tables/frameworks.md":
                validate_framework_row(relative_path, index, row, errors)
            elif relative_path == "tables/benchmarks.md":
                validate_benchmark_row(relative_path, index, row, errors)
            elif relative_path == "tables/industrial_reports.md":
                validate_industrial_row(relative_path, index, row, errors)
            elif relative_path == "tables/adjacent_work.md":
                validate_adjacent_row(relative_path, index, row, errors)

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
