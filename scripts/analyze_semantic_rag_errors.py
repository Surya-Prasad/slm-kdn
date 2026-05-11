import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_semantic_rag import expected_frame, failure_stage  # noqa: E402
from utils import read_jsonl  # noqa: E402

try:
    from validate_output import extract_entities
except Exception:  # pragma: no cover
    extract_entities = None


def main(args):
    rows = read_jsonl(args.pred_file)
    summary = Counter()
    grouped_failures = Counter()
    failures = []
    entity_warning = None

    if extract_entities is None:
        entity_warning = "extract_entities helper not available; entity preservation failures skipped"

    for row in rows:
        expected = expected_frame(row)
        stage = failure_stage(row, expected)
        context = row.get("command_context") or {}
        warnings = row.get("context_warnings") or []

        summary["parse_errors"] += int(bool(row.get("semantic_parse_error")))
        summary["template_not_found"] += int(context.get("reason") == "template_not_found")
        summary["missing_parameters"] += int(str(row.get("assembly_error") or "").startswith("missing_parameter"))
        summary["guardrail_modifications"] += int(bool(row.get("guardrails_applied")))
        summary["commit_suppressed_for_operational_action"] += int("commit_suppressed_for_operational_action" in warnings)

        if extract_entities is not None:
            entities = extract_entities(row.get("intent", ""))
            missing = [e for e in entities if e not in row.get("prediction", "").lower()]
            summary["entity_preservation_failures"] += int(bool(missing))

        if stage != "ok":
            key = f"{expected['action']}/{expected['domain']}/{expected['sub_domain']}"
            grouped_failures[key] += 1
            failures.append({**row, "failure_stage": stage, "expected_template_key": expected["template_key"]})

    payload = {
        "total": len(rows),
        **dict(summary),
        "exact_match_failures_by_action_domain_sub_domain": dict(grouped_failures),
    }
    if entity_warning:
        payload["warning"] = entity_warning

    summary_path = Path(args.summary_file)
    failures_path = Path(args.failures_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with failures_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze semantic-RAG prediction failures.")
    parser.add_argument("--pred_file", default="results/predictions/semantic_rag_predictions.jsonl")
    parser.add_argument("--summary_file", default="results/error_analysis/semantic_rag_error_summary.json")
    parser.add_argument("--failures_file", default="results/error_analysis/semantic_rag_failures.jsonl")
    main(parser.parse_args())
