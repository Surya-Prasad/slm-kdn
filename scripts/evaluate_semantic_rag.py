import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_perfect_datastore_v2 import command_variant, infer_domain_subdomain, split_commit  # noqa: E402
from parameter_binding import unresolved_placeholders  # noqa: E402
from semantic_parser import command_to_semantic_frame  # noqa: E402
from utils import read_jsonl, tokenize  # noqa: E402
from validate_output import extract_entities, validate  # noqa: E402

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover
    SmoothingFunction = None
    sentence_bleu = None


def token_f1(pred, gold):
    p, g = tokenize(pred), tokenize(gold)
    pc, gc = Counter(p), Counter(g)
    tp = sum((pc & gc).values())
    if tp == 0:
        return 0.0
    pr = tp / max(len(p), 1)
    rc = tp / max(len(g), 1)
    return 2 * pr * rc / (pr + rc)


def normalize_cli_command(command: str) -> str:
    text = str(command or "")
    text = text.replace("\\\\n", "\n").replace("\\n", "\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def command_has_commit(command):
    return bool(re.search(r"\ncommit\s*$|^commit$", normalize_cli_command(command), flags=re.I))


def expected_frame(row):
    body, requires_commit = split_commit(row.get("target_command", ""))
    action = body.split()[0].lower() if body else ""
    domain, sub_domain = infer_domain_subdomain(body)
    frame = command_to_semantic_frame(body)
    operation = str(frame.get("operation", "general"))
    variant = command_variant(body)
    return {
        "action": action,
        "domain": domain,
        "sub_domain": sub_domain,
        "operation": operation,
        "variant": variant,
        "template_key": "/".join((action, domain, sub_domain)),
        "template_variant_key": "/".join((action, domain, sub_domain, operation, variant)),
        "requires_commit": requires_commit,
    }


def parameter_scores(row):
    parsed = row.get("semantic_json") or {}
    params = parsed.get("parameters") if isinstance(parsed, dict) else {}
    if not isinstance(params, dict):
        return 0.0, 0.0, 0.0
    gold_text = f"{row.get('intent', '')} {row.get('context', '')} {row.get('target_command', '')}".lower()
    predicted = {str(v).lower() for v in params.values() if v not in (None, "")}
    if not predicted:
        return 1.0, 1.0, 1.0
    tp = sum(1 for value in predicted if value in gold_text)
    precision = tp / max(len(predicted), 1)
    recall = precision
    f1 = precision if precision == recall else 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def failure_stage(row, expected):
    context = row.get("command_context") or {}
    pred_norm = normalize_cli_command(row.get("prediction", ""))
    gold_norm = normalize_cli_command(row.get("target_command", ""))
    if row.get("semantic_parse_error"):
        return "semantic_parse_error"
    if context.get("reason") == "template_not_found":
        return "template_not_found"
    if context.get("reason") == "ambiguous_template":
        return "ambiguous_template"
    if row.get("template_key") and row.get("template_key") != expected["template_key"]:
        return "wrong_template"
    if str(row.get("assembly_error") or "").startswith("missing_parameter"):
        return "missing_parameter"
    if command_has_commit(row.get("prediction", "")) != expected["requires_commit"]:
        return "commit_error"
    if pred_norm != gold_norm:
        return "final_command_mismatch"
    return "ok"


def evaluate_rows(rows):
    counts = defaultdict(float)
    failure_rows = []
    stage_counts = Counter()
    grouped_failures = Counter()
    smooth = SmoothingFunction().method1 if SmoothingFunction else None

    for row in rows:
        expected = expected_frame(row)
        parsed = row.get("semantic_json")
        context = row.get("command_context") or {}
        pred = row.get("prediction", "")
        gold = row.get("target_command", "")
        stage = failure_stage(row, expected)
        stage_counts[stage] += 1
        if stage != "ok":
            grouped_failures[f"{expected['action']}/{expected['domain']}/{expected['sub_domain']}"] += 1
            failure_rows.append({**row, "failure_stage": stage, "expected_template_key": expected["template_key"]})

        json_valid = isinstance(parsed, dict) and not row.get("semantic_parse_error")
        counts["json_valid_rate"] += float(json_valid)
        parse_warnings = []
        if isinstance(parsed, dict):
            parse_warnings = list(parsed.get("_parse_warnings") or row.get("semantic_parse_warnings") or [])
        counts["repaired_parse_rate"] += float("repaired_full_command_action" in parse_warnings)
        counts["unrepaired_parse_error_rate"] += float(bool(row.get("semantic_parse_error")))
        counts["action_accuracy"] += float(json_valid and parsed.get("action") == expected["action"])
        counts["domain_accuracy"] += float(json_valid and parsed.get("domain") == expected["domain"])
        counts["sub_domain_accuracy"] += float(json_valid and parsed.get("sub_domain") == expected["sub_domain"])
        counts["semantic_frame_exact_match"] += float(
            json_valid
            and parsed.get("action") == expected["action"]
            and parsed.get("domain") == expected["domain"]
            and parsed.get("sub_domain") == expected["sub_domain"]
        )
        counts["operation_accuracy"] += float(json_valid and parsed.get("operation") == expected["operation"])
        counts["operation_inferred_rate"] += float(
            isinstance(parsed, dict)
            and ("inferred_operation" in parse_warnings or bool(parsed.get("_operation_inferred")))
        )
        pp, pr, pf = parameter_scores(row)
        counts["parameter_precision"] += pp
        counts["parameter_recall"] += pr
        counts["parameter_f1"] += pf
        entities = extract_entities(row.get("intent", ""))
        counts["entity_preservation"] += float(all(e in pred.lower() for e in entities)) if entities else 1.0

        counts["template_hit_rate"] += float(bool(context.get("found")))
        counts["correct_template_rate"] += float(row.get("template_key") == expected["template_key"])
        counts["family_hit_rate"] += float(row.get("template_key") == expected["template_key"])
        counts["variant_hit_rate"] += float(context.get("template_variant_key") == expected["template_variant_key"])
        counts["template_variant_accuracy"] += float(context.get("template_variant_key") == expected["template_variant_key"])
        counts["template_not_found_rate"] += float(context.get("reason") == "template_not_found")
        counts["ambiguous_template_rate"] += float(context.get("reason") == "ambiguous_template")

        counts["assembly_success_rate"] += float(not row.get("assembly_error"))
        counts["missing_parameter_rate"] += float(str(row.get("assembly_error") or "").startswith("missing_parameter"))
        counts["unresolved_placeholder_rate"] += float(bool(unresolved_placeholders(pred)))
        counts["commit_decision_accuracy"] += float(command_has_commit(pred) == expected["requires_commit"])
        counts["commit_false_positive_rate"] += float(command_has_commit(pred) and not expected["requires_commit"])
        counts["commit_false_negative_rate"] += float((not command_has_commit(pred)) and expected["requires_commit"])
        counts["guardrail_application_rate"] += float(bool(row.get("guardrails_applied")))

        counts["raw_exact_match"] += float(pred.strip() == gold.strip())
        counts["normalized_exact_match"] += float(normalize_cli_command(pred) == normalize_cli_command(gold))
        counts["exact_match"] += float(normalize_cli_command(pred) == normalize_cli_command(gold))
        counts["token_f1"] += token_f1(pred, gold)
        val = validate(row)
        counts["valid_rate"] += float(val["is_valid"])
        if sentence_bleu:
            counts["bleu"] += sentence_bleu([tokenize(gold)], tokenize(pred), smoothing_function=smooth)

    n = max(len(rows), 1)
    metrics = {key: value / n for key, value in counts.items()}
    metrics["invalid_output_rate"] = 1 - metrics.get("valid_rate", 0.0)
    metrics["failure_stage_counts"] = dict(stage_counts)
    metrics["exact_match_failures_by_action_domain_sub_domain"] = dict(grouped_failures)
    return metrics, failure_rows


def main(args):
    rows = read_jsonl(args.pred_file)
    metrics, failures = evaluate_rows(rows)
    out_metrics = Path(args.out_file)
    out_failures = Path(args.failures_file)
    out_summary = Path(args.summary_file)
    for path in (out_metrics, out_failures, out_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps({"overall": metrics}, indent=2), encoding="utf-8")
    with out_failures.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "total": len(rows),
        "failure_stage_counts": metrics.get("failure_stage_counts", {}),
        "parse_errors": metrics.get("failure_stage_counts", {}).get("semantic_parse_error", 0),
        "template_not_found": metrics.get("failure_stage_counts", {}).get("template_not_found", 0),
        "missing_parameter": metrics.get("failure_stage_counts", {}).get("missing_parameter", 0),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"overall": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic-RAG predictions by stage.")
    parser.add_argument("--pred_file", default="results/predictions/semantic_rag_predictions.jsonl")
    parser.add_argument("--out_file", default="results/metrics/semantic_rag_metrics.json")
    parser.add_argument("--failures_file", default="results/error_analysis/semantic_rag_failures.jsonl")
    parser.add_argument("--summary_file", default="results/error_analysis/semantic_rag_error_summary.json")
    main(parser.parse_args())
