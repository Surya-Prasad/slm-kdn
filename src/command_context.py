from typing import Dict, List

from semantic_parser import infer_operation
from rag_store import OPERATIONAL_ACTIONS, all_templates


def _key(action: str, domain: str, sub_domain: str) -> str:
    return "/".join((str(action or "").lower(), str(domain or "").lower(), str(sub_domain or "").lower()))


def _record_context(record, reason: str = "") -> Dict[str, object]:
    payload = record.to_dict()
    payload.update(
        {
            "found": True,
            "reason": reason,
            "template_key": _key(record.action, record.domain, record.sub_domain),
            "template_variant_key": "/".join((record.action, record.domain, record.sub_domain, record.operation, record.variant)),
            "warnings": [],
        }
    )
    return payload


def _error_context(parsed_json: dict, reason: str, warnings: List[str] | None = None) -> Dict[str, object]:
    return {
        "found": False,
        "reason": reason,
        "template_key": "",
        "action": str(parsed_json.get("action", "")).lower(),
        "domain": str(parsed_json.get("domain", "")).lower(),
        "sub_domain": str(parsed_json.get("sub_domain", "")).lower(),
        "template": "",
        "mode": "unknown",
        "requires_commit": False,
        "default_params": {},
        "allowed_params": [],
        "description": "",
        "intent_examples": [],
        "negative_rules": [],
        "validation_rules": {},
        "operation": str(parsed_json.get("operation", "")).lower(),
        "variant": "plain",
        "positive_cues": [],
        "negative_cues": [],
        "required_params": [],
        "forbidden_params": [],
        "warnings": warnings or [],
    }


def _action_compatible(parsed_action: str, candidate_action: str) -> bool:
    if parsed_action == candidate_action:
        return True
    if parsed_action in OPERATIONAL_ACTIONS and candidate_action in OPERATIONAL_ACTIONS:
        return parsed_action == candidate_action
    return False


def _score_candidate(record, parsed_json: dict, desired_variant: str, intent_context: str) -> int:
    params = parsed_json.get("parameters") if isinstance(parsed_json.get("parameters"), dict) else {}
    operation = str(parsed_json.get("operation", "")).strip().lower()
    score = 0
    if record.operation == operation:
        score += 3
    if record.variant == desired_variant:
        score += 1
    for cue in record.positive_cues:
        if str(cue).lower() in intent_context:
            score += 2
    if record.required_params and all(param in params and params[param] not in (None, "") for param in record.required_params):
        score += 2
    for cue in record.negative_cues:
        if str(cue).lower() in intent_context:
            score -= 4
    if "disable" in record.template.lower() and not any(term in intent_context for term in ("disable", "deactivate", "turn off", "stop")):
        score -= 5
    missing = [param for param in record.required_params if param not in params or params[param] in (None, "")]
    if missing:
        score -= 5
    return score


def _select_candidate(candidates, parsed_json: dict, desired_variant: str, intent_context: str):
    if not candidates:
        return None, "template_not_found"
    scored = sorted(
        ((_score_candidate(candidate, parsed_json, desired_variant, intent_context), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        # Prefer exact operation if the score tie is otherwise ambiguous.
        operation = str(parsed_json.get("operation", "")).lower()
        exact = [candidate for _, candidate in scored if candidate.operation == operation]
        if len(exact) == 1:
            return exact[0], "selected_by_operation_tie_break"
    return scored[0][1], "selected_by_operation_score"


def retrieve_command_context(parsed_json: dict) -> Dict[str, object]:
    try:
        action = str(parsed_json.get("action", "")).strip().lower()
        domain = str(parsed_json.get("domain", "")).strip().lower()
        sub_domain = str(parsed_json.get("sub_domain", "")).strip().lower()
        intent_context = str(parsed_json.get("_intent_context", "")).lower()
        desired_variant = "display_set" if ("display set" in intent_context or "set format" in intent_context) else "plain"
        if not str(parsed_json.get("operation", "")).strip():
            parsed_json["operation"] = infer_operation(
                " ".join((intent_context, action, domain, sub_domain, " ".join(str(v) for v in dict(parsed_json.get("parameters", {})).values()))),
                action,
                domain,
                sub_domain,
            )
            parsed_json["_operation_inferred"] = True

        exact_candidates = [
            item for item in all_templates()
            if item.action == action and item.domain == domain and item.sub_domain == sub_domain
        ]
        if exact_candidates:
            variant_matches = [item for item in exact_candidates if item.variant == desired_variant] or exact_candidates
            record, reason = _select_candidate(variant_matches, parsed_json, desired_variant, intent_context)
            return _record_context(record, reason=reason)

        candidates = [
            item for item in all_templates()
            if item.action == action and item.domain == domain
        ]
        if len(candidates) == 1:
            return _record_context(candidates[0], reason="fallback_same_action_domain")
        if candidates:
            record, reason = _select_candidate(candidates, parsed_json, desired_variant, intent_context)
            return _record_context(record, reason=f"fallback_same_action_domain_{reason}")

        candidates = [
            item for item in all_templates()
            if item.domain == domain
            and item.sub_domain == sub_domain
            and _action_compatible(action, item.action)
        ]
        if len(candidates) == 1:
            return _record_context(candidates[0], reason="fallback_same_domain_sub_domain")
        if candidates:
            record, reason = _select_candidate(candidates, parsed_json, desired_variant, intent_context)
            return _record_context(record, reason=f"fallback_same_domain_sub_domain_{reason}")
        if len(candidates) > 1:
            return _error_context(parsed_json, "ambiguous_template")
        return _error_context(parsed_json, "template_not_found")
    except Exception as exc:
        return _error_context(parsed_json, "context_lookup_error", [str(exc)])


def validate_context(parsed_json: dict, context: dict) -> List[str]:
    warnings = list(context.get("warnings") or [])
    action = str(parsed_json.get("action", "")).lower()
    mode = str(context.get("mode", "unknown")).lower()
    if action in {"set", "delete", "load"} and mode == "operational":
        warnings.append("inconsistent_action_mode")
    if action in OPERATIONAL_ACTIONS and bool(context.get("requires_commit", False)):
        warnings.append("commit_suppressed_for_operational_action")
        context["requires_commit"] = False
    context["warnings"] = warnings
    return warnings


def should_commit(parsed_json: dict, context: dict) -> bool:
    action = str(parsed_json.get("action", "")).lower()
    mode = str(context.get("mode", "unknown")).lower()
    if mode == "operational" or action in OPERATIONAL_ACTIONS:
        return False
    return mode == "configuration" and bool(context.get("requires_commit", False))
