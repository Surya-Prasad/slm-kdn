from typing import Dict, List

from rag_store import OPERATIONAL_ACTIONS, all_templates, retrieve_template


def _key(action: str, domain: str, sub_domain: str) -> str:
    return "/".join((str(action or "").lower(), str(domain or "").lower(), str(sub_domain or "").lower()))


def _record_context(record, reason: str = "") -> Dict[str, object]:
    payload = record.to_dict()
    payload.update(
        {
            "found": True,
            "reason": reason,
            "template_key": _key(record.action, record.domain, record.sub_domain),
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
        "warnings": warnings or [],
    }


def _action_compatible(parsed_action: str, candidate_action: str) -> bool:
    if parsed_action == candidate_action:
        return True
    if parsed_action in OPERATIONAL_ACTIONS and candidate_action in OPERATIONAL_ACTIONS:
        return parsed_action == candidate_action
    return False


def retrieve_command_context(parsed_json: dict) -> Dict[str, object]:
    try:
        action = str(parsed_json.get("action", "")).strip().lower()
        domain = str(parsed_json.get("domain", "")).strip().lower()
        sub_domain = str(parsed_json.get("sub_domain", "")).strip().lower()

        record = retrieve_template(action, domain, sub_domain)
        if record:
            return _record_context(record)

        candidates = [
            item for item in all_templates()
            if item.action == action and item.domain == domain
        ]
        if len(candidates) == 1:
            return _record_context(candidates[0], reason="fallback_same_action_domain")

        candidates = [
            item for item in all_templates()
            if item.domain == domain
            and item.sub_domain == sub_domain
            and _action_compatible(action, item.action)
        ]
        if len(candidates) == 1:
            return _record_context(candidates[0], reason="fallback_same_domain_sub_domain")
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
