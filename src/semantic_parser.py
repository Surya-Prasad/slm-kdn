import json
import re
from typing import Any, Dict, Optional, Tuple


REQUIRED_JSON_KEYS = {"action", "domain", "sub_domain", "parameters"}
ALLOWED_ACTIONS = {"set", "delete", "show", "clear", "request", "ping", "traceroute", "monitor", "load"}
ACTION_SYNONYMS = {
    "display": "show",
    "get": "show",
    "remove": "delete",
    "configure": "set",
    "enable": "set",
    "create": "set",
    "make": "set",
    "put": "set",
    "block": "set",
    "notify": "set",
}
DOMAIN_SYNONYMS = {
    "protocol": "protocols",
    "interface": "interfaces",
    "ethernet switching": "ethernet-switching",
    "ethernet_switching": "ethernet-switching",
    "chassis lcd": "chassis",
}


def semantic_prompt(intent: str, context: str = "") -> str:
    return (
        "You are a semantic intent parser for Juniper Junos intents.\n"
        "Return only strict JSON with exactly these top-level keys:\n"
        '{"action":"...","domain":"...","sub_domain":"...","parameters":{...}}\n'
        "Do not return a CLI command. Do not include markdown or explanation.\n\n"
        f"Intent: {intent}\n"
        f"Context: {context}\n"
        "JSON:"
    )


def _extract_json_object(raw: str) -> Dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no_json_object_found")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json_not_object")
    return parsed


def _normalize_domain(domain: str, sub_domain: str) -> tuple[str, str]:
    domain_norm = str(domain or "").strip().lower().replace("_", "-")
    sub_norm = str(sub_domain or "").strip().lower().replace("_", "-")
    if domain_norm in {"chassis lcd", "chassis-lcd", "chassis/lcd"}:
        return "chassis", sub_norm or "lcd"
    domain_norm = DOMAIN_SYNONYMS.get(domain_norm, domain_norm)
    if domain_norm == "chassis" and sub_norm == "":
        return "chassis", sub_norm
    return domain_norm, sub_norm


def normalize_semantic_frame(parsed: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(parsed)
    action = str(normalized.get("action", "")).strip().lower().replace("_", "-")
    normalized["action"] = ACTION_SYNONYMS.get(action, action)

    domain, sub_domain = _normalize_domain(
        str(normalized.get("domain", "")),
        str(normalized.get("sub_domain", "")),
    )
    normalized["domain"] = domain
    normalized["sub_domain"] = sub_domain

    params = normalized.get("parameters")
    if not isinstance(params, dict):
        params = {}
    params = dict(params)
    for key in ("vlan_id", "unit", "limit"):
        if key in params and params[key] is not None:
            text = str(params[key]).strip()
            if re.fullmatch(r"\d+", text):
                params[key] = int(text)
    normalized["parameters"] = params
    return normalized


def parse_semantic_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        parsed = _extract_json_object(raw)
        missing = REQUIRED_JSON_KEYS - set(parsed.keys())
        if missing:
            raise ValueError("missing_keys:" + ",".join(sorted(missing)))
        if not isinstance(parsed.get("parameters"), dict):
            raise ValueError("parameters_not_object")
        parsed = normalize_semantic_frame(parsed)
        if parsed.get("action") not in ALLOWED_ACTIONS:
            raise ValueError("invalid_enum_action")
        for key in REQUIRED_JSON_KEYS:
            if key != "parameters" and not str(parsed.get(key, "")).strip():
                raise ValueError(f"empty_required_key:{key}")
        return parsed, None
    except Exception as exc:
        return None, str(exc)
