import json
import re
from typing import Any, Dict, Optional, Tuple

from parameter_binding import extract_params_from_text


REQUIRED_JSON_KEYS = {"action", "domain", "sub_domain", "parameters"}
ALLOWED_ACTIONS = {"set", "delete", "show", "clear", "request", "ping", "traceroute", "monitor", "load", "start", "run", "commit"}
COMMAND_ACTION_TOKENS = ALLOWED_ACTIONS
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
KNOWN_DOMAINS = {
    "protocols",
    "interfaces",
    "system",
    "snmp",
    "chassis",
    "virtual-chassis",
    "ethernet-switching-options",
    "ethernet-switching",
    "vlans",
    "poe",
    "configuration",
    "lldp",
    "ospf",
    "dhcp",
    "cli",
    "factory-default",
}
KNOWN_MULTI_TOKEN_DOMAINS = {"ethernet switching"}


def infer_operation(text: str, action: str = "", domain: str = "", sub_domain: str = "") -> str:
    q = str(text or "").lower().replace("_", "-")
    q = re.sub(r"\s+", " ", q)
    if "traceoptions" in q and "flag" in q:
        if re.search(r"\b(disable|deactivate|turn off|stop)\b", q):
            return "traceoptions_flag_disable"
        return "traceoptions_flag_enable"
    if "sflow" in q and re.search(r"\binterface\s+[a-z]{2}-\d+/\d+/\d+\b", q):
        return "interface_enable"
    if "sample-rate" in q and "ingress" in q:
        return "sample_rate_ingress"
    if "sample-rate" in q and "egress" in q:
        return "sample_rate_egress"
    if "polling interval" in q or "polling-interval" in q:
        return "polling_interval"
    if "mac moving limit" in q or "mac-move-limit" in q:
        return "mac_move_limit"
    if "mac limit" in q and "action log" in q:
        return "mac_limit_action_log"
    if "arp inspection" in q:
        return "arp_inspection"
    if "dhcp trusted" in q or "trusted dhcp" in q:
        return "dhcp_trusted"
    if "no-examine-dhcp" in q:
        return "no_examine_dhcp"
    if "lcd" in q and ("menu" in q or "active menu" in q):
        return "lcd_menu"
    if "clear ethernet-switching-table" in q or ("clear" in q and "ethernet-switching-table" in q):
        return "clear_table"
    if "disable" in q:
        return "disable"
    if "enable" in q:
        return "enable"
    if action and domain and sub_domain:
        return "_".join(part for part in (action, domain, sub_domain) if part).replace("-", "_").replace("/", "_")
    return "general"


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


def _strip_pre_json_noise(raw: str) -> tuple[str, list[str]]:
    text = str(raw or "").strip()
    warnings = []
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(Command:|Output:)\s*", "", text, flags=re.I)
    start = text.find("{")
    end = text.rfind("}")
    if end != -1 and text[end + 1 :].strip():
        warnings.append("stripped_trailing_text_after_json")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text, warnings


def _extract_json_object(raw: str) -> tuple[Dict[str, Any], list[str]]:
    cleaned, warnings = _strip_pre_json_noise(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no_json_object_found")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json_not_object")
    return parsed, warnings


def _normalize_domain(domain: str, sub_domain: str) -> tuple[str, str]:
    domain_norm = str(domain or "").strip().lower().replace("_", "-")
    sub_norm = str(sub_domain or "").strip().lower().replace("_", "-")
    if domain_norm in {"chassis lcd", "chassis-lcd", "chassis/lcd"}:
        return "chassis", sub_norm or "lcd"
    domain_norm = DOMAIN_SYNONYMS.get(domain_norm, domain_norm)
    if domain_norm == "chassis" and sub_norm == "":
        return "chassis", sub_norm
    return domain_norm, sub_norm


def _infer_domain_subdomain_from_tokens(tokens: list[str]) -> tuple[str, str]:
    action = tokens[0] if tokens else ""
    if action == "show":
        if tokens[:4] == ["show", "configuration", "protocols", "sflow"]:
            return "configuration", "protocols"
        return (tokens[1], tokens[2] if len(tokens) > 2 else "general") if len(tokens) > 1 else ("unknown", "general")
    if action in {"set", "delete"}:
        if tokens[:3] == ["set", "ethernet-switching-options", "secure-access-port"]:
            return "ethernet-switching-options", "secure-access-port"
        if tokens[:3] == ["set", "virtual-chassis", "traceoptions"]:
            return "virtual-chassis", "traceoptions"
        if len(tokens) > 2 and tokens[1] == "protocols":
            return "protocols", tokens[2]
        return (tokens[1], tokens[2] if len(tokens) > 2 else "general") if len(tokens) > 1 else ("unknown", "general")
    if action == "clear":
        if len(tokens) > 1 and tokens[1] == "ethernet-switching-table":
            return "ethernet-switching", "table"
        return (tokens[1], tokens[2] if len(tokens) > 2 else "general") if len(tokens) > 1 else ("unknown", "general")
    return (tokens[1], tokens[2] if len(tokens) > 2 else "general") if len(tokens) > 1 else ("unknown", "general")


def _extract_command_parameters(command: str) -> Dict[str, Any]:
    params: Dict[str, Any] = extract_params_from_text(command)
    text = str(command or "")
    interface = re.search(r"\b[a-z]{2}-\d+/\d+/\d+\b", text)
    if interface:
        params["interface"] = interface.group(0)
    ip_address = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    if ip_address:
        params["ip_address"] = ip_address.group(0)
    unit = re.search(r"\bunit\s+(\d+)\b", text, flags=re.I)
    if unit:
        params["unit"] = int(unit.group(1))
    vlan = re.search(r"\bvlan\s+([A-Za-z0-9_-]+)\b", text, flags=re.I)
    if vlan:
        value = vlan.group(1)
        if value.isdigit():
            params["vlan_id"] = int(value)
        else:
            params["vlan_name"] = value
    limit = re.search(r"\b(?:limit|mac-move-limit)\s+(\d+)\b", text, flags=re.I)
    if limit:
        params["limit"] = int(limit.group(1))
    flag = re.search(r"\bflag\s+([A-Za-z0-9_-]+)\b", text, flags=re.I)
    if flag:
        params["flag"] = flag.group(1)
    show_config_protocol = re.search(r"\bshow\s+configuration\s+protocols\s+([A-Za-z0-9_-]+)\b", text, flags=re.I)
    if show_config_protocol:
        params["protocol"] = show_config_protocol.group(1)
    username = re.search(r"\buser\s+([A-Za-z0-9_*.-]+)\b", text, flags=re.I)
    if username:
        params["username"] = username.group(1)
    community = re.search(r"\bcommunity\s+([A-Za-z0-9_-]+)\b", text, flags=re.I)
    if community:
        params["community_name"] = community.group(1)
    return params


def _looks_like_full_command(value: str) -> bool:
    tokens = str(value or "").strip().lower().split()
    return bool(tokens and tokens[0] in COMMAND_ACTION_TOKENS and len(tokens) > 1)


def _untrusted_model_field(value: str, raw_action: str, action: str, is_domain: bool) -> bool:
    text = str(value or "").strip().lower()
    raw = str(raw_action or "").strip().lower()
    if not text:
        return True
    if text == action or text == raw:
        return True
    if text.startswith(action + " "):
        return True
    if raw and (raw in text or text in raw and len(text.split()) > 1):
        return True
    if is_domain and " " in text and text not in KNOWN_MULTI_TOKEN_DOMAINS:
        return True
    if is_domain and text not in KNOWN_DOMAINS and len(text.split()) > 1:
        return True
    if not is_domain and len(text.split()) > 4:
        return True
    return False


def _repair_full_command_action(parsed: Dict[str, Any], warnings: list[str]) -> Dict[str, Any]:
    raw_action = str(parsed.get("action", "")).strip()
    tokens = raw_action.lower().split()
    if not tokens or tokens[0] not in COMMAND_ACTION_TOKENS or tokens[0] in ACTION_SYNONYMS:
        return parsed
    current_action = ACTION_SYNONYMS.get(tokens[0], tokens[0])
    if raw_action.strip().lower() == current_action:
        return parsed

    repaired = dict(parsed)
    repaired["action"] = current_action
    warnings.append("repaired_full_command_action")
    inferred_domain, inferred_sub_domain = _infer_domain_subdomain_from_tokens(tokens)
    if (
        _untrusted_model_field(str(repaired.get("domain", "")), raw_action, current_action, is_domain=True)
        or str(repaired.get("domain", "")).strip().lower() != inferred_domain
    ):
        repaired["domain"] = inferred_domain
        warnings.append("inferred_domain_sub_domain")
    if (
        _untrusted_model_field(str(repaired.get("sub_domain", "")), raw_action, current_action, is_domain=False)
        or str(repaired.get("sub_domain", "")).strip().lower() != inferred_sub_domain
    ):
        repaired["sub_domain"] = inferred_sub_domain
        if "inferred_domain_sub_domain" not in warnings:
            warnings.append("inferred_domain_sub_domain")

    params = repaired.get("parameters") if isinstance(repaired.get("parameters"), dict) else {}
    command_params = _extract_command_parameters(raw_action)
    merged_params = dict(command_params)
    merged_params.update(params)
    repaired["parameters"] = merged_params
    if not str(repaired.get("operation", "")).strip():
        repaired["operation"] = infer_operation(
            raw_action,
            current_action,
            str(repaired.get("domain", "")),
            str(repaired.get("sub_domain", "")),
        )
        warnings.append("inferred_operation")
    return repaired


def command_to_semantic_frame(command: str) -> Dict[str, Any]:
    body = str(command or "").replace("\\n", "\n").strip()
    body = re.sub(r"\n\s*commit\s*$", "", body, flags=re.I).strip()
    body = re.sub(r"\s+", " ", body)
    tokens = body.lower().split()
    if not tokens:
        return {"action": "", "domain": "", "sub_domain": "", "parameters": {}}
    action = ACTION_SYNONYMS.get(tokens[0], tokens[0])
    domain, sub_domain = _infer_domain_subdomain_from_tokens(tokens)
    return normalize_semantic_frame(
        {
            "action": action,
            "domain": domain,
            "sub_domain": sub_domain,
            "parameters": _extract_command_parameters(body),
            "operation": infer_operation(body, action, domain, sub_domain),
        },
        [],
    )


def normalize_semantic_frame(parsed: Dict[str, Any], warnings: Optional[list[str]] = None) -> Dict[str, Any]:
    warnings = warnings if warnings is not None else []
    parsed = _repair_full_command_action(parsed, warnings)
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
    raw_operation = str(normalized.get("operation", "")).strip()
    operation = raw_operation.lower().replace("-", "_")
    if not operation:
        operation = infer_operation(
            " ".join(
                str(part)
                for part in (
                    normalized.get("action", ""),
                    normalized.get("domain", ""),
                    normalized.get("sub_domain", ""),
                    " ".join(str(v) for v in params.values()),
                )
            ),
            normalized.get("action", ""),
            normalized.get("domain", ""),
            normalized.get("sub_domain", ""),
        )
        warnings.append("inferred_operation")
    normalized["operation"] = operation
    if warnings:
        normalized["_parse_warnings"] = list(dict.fromkeys(warnings))
    return normalized


def parse_semantic_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        parsed, warnings = _extract_json_object(raw)
        missing = REQUIRED_JSON_KEYS - set(parsed.keys())
        if missing:
            raise ValueError("missing_keys:" + ",".join(sorted(missing)))
        if not isinstance(parsed.get("parameters"), dict):
            raise ValueError("parameters_not_object")
        parsed = normalize_semantic_frame(parsed, warnings)
        if parsed.get("action") not in ALLOWED_ACTIONS:
            raise ValueError("invalid_enum_action")
        for key in REQUIRED_JSON_KEYS:
            if key != "parameters" and not str(parsed.get(key, "")).strip():
                raise ValueError(f"empty_required_key:{key}")
        return parsed, None
    except Exception as exc:
        return None, str(exc)
