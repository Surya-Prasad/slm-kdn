from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TemplateRecord:
    action: str
    domain: str
    sub_domain: str
    template: str
    mode: str = "unknown"
    requires_commit: bool = False
    default_params: Dict[str, object] = field(default_factory=dict)
    allowed_params: List[str] = field(default_factory=list)
    description: str = ""
    intent_examples: List[str] = field(default_factory=list)
    negative_rules: List[str] = field(default_factory=list)
    validation_rules: Dict[str, object] = field(default_factory=dict)
    variant: str = "plain"

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "template": self.template,
            "mode": self.mode,
            "requires_commit": self.requires_commit,
            "default_params": dict(self.default_params),
            "allowed_params": list(self.allowed_params),
            "description": self.description,
            "intent_examples": list(self.intent_examples),
            "negative_rules": list(self.negative_rules),
            "validation_rules": dict(self.validation_rules),
            "variant": self.variant,
        }


_TEMPLATE_STORE: Dict[tuple[str, str, str, str], TemplateRecord] = {}

OPERATIONAL_ACTIONS = {"show", "clear", "request", "ping", "traceroute", "monitor"}
CONFIGURATION_ACTIONS = {"set", "delete", "load"}
VALID_MODES = {"operational", "configuration", "maintenance", "unknown"}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower()


def _infer_mode(action: str, explicit_mode: object = None) -> str:
    mode = _normalize_key(explicit_mode or "")
    if mode in VALID_MODES:
        return mode
    if action in OPERATIONAL_ACTIONS:
        return "operational"
    if action in CONFIGURATION_ACTIONS:
        return "configuration"
    return "unknown"


def _infer_requires_commit(action: str, mode: str, explicit_value: object = None) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    if action in OPERATIONAL_ACTIONS or mode == "operational":
        return False
    return action in CONFIGURATION_ACTIONS and mode == "configuration"


def load_datastore(filepath: str = "data/juniper_templates.json") -> None:
    path = Path(filepath)
    if not path.exists():
        _TEMPLATE_STORE.clear()
        return
    payload = json.loads(path.read_text(encoding="utf-8"))

    records = payload.items() if isinstance(payload, dict) else [(None, item) for item in payload]

    store: Dict[tuple[str, str, str, str], TemplateRecord] = {}
    for raw_key, entry in records:
        if not isinstance(entry, dict):
            continue

        key_parts = []
        if raw_key:
            key_parts = [_normalize_key(part) for part in str(raw_key).split("/") if part]

        action = _normalize_key(entry.get("action", "")) or (key_parts[0] if len(key_parts) > 0 else "")
        domain = _normalize_key(entry.get("domain", "")) or (key_parts[1] if len(key_parts) > 1 else "")
        sub_domain = _normalize_key(entry.get("sub_domain", "")) or (key_parts[2] if len(key_parts) > 2 else "")

        if not action or not domain or not sub_domain:
            continue

        mode = _infer_mode(action, entry.get("mode"))
        record = TemplateRecord(
            action=action,
            domain=domain,
            sub_domain=sub_domain,
            template=str(entry.get("template", "")),
            mode=mode,
            requires_commit=_infer_requires_commit(action, mode, entry.get("requires_commit")),
            default_params=dict(entry.get("default_params", {})),
            allowed_params=list(entry.get("allowed_params", [])),
            description=str(entry.get("description", "")),
            intent_examples=list(entry.get("intent_examples", [])),
            negative_rules=list(entry.get("negative_rules", [])),
            validation_rules=dict(entry.get("validation_rules", {})),
            variant=str(entry.get("variant", "plain") or "plain"),
        )
        store[(action, domain, sub_domain, record.variant)] = record

    _TEMPLATE_STORE.clear()
    _TEMPLATE_STORE.update(store)


def retrieve_template(action: str, domain: str, sub_domain: str) -> Optional[TemplateRecord]:
    if not _TEMPLATE_STORE:
        load_datastore()

    key = (_normalize_key(action), _normalize_key(domain), _normalize_key(sub_domain), "plain")
    if key in _TEMPLATE_STORE:
        return _TEMPLATE_STORE[key]
    matches = [
        record for (a, d, s, _), record in _TEMPLATE_STORE.items()
        if (a, d, s) == key[:3]
    ]
    return matches[0] if len(matches) == 1 else None


def all_templates() -> List[TemplateRecord]:
    if not _TEMPLATE_STORE:
        load_datastore()
    return list(_TEMPLATE_STORE.values())


def get_command_context(action: str, domain: str, sub_domain: str) -> Dict[str, object]:
    record = retrieve_template(action, domain, sub_domain)
    if record is None:
        return {
            "found": False,
            "reason": "template_not_found",
            "action": _normalize_key(action),
            "domain": _normalize_key(domain),
            "sub_domain": _normalize_key(sub_domain),
        }
    payload = record.to_dict()
    payload["found"] = True
    payload["template_key"] = "/".join((record.action, record.domain, record.sub_domain))
    return payload
