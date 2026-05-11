from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class TemplateRecord:
    action: str
    domain: str
    sub_domain: str
    template: str
    requires_commit: bool
    default_params: Dict[str, object] = field(default_factory=dict)


_TEMPLATE_STORE: Dict[tuple[str, str, str], TemplateRecord] = {}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower()


def load_datastore(filepath: str = "data/juniper_templates.json") -> None:
    path = Path(filepath)
    payload = json.loads(path.read_text(encoding="utf-8"))

    records = payload.values() if isinstance(payload, dict) else payload

    store: Dict[tuple[str, str, str], TemplateRecord] = {}
    for entry in records:
        if not isinstance(entry, dict):
            continue

        action = _normalize_key(entry.get("action", ""))
        domain = _normalize_key(entry.get("domain", ""))
        sub_domain = _normalize_key(entry.get("sub_domain", ""))

        if not action or not domain or not sub_domain:
            continue

        record = TemplateRecord(
            action=action,
            domain=domain,
            sub_domain=sub_domain,
            template=str(entry.get("template", "")),
            requires_commit=bool(entry.get("requires_commit", False)),
            default_params=dict(entry.get("default_params", {})),
        )
        store[(action, domain, sub_domain)] = record

    _TEMPLATE_STORE.clear()
    _TEMPLATE_STORE.update(store)


def retrieve_template(action: str, domain: str, sub_domain: str) -> Optional[TemplateRecord]:
    if not _TEMPLATE_STORE:
        load_datastore()

    key = (_normalize_key(action), _normalize_key(domain), _normalize_key(sub_domain))
    return _TEMPLATE_STORE.get(key)
