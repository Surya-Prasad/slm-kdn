from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TemplateRecord:
    action: str
    target_type: str
    template: str
    requires_commit: bool


# Local deterministic store. Extendable without changing retrieval logic.
_TEMPLATE_STORE: Dict[tuple[str, str], TemplateRecord] = {
    ("set", "interface"): TemplateRecord(
        action="set",
        target_type="interface",
        template="set interfaces {target} unit {unit} family ethernet-switching vlan members {vlan_id}",
        requires_commit=True,
    ),
    ("delete", "interface"): TemplateRecord(
        action="delete",
        target_type="interface",
        template="delete interfaces {target} unit {unit} family ethernet-switching vlan members {vlan_id}",
        requires_commit=True,
    ),
    ("show", "interface"): TemplateRecord(
        action="show",
        target_type="interface",
        template="show interfaces {target} terse",
        requires_commit=False,
    ),
    ("set", "vlan"): TemplateRecord(
        action="set",
        target_type="vlan",
        template="set vlans {vlan_name} vlan-id {vlan_id}",
        requires_commit=True,
    ),
    ("delete", "vlan"): TemplateRecord(
        action="delete",
        target_type="vlan",
        template="delete vlans {vlan_name}",
        requires_commit=True,
    ),
    ("show", "vlan"): TemplateRecord(
        action="show",
        target_type="vlan",
        template="show vlans",
        requires_commit=False,
    ),
    ("show", "route"): TemplateRecord(
        action="show",
        target_type="route",
        template="show route {prefix}",
        requires_commit=False,
    ),
}


def retrieve_template(action: str, target_type: str) -> Optional[TemplateRecord]:
    """Retrieve template by normalized action/target key."""
    key = (str(action).strip().lower(), str(target_type).strip().lower())
    return _TEMPLATE_STORE.get(key)
