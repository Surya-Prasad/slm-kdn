from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from juniper_docs_store import JuniperDoc, retrieve_doc


@dataclass(frozen=True)
class TemplateRecord:
    action: str
    target_type: str
    template: str
    requires_commit: bool
    default_params: Dict[str, object] = field(default_factory=dict)


_TEMPLATE_STORE: Dict[tuple[str, str], TemplateRecord] = {
    ("set", "interface"): TemplateRecord(
        action="set",
        target_type="interface",
        template="set interfaces {target} unit {unit} family ethernet-switching vlan members {vlan_id}",
        requires_commit=True,
        default_params={"unit": 0, "vlan_id": None, "target": ""},
    ),
    ("delete", "interface"): TemplateRecord(
        action="delete",
        target_type="interface",
        template="delete interfaces {target} unit {unit} family ethernet-switching vlan members {vlan_id}",
        requires_commit=True,
        default_params={"unit": 0, "vlan_id": None, "target": ""},
    ),
    ("show", "interface"): TemplateRecord(
        action="show",
        target_type="interface",
        template="show interfaces {target} terse",
        requires_commit=False,
        default_params={"target": ""},
    ),
    ("set", "vlan"): TemplateRecord(
        action="set",
        target_type="vlan",
        template="set vlans {vlan_name} vlan-id {vlan_id}",
        requires_commit=True,
        default_params={"vlan_name": "", "vlan_id": None},
    ),
    ("delete", "vlan"): TemplateRecord(
        action="delete",
        target_type="vlan",
        template="delete vlans {vlan_name}",
        requires_commit=True,
        default_params={"vlan_name": ""},
    ),
    ("show", "vlan"): TemplateRecord(
        action="show",
        target_type="vlan",
        template="show vlans",
        requires_commit=False,
        default_params={},
    ),
    ("show", "route"): TemplateRecord(
        action="show",
        target_type="route",
        template="show route {prefix}",
        requires_commit=False,
        default_params={"prefix": ""},
    ),
}


def retrieve_template(action: str, target_type: str) -> Optional[TemplateRecord]:
    key = (str(action).strip().lower(), str(target_type).strip().lower())
    return _TEMPLATE_STORE.get(key)


def retrieve_template_with_doc(action: str, target_type: str) -> Tuple[Optional[TemplateRecord], Optional[JuniperDoc]]:
    """Retrieve deterministic CLI template and matching Juniper doc snippet datastore entry."""
    template_record = retrieve_template(action, target_type)
    doc_record = retrieve_doc(action, target_type)
    return template_record, doc_record
