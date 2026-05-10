"""Local Juniper documentation datastore for low-latency RAG retrieval.

This module stores concise, curated documentation snippets and source links.
Retrieval remains fully local/deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class JuniperDoc:
    doc_id: str
    action: str
    target_type: str
    title: str
    snippet: str
    source_url: str


_DOCS: Dict[tuple[str, str], JuniperDoc] = {
    ("set", "interface"): JuniperDoc(
        doc_id="jnpr_if_vlan_set",
        action="set",
        target_type="interface",
        title="Configure VLAN membership under an interface unit",
        snippet=(
            "Use 'set interfaces <interface> unit <unit> family ethernet-switching vlan members <vlan-id>' "
            "to assign an interface logical unit to VLAN membership."
        ),
        source_url="https://www.juniper.net/documentation/",
    ),
    ("delete", "interface"): JuniperDoc(
        doc_id="jnpr_if_vlan_delete",
        action="delete",
        target_type="interface",
        title="Remove VLAN membership from an interface unit",
        snippet=(
            "Use 'delete interfaces <interface> unit <unit> family ethernet-switching vlan members <vlan-id>' "
            "to remove interface VLAN membership."
        ),
        source_url="https://www.juniper.net/documentation/",
    ),
    ("set", "vlan"): JuniperDoc(
        doc_id="jnpr_vlan_create",
        action="set",
        target_type="vlan",
        title="Create VLAN and assign VLAN ID",
        snippet="Use 'set vlans <name> vlan-id <id>' to define a VLAN in configuration mode.",
        source_url="https://www.juniper.net/documentation/",
    ),
    ("show", "interface"): JuniperDoc(
        doc_id="jnpr_show_interface_terse",
        action="show",
        target_type="interface",
        title="Operational interface status",
        snippet="Use 'show interfaces <interface> terse' to view concise operational interface state.",
        source_url="https://www.juniper.net/documentation/",
    ),
    ("show", "route"): JuniperDoc(
        doc_id="jnpr_show_route",
        action="show",
        target_type="route",
        title="Display route information",
        snippet="Use 'show route <prefix>' to inspect routing table entries for a destination prefix.",
        source_url="https://www.juniper.net/documentation/",
    ),
}


def retrieve_doc(action: str, target_type: str) -> Optional[JuniperDoc]:
    key = (str(action).strip().lower(), str(target_type).strip().lower())
    return _DOCS.get(key)
