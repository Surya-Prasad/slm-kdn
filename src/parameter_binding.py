import re
from typing import Any, Dict, List, Tuple


ANGLE_PLACEHOLDER_RE = re.compile(r"<([^<>]+)>")
BRACE_PLACEHOLDER_RE = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)}")


def normalize_placeholder_name(name: str) -> str:
    key = str(name or "").strip().strip("<>").lower()
    key = key.replace("_", "-")
    aliases = {
        "interface-name": "interface",
        "interface": "interface",
        "vlan-name": "vlan_name",
        "vlan-id": "vlan_id",
        "vlan-id-or-name": "vlan_id_or_name",
        "vlan-name-or-id": "vlan_id_or_name",
        "limit": "limit",
        "rate": "rate",
        "unit": "unit",
        "unit-number": "unit",
        "ntp-server-address": "ip_address",
        "collector-ip": "ip_address",
        "udp-port": "udp_port",
        "area-number": "area",
        "interval": "interval",
        "max-age-interval": "interval",
        "minutes": "minutes",
        "description": "description",
    }
    return aliases.get(key, key.replace("-", "_"))


def required_params_from_template(template: str) -> List[str]:
    params = {normalize_placeholder_name(name) for name in ANGLE_PLACEHOLDER_RE.findall(str(template or ""))}
    params.update(BRACE_PLACEHOLDER_RE.findall(str(template or "")))
    return sorted(params)


def unresolved_placeholders(command: str) -> List[str]:
    return [normalize_placeholder_name(name) for name in ANGLE_PLACEHOLDER_RE.findall(str(command or ""))]


def extract_params_from_text(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    params: Dict[str, Any] = {}

    interface_range = re.search(
        r"\b([a-z]{2}-\d+/\d+/\d+)\s+(?:to|through|-)\s+([a-z]{2}-\d+/\d+/\d+)\b",
        raw,
        flags=re.I,
    )
    if interface_range:
        params["interface_range_start"] = interface_range.group(1)
        params["interface_range_end"] = interface_range.group(2)

    interface = re.search(r"\b(?:interface\s+)?([a-z]{2}-\d+/\d+/\d+)\b", raw, flags=re.I)
    if interface:
        params["interface"] = interface.group(1)

    ip_address = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw)
    if ip_address:
        params["ip_address"] = ip_address.group(0)

    vlan_id = re.search(r"\bvlan(?:\s+id)?\s+(?:of\s+)?(\d+)\b", raw, flags=re.I)
    if vlan_id:
        params["vlan_id"] = int(vlan_id.group(1))
    vlan_name = re.search(r"\bvlan\s+([A-Za-z][A-Za-z0-9_-]*)\b", raw, flags=re.I)
    if vlan_name:
        params["vlan_name"] = vlan_name.group(1)

    rate = re.search(r"\b(?:sampling\s+rate|sample-rate|rate)\s+(?:to\s+|of\s+)?(\d+)\b", raw, flags=re.I)
    if rate:
        params["rate"] = int(rate.group(1))

    limit = re.search(r"\b(?:mac\s+moving\s+limit|mac-move-limit|mac\s+limit|limit)\s+(?:of\s+|to\s+)?(\d+)\b", raw, flags=re.I)
    if limit:
        params["limit"] = int(limit.group(1))

    unit = re.search(r"\bunit\s+(\d+)\b", raw, flags=re.I)
    if unit:
        params["unit"] = int(unit.group(1))

    udp_port = re.search(r"\b(?:udp-port|udp\s+port|port\s+number)\s+(?:to\s+|of\s+)?(\d+)\b", raw, flags=re.I)
    if udp_port:
        params["udp_port"] = int(udp_port.group(1))

    area = re.search(r"\barea(?:\s+number)?\s+([0-9.]+)\b", raw, flags=re.I)
    if area:
        params["area"] = area.group(1)

    interval = re.search(r"\b(?:interval|max-age-interval)\s+(?:to\s+|of\s+)?(\d+)\b", raw, flags=re.I)
    if interval:
        params["interval"] = int(interval.group(1))

    minutes = re.search(r"\bin\s+(\d+)\s+minutes?\b|\bminutes?\s+(?:to\s+|of\s+)?(\d+)\b", raw, flags=re.I)
    if minutes:
        params["minutes"] = int(next(group for group in minutes.groups() if group))

    quoted = re.search(r"['\"]([^'\"]+)['\"]", raw)
    if quoted:
        params["description"] = quoted.group(1)
    elif "description" in raw.lower():
        desc = re.search(r"\bdescription\s+(?:to\s+|as\s+)?(.+)$", raw, flags=re.I)
        if desc:
            params["description"] = desc.group(1).strip()

    return params


def _value_for_placeholder(name: str, params: Dict[str, Any]) -> Tuple[bool, Any]:
    canonical = normalize_placeholder_name(name)
    if canonical == "vlan_id_or_name":
        if params.get("vlan_id") not in (None, ""):
            return True, params["vlan_id"]
        if params.get("vlan_name") not in (None, ""):
            return True, params["vlan_name"]
        return False, None
    if canonical in params and params[canonical] not in (None, ""):
        return True, params[canonical]
    return False, None


def bind_template(template: str, params: Dict[str, Any], source_text: str = "") -> Tuple[str, List[str]]:
    text = str(template or "")
    missing: List[str] = []

    def replace_angle(match: re.Match) -> str:
        raw_name = match.group(1)
        found, value = _value_for_placeholder(raw_name, params)
        if found:
            return str(value)
        placeholder = match.group(0)
        if placeholder.lower() in str(source_text or "").lower():
            return placeholder
        missing.append(normalize_placeholder_name(raw_name))
        return placeholder

    text = ANGLE_PLACEHOLDER_RE.sub(replace_angle, text)
    if missing:
        return text, sorted(set(missing))

    try:
        text = text.format(**params)
    except KeyError as exc:
        return text, [str(exc).strip("'")]

    remaining = unresolved_placeholders(text)
    if remaining:
        return text, sorted(set(remaining))
    return text, []
