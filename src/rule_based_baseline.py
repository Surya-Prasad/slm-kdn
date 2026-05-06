import re

def predict(intent: str, context: str = "") -> str:
    s = f"{context} {intent}".lower()
    iface = re.search(r"(ge-\d+/\d+/\d+|eth\d+|port\s+\d+)", s)
    iface = iface.group(1) if iface else "interface ge-0/0/0"
    if "dhcp" in s and "trust" in s:
        return f"set interfaces {iface} dhcp-trusted"
    if "shutdown" in s or "disable" in s:
        return f"set interfaces {iface} disable"
    if "enable" in s or "no shutdown" in s:
        return f"delete interfaces {iface} disable"
    vlan = re.search(r"vlan\s*(\d+)", s)
    if vlan:
        return f"set interfaces {iface} unit 0 family ethernet-switching vlan members {vlan.group(1)}"
    return f"set interfaces {iface} description configured-by-rule-baseline"
