import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_store import CONFIGURATION_ACTIONS, OPERATIONAL_ACTIONS  # noqa: E402
from semantic_parser import command_to_semantic_frame  # noqa: E402
from utils import read_jsonl  # noqa: E402


def split_commit(command: str) -> tuple[str, bool]:
    text = str(command or "").replace("\\n", "\n").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    requires_commit = False
    while lines and lines[-1].lower() == "commit":
        requires_commit = True
        lines.pop()
    return " ".join(lines).strip(), requires_commit


def infer_mode(action: str) -> str:
    if action in OPERATIONAL_ACTIONS:
        return "operational"
    if action in CONFIGURATION_ACTIONS:
        return "configuration"
    return "unknown"


def infer_requires_commit(action: str, mode: str, command_had_commit: bool) -> bool:
    if action in OPERATIONAL_ACTIONS or mode == "operational":
        return False
    return command_had_commit or (action in CONFIGURATION_ACTIONS and mode == "configuration")


def infer_domain_subdomain(body: str) -> tuple[str, str]:
    frame = command_to_semantic_frame(body)
    return str(frame.get("domain", "")), str(frame.get("sub_domain", ""))


def command_variant(body: str) -> str:
    return "display_set" if "| display set" in body.lower() else "plain"


def plain_body_for_variant(body: str) -> str:
    return re.sub(r"\s+\|\s+display\s+set\b.*$", "", body, flags=re.I).strip()


def valid_key_fields(action: str, domain: str, sub_domain: str, body: str) -> bool:
    body_norm = re.sub(r"\s+", " ", body.lower()).strip()
    if not action or not domain or not sub_domain:
        return False
    if domain == action:
        return False
    if domain in body_norm and len(domain.split()) > 1:
        return False
    if sub_domain in body_norm and len(sub_domain.split()) > 4:
        return False
    if domain.startswith(action + " ") or sub_domain.startswith(action + " "):
        return False
    return True


def operation_cues(operation: str) -> tuple[list[str], list[str]]:
    positive = {
        "traceoptions_flag_enable": ["trace", "traceoptions", "flag"],
        "traceoptions_flag_disable": ["disable", "traceoptions", "flag"],
        "interface_enable": ["interface"],
        "sample_rate_ingress": ["sample-rate", "ingress"],
        "sample_rate_egress": ["sample-rate", "egress"],
        "polling_interval": ["polling interval", "polling-interval"],
        "mac_move_limit": ["mac moving limit", "mac-move-limit"],
        "mac_limit_action_log": ["mac limit", "action log"],
        "arp_inspection": ["arp inspection"],
        "dhcp_trusted": ["dhcp trusted", "trusted dhcp"],
        "no_examine_dhcp": ["no-examine-dhcp"],
        "lcd_menu": ["lcd", "menu"],
        "clear_table": ["clear", "ethernet-switching-table"],
    }.get(operation, [operation.replace("_", " ")])
    negative = []
    if operation != "traceoptions_flag_disable":
        negative.append("disable")
    if operation != "no_examine_dhcp":
        negative.append("no-examine-dhcp")
    if operation != "mac_move_limit":
        negative.append("mac-move-limit")
    return positive, negative


def placeholders(template: str) -> list[str]:
    return sorted(set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", template)))


def parameterize(body: str) -> tuple[str, list[str]]:
    params = []
    template = body

    replacements = [
        (r"\b[a-z]{2}-\d+/\d+/\d+\b", "{interface}"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", "{ip_address}"),
    ]
    for pattern, repl in replacements:
        if re.search(pattern, template):
            params.append(repl.strip("{}"))
            template = re.sub(pattern, repl, template, count=1)

    if re.search(r"\bunit\s+\d+\b", template):
        params.append("unit")
        template = re.sub(r"\bunit\s+\d+\b", "unit {unit}", template, count=1)
    if re.search(r"\bvlan(?:-id)?\s+\d+\b", template):
        params.append("vlan_id")
        template = re.sub(r"\b(vlan(?:-id)?)\s+\d+\b", r"\1 {vlan_id}", template, count=1)
    if re.search(r"\bvlan\s+[A-Za-z][A-Za-z0-9_-]*\b", template):
        params.append("vlan_name")
        template = re.sub(r"\bvlan\s+[A-Za-z][A-Za-z0-9_-]*\b", "vlan {vlan_name}", template, count=1)
    if re.search(r"\b(?:limit|mac-move-limit)\s+\d+\b", template):
        params.append("limit")
        template = re.sub(r"\b(limit|mac-move-limit)\s+\d+\b", r"\1 {limit}", template, count=1)
    if re.search(r"\bflag\s+[A-Za-z0-9_-]+\b", template):
        params.append("flag")
        template = re.sub(r"\bflag\s+[A-Za-z0-9_-]+\b", "flag {flag}", template, count=1)
    if re.search(r"\bsnmp community [A-Za-z0-9_-]+\b", template):
        params.append("community")
        template = re.sub(r"\bsnmp community [A-Za-z0-9_-]+\b", "snmp community {community}", template, count=1)
    if re.search(r"\bsystem syslog user [A-Za-z0-9_-]+\b", template) and "user *" not in template:
        params.append("user")
        template = re.sub(r"\bsystem syslog user [A-Za-z0-9_-]+\b", "system syslog user {user}", template, count=1)

    return template, sorted(set(params))


def load_rows(input_paths):
    rows = []
    for path in input_paths:
        if path.exists():
            for row in read_jsonl(str(path)):
                row["_source_file"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def main(args):
    data_dir = ROOT / "data" / "processed"
    paths = [data_dir / "train.jsonl", data_dir / "val.jsonl"]
    if args.allow_test:
        paths.append(data_dir / "test.jsonl")

    records = {}
    conflicts = []
    examples = defaultdict(list)

    for row in load_rows(paths):
        target = row.get("target_command", "")
        body, had_commit = split_commit(target)
        if not body:
            continue
        variant_bodies = [(body, command_variant(body))]
        if command_variant(body) == "display_set":
            variant_bodies.append((plain_body_for_variant(body), "plain"))

        for variant_body, variant in variant_bodies:
            frame = command_to_semantic_frame(variant_body)
            action = str(frame.get("action", "")).lower()
            domain = str(frame.get("domain", "")).lower()
            sub_domain = str(frame.get("sub_domain", "")).lower()
            if not valid_key_fields(action, domain, sub_domain, variant_body):
                continue
            mode = infer_mode(action)
            requires_commit = infer_requires_commit(action, mode, had_commit)
            template, allowed_params = parameterize(variant_body)
            operation = str(frame.get("operation", "general") or "general")
            positive_cues, negative_cues = operation_cues(operation)
            required_params = placeholders(template)
            key = f"{action}/{domain}/{sub_domain}/{operation}/{variant}"
            public_key = f"{action}/{domain}/{sub_domain}"
            record = {
                "action": action,
                "domain": domain,
                "sub_domain": sub_domain,
                "template": template,
                "mode": mode,
                "requires_commit": requires_commit,
                "default_params": {},
                "allowed_params": sorted(set(allowed_params) | set(frame.get("parameters", {}).keys())),
                "required_params": required_params,
                "forbidden_params": [],
                "description": f"Template inferred from {row.get('_source_file', 'processed data')}",
                "intent_examples": [],
                "operation": operation,
                "positive_cues": positive_cues,
                "negative_cues": negative_cues,
                "negative_rules": [
                    "never_append_commit_for_operational_mode",
                    "commit_required_only_when_requires_commit_true",
                ],
                "validation_rules": {
                    "action": action,
                    "mode": mode,
                },
                "variant": variant,
            }
            examples[key].append(str(row.get("intent", "")))
            if key in records and (
                records[key]["template"] != template
                or records[key]["requires_commit"] != requires_commit
                or records[key]["mode"] != mode
            ):
                conflicts.append(
                    {
                        "key": public_key,
                        "variant": variant,
                        "existing": records[key],
                        "candidate": record,
                        "source_file": row.get("_source_file"),
                        "target_command": target,
                    }
                )
                continue
            records[key] = record

    for key, intents in examples.items():
        if key in records:
            records[key]["intent_examples"] = list(dict.fromkeys([i for i in intents if i]))[:10]

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")

    conflict_path = ROOT / "outputs" / "datastore_conflicts.jsonl"
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    with conflict_path.open("w", encoding="utf-8") as f:
        for row in conflicts:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(records)} templates to {output}")
    print(f"Wrote {len(conflicts)} conflicts to {conflict_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local command-context template store from train/val processed data.")
    parser.add_argument("--output", default="data/juniper_templates.json")
    parser.add_argument("--allow-test", action="store_true", help="Also include data/processed/test.jsonl. Off by default.")
    main(parser.parse_args())
