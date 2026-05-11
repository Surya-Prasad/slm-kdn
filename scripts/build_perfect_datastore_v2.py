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
    toks = body.lower().split()
    if not toks:
        return "unknown", "unknown"
    action = toks[0]
    rest = toks[1:]
    if action in {"ping", "traceroute"}:
        return "network", action
    if not rest:
        return "unknown", "unknown"
    if rest[:2] == ["protocols", "ospf"]:
        return "protocols", "ospf"
    if rest[:2] == ["protocols", "igmp-snooping"] or rest[:1] == ["igmp-snooping"]:
        return "protocols", "igmp-snooping"
    if rest[:2] == ["system", "syslog"]:
        return "system", "syslog"
    if rest[:2] == ["snmp", "community"]:
        return "snmp", "community"
    if rest[:2] == ["chassis", "lcd"]:
        return "chassis", "lcd"
    if rest[:2] == ["chassis", "pic-mode"]:
        return "chassis", "pic-mode"
    if rest[:1] == ["chassis"]:
        return "chassis", rest[1] if len(rest) > 1 else "status"
    if rest[:1] == ["ethernet-switching-table"]:
        return "ethernet-switching", "table"
    if rest[:1] == ["ethernet-switching"]:
        return "ethernet-switching", rest[1] if len(rest) > 1 else "general"
    if rest[:1] == ["interfaces"]:
        return "interfaces", rest[2] if len(rest) > 2 and rest[2] != "unit" else "interface"
    if rest[:1] == ["virtual-chassis"]:
        return "virtual-chassis", rest[1] if len(rest) > 1 else "general"
    return rest[0], rest[1] if len(rest) > 1 else "general"


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
        action = body.split()[0].lower()
        domain, sub_domain = infer_domain_subdomain(body)
        mode = infer_mode(action)
        requires_commit = infer_requires_commit(action, mode, had_commit)
        template, allowed_params = parameterize(body)
        key = f"{action}/{domain}/{sub_domain}"
        record = {
            "action": action,
            "domain": domain,
            "sub_domain": sub_domain,
            "template": template,
            "mode": mode,
            "requires_commit": requires_commit,
            "default_params": {},
            "allowed_params": allowed_params,
            "description": f"Template inferred from {row.get('_source_file', 'processed data')}",
            "intent_examples": [],
            "negative_rules": [
                "never_append_commit_for_operational_mode",
                "commit_required_only_when_requires_commit_true",
            ],
            "validation_rules": {
                "action": action,
                "mode": mode,
            },
        }
        examples[key].append(str(row.get("intent", "")))
        if key in records and (
            records[key]["template"] != template
            or records[key]["requires_commit"] != requires_commit
            or records[key]["mode"] != mode
        ):
            conflicts.append(
                {
                    "key": key,
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
