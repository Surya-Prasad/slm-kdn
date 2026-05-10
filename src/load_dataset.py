import argparse
import json
import re
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from utils import ensure_dir, load_config, write_jsonl


def infer_columns(columns):
    lc = {c.lower(): c for c in columns}
    intent = next((lc[k] for k in lc if "intent" in k or "query" in k or "question" in k), None)
    cmd = next((lc[k] for k in lc if "command" in k or "config" in k or "label" in k or "answer" in k), None)
    context = next((lc[k] for k in lc if "context" in k or "description" in k or "desc" in k), None)
    return intent, context, cmd


def command_to_semantic_json(command: str) -> str:
    clean = re.sub(r"Use the following command.*?:\s*", "", command, flags=re.IGNORECASE | re.DOTALL)
    clean = clean.replace("`", "").strip()
    cmd = clean.split("\\n")[0].strip()

    action = cmd.split()[0].lower() if cmd else "unknown"
    target_type = "unknown"

    if "interfaces" in cmd:
        target_type = "interface"
    elif "vlans" in cmd or "vlan" in cmd:
        target_type = "vlan"
    elif "route" in cmd:
        target_type = "route"

    target = ""
    iface = re.search(r"interfaces\s+([^\s]+)", cmd, flags=re.I)
    if iface:
        target = iface.group(1)

    vlan_id = None
    vlan_id_match = re.search(r"vlan(?:-id|\s+members)?\s+(\d+)", cmd, flags=re.I)
    if vlan_id_match:
        vlan_id = int(vlan_id_match.group(1))

    vlan_name = ""
    vlan_name_match = re.search(r"vlans\s+([^\s]+)", cmd, flags=re.I)
    if vlan_name_match:
        vlan_name = vlan_name_match.group(1)

    unit = 0
    unit_match = re.search(r"unit\s+(\d+)", cmd, flags=re.I)
    if unit_match:
        unit = int(unit_match.group(1))

    prefix = ""
    prefix_match = re.search(r"route\s+([^\s]+)", cmd, flags=re.I)
    if prefix_match:
        prefix = prefix_match.group(1)

    payload = {
        "action": action,
        "target": target,
        "target_type": target_type,
        "parameters": {
            "vlan_id": vlan_id,
            "vlan_name": vlan_name,
            "unit": unit,
            "prefix": prefix,
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def map_record(row, intent_col, context_col, cmd_col):
    raw_cmd = str(row.get(cmd_col, "")).strip()
    clean_cmd = re.sub(r"Use the following command.*?:\s*", "", raw_cmd, flags=re.IGNORECASE | re.DOTALL)
    clean_cmd = clean_cmd.replace("`", "").strip()

    return {
        "intent": str(row.get(intent_col, "")).strip(),
        "context": str(row.get(context_col, "")).strip() if context_col else "",
        "target_command": clean_cmd,
        "target_json": command_to_semantic_json(clean_cmd),
        "category": str(row.get("category", "")),
    }


def main(args):
    cfg = load_config(args.config)
    ds = load_dataset(cfg["data"]["dataset_name"])
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    sample_split = next(iter(ds.keys()))
    intent_col, context_col, cmd_col = infer_columns(ds[sample_split].column_names)

    mapped = []
    for split_name in ds.keys():
        for row in ds[split_name]:
            mapped.append(map_record(row, intent_col, context_col, cmd_col))

    filtered = [r for r in mapped if r["intent"] and r["target_command"]]
    train_val, test = train_test_split(filtered, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["seed"])
    train, val = train_test_split(train_val, test_size=cfg["data"]["val_size"], random_state=cfg["data"]["seed"])

    out = cfg["data"]["output_dir"]
    ensure_dir(out)
    write_jsonl(f"{out}/train.jsonl", train)
    write_jsonl(f"{out}/val.jsonl", val)
    write_jsonl(f"{out}/test.jsonl", test)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args())
