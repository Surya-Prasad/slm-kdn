import argparse
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

from utils import ensure_dir, load_config, write_jsonl


def infer_columns(columns):
    lc = {c.lower(): c for c in columns}
    intent = next((lc[k] for k in lc if "intent" in k or "query" in k), None)
    cmd = next((lc[k] for k in lc if "command" in k or "config" in k or "label" in k), None)
    context = next((lc[k] for k in lc if "context" in k or "description" in k or "desc" in k), None)
    return intent, context, cmd


def map_record(row, intent_col, context_col, cmd_col):
    return {
        "intent": str(row.get(intent_col, "")).strip(),
        "context": str(row.get(context_col, "")).strip() if context_col else "",
        "target_command": str(row.get(cmd_col, "")).strip(),
        "category": str(row.get("category", "")),
    }


def main(args):
    cfg = load_config(args.config)
    ds = load_dataset(cfg["data"]["dataset_name"])
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    sample_split = next(iter(ds.keys()))
    intent_col, context_col, cmd_col = infer_columns(ds[sample_split].column_names)
    print({"intent": intent_col, "context": context_col, "target_command": cmd_col})

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
    p.add_argument("--config", default="micro_kdn_llama/config.yaml")
    main(p.parse_args())
