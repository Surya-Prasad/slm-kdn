import argparse
import random
import re

from utils import read_jsonl, write_jsonl


IFACE_RE = re.compile(r"\b(?:ge|xe|et|fe)-\d+/\d+/\d+\b")


def make_interface_pool(fpcs=(0, 1), pics=(0,), ports=range(0, 48), prefixes=("ge", "xe")):
    return [f"{prefix}-{fpc}/{pic}/{port}" for prefix in prefixes for fpc in fpcs for pic in pics for port in ports]


def replace_interfaces(text, mapping):
    if not text:
        return text
    return IFACE_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def augment_row(row, interface_pool, rng):
    found = sorted(
        set(
            IFACE_RE.findall(" ".join(str(row.get(k, "")) for k in ("intent", "context", "target_command")))
        )
    )
    if not found:
        return None

    replacements = rng.sample(interface_pool, k=min(len(found), len(interface_pool)))
    mapping = dict(zip(found, replacements))
    return {
        **row,
        "intent": replace_interfaces(row.get("intent", ""), mapping),
        "context": replace_interfaces(row.get("context", ""), mapping),
        "target_command": replace_interfaces(row.get("target_command", ""), mapping),
        "category": f"{row.get('category', 'all')}_iface_aug",
    }


def main(args):
    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_file)
    pool = make_interface_pool()
    augmented = []
    for row in rows:
        augmented.append(row)
        for _ in range(args.copies):
            new_row = augment_row(row, pool, rng)
            if new_row:
                augmented.append(new_row)
    write_jsonl(args.output_file, augmented)
    print(f"Wrote {len(augmented)} rows to {args.output_file} from {len(rows)} source rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Junos interface names in NIT-style JSONL rows.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
