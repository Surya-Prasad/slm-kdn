import argparse
from utils import load_config, read_jsonl, write_jsonl

INSTR = (
    "You are a network semantic parser. Convert the user intent into strict JSON only. "
    "JSON schema: {\"action\": str, \"target\": str, \"target_type\": str, "
    "\"parameters\": object}. Do not output explanations."
)


def build_prompt(intent, context, mode):
    if mode == "intent_only":
        return f"{INSTR}\n\nIntent:\n{intent}\n\nJSON:"
    return f"{INSTR}\n\nContext:\n{context}\n\nIntent:\n{intent}\n\nJSON:"


def main(args):
    cfg = load_config(args.config)
    mode = args.mode or cfg["prompt"]["mode"]
    for split in ["train", "val", "test"]:
        rows = read_jsonl(f"{cfg['data']['output_dir']}/{split}.jsonl")
        out = []
        for r in rows:
            prompt = build_prompt(r["intent"], r.get("context", ""), mode)
            target = r.get("target_json") or r.get("target_command", "")
            out.append({**r, "prompt": prompt, "text": prompt + " " + target})
        write_jsonl(f"{cfg['data']['output_dir']}/{split}_{mode}.jsonl", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["intent_only", "intent_with_context"], default=None)
    main(ap.parse_args())
