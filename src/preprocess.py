import argparse
from utils import load_config, read_jsonl, write_jsonl

INSTR = "You are a network intent translation assistant. Convert the user intent into the correct network configuration command. Output only the command."

def build_prompt(intent, context, mode):
    if mode == "intent_only":
        return f"{INSTR}\n\nIntent:\n{intent}\n\nCommand:"
    return f"{INSTR}\n\nContext:\n{context}\n\nIntent:\n{intent}\n\nCommand:"

def main(args):
    cfg = load_config(args.config)
    mode = args.mode or cfg["prompt"]["mode"]
    for split in ["train", "val", "test"]:
        rows = read_jsonl(f"{cfg['data']['output_dir']}/{split}.jsonl")
        out = []
        for r in rows:
            prompt = build_prompt(r["intent"], r.get("context", ""), mode)
            out.append({**r, "prompt": prompt, "text": prompt + " " + r["target_command"]})
        write_jsonl(f"{cfg['data']['output_dir']}/{split}_{mode}.jsonl", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="micro_kdn_llama/config.yaml")
    ap.add_argument("--mode", choices=["intent_only", "intent_with_context"], default=None)
    main(ap.parse_args())
