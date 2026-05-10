# Meant for use with an A100 (unquantized fp16 path)
import argparse
import json

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess import build_prompt
from rag_store import retrieve_template_with_doc
from utils import load_config, read_jsonl, write_jsonl


REQUIRED_JSON_KEYS = {"action", "target", "target_type", "parameters"}
ALLOWED_ACTIONS = {"set", "delete", "show"}
ALLOWED_TARGET_TYPES = {"interface", "vlan", "route"}


def parse_semantic_json(raw: str):
    """Parse potentially noisy model output into a semantic JSON object with safe type normalization."""
    errors = []
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no_json_object_found")

        candidate = raw[start : end + 1]
        parsed = json.loads(candidate)

        if not isinstance(parsed, dict):
            raise ValueError("json_not_object")

        missing = REQUIRED_JSON_KEYS - set(parsed.keys())
        if missing:
            raise ValueError(f"missing_keys:{','.join(sorted(missing))}")

        if not isinstance(parsed.get("parameters"), dict):
            raise ValueError("parameters_not_object")

        if parsed.get("action") not in ALLOWED_ACTIONS:
            errors.append("invalid_enum_action")
        if parsed.get("target_type") not in ALLOWED_TARGET_TYPES:
            errors.append("invalid_enum_target_type")

        params = parsed["parameters"]
        for key in ("vlan_id", "unit"):
            if key in params and params[key] is not None:
                try:
                    params[key] = int(params[key])
                except (TypeError, ValueError):
                    params[key] = None
                    errors.append(f"invalid_type_{key}")

        error_str = ";".join(errors) if errors else None
        return parsed, error_str

    except Exception as exc:  # robust boundary for hallucinated formatting
        return None, str(exc)


def assemble_command(parsed):
    record, doc = retrieve_template_with_doc(parsed.get("action", ""), parsed.get("target_type", ""))
    if record is None:
        return "", "template_not_found", None

    fields = dict(record.default_params)
    fields.update(dict(parsed.get("parameters", {})))
    fields["target"] = parsed.get("target", fields.get("target", ""))

    try:
        command = record.template.format(**fields).strip()
    except KeyError as exc:
        return "", f"missing_template_parameter:{exc}", doc

    if record.requires_commit:
        command = f"{command}\\ncommit"

    return command, None, doc


def main(a):
    c = load_config(a.config)
    t = c["training"]
    ic = c["inference"]

    tok = AutoTokenizer.from_pretrained(t["base_model"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(t["base_model"], device_map="auto", torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, t["output_dir"])

    rows = read_jsonl(a.input_file)
    out = []

    print(f"\n[INFO] Starting BATCHED inference on {len(rows)} instances for {a.input_file.split('/')[-1]}...")

    for i in tqdm(range(0, len(rows), a.batch_size), desc="Batches"):
        batch_rows = rows[i : i + a.batch_size]
        prompts = [build_prompt(r["intent"], r.get("context", ""), a.mode) for r in batch_rows]

        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            gens = model.generate(**inputs, max_new_tokens=ic["max_new_tokens"], do_sample=False)

        for j, gen in enumerate(gens):
            text = tok.decode(gen, skip_special_tokens=True)
            raw_json = text[len(prompts[j]) :].strip()
            parsed, parse_error = parse_semantic_json(raw_json)

            assembled = ""
            assembly_error = None
            retrieved_doc = None
            if parsed is not None:
                assembled, assembly_error, retrieved_doc = assemble_command(parsed)

            out.append(
                {
                    **batch_rows[j],
                    "prediction_json_raw": raw_json,
                    "prediction_json": parsed,
                    "prediction": assembled,
                    "parse_error": parse_error,
                    "assembly_error": assembly_error,
                    "retrieved_doc": None if retrieved_doc is None else {
                        "doc_id": retrieved_doc.doc_id,
                        "title": retrieved_doc.title,
                        "snippet": retrieved_doc.snippet,
                        "source_url": retrieved_doc.source_url,
                    },
                }
            )

    write_jsonl(a.output_file, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--mode", default="intent_with_context")
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
