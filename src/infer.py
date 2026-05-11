# Meant for use with an A100
import argparse, json, re, torch
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess import build_prompt
from rag import assert_no_test_leakage, build_rag_prompt, format_retrieval_debug, get_or_build_index
from utils import load_config, read_jsonl, write_jsonl


REQUIRED_JSON_KEYS = {"action", "domain", "sub_domain", "parameters"}
ALLOWED_ACTIONS = {"set", "delete", "show"}


def clean(s):
    s = re.sub(r'^(Command:|Output:)\s*', '', s.strip(), flags=re.I)
    s = s.replace('\n', '\\n')
    parts = s.split('\\n')
    if not parts:
        return ""

    cmd = parts[0].strip()
    read_only_prefixes = ('show ', 'ping ', 'traceroute ', 'monitor ', 'clear ', 'request ')
    is_read_only = cmd.lower().startswith(read_only_prefixes)
    if len(parts) > 1 and parts[1].strip().lower() == 'commit' and not is_read_only:
        cmd += '\\ncommit'

    return cmd


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
    record = retrieve_template(
        parsed.get("action", ""),
        parsed.get("domain", ""),
        parsed.get("sub_domain", ""),
    )
    if record is None:
        return "", "template_not_found"

    fields = dict(record.default_params)
    fields.update(dict(parsed.get("parameters", {})))

    try:
        command = record.template.format(**fields).strip()
    except KeyError as exc:
        return "", f"missing_template_parameter:{exc}"

    if record.requires_commit:
        command = f"{command}\\ncommit"

    return command, None


def clean_answer(s):
    return re.sub(r"\s+", " ", s.strip())

def main(a):
    c=load_config(a.config); t=c['training']; ic=c['inference']
    batch_size = ic.get("batch_size", 1) 
    use_rag = a.use_rag or bool(c.get("rag", {}).get("enabled", False))
    rag_index = get_or_build_index(c, rebuild=a.rebuild_rag) if use_rag else None
    eval_mode = Path(a.input_file).name in {"test.jsonl", "clean_test.jsonl", "rag_smoke.jsonl"}
    source_counts = Counter()
    
    tok=AutoTokenizer.from_pretrained(t['base_model'])
    tok.padding_side = 'left' 
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(t["base_model"], device_map="auto", torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, t["output_dir"])

    rows = read_jsonl(a.input_file)
    out = []

    print(f"\n[INFO] Starting BATCHED inference on {len(rows)} instances for {a.input_file.split('/')[-1]}...")
    
    for i in tqdm(range(0, len(rows), batch_size), desc="Batches"):
        batch_rows = rows[i:i+batch_size]
        prompts = []
        retrievals = []
        for r in batch_rows:
            question = r['intent']
            if rag_index:
                chunks = rag_index.retrieve(question, top_k=int(c.get("rag", {}).get("top_k", 5)))
                if eval_mode and not a.allow_test_rag:
                    assert_no_test_leakage(chunks)
                for chunk in chunks:
                    source_counts[chunk.metadata.get("source_file", "unknown")] += 1
                retrievals.append(chunks)
                if a.rag_debug:
                    print(format_retrieval_debug(question, chunks))
                prompts.append(build_rag_prompt(question, chunks))
            else:
                retrievals.append([])
                prompts.append(build_prompt(question, r.get('context',''), a.mode))
        
        inputs = tok(prompts, return_tensors='pt', padding=True).to(model.device)
        
        with torch.no_grad():
            gens = model.generate(**inputs, max_new_tokens=ic["max_new_tokens"], do_sample=False)

        for j, gen in enumerate(gens):
            text = tok.decode(gen, skip_special_tokens=True)
            raw_pred = text[len(prompts[j]):]
            pred = clean_answer(raw_pred) if rag_index else clean(raw_pred)
            rag_sources = [
                {
                    "source_file": chunk.metadata.get("source_file"),
                    "page": chunk.metadata.get("page"),
                    "score": chunk.score,
                }
                for chunk in retrievals[j]
            ]
            row = {**batch_rows[j], 'prediction': pred}
            if rag_index:
                row["rag_sources"] = rag_sources
            out.append(row)
            
    write_jsonl(a.output_file,out)
    if rag_index:
        print("[RAG] retrieved chunk counts:")
        for source, count in sorted(source_counts.items()):
            print(f"[RAG]   {source}: {count}")

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); p.add_argument('--input_file',required=True); p.add_argument('--output_file',required=True); p.add_argument('--mode',default='intent_with_context'); p.add_argument('--use_rag',action='store_true'); p.add_argument('--rebuild_rag',action='store_true'); p.add_argument('--rebuild_index',action='store_true'); p.add_argument('--rag_debug',action='store_true'); p.add_argument('--allow_test_rag',action='store_true'); args=p.parse_args(); args.rebuild_rag = args.rebuild_rag or args.rebuild_index; main(args)
