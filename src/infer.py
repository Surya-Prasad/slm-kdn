# Meant for use with an A100
import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from preprocess import build_prompt
from utils import load_config, read_jsonl, write_jsonl
from tqdm import tqdm

def clean(s):
    # Remove conversational prefixes
    s = re.sub(r'^(Command:|Output:)\s*', '', s.strip(), flags=re.I)
    
    # CRITICAL FIX: Convert any actual newlines into literal '\n' strings 
    # so we can process everything uniformly
    s = s.replace('\n', '\\n')
    
    # Split by the literal '\n' string
    parts = s.split('\\n')
    if not parts: return ""
    
    cmd = parts[0].strip()
    
    # If the model logically output 'commit' as the second command, append it exactly 
    # as the ground-truth dataset expects it: with a literal '\n'
    if len(parts) > 1 and parts[1].strip().lower() == 'commit':
        cmd += '\\ncommit'
        
    return cmd

def main(a):
    c=load_config(a.config); t=c['training']; ic=c['inference']
    
    tok=AutoTokenizer.from_pretrained(t['base_model'])
    tok.padding_side = 'left' 
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    base=AutoModelForCausalLM.from_pretrained(t['base_model'],device_map='auto',torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model=PeftModel.from_pretrained(base, t['output_dir'])
    rows=read_jsonl(a.input_file)
    out=[]
    
    batch_size = 32 
    print(f"\n[INFO] Starting BATCHED inference on {len(rows)} instances for {a.input_file.split('/')[-1]}...")
    
    for i in tqdm(range(0, len(rows), batch_size), desc="Batches"):
        batch_rows = rows[i:i+batch_size]
        prompts = [build_prompt(r['intent'], r.get('context',''), a.mode) for r in batch_rows]
        
        inputs = tok(prompts, return_tensors='pt', padding=True).to(model.device)
        
        with torch.no_grad():
            gens = model.generate(**inputs, max_new_tokens=ic['max_new_tokens'], do_sample=False)
        
        for j, gen in enumerate(gens):
            text = tok.decode(gen, skip_special_tokens=True)
            pred = clean(text[len(prompts[j]):])
            out.append({**batch_rows[j], 'prediction': pred})
            
    write_jsonl(a.output_file,out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); p.add_argument('--input_file',required=True); p.add_argument('--output_file',required=True); p.add_argument('--mode',default='intent_with_context'); main(p.parse_args())