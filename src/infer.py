import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from preprocess import build_prompt
from utils import load_config, read_jsonl, write_jsonl

def clean(s):
    s=s.strip().split('\n')[0]
    s=re.sub(r'^(Command:|Output:)\s*','',s,flags=re.I)
    return s.strip()

def main(a):
    c=load_config(a.config); t=c['training']; ic=c['inference']
    tok=AutoTokenizer.from_pretrained(t['base_model'])
    base=AutoModelForCausalLM.from_pretrained(t['base_model'],device_map='auto',torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model=PeftModel.from_pretrained(base, t['output_dir'])
    rows=read_jsonl(a.input_file)
    out=[]
    for r in rows:
        prompt=build_prompt(r['intent'], r.get('context',''), a.mode)
        ids=tok(prompt, return_tensors='pt').to(model.device)
        gen=model.generate(**ids,max_new_tokens=ic['max_new_tokens'],temperature=ic['temperature'],top_p=ic['top_p'])
        text=tok.decode(gen[0], skip_special_tokens=True)
        pred=clean(text[len(prompt):])
        out.append({**r,'prediction':pred})
    write_jsonl(a.output_file,out)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); p.add_argument('--input_file',required=True); p.add_argument('--output_file',required=True); p.add_argument('--mode',default='intent_with_context'); main(p.parse_args())
