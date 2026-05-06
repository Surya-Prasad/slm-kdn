import argparse, json
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import normalize_command, read_jsonl, tokenize, load_config, ensure_dir
from validate_output import validate, extract_entities

def f1(pred, gold):
    p=tokenize(pred); g=tokenize(gold)
    pc, gc = Counter(p), Counter(g)
    tp=sum((pc&gc).values())
    if tp==0: return 0.0
    pr=tp/max(len(p),1); rc=tp/max(len(g),1)
    return 2*pr*rc/(pr+rc)

def main(a):
    rows=read_jsonl(a.pred_file)
    m=defaultdict(float); per_cat=defaultdict(lambda: defaultdict(float)); n=len(rows)
    smooth=SmoothingFunction().method1
    for r in rows:
        pred=r['prediction']; gold=r['target_command']; cat=r.get('category','all')
        em=float(pred.strip()==gold.strip()); nem=float(normalize_command(pred)==normalize_command(gold))
        fv=f1(pred,gold); bleu=sentence_bleu([tokenize(gold)], tokenize(pred), smoothing_function=smooth)
        val=validate(pred, r.get('intent','')); ent=extract_entities(r.get('intent',''))
        ent_ok=float(all(e in pred.lower() for e in ent)) if ent else 1.0
        for d in [m, per_cat[cat]]:
            d['exact_match']+=em; d['normalized_exact_match']+=nem; d['token_f1']+=fv; d['bleu']+=bleu; d['valid_rate']+=float(val['is_valid']); d['entity_preservation']+=ent_ok
    for d,cnt in [(m,n)]+[(v,sum(1 for r in rows if r.get('category','all')==k)) for k,v in per_cat.items()]:
        for k in list(d.keys()): d[k]/=max(cnt,1)
        d['invalid_output_rate']=1-d['valid_rate']
    out={'overall':dict(m),'per_category':{k:dict(v) for k,v in per_cat.items()}}
    ensure_dir(a.out_dir)
    with open(a.out_file,'w') as f: json.dump(out,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='micro_kdn_llama/config.yaml'); p.add_argument('--pred_file',default='micro_kdn_llama/results/predictions/predictions.jsonl'); p.add_argument('--out_dir',default='micro_kdn_llama/results/metrics'); p.add_argument('--out_file',default='micro_kdn_llama/results/metrics/eval_metrics.json'); main(p.parse_args())
