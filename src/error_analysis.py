import argparse, csv, json
from collections import Counter
from utils import read_jsonl
from validate_output import validate

def classify(r):
    p=r['prediction'].lower(); g=r['target_command'].lower()
    if not p.strip(): return 'hallucinated_command'
    v=validate(p,r.get('intent',''))
    if not v['is_valid'] and any('contains_explanation' in e for e in v['errors']): return 'extra_explanation_text'
    if not v['is_valid']: return 'syntax_invalid'
    if p==g: return 'correct'
    if 'ge-' in g and 'ge-' in p and p.split('ge-')[1][:5]!=g.split('ge-')[1][:5]: return 'wrong_interface'
    if p.split()[:1]!=g.split()[:1]: return 'wrong_action'
    if len(p.split())<len(g.split())*0.7: return 'missing_parameter'
    return 'semantically_close_but_not_exact'

def main(a):
    rows=read_jsonl(a.pred_file)
    for r in rows: r['error_type']=classify(r)
    counts=Counter(r['error_type'] for r in rows)
    with open(a.out_json,'w') as f: json.dump({'counts':counts,'total':len(rows)},f,indent=2,default=int)
    with open(a.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['intent','target_command','prediction','error_type']); w.writeheader();
        for r in rows: w.writerow({k:r.get(k,'') for k in w.fieldnames})
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--pred_file',required=True); p.add_argument('--out_json',default='results/error_analysis/error_summary.json'); p.add_argument('--out_csv',default='results/error_analysis/errors.csv'); main(p.parse_args())
