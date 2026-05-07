import argparse, json, time, numpy as np
from rule_based_baseline import predict
from utils import read_jsonl

def main(a):
    rows=read_jsonl(a.input_file)[:a.num_samples]
    l=[]
    for r in rows:
        t=time.perf_counter(); out=predict(r['intent'], r.get('context','')); _=len(out.split()); l.append(time.perf_counter()-t)
    res={'avg_latency_ms':float(np.mean(l)*1000),'p50_ms':float(np.percentile(l,50)*1000),'p95_ms':float(np.percentile(l,95)*1000)}
    with open(a.out_file,'w') as f: json.dump(res,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input_file',required=True); p.add_argument('--num_samples',type=int,default=200); p.add_argument('--out_file',default='results/metrics/latency.json'); main(p.parse_args())
