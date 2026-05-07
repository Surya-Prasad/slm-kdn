import argparse, re, random
from utils import load_config, read_jsonl, write_jsonl
SYN={"set":"configure","enable":"turn on","disable":"turn off","allow":"permit"}

def paraphrase(t):
    for k,v in SYN.items(): t=re.sub(rf"\b{k}\b",v,t,flags=re.I)
    return t

def noisy(t):
    t = t.upper() if random.random()<0.3 else t.lower()
    t = re.sub(r"\s+","  ",t) if random.random()<0.4 else t
    return t + random.choice(["", ".", " !"])

def main(a):
    c=load_config(a.config); rows=read_jsonl(f"{c['data']['output_dir']}/test.jsonl")
    write_jsonl(f"{c['data']['output_dir']}/clean_test.jsonl", rows)
    write_jsonl(f"{c['data']['output_dir']}/paraphrased_test.jsonl", [{**r,"intent":paraphrase(r['intent'])} for r in rows])
    write_jsonl(f"{c['data']['output_dir']}/noisy_test.jsonl", [{**r,"intent":noisy(r['intent'])} for r in rows])
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); main(p.parse_args())
