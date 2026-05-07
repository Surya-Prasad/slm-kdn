import argparse, json, os, time, psutil, torch
from transformers import AutoModelForCausalLM

def main(a):
    p=psutil.Process(os.getpid()); before=p.memory_info().rss
    t0=time.time(); model=AutoModelForCausalLM.from_pretrained(a.model_name, device_map='auto'); load_s=time.time()-t0
    after=p.memory_info().rss
    gpu_peak=None
    if torch.cuda.is_available(): gpu_peak=torch.cuda.max_memory_allocated()
    res={'model_load_time_sec':load_s,'cpu_mem_mb':(after-before)/(1024**2),'gpu_peak_bytes':gpu_peak}
    with open(a.out_file,'w') as f: json.dump(res,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model_name',default='meta-llama/Meta-Llama-3-8B'); p.add_argument('--out_file',default='results/metrics/resource.json'); main(p.parse_args())
