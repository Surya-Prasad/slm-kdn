#!/usr/bin/env bash
set -e
for f in clean_test paraphrased_test noisy_test; do
  python micro_kdn_llama/src/infer.py --input_file micro_kdn_llama/data/processed/${f}.jsonl --output_file micro_kdn_llama/results/predictions/${f}_pred.jsonl
  python micro_kdn_llama/src/evaluate.py --pred_file micro_kdn_llama/results/predictions/${f}_pred.jsonl --out_file micro_kdn_llama/results/metrics/${f}_metrics.json
done
