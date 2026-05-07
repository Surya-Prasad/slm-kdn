#!/usr/bin/env bash
set -e
for f in clean_test paraphrased_test noisy_test; do
  python src/infer.py --input_file data/processed/${f}.jsonl --output_file results/predictions/${f}_pred.jsonl
  python src/evaluate.py --pred_file results/predictions/${f}_pred.jsonl --out_file results/metrics/${f}_metrics.json
done
