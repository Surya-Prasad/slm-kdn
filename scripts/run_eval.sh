#!/usr/bin/env bash
set -e
python micro_kdn_llama/src/infer.py --input_file micro_kdn_llama/data/processed/clean_test.jsonl --output_file micro_kdn_llama/results/predictions/predictions.jsonl
python micro_kdn_llama/src/evaluate.py --pred_file micro_kdn_llama/results/predictions/predictions.jsonl
python micro_kdn_llama/src/error_analysis.py --pred_file micro_kdn_llama/results/predictions/predictions.jsonl
