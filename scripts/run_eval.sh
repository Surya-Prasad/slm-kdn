#!/usr/bin/env bash
set -e
python src/infer.py --input_file data/processed/clean_test.jsonl --output_file results/predictions/predictions.jsonl
python src/evaluate.py --pred_file results/predictions/predictions.jsonl
python src/error_analysis.py --pred_file results/predictions/predictions.jsonl
