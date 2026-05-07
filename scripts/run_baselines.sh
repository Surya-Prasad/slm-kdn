#!/usr/bin/env bash
set -e
python src/latency_benchmark.py --input_file data/processed/clean_test.jsonl
