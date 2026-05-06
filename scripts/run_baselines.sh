#!/usr/bin/env bash
set -e
python micro_kdn_llama/src/latency_benchmark.py --input_file micro_kdn_llama/data/processed/clean_test.jsonl
