#!/usr/bin/env bash
set -e
python micro_kdn_llama/src/load_dataset.py
python micro_kdn_llama/src/preprocess.py --mode intent_only
python micro_kdn_llama/src/preprocess.py --mode intent_with_context
python micro_kdn_llama/src/augment.py
