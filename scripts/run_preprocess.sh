#!/usr/bin/env bash
set -e
python src/load_dataset.py
python src/preprocess.py --mode intent_only
python src/preprocess.py --mode intent_with_context
python src/augment.py
