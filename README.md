# SLM-driven Micro-KDN for Edge Networking Environments

Research-grade skeleton for intent-to-network-command generation with Llama-3-8B + (Q)LoRA.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline
1. Preprocess and create robustness sets:
```bash
scripts/run_preprocess.sh
```
2. Train LoRA adapter:
```bash
scripts/run_train.sh
```
3. Infer + evaluate + error analysis:
```bash
scripts/run_eval.sh
```
4. Robustness:
```bash
scripts/run_robustness.sh
```

## RAG over EX3300 guide
The RAG index combines processed NIT JSONL files, when present, with documents under `rag-doc/`.
The included Juniper guide is `rag-doc/ex3300.pdf`.

Build or rebuild the retrieval index and run a quick EX3300 query:
```bash
python src/rag_query.py --rebuild --query "What are the front panel ports on a Juniper EX3300 switch?"
```

Run the default retrieval smoke tests:
```bash
python src/rag_query.py --rebuild
```

Use retrieved EX3300 context during model inference:
```bash
python src/infer.py \
  --input_file data/processed/test.jsonl \
  --output_file results/predictions/rag_predictions.jsonl \
  --use_rag \
  --rebuild_rag \
  --rag_debug
```

RAG settings live in `config.yaml` under `rag`. Set `--rebuild` or `--rebuild_rag` after changing files in
`rag-doc/`; the index also rebuilds automatically when source files or chunking settings change.

## Final Architecture: Semantic RAG Micro-KDN
The final research path treats the fine-tuned Llama-3-8B adapter as a semantic intent parser, not a free-form CLI generator. With `--semantic_rag`, the model is prompted to output strict JSON only:
```json
{"action":"...","domain":"...","sub_domain":"...","parameters":{}}
```

The local command-context store then retrieves command metadata: template, operational/configuration mode, allowed parameters, validation rules, and commit behavior. Python assembles the final Junos CLI deterministically, and guardrails clean up commit handling so operational commands such as `show`, `clear`, `request`, `ping`, `traceroute`, and `monitor` never receive `commit`.

The older `--use_rag` path is retained as a baseline for comparison, but `--semantic_rag` is the final Semantic RAG Micro-KDN evaluation path.

Build the local command knowledge store from train/val only:
```bash
python scripts/build_perfect_datastore_v2.py
```

Run semantic RAG inference and evaluation:
```bash
scripts/run_semantic_rag_eval.sh
```

Analyze semantic RAG failures:
```bash
python scripts/analyze_semantic_rag_errors.py
```

## Notes
- Dataset: `Smarneh/NIT` via HuggingFace `datasets`.
- Prompt modes: `intent_only`, `intent_with_context`.
- QLoRA enabled by default in `config.yaml`.
- Output artifacts include processed JSONL, predictions, metrics, latency/resource logs, and error summaries.
