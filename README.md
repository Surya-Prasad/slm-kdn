# SLM-driven Micro-KDN for Edge Networking Environments

Research-grade skeleton for intent-to-network-command generation with Llama-3-8B + (Q)LoRA.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r micro_kdn_llama/requirements.txt
```

## Pipeline
1. Preprocess and create robustness sets:
```bash
micro_kdn_llama/scripts/run_preprocess.sh
```
2. Train LoRA adapter:
```bash
micro_kdn_llama/scripts/run_train.sh
```
3. Infer + evaluate + error analysis:
```bash
micro_kdn_llama/scripts/run_eval.sh
```
4. Robustness:
```bash
micro_kdn_llama/scripts/run_robustness.sh
```

## Notes
- Dataset: `Smarneh/NIT` via HuggingFace `datasets`.
- Prompt modes: `intent_only`, `intent_with_context`.
- QLoRA enabled by default in `config.yaml`.
- Output artifacts include processed JSONL, predictions, metrics, latency/resource logs, and error summaries.
