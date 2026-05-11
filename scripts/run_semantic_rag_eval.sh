#!/usr/bin/env bash
set -e

mkdir -p results/predictions results/metrics results/error_analysis

run_one() {
  local input_file="$1"
  local name="$2"
  local pred_file="results/predictions/${name}_predictions.jsonl"
  local metrics_file="results/metrics/${name}_metrics.json"
  local failures_file="results/error_analysis/${name}_failures.jsonl"
  local summary_file="results/error_analysis/${name}_error_summary.json"

  python src/infer.py \
    --input_file "${input_file}" \
    --output_file "${pred_file}" \
    --semantic_rag \
    --mode intent_with_context

  python scripts/evaluate_semantic_rag.py \
    --pred_file "${pred_file}" \
    --out_file "${metrics_file}" \
    --failures_file "${failures_file}" \
    --summary_file "${summary_file}"

  python src/evaluate.py \
    --pred_file "${pred_file}" \
    --out_file "results/metrics/${name}_legacy_eval_metrics.json"
}

run_one data/processed/test.jsonl semantic_rag

for variant in clean_test paraphrased_test noisy_test; do
  if [[ -f "data/processed/${variant}.jsonl" ]]; then
    run_one "data/processed/${variant}.jsonl" "${variant}"
  fi
done

python scripts/analyze_semantic_rag_errors.py \
  --pred_file results/predictions/semantic_rag_predictions.jsonl
