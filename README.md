# SLM-driven Micro-KDN for Edge Networking Environments

This repository implements a Micro-KDN research prototype for translating natural-language networking intents into Juniper Junos CLI commands under edge-compute constraints.

The project began as an end-to-end Llama-3-8B command generator and evolved into the current final architecture:

```text
User intent + optional context
        |
Fine-tuned Llama-3-8B semantic parser
        |
Strict semantic JSON
        |
Local command-context/template store
        |
Deterministic Python assembler
        |
Mode/commit/placeholder guardrails
        |
Final Junos CLI
```

The final research path is `--semantic_rag`: the model parses intent, while Python retrieves command metadata and assembles the final CLI. The model is not responsible for rendering final commands in the final architecture.

## Research Goal

Evaluate whether a local SLM, fine-tuned with LoRA and deployable with 4-bit quantization, can support a reliable Knowledge-Defined Networking control plane without depending on cloud LLM APIs.

Core target:
- Base model: `meta-llama/Meta-Llama-3-8B`
- Fine-tuning: LoRA / QLoRA
- Domain: Juniper Junos CLI, especially EX3300-style NIT data
- Target metric: >94% normalized exact match
- Constraints: deterministic edge operation, no external vector database, no cloud inference dependency

## Current Architecture

### 1. Semantic Parser

`src/infer.py --semantic_rag` prompts the fine-tuned Llama-3-8B adapter to output only strict JSON:

```json
{"action":"...","domain":"...","sub_domain":"...","operation":"...","parameters":{}}
```

`src/semantic_parser.py` repairs common model failure modes:
- strips markdown fences and prefixes such as `Command:` or `Output:`
- extracts JSON between the first `{` and last `}`
- ignores trailing text such as `commit`
- repairs full CLI strings placed in `action`
- infers `domain`, `sub_domain`, and `operation`
- extracts obvious parameters from command-like model output

### 2. Command Context Store

`src/rag_store.py` loads `data/juniper_templates.json` as a local command knowledge store. It stores command metadata rather than raw prompt context:

- `action`
- `domain`
- `sub_domain`
- `operation`
- `template`
- `mode`
- `requires_commit`
- `variant`
- `allowed_params`
- `required_params`
- `positive_cues`
- `negative_cues`
- `validation_rules`

`src/command_context.py` retrieves candidate templates by command family and then selects the correct operation-level template using deterministic scoring.

### 3. Deterministic Assembly

`src/parameter_binding.py` binds values from model JSON plus the original intent/context into template placeholders. It handles both Python placeholders and Junos-style angle placeholders:

```text
{interface}
<interface-name>
<vlan-name-or-id>
<rate>
<limit>
```

If a placeholder cannot be resolved, assembly fails with `missing_parameter` rather than emitting an invalid command containing `<...>`.

### 4. Guardrails

`src/guardrails.py` enforces final command validity:

- operational commands never receive `commit`
- configuration commands receive exactly one `commit` only when metadata requires it
- duplicate commits are collapsed
- literal newline formatting is normalized for evaluation compatibility
- markdown, prefixes, and model explanations are removed if they appear

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Preprocess the NIT dataset:

```bash
scripts/run_preprocess.sh
```

Train the LoRA adapter:

```bash
scripts/run_train.sh
```

Build the semantic-RAG command store from train/val only:

```bash
python scripts/build_perfect_datastore_v2.py
```

Run the final semantic-RAG evaluation:

```bash
scripts/run_semantic_rag_eval.sh
```

Analyze failures:

```bash
python scripts/analyze_semantic_rag_errors.py
```

Run lightweight semantic-RAG tests:

```bash
python tests/test_semantic_rag.py
```

If `pytest` is available:

```bash
python -m pytest tests/test_semantic_rag.py
```

## Colab Workflows

Use this notebook for the final architecture:

```text
notebooks/Colab_A100_Semantic_RAG_Micro_KDN.ipynb
```

Use this notebook for legacy RAG/baseline comparison:

```text
notebooks/Colab_A100_RAG_Train_Test.ipynb
```

Recommended Colab setup:
- Colab Pro A100 runtime
- Hugging Face token with access to `meta-llama/Meta-Llama-3-8B`
- run preprocessing before building the datastore
- do not pass `--allow-test` to the datastore builder for final evaluation

Large Colab run artifacts are stored outside git:

```text
https://drive.google.com/drive/folders/1XPHHBGRtsawkQ0yURVP18A6055VgSD2e?usp=sharing
```

The repository keeps only lightweight evaluation metrics from generated runs. Model adapters, checkpoints, optimizer states, predictions, RAG indexes, and copied Colab result bundles should remain in Google Drive or local storage.

## Main Commands

### Legacy Direct Generation

```bash
python src/infer.py \
  --input_file data/processed/test.jsonl \
  --output_file results/predictions/predictions.jsonl \
  --mode intent_with_context

python src/evaluate.py --pred_file results/predictions/predictions.jsonl
```

### Legacy Free-form RAG Baseline

```bash
python src/infer.py \
  --input_file data/processed/test.jsonl \
  --output_file results/predictions/rag_predictions_baseline.jsonl \
  --use_rag \
  --rag-corpus train,rag_docs

python src/evaluate.py \
  --pred_file results/predictions/rag_predictions_baseline.jsonl \
  --out_file results/metrics/rag_baseline_eval_metrics.json
```

### Final Semantic-RAG Path

```bash
python scripts/build_perfect_datastore_v2.py

python src/infer.py \
  --input_file data/processed/test.jsonl \
  --output_file results/predictions/semantic_rag_predictions.jsonl \
  --semantic_rag \
  --mode intent_with_context

python scripts/evaluate_semantic_rag.py \
  --pred_file results/predictions/semantic_rag_predictions.jsonl \
  --out_file results/metrics/semantic_rag_metrics.json \
  --failures_file results/error_analysis/semantic_rag_failures.jsonl \
  --summary_file results/error_analysis/semantic_rag_error_summary.json
```

### RAG Retrieval Diagnostics

```bash
python src/rag_query.py --rebuild-index --rag-corpus train,rag_docs --regression
python scripts/check_rag_corpus_coverage.py
```

## Evaluation Outputs

Semantic-RAG writes:

```text
results/predictions/semantic_rag_predictions.jsonl
results/metrics/semantic_rag_metrics.json
results/error_analysis/semantic_rag_failures.jsonl
results/error_analysis/semantic_rag_error_summary.json
```

Archived RAG and semantic-RAG Colab outputs are available in the project Google Drive folder:

```text
https://drive.google.com/drive/folders/1XPHHBGRtsawkQ0yURVP18A6055VgSD2e?usp=sharing
```

Only metric files from these runs are intended to be tracked in git:

```text
results/metrics/**
rag-results/results-rag-*/metrics/**
rag-results/semantic-rag-results-*/metrics/**
```

The semantic-RAG evaluator reports stage-aware metrics:

Semantic parsing:
- `json_valid_rate`
- `repaired_parse_rate`
- `unrepaired_parse_error_rate`
- `action_accuracy`
- `domain_accuracy`
- `sub_domain_accuracy`
- `semantic_frame_exact_match`
- `operation_accuracy`
- `operation_inferred_rate`
- `parameter_precision`
- `parameter_recall`
- `parameter_f1`
- `entity_preservation`

Template/context retrieval:
- `template_hit_rate`
- `family_hit_rate`
- `variant_hit_rate`
- `template_variant_accuracy`
- `template_not_found_rate`
- `ambiguous_template_rate`

Assembly and guardrails:
- `assembly_success_rate`
- `missing_parameter_rate`
- `unresolved_placeholder_rate`
- `commit_decision_accuracy`
- `commit_false_positive_rate`
- `commit_false_negative_rate`
- `guardrail_application_rate`

Final command:
- `raw_exact_match`
- `exact_match`
- `normalized_exact_match`
- `token_f1`
- `valid_rate`
- `invalid_output_rate`
- `bleu`

Failure stages:
- `semantic_parse_error`
- `template_not_found`
- `wrong_template`
- `missing_parameter`
- `commit_error`
- `final_command_mismatch`

Important evaluator fix: semantic-RAG command matching now normalizes literal `\\n` and real newlines before normalized exact match and failure staging. This avoids undercounting rows where the prediction stores literal `\\ncommit` and the target stores an actual newline.

## Chronological Research Progress

### Phase 1: Direct Llama-3-8B CLI Generation

The first system fine-tuned Llama-3-8B with LoRA to generate final Junos CLI commands directly.

Observed direct-generation baseline:

| Metric | Approx. result |
| --- | ---: |
| True normalized exact match | 65.33% |
| Token F1 | 84.7% |
| BLEU | 0.707 |
| Entity preservation | 99.33% |
| Valid rate | 93.33% |

Key finding: the model preserved entities extremely well but was brittle at exact command rendering.

### Phase 2: Commit Guardrail

The model frequently appended `commit` to operational commands such as:

```text
show chassis led
commit
```

This revealed that the model learned Junos syntax patterns without reliably distinguishing operational mode from configuration mode.

Guardrails were added to remove commit from read-only commands and preserve it only for configuration actions. This supported the hybrid Micro-KDN thesis: use the SLM for language understanding and deterministic code for command safety.

### Phase 3: Batch Inference Optimization

Sequential inference on A100 was too slow. `src/infer.py` was refactored for batched inference:

- `inference.batch_size: 32`
- left padding for batched generation
- `tqdm` progress tracking

Result: test-set inference dropped from roughly one hour to under three minutes on A100.

### Phase 4: Quantization and Edge Simulation

4-bit inference on an AWS G4/T4-style edge environment exposed a quantization tax:

| Metric | Approx. result |
| --- | ---: |
| Exact match | 29.33% |
| Token F1 | 89.7% |
| Entity preservation | 99.33% |

Interpretation: quantization hurt exact syntax more than semantic/entity extraction. This strengthened the decision to make the SLM a semantic parser rather than a final CLI renderer.

### Phase 5: Robustness Testing

Robustness sets such as clean, paraphrased, and noisy inputs showed that exact command generation dropped sharply under input variation, while entity preservation remained near 99.33%.

Conclusion: direct sequence-to-sequence command generation is fragile for strict network automation, but the model is robust as an intent/entity parser.

### Phase 6: Initial Semantic-RAG Schema Failure

The first semantic-RAG attempt failed because the model and dataset still used an old schema:

```json
{"action":"show","target":"","target_type":"unknown","parameters":{}}
```

The new pipeline expected:

```json
{"action":"...","domain":"...","sub_domain":"...","parameters":{}}
```

This produced 0% exact match and 100% invalid output in that attempt. The training/evaluation schema was then aligned around `action`, `domain`, `sub_domain`, and `parameters`.

### Phase 7: Raw RAG Over Train/PDF Context

The next system tried:

```text
Intent -> RAG context -> Llama generates final CLI
```

RAG indexed:
- processed NIT train rows
- `rag-doc/ex3300.pdf`

Evaluation leakage was discovered because the retriever could index `test.jsonl`. The corpus logic was fixed:

- strict default: `train,rag_docs`
- relaxed mode: `train,val,rag_docs`
- `test.jsonl` is never included by default
- strict mode treats `val.jsonl` as leakage
- runtime guard raises `Evaluation leakage detected`

Representative post-fix RAG result:

| Metric | Approx. result |
| --- | ---: |
| Exact match | 18-19% |
| Normalized exact match | 60-64% |
| Token F1 | 84-87% |
| Entity preservation | 98% |

Finding: RAG as raw prompt context improved semantic similarity but did not reliably produce exact CLI.

### Phase 8: Hybrid Retrieval and Reranker Diagnostics

Dense retrieval confused neighboring networking concepts:

- OSPF neighbor vs LLDP neighbor
- disable protocol vs show interface
- clear MAC table vs clear port-error/BPDU-error

The retriever was upgraded with:

- dense TF-IDF and lexical TF-IDF candidate pools
- command-aware scoring
- contradiction penalties
- path/object/value boosts
- source leakage guards
- corpus coverage diagnostics
- candidate recall diagnostics
- query normalization
- optional high-confidence template fallback

Key scripts:

```text
src/rag.py
src/rag_query.py
scripts/check_rag_corpus_coverage.py
```

This phase showed that blind reranker tuning was not enough when the correct command was missing from the candidate pool or corpus.

### Phase 9: Architectural Pivot to Semantic-RAG

The final architecture changed from:

```text
Intent -> RAG -> Llama generates final CLI
```

to:

```text
Intent -> Llama semantic JSON -> command context store -> deterministic assembler -> guardrails
```

Implemented modules:

| Module | Role |
| --- | --- |
| `src/semantic_parser.py` | parse and repair strict semantic JSON |
| `src/rag_store.py` | load local command templates and metadata |
| `src/command_context.py` | retrieve and score command contexts |
| `src/parameter_binding.py` | extract and bind parameters into templates |
| `src/guardrails.py` | enforce mode/commit validity |
| `scripts/evaluate_semantic_rag.py` | stage-aware semantic-RAG evaluation |
| `scripts/analyze_semantic_rag_errors.py` | semantic-RAG error analysis |
| `scripts/build_perfect_datastore_v2.py` | build command store from train/val |

### Phase 10: Semantic Parser Repair

The model often produced JSON-shaped command output such as:

```json
{"action":"show chassis led","domain":"chassis","sub_domain":"led","parameters":{}}
```

The parser now repairs these cases:

- `show chassis led` -> `action=show`, `domain=chassis`, `sub_domain=led`
- `set protocols sflow traceoptions flag all` -> `operation=traceoptions_flag_enable`
- `clear ethernet-switching-table` -> `operation=clear_table`

Recent observed semantic-RAG parser metrics after repair:

| Metric | Approx. result |
| --- | ---: |
| JSON validity | 91% |
| Semantic frame exact match | 88% |
| Template hit rate | 89% |
| Assembly success | 80% |

### Phase 11: Operation-Level Template Selection

The first semantic-RAG command store keyed templates too coarsely by:

```text
action/domain/sub_domain
```

For example, `set/protocols/sflow` can mean many operations:

- traceoptions flag enable
- traceoptions flag disable
- sample-rate ingress
- sample-rate egress
- polling interval
- source IP
- interface enable

`TemplateRecord` was extended with:

- `operation`
- `positive_cues`
- `negative_cues`
- `required_params`
- `forbidden_params`

Template selection now scores operation-level variants. This prevents selecting templates such as a `disable` variant when the intent only asks to trace all events.

### Phase 12: Placeholder Binding and Exact-Match Normalization

Many final mismatches were assembled templates with unresolved placeholders, such as:

```text
set protocols sflow sample-rate egress <rate>
```

`src/parameter_binding.py` now:

- treats angle placeholders as required parameters
- normalizes placeholder names
- extracts parameters from intent and context
- prefers concrete values over placeholder defaults
- returns `missing_parameter` instead of emitting `<...>`

Examples now handled:

```text
"set the sflow egress sampling rate to 1000"
-> set protocols sflow sample-rate egress 1000\ncommit

"set a mac moving limit of 2 on vlan HR"
-> set ethernet-switching-options secure-access-port vlan HR mac-move-limit 2\ncommit

"put a mac limit of 1 on interface ge-0/0/15 and log..."
-> set ethernet-switching-options secure-access-port interface ge-0/0/15 mac-limit 1 action log\ncommit
```

The evaluator also now reports both raw and normalized exact match. Normalized exact match handles literal `\\n` vs real newline differences.

## Data Leakage Policy

Final datastore construction uses train/val only by default:

```bash
python scripts/build_perfect_datastore_v2.py
```

`test.jsonl` is excluded unless explicitly requested:

```bash
python scripts/build_perfect_datastore_v2.py --allow-test
```

Do not use `--allow-test` for final reported metrics.

## Known Limitations

- Full semantic-RAG performance depends on a datastore built after preprocessing.
- If train/val do not contain a command family, semantic-RAG returns `template_not_found`.
- Exact operation inference is rule-based and will need more coverage as new Junos families are added.
- The current datastore is a local command knowledge store, not a documentation-only RAG corpus.
- The legacy `--use_rag` path is retained for ablation and comparison, but it is not the final architecture.

## Repository Map

```text
config.yaml
requirements.txt
src/
  infer.py
  train_lora.py
  preprocess.py
  rag.py
  rag_query.py
  rag_store.py
  semantic_parser.py
  command_context.py
  parameter_binding.py
  guardrails.py
  evaluate.py
  validate_output.py
scripts/
  run_preprocess.sh
  run_train.sh
  run_eval.sh
  run_robustness.sh
  run_semantic_rag_eval.sh
  build_perfect_datastore_v2.py
  evaluate_semantic_rag.py
  analyze_semantic_rag_errors.py
  check_rag_corpus_coverage.py
notebooks/
  Colab_A100_Semantic_RAG_Micro_KDN.ipynb
  Colab_A100_RAG_Train_Test.ipynb
rag-doc/
  ex3300.pdf
tests/
  test_semantic_rag.py
```

## Research Takeaway

The main result is architectural: the local Llama-3-8B adapter is strongest as a semantic parser, not as the final CLI renderer. Direct generation preserves entities but remains fragile under quantization, paraphrase, and strict syntax. Semantic-RAG converts the problem into structured parsing plus deterministic command assembly, making the system more suitable for reliable edge KDN operation.
