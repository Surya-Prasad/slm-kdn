import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag import normalize_command_for_match, normalize_retrieval_query  # noqa: E402
from utils import load_config, read_jsonl  # noqa: E402


EXPECTED_PATTERNS = [
    "set protocols ospf disable commit",
    "clear ethernet-switching-table",
    "show igmp-snooping flows detail",
    "set system syslog user * any emergency commit",
    "show chassis pic-mode all-members",
    "show chassis lcd menu",
]


def command_terms(command):
    normalized = normalize_command_for_match(command)
    terms = re.findall(r"[\w*/-]+", normalized)
    stop = {"set", "show", "clear", "delete", "commit", "user", "any", "*"}
    return [term for term in terms if term not in stop]


def similarity(left, right):
    return SequenceMatcher(None, normalize_command_for_match(left), normalize_command_for_match(right)).ratio()


def row_record(row, source_file, index, score=None):
    record = {
        "source_file": source_file,
        "line_number": index,
        "record_index": index,
        "target_command": row.get("target_command", ""),
        "intent": row.get("intent", ""),
        "context": row.get("context", ""),
    }
    if score is not None:
        record["similarity"] = round(score, 4)
    return record


def check_pattern(pattern, rows, source_file):
    expected_norm = normalize_command_for_match(pattern)
    exact_matches = []
    close_matches = []
    text_matches = []
    terms = command_terms(pattern)

    for index, row in enumerate(rows, start=1):
        target = row.get("target_command", "")
        target_norm = normalize_command_for_match(target)
        if target_norm == expected_norm:
            exact_matches.append(row_record(row, source_file, index))

        score = similarity(target, pattern)
        shared_terms = sum(1 for term in terms if term in target_norm)
        if target_norm and (score >= 0.62 or shared_terms >= max(2, min(3, len(terms)))):
            close_matches.append(row_record(row, source_file, index, score=score))

        intent_context = normalize_retrieval_query(f"{row.get('intent', '')} {row.get('context', '')}").lower()
        if terms and sum(1 for term in terms if term in intent_context) >= max(2, min(3, len(terms))):
            text_matches.append(row_record(row, source_file, index))

    close_matches.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return {
        "pattern": pattern,
        "exact_target_command_match_found": bool(exact_matches),
        "exact_matches": exact_matches,
        "close_command_matches": close_matches[:10],
        "matching_intent_context_rows": text_matches[:10],
        "source_file": source_file,
    }


def main(args):
    cfg = load_config(args.config)
    train_path = Path(args.train_file) if args.train_file else ROOT / cfg["data"]["output_dir"] / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train corpus: {train_path}")

    rows = read_jsonl(str(train_path))
    source_file = str(train_path.relative_to(ROOT)) if train_path.is_relative_to(ROOT) else str(train_path)
    results = [check_pattern(pattern, rows, source_file) for pattern in EXPECTED_PATTERNS]

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether expected RAG command patterns exist in train.jsonl.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-file", default=None, help="Override the processed train.jsonl path.")
    parser.add_argument("--output", default="outputs/rag_corpus_coverage.json")
    main(parser.parse_args())
