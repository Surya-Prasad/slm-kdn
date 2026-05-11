import argparse
import json
from pathlib import Path

from rag import (
    apply_rag_corpus,
    assert_no_eval_leakage,
    build_rag_prompt,
    extractive_answer,
    format_retrieval_debug,
    get_or_build_index,
)
from utils import load_config


EXAMPLE_QUERIES = [
    "What are the front panel ports on a Juniper EX3300 switch?",
    "How do I interpret the LEDs on an EX3300?",
    "What power supply information does the EX3300 guide provide?",
    "What are the console port details for EX3300?",
]

SANITY_QUERIES = [
    ("How to display the LED status", None),
    ("Show sFlow protocol configurations in the operational mode", None),
    ("What are the front panel ports on an EX3300 switch?", "ex3300.pdf"),
]

REGRESSION_CASES = [
    ("how to disable the OSPF protocol", "set protocols ospf disable", ("set interfaces <interface-name> disable", "show ospf interface", "show ospf route")),
    ("Clear all MAC address entries in the ethernet switching table", "clear ethernet-switching-table", ("clear ethernet-switching port-error", "clear ethernet-switching bpdu-error")),
    ("Display information about OSPF neighbors", "show ospf neighbor", ("show lldp neighbors", "show lldp neighbor")),
    ("create SNMP community with the name CAMPUS-COMMUNITY and set authorization to read only", "read-only", ("read-write",)),
    ("Display IGMP snooping detailed flows information", "show igmp-snooping flows detail", None),
    ("Notify all logged users when any emergency level event occurs", "set system syslog user * any emergency", ("show log user", "kernel any", "ntp any")),
    ("Show LCD active menu items", None, None),
]


def _contains_expected(chunks, expected):
    if not expected:
        return True
    expected = expected.lower()
    for chunk in chunks:
        meta = chunk.metadata or {}
        target = str(meta.get("target_command", ""))
        if expected in target.lower() or expected in chunk.text.lower():
            return True
    return False


def _source_present(chunks, source_file):
    return any((chunk.metadata or {}).get("source_file") == source_file for chunk in chunks)


def main(args):
    cfg = load_config(args.config)
    apply_rag_corpus(cfg, args.rag_corpus)
    index = get_or_build_index(cfg, rebuild=args.rebuild or args.rebuild_index)
    queries = args.query or EXAMPLE_QUERIES
    top_k = args.top_k or int(cfg.get("rag", {}).get("top_k", 5))
    strict = not cfg.get("rag", {}).get("include_val_in_rag", False)

    if args.sanity:
        for query, required_source in SANITY_QUERIES:
            chunks = index.retrieve(query, top_k=top_k)
            print("=" * 80)
            print(format_retrieval_debug(query, chunks))
            assert_no_eval_leakage(chunks, strict=strict)
            sources = {chunk.metadata.get("source_file") for chunk in chunks}
            if required_source and required_source not in sources:
                raise RuntimeError(f"Expected retrieval source {required_source} for query: {query}")
        print("[RAG] sanity checks passed: test.jsonl was not retrieved.")
        return

    if args.regression:
        has_train = Path(cfg["data"]["output_dir"], "train.jsonl").exists()
        regression_results = []
        for query, expected, bad_patterns in REGRESSION_CASES:
            chunks = index.retrieve(query, top_k=5)
            print("=" * 80)
            print(format_retrieval_debug(query, chunks))
            assert_no_eval_leakage(chunks, strict=strict)
            passed = True
            error = None
            if expected and has_train and not _contains_expected(chunks, expected):
                passed = False
                error = f"Expected top-5 to contain `{expected}`"
            if bad_patterns and has_train:
                targets = [str((chunk.metadata or {}).get("target_command", "")).lower() for chunk in chunks]
                bad_rank = 0 if targets and any(bad in targets[0] for bad in bad_patterns) else None
                good_rank = next((i for i, target in enumerate(targets) if expected and expected in target), None)
                if bad_rank is not None and (good_rank is None or bad_rank < good_rank):
                    passed = False
                    error = f"Forbidden top-1 outranked `{expected}`"
            if "lcd" in query.lower() and not _source_present(chunks, "ex3300.pdf"):
                passed = False
                error = "Expected ex3300.pdf for LCD query"
            if "led" in query.lower() and not (_source_present(chunks, "ex3300.pdf") or _contains_expected(chunks, "show chassis led")):
                passed = False
                error = "Expected ex3300.pdf LED page or show chassis led example"
            record = {
                "query": query,
                "passed": passed,
                "error": error,
                "expected_phrases": [expected] if expected else [],
                "forbidden_top1_phrases": list(bad_patterns or []),
                "top5_sources": [chunk.metadata.get("source_file") for chunk in chunks],
                "top5_previews": [str(chunk.metadata.get("target_command") or chunk.text).replace("\n", " ")[:240] for chunk in chunks],
                "top5_scores": [
                    {"score": chunk.score, "dense": chunk.dense_score, "lexical": chunk.lexical_score}
                    for chunk in chunks
                ],
            }
            regression_results.append(record)
            if not passed:
                Path("outputs").mkdir(exist_ok=True)
                Path("outputs/retrieval_regression.json").write_text(json.dumps(regression_results, indent=2), encoding="utf-8")
                raise RuntimeError(f"{error} for query: {query}")
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/retrieval_regression.json").write_text(json.dumps(regression_results, indent=2), encoding="utf-8")
        print("[RAG] regression checks passed.")
        return

    for query in queries:
        chunks = index.retrieve(query, top_k=top_k)
        assert_no_eval_leakage(chunks, strict=strict)
        print("=" * 80)
        print(format_retrieval_debug(query, chunks))
        print("\n[RAG] prompt:\n")
        print(build_rag_prompt(query, chunks))
        print("\n[RAG] final answer preview:\n")
        print(extractive_answer(query, chunks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval over NIT data and rag-doc/ documents.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", action="append", help="Question to retrieve for. Repeat for multiple queries.")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the local RAG index.")
    parser.add_argument("--rebuild-index", dest="rebuild_index", action="store_true", help="Alias for --rebuild.")
    parser.add_argument("--rag-corpus", dest="rag_corpus", default=None, help="Comma-separated corpus, e.g. train,rag_docs or train,val,rag_docs.")
    parser.add_argument("--sanity", action="store_true", help="Run retrieval leakage and EX3300 sanity checks.")
    parser.add_argument("--regression", action="store_true", help="Run command-aware retrieval regression checks.")
    main(parser.parse_args())
