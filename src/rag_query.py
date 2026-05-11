import argparse

from rag import assert_no_test_leakage, build_rag_prompt, extractive_answer, format_retrieval_debug, get_or_build_index
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


def main(args):
    cfg = load_config(args.config)
    index = get_or_build_index(cfg, rebuild=args.rebuild or args.rebuild_index)
    queries = args.query or EXAMPLE_QUERIES
    top_k = args.top_k or int(cfg.get("rag", {}).get("top_k", 5))

    if args.sanity:
        for query, required_source in SANITY_QUERIES:
            chunks = index.retrieve(query, top_k=top_k)
            print("=" * 80)
            print(format_retrieval_debug(query, chunks))
            assert_no_test_leakage(chunks)
            sources = {chunk.metadata.get("source_file") for chunk in chunks}
            if required_source and required_source not in sources:
                raise RuntimeError(f"Expected retrieval source {required_source} for query: {query}")
        print("[RAG] sanity checks passed: test.jsonl was not retrieved.")
        return

    for query in queries:
        chunks = index.retrieve(query, top_k=top_k)
        assert_no_test_leakage(chunks)
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
    parser.add_argument("--sanity", action="store_true", help="Run retrieval leakage and EX3300 sanity checks.")
    main(parser.parse_args())
