import argparse

from rag import build_rag_prompt, extractive_answer, format_retrieval_debug, get_or_build_index
from utils import load_config


EXAMPLE_QUERIES = [
    "What are the front panel ports on a Juniper EX3300 switch?",
    "How do I interpret the LEDs on an EX3300?",
    "What power supply information does the EX3300 guide provide?",
    "What are the console port details for EX3300?",
]


def main(args):
    cfg = load_config(args.config)
    index = get_or_build_index(cfg, rebuild=args.rebuild)
    queries = args.query or EXAMPLE_QUERIES
    top_k = args.top_k or int(cfg.get("rag", {}).get("top_k", 5))

    for query in queries:
        chunks = index.retrieve(query, top_k=top_k)
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
    main(parser.parse_args())
