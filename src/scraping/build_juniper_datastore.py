from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import write_jsonl
from scraping.juniper_cli_explorer_scraper import scrape_cli_explorer_paths
from scraping.juniper_cli_reference_scraper import scrape_cli_reference
from scraping.juniper_pdf_context_scraper import build_training_context


def _template_key(template: str) -> str:
    parts = template.split()
    action = parts[0] if parts else "unknown"
    domain = parts[1] if len(parts) > 1 else "general"
    sub_domain = parts[2] if len(parts) > 2 else "general"
    return f"{action}_{domain}_{sub_domain}".replace("-", "_")


def main(args):
    templates = scrape_cli_reference(args.cli_reference_urls)
    explorer_paths = scrape_cli_explorer_paths(args.cli_explorer_url)

    explorer_index = {(p[0], p[1]): p for p in explorer_paths if len(p) >= 2}

    payload = {}
    for t in templates:
        key = _template_key(t.template)
        domain = t.domain
        sub_domain = t.sub_domain
        match = explorer_index.get((domain, sub_domain))
        if match:
            domain, sub_domain = match[0], match[1]

        payload[key] = {
            "action": t.action,
            "domain": domain,
            "sub_domain": sub_domain,
            "template": t.template,
            "requires_commit": t.requires_commit,
            "default_params": t.default_params,
        }

    out_json = Path(args.templates_output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    contexts = build_training_context(args.pdf_urls, cache_dir=args.pdf_cache_dir)
    write_jsonl(args.training_context_output, contexts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cli_reference_urls", nargs="+", required=True)
    p.add_argument("--cli_explorer_url", default="https://apps.juniper.net/cli-explorer/")
    p.add_argument("--pdf_urls", nargs="+", required=True)
    p.add_argument("--templates_output", default="data/juniper_templates.json")
    p.add_argument("--training_context_output", default="data/training_context.jsonl")
    p.add_argument("--pdf_cache_dir", default="data/pdfs")
    main(p.parse_args())
