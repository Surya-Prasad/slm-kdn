from __future__ import annotations

import re
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

from .juniper_schema import JuniperTemplate

_ACTION_PREFIXES = ("set ", "delete ", "show ", "request ")


def _to_snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parameterize_command(raw_command: str) -> tuple[str, dict]:
    defaults = {}

    def repl(match: re.Match) -> str:
        variable = _to_snake(match.group(1))
        defaults[variable] = None
        return "{" + variable + "}"

    template = re.sub(r"<\s*([^>]+?)\s*>", repl, " ".join(raw_command.split()))
    return template, defaults


def infer_domain_subdomain(template: str) -> tuple[str, str]:
    parts = template.split()
    domain = parts[1] if len(parts) > 1 else "unknown"
    sub_domain = parts[2] if len(parts) > 2 else "general"
    return domain, sub_domain


def extract_cli_templates_from_html(html: str) -> List[JuniperTemplate]:
    soup = BeautifulSoup(html, "html.parser")
    templates: List[JuniperTemplate] = []

    for tag in soup.find_all(["code", "pre"]):
        text = " ".join(tag.get_text(" ").split()).strip()
        if not text:
            continue
        lowered = text.lower()
        if not lowered.startswith(_ACTION_PREFIXES):
            continue

        template, defaults = parameterize_command(text)
        action = template.split()[0].lower()
        domain, sub_domain = infer_domain_subdomain(template)
        templates.append(
            JuniperTemplate(
                action=action,
                domain=domain,
                sub_domain=sub_domain,
                template=template,
                requires_commit=action in {"set", "delete"},
                default_params=defaults,
            )
        )
    return templates


def scrape_cli_reference(urls: Iterable[str], timeout: int = 20) -> List[JuniperTemplate]:
    out: List[JuniperTemplate] = []
    for url in urls:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        out.extend(extract_cli_templates_from_html(response.text))
    return out
