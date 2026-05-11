from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pdfplumber
import requests


def download_pdf(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / Path(url).name
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    file_path.write_bytes(resp.content)
    return file_path


def _clean_text(raw: str) -> str:
    lines = [ln.strip() for ln in raw.splitlines()]
    filtered = [ln for ln in lines if ln and not re.fullmatch(r"\d+", ln)]
    filtered = [ln for ln in filtered if not re.search(r"copyright|juniper networks", ln, flags=re.I)]
    return "\n".join(filtered)


def extract_context_chunks(pdf_path: Path, min_chunk_len: int = 240) -> List[str]:
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            cleaned = _clean_text(page.extract_text() or "")
            if not cleaned:
                continue
            for part in re.split(r"\n\s*\n+", cleaned):
                paragraph = " ".join(part.split())
                if len(paragraph) >= min_chunk_len:
                    chunks.append(paragraph)
    return chunks


def build_training_context(pdf_urls: Iterable[str], cache_dir: str = "data/pdfs") -> List[dict]:
    out = []
    root = Path(cache_dir)
    for url in pdf_urls:
        local = download_pdf(url, root)
        for idx, chunk in enumerate(extract_context_chunks(local)):
            out.append({"source_url": url, "source_file": str(local), "chunk_id": idx, "text": chunk})
    return out
