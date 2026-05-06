import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_config(path: str = "micro_kdn_llama/config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def normalize_command(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.;,:!?]+$", "", text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())
