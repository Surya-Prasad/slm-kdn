import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import read_jsonl


RAG_PROMPT_TEMPLATE = """You are a network intent translation assistant. Convert the user intent into the correct Juniper Junos CLI command.

Use retrieved NIT examples for command pattern matching.
Use retrieved documentation for hardware/device-specific grounding.
Prefer the Juniper EX3300 hardware config guide when the question is about EX3300 hardware, ports, LEDs, power, installation, interfaces, LCDs, or CLI/config behavior.

Rules:
- Output only the final CLI command.
- Do not explain.
- Do not include source citations in the answer.
- If the retrieved context does not contain enough information for a CLI command, output the best command implied by the user intent and NIT examples.

Retrieved NIT examples:
{examples}

Retrieved documentation:
{docs}

Question:
{question}

Answer:
"""


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROTOCOL_TERMS = ("ospf", "rstp", "igmp", "igmp-snooping", "sflow", "snmp", "lldp")
ACTION_TERMS = (
    "show",
    "display",
    "get",
    "determine",
    "set",
    "configure",
    "enable",
    "create",
    "make",
    "put",
    "block",
    "notify",
    "disable",
    "delete",
    "remove",
    "clear",
    "override",
    "load",
    "restore",
    "trace",
)
COMMAND_NOUNS = (
    "neighbor",
    "neighbors",
    "interface",
    "route",
    "flow",
    "flows",
    "detail",
    "syslog",
    "trap-group",
    "community",
    "authorization",
    "mac",
    "vlan",
    "telemetry",
)
HARDWARE_TERMS = (
    "ex3300",
    "front panel",
    "rear panel",
    "lcd",
    "led",
    "port",
    "console",
    "management port",
    "power",
    "poe",
    "rack",
    "mounting",
    "chassis",
)


def apply_rag_corpus(cfg: Dict[str, Any], corpus: Optional[str]) -> Dict[str, Any]:
    if not corpus:
        corpus = cfg.get("rag", {}).get("corpus")
    if not corpus:
        return cfg
    tokens = {part.strip().lower() for part in corpus.split(",") if part.strip()}
    rag_cfg = cfg.setdefault("rag", {})
    rag_cfg["corpus"] = ",".join(sorted(tokens))
    rag_cfg["include_train_in_rag"] = "train" in tokens
    rag_cfg["include_val_in_rag"] = "val" in tokens
    rag_cfg["include_test_in_rag"] = "test" in tokens
    rag_cfg["include_rag_docs"] = "rag_docs" in tokens or "rag-docs" in tokens or "docs" in tokens
    return cfg


def _has_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.I) is not None


def _matched_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if _has_term(text, term)]


def _is_hardware_query(query: str) -> bool:
    return bool(_matched_terms(query, HARDWARE_TERMS))


def _norm_query(text: str) -> str:
    text = text.replace("read only", "read-only").replace("read write", "read-write")
    fixes = {
        "igmp snooping": "igmp-snooping",
        "mac moving": "mac-move",
        "taceoptions": "traceoptions",
        "systme": "system",
        "interfacs": "interfaces",
        "dipslay": "display",
    }
    for bad, good in fixes.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text, flags=re.I)
    return text


def _infer_query_action(query: str) -> str:
    q = _norm_query(query)
    action_groups = [
        ("delete", ("remove", "delete")),
        ("disable", ("disable",)),
        ("clear", ("clear",)),
        ("show", ("show", "display", "get", "determine")),
        ("set", ("set", "configure", "enable", "create", "make", "put", "block", "notify")),
        ("load", ("override", "load", "restore")),
    ]
    for action, terms in action_groups:
        if any(_has_term(q, term) for term in terms):
            return action
    return ""


def _command_action(command: str) -> str:
    first = command.strip().split(maxsplit=1)[0].lower() if command.strip() else ""
    if first in {"set", "delete", "show", "clear", "load", "request"}:
        return first
    return ""


def _extract_values(text: str) -> List[str]:
    text = _norm_query(text)
    values = set()
    patterns = [
        r"\b[a-z]{2}-\d+/\d+/\d+\b",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\bvlan\s+([A-Za-z][A-Za-z0-9_-]*)\b",
        r"\b[A-Z][A-Z0-9_-]{2,}\b",
        r"\b\d+\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            values.add(match if isinstance(match, str) else match[0])
    for term in ("emergency", "kernel", "any", "all", "ingress", "egress", "read-only", "read-write"):
        if _has_term(text, term):
            values.add(term)
    for username in re.findall(r"\b(?:user|username|notify user)\s+([A-Za-z][A-Za-z0-9_-]*)\b", text, flags=re.I):
        values.add(username)
    return sorted(v.lower() for v in values if v)


def _minmax(values):
    if len(values) == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return values * 0
    return (values - low) / (high - low)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def load_rag_doc(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires pypdf. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        reader = PdfReader(str(path))
        docs = []
        for idx, page in enumerate(reader.pages, start=1):
            text = _normalize_text(page.extract_text() or "")
            if text.startswith("Table of Contents"):
                continue
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(path),
                            "source_file": path.name,
                            "page": idx,
                            "doc_type": "rag-doc",
                        },
                    )
                )
        return docs

    if suffix in {".txt", ".md", ".markdown"}:
        return [
            Document(
                text=_normalize_text(path.read_text(encoding="utf-8")),
                metadata={
                    "source": str(path),
                    "source_file": path.name,
                    "page": None,
                    "doc_type": "rag-doc",
                },
            )
        ]

    return []


def load_rag_documents(root: Path, rag_dir: str) -> List[Document]:
    base = root / rag_dir
    if not base.exists():
        return []
    docs: List[Document] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}:
            loaded = load_rag_doc(path)
            for doc in loaded:
                doc.metadata["source"] = str(path.relative_to(root))
            docs.extend(loaded)
    return docs


def nit_splits_for_rag(cfg: Dict[str, Any]) -> List[str]:
    apply_rag_corpus(cfg, None)
    rag_cfg = cfg.get("rag", {})
    splits = []
    if rag_cfg.get("include_train_in_rag", True):
        splits.append("train")
    if rag_cfg.get("include_val_in_rag", True):
        splits.append("val")
    if rag_cfg.get("include_test_in_rag", False):
        splits.append("test")
    return splits


def load_nit_documents(root: Path, data_dir: str, splits: Iterable[str]) -> List[Document]:
    docs: List[Document] = []
    base = root / data_dir
    for split in splits:
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue
        for idx, row in enumerate(read_jsonl(str(path)), start=1):
            parts = [
                f"Intent: {row.get('intent', '')}",
                f"Context: {row.get('context', '')}",
                f"Target command: {row.get('target_command', '')}",
            ]
            text = _normalize_text("\n".join(p for p in parts if p.strip()))
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(path.relative_to(root)),
                            "source_file": path.name,
                            "record": idx,
                            "split": split,
                            "page": None,
                            "doc_type": "nit",
                            "intent": row.get("intent", ""),
                            "context": row.get("context", ""),
                            "target_command": row.get("target_command", ""),
                        },
                    )
                )
    return docs


def load_documents(cfg: Dict[str, Any], root: Optional[Path] = None) -> List[Document]:
    root = root or project_root()
    rag_cfg = cfg.get("rag", {})
    docs = load_nit_documents(root, cfg["data"]["output_dir"], nit_splits_for_rag(cfg))
    if rag_cfg.get("include_rag_docs", True):
        docs.extend(load_rag_documents(root, rag_cfg.get("doc_dir", "rag-doc")))
    return docs


def indexed_source_files(cfg: Dict[str, Any], root: Optional[Path] = None) -> List[Path]:
    root = root or project_root()
    paths: List[Path] = []
    data_base = root / cfg["data"]["output_dir"]
    for split in nit_splits_for_rag(cfg):
        path = data_base / f"{split}.jsonl"
        if path.exists():
            paths.append(path)
    rag_cfg = cfg.get("rag", {})
    rag_base = root / rag_cfg.get("doc_dir", "rag-doc")
    if rag_cfg.get("include_rag_docs", True) and rag_base.exists():
        paths.extend(
            p
            for p in rag_base.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
        )
    return sorted(paths)


def _split_paragraphs(text: str) -> List[str]:
    lines = text.splitlines()
    blocks: List[str] = []
    current: List[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if not stripped and current and not in_code:
            blocks.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
    blocks = _split_paragraphs(doc.text)
    chunks: List[Document] = []
    current: List[str] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            meta = dict(doc.metadata)
            meta["chunk"] = len(chunks)
            chunks.append(Document(text=chunk_text, metadata=meta))
        if chunk_overlap > 0:
            overlap: List[str] = []
            overlap_len = 0
            for block in reversed(current):
                block_len = len(block)
                if overlap and overlap_len + block_len > chunk_overlap:
                    break
                overlap.insert(0, block)
                overlap_len += block_len
            current = overlap
            current_len = overlap_len
        else:
            current = []
            current_len = 0

    for block in blocks:
        block_len = len(block)
        if block_len > chunk_size:
            emit()
            for start in range(0, block_len, max(chunk_size - chunk_overlap, 1)):
                piece = block[start : start + chunk_size].strip()
                if piece:
                    meta = dict(doc.metadata)
                    meta["chunk"] = len(chunks)
                    chunks.append(Document(text=piece, metadata=meta))
            continue
        if current and current_len + block_len + 2 > chunk_size:
            emit()
        current.append(block)
        current_len += block_len + 2
    emit()
    return chunks


def chunk_documents(docs: Iterable[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunks: List[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks


def _fingerprint(cfg: Dict[str, Any], root: Path) -> str:
    rag_cfg = cfg.get("rag", {})
    h = hashlib.sha256()
    h.update(json.dumps(rag_cfg, sort_keys=True).encode("utf-8"))
    h.update(str(cfg["data"]["output_dir"]).encode("utf-8"))
    paths = indexed_source_files(cfg, root)
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(_sha256_file(path).encode("utf-8"))
    return h.hexdigest()


class RagIndex:
    def __init__(
        self,
        chunks: List[Document],
        dense_vectorizer: TfidfVectorizer,
        dense_matrix: Any,
        lexical_vectorizer: TfidfVectorizer,
        lexical_matrix: Any,
        fingerprint: str,
        rag_doc_boost: float = 0.0,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
        action_weight: float = 0.20,
        protocol_weight: float = 0.20,
        value_weight: float = 0.25,
        contradiction_weight: float = 0.40,
    ):
        self.chunks = chunks
        self.vectorizer = dense_vectorizer
        self.matrix = dense_matrix
        self.dense_vectorizer = dense_vectorizer
        self.dense_matrix = dense_matrix
        self.lexical_vectorizer = lexical_vectorizer
        self.lexical_matrix = lexical_matrix
        self.fingerprint = fingerprint
        self.rag_doc_boost = rag_doc_boost
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.action_weight = action_weight
        self.protocol_weight = protocol_weight
        self.value_weight = value_weight
        self.contradiction_weight = contradiction_weight

    @classmethod
    def build(cls, cfg: Dict[str, Any], root: Optional[Path] = None) -> "RagIndex":
        root = root or project_root()
        rag_cfg = cfg.get("rag", {})
        docs = load_documents(cfg, root)
        chunks = chunk_documents(
            docs,
            chunk_size=int(rag_cfg.get("chunk_size", 1400)),
            chunk_overlap=int(rag_cfg.get("chunk_overlap", 200)),
        )
        if not chunks:
            raise ValueError("No RAG documents were loaded. Check data/processed and rag-doc/.")
        dense_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=int(rag_cfg.get("max_features", 50000)),
        )
        lexical_vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w/-]+\b",
            ngram_range=(1, 3),
            lowercase=True,
            max_features=int(rag_cfg.get("max_features", 50000)),
        )
        texts = [c.text for c in chunks]
        dense_matrix = dense_vectorizer.fit_transform(texts)
        lexical_matrix = lexical_vectorizer.fit_transform(texts)
        return cls(
            chunks=chunks,
            dense_vectorizer=dense_vectorizer,
            dense_matrix=dense_matrix,
            lexical_vectorizer=lexical_vectorizer,
            lexical_matrix=lexical_matrix,
            fingerprint=_fingerprint(cfg, root),
            rag_doc_boost=float(rag_cfg.get("rag_doc_boost", 0.0)),
            dense_weight=float(rag_cfg.get("dense_weight", 0.65)),
            lexical_weight=float(rag_cfg.get("lexical_weight", 0.35)),
            action_weight=float(rag_cfg.get("action_weight", 0.20)),
            protocol_weight=float(rag_cfg.get("protocol_weight", 0.20)),
            value_weight=float(rag_cfg.get("value_weight", 0.25)),
            contradiction_weight=float(rag_cfg.get("contradiction_weight", 0.40)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RagIndex":
        with path.open("rb") as f:
            return pickle.load(f)

    def _command_scores(self, query: str, chunk: Document) -> tuple[float, float, float, float]:
        meta = chunk.metadata or {}
        if meta.get("doc_type") != "nit":
            return 0.0, 0.0, 0.0, 0.0

        target = str(meta.get("target_command", ""))
        intent = str(meta.get("intent", ""))
        context = str(meta.get("context", ""))
        query_norm = _norm_query(query)
        target_norm = _norm_query(target)
        haystack = _norm_query(f"{target}\n{context}\n{intent}")
        query_action = _infer_query_action(query_norm)
        target_action = _command_action(target_norm)
        query_protocols = _matched_terms(query_norm, PROTOCOL_TERMS)
        target_protocols = _matched_terms(target_norm, PROTOCOL_TERMS)

        action_score = 0.0
        contradiction = 0.0
        if query_action == "show":
            action_score = self.action_weight if target_action == "show" else 0.0
            contradiction += 0.35 if target_action in {"set", "delete", "clear"} else 0.0
        elif query_action == "set":
            action_score = self.action_weight if target_action == "set" else 0.0
            contradiction += 0.35 if target_action in {"show", "delete", "clear"} else 0.0
        elif query_action == "disable":
            if _has_term(target_norm, "disable"):
                action_score = self.action_weight
            elif target_action == "delete":
                action_score = self.action_weight * 0.6
            elif target_action == "show" or (target_action == "set" and not _has_term(target_norm, "disable")):
                contradiction += 0.35
        elif query_action == "delete":
            if target_action == "delete":
                action_score = self.action_weight
            elif _has_term(target_norm, "disable"):
                action_score = self.action_weight * 0.4
            elif target_action in {"show", "set"}:
                contradiction += 0.35
        elif query_action == "clear":
            action_score = self.action_weight if target_action == "clear" else 0.0
            contradiction += 0.35 if target_action in {"show", "set", "delete"} else 0.0
        elif query_action == "load":
            action_score = self.action_weight if target_action in {"load", "request"} else 0.0

        protocol_score = 0.0
        if query_protocols:
            for protocol in query_protocols:
                if _has_term(target_norm, protocol):
                    protocol_score = max(protocol_score, min(self.protocol_weight, 0.15))
                elif target_protocols:
                    contradiction += 0.30
            if "igmp-snooping" in query_protocols and "igmp" in target_protocols and "igmp-snooping" not in target_protocols:
                contradiction += 0.30

        value_score = 0.0
        for value in _extract_values(query_norm):
            if _has_term(haystack, value):
                value_score += 0.05
        value_score = min(value_score, self.value_weight, 0.25)

        query_nouns = _matched_terms(query_norm, COMMAND_NOUNS)
        for noun in query_nouns:
            singular = noun[:-1] if noun.endswith("s") else noun
            if _has_term(haystack, noun) or _has_term(haystack, singular):
                value_score = min(value_score + 0.03, self.value_weight, 0.25)

        opposites = (
            ("enable", "disable"),
            ("disable", "enable"),
            ("read-only", "read-write"),
            ("read-write", "read-only"),
            ("ingress", "egress"),
            ("egress", "ingress"),
            ("interface", "vlan"),
            ("vlan", "interface"),
        )
        for want, bad in opposites:
            if _has_term(query_norm, want) and _has_term(haystack, bad) and not _has_term(haystack, want):
                contradiction += 0.30

        if _has_term(query_norm, "emergency"):
            if _has_term(haystack, "emergency"):
                value_score = min(value_score + 0.08, self.value_weight, 0.25)
            elif any(_has_term(haystack, sev) for sev in ("critical", "panic", "warning")):
                contradiction += 0.20
        if _has_term(query_norm, "all") and _has_term(haystack, "interface") and not _has_term(haystack, "all"):
            contradiction += 0.12
        if _has_term(query_norm, "all") and "system syslog user *" in target_norm:
            value_score = min(value_score + 0.10, self.value_weight, 0.25)
        if _has_term(query_norm, "read-only") and _has_term(target_norm, "read-only"):
            value_score = min(value_score + 0.10, self.value_weight, 0.25)
        if _has_term(query_norm, "read-write") and _has_term(target_norm, "read-write"):
            value_score = min(value_score + 0.10, self.value_weight, 0.25)

        return action_score, protocol_score, value_score, min(contradiction, self.contradiction_weight, 0.70)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        dense_q = self.dense_vectorizer.transform([query])
        lexical_q = self.lexical_vectorizer.transform([query])
        dense_raw = cosine_similarity(dense_q, self.dense_matrix).ravel()
        lexical_raw = cosine_similarity(lexical_q, self.lexical_matrix).ravel()
        dense_scores = _minmax(dense_raw)
        lexical_scores = _minmax(lexical_raw)
        sims = (self.dense_weight * dense_scores) + (self.lexical_weight * lexical_scores)
        hardware_query = _is_hardware_query(query)
        if not hardware_query:
            for i, chunk in enumerate(self.chunks):
                if chunk.metadata.get("doc_type") == "rag-doc":
                    sims[i] *= 0.65
        if re.search(r"\bex3300\b", query, re.I):
            other_model = re.compile(r"\bEX(?!3300\b)\d{4}[A-Z-]*\b")
            for i, chunk in enumerate(self.chunks):
                if other_model.search(chunk.text) and not re.search(r"\bEX3300\b", chunk.text):
                    sims[i] *= 0.35
        q_lower = query.lower()
        for i, chunk in enumerate(self.chunks):
            text = chunk.text
            t_lower = text.lower()
            if "led" in q_lower and any(term in q_lower for term in ("interpret", "alm", "sys", "mst", "status")):
                if "table 7: chassis status leds" in t_lower:
                    sims[i] += 0.15
                elif "chassis status leds in ex3300" in t_lower:
                    sims[i] += 0.10
                if "chassis physical specifications" in t_lower:
                    sims[i] *= 0.5
                if text.count("ALM") > 4 and "Table 7: Chassis Status LEDs" not in text:
                    sims[i] *= 0.45
            if "console" in q_lower:
                if "console port connector pinout information" in t_lower or "default baud rate" in t_lower:
                    sims[i] += 0.15
                if "to connect and configure the switch from the console" in t_lower:
                    sims[i] += 0.08
            if "power" in q_lower and "supply" in q_lower:
                if "power supply in ex3300 switches" in t_lower or "power specifications for ex3300 switches" in t_lower:
                    sims[i] += 0.15
                if "connecting dc power" in t_lower:
                    sims[i] *= 0.8
            if chunk.metadata.get("doc_type") == "rag-doc":
                if hardware_query:
                    sims[i] += 0.15
                continue
            action_score, protocol_score, value_score, contradiction = self._command_scores(query, chunk)
            sims[i] += action_score + protocol_score + value_score - contradiction
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                metadata=self.chunks[i].metadata,
                score=float(sims[i]),
                dense_score=float(dense_raw[i]),
                lexical_score=float(lexical_raw[i]),
            )
            for i in order
            if sims[i] > 0
        ]


def get_or_build_index(cfg: Dict[str, Any], rebuild: bool = False, root: Optional[Path] = None) -> RagIndex:
    root = root or project_root()
    apply_rag_corpus(cfg, None)
    rag_cfg = cfg.get("rag", {})
    index_path = root / rag_cfg.get("index_path", "results/rag_index.pkl")
    expected_fingerprint = _fingerprint(cfg, root)
    if rebuild and index_path.exists():
        index_path.unlink()
    if index_path.exists() and not rebuild:
        index = RagIndex.load(index_path)
        if index.fingerprint == expected_fingerprint:
            return index
        print("[RAG] Existing index is stale; rebuilding because source documents or settings changed.")
    print("[RAG] Building index from:")
    indexed = indexed_source_files(cfg, root)
    for path in indexed:
        print(f"[RAG]   include {path.relative_to(root)}")
    data_base = root / cfg["data"]["output_dir"]
    for split in ("val", "test"):
        excluded = data_base / f"{split}.jsonl"
        if excluded.exists() and not rag_cfg.get(f"include_{split}_in_rag", False):
            print(f"[RAG]   exclude {excluded.relative_to(root)}")
    index = RagIndex.build(cfg, root)
    index.save(index_path)
    return index


def format_retrieval_debug(query: str, chunks: List[RetrievedChunk]) -> str:
    lines = [f"[RAG] query: {query}"]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        preview = re.sub(r"\s+", " ", chunk.text)[:200]
        page = meta.get("page")
        page_text = f", page={page}" if page else ""
        lines.append(
            f"[RAG] {i}. source={meta.get('source_file')}{page_text}, "
            f"score={chunk.score:.4f}, dense={chunk.dense_score:.4f}, "
            f"lexical={chunk.lexical_score:.4f}, preview={preview}"
        )
    return "\n".join(lines)


def assert_no_eval_leakage(chunks: List[RetrievedChunk], strict: bool = False) -> None:
    for chunk in chunks:
        source = chunk.metadata.get("source_file")
        if source == "test.jsonl" or (strict and source == "val.jsonl"):
            raise RuntimeError(f"Evaluation leakage detected: retrieved source={source}")


def assert_no_test_leakage(chunks: List[RetrievedChunk]) -> None:
    assert_no_eval_leakage(chunks, strict=False)


def build_rag_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    example_parts = []
    doc_parts = []
    for chunk in chunks:
        meta = chunk.metadata
        page = meta.get("page")
        cite = meta.get("source_file", "unknown")
        if page:
            cite = f"{cite}, page {page}"
        part = f"[Source: {cite}]\n{chunk.text}"
        if meta.get("doc_type") == "nit":
            example_parts.append(part)
        else:
            doc_parts.append(part)
    examples = "\n\n---\n\n".join(example_parts) if example_parts else "None retrieved."
    docs = "\n\n---\n\n".join(doc_parts) if doc_parts else "None retrieved."
    return RAG_PROMPT_TEMPLATE.format(examples=examples, docs=docs, question=question)


def extractive_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "I could not find this in the retrieved EX3300 guide context."
    chunk = chunks[0]
    meta = chunk.metadata
    page = meta.get("page")
    cite = meta.get("source_file", "unknown source")
    if page:
        cite = f"{cite}, page {page}"
    preview = re.sub(r"\s+", " ", chunk.text).strip()[:900]
    return f"From {cite}: {preview}"
