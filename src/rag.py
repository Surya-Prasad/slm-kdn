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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
                        },
                    )
                )
    return docs


def load_documents(cfg: Dict[str, Any], root: Optional[Path] = None) -> List[Document]:
    root = root or project_root()
    rag_cfg = cfg.get("rag", {})
    docs = load_nit_documents(root, cfg["data"]["output_dir"], nit_splits_for_rag(cfg))
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
    rag_base = root / cfg.get("rag", {}).get("doc_dir", "rag-doc")
    if rag_base.exists():
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
        vectorizer: TfidfVectorizer,
        matrix: Any,
        fingerprint: str,
        rag_doc_boost: float = 0.0,
    ):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.fingerprint = fingerprint
        self.rag_doc_boost = rag_doc_boost

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
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=int(rag_cfg.get("max_features", 50000)),
        )
        matrix = vectorizer.fit_transform([c.text for c in chunks])
        return cls(
            chunks=chunks,
            vectorizer=vectorizer,
            matrix=matrix,
            fingerprint=_fingerprint(cfg, root),
            rag_doc_boost=float(rag_cfg.get("rag_doc_boost", 0.0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RagIndex":
        with path.open("rb") as f:
            return pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        if self.rag_doc_boost and re.search(r"\b(ex3300|hardware|front panel|lcd|led|power|console)\b", query, re.I):
            for i, chunk in enumerate(self.chunks):
                if chunk.metadata.get("doc_type") == "rag-doc":
                    sims[i] += self.rag_doc_boost
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
                    sims[i] += 0.6
                elif "chassis status leds in ex3300" in t_lower:
                    sims[i] += 0.2
                if "chassis physical specifications" in t_lower:
                    sims[i] *= 0.5
                if text.count("ALM") > 4 and "Table 7: Chassis Status LEDs" not in text:
                    sims[i] *= 0.45
            if "console" in q_lower:
                if "console port connector pinout information" in t_lower or "default baud rate" in t_lower:
                    sims[i] += 0.25
                if "to connect and configure the switch from the console" in t_lower:
                    sims[i] += 0.12
            if "power" in q_lower and "supply" in q_lower:
                if "power supply in ex3300 switches" in t_lower or "power specifications for ex3300 switches" in t_lower:
                    sims[i] += 0.25
                if "connecting dc power" in t_lower:
                    sims[i] *= 0.8
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                metadata=self.chunks[i].metadata,
                score=float(sims[i]),
            )
            for i in order
            if sims[i] > 0
        ]


def get_or_build_index(cfg: Dict[str, Any], rebuild: bool = False, root: Optional[Path] = None) -> RagIndex:
    root = root or project_root()
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
    excluded_test = root / cfg["data"]["output_dir"] / "test.jsonl"
    if excluded_test.exists() and not rag_cfg.get("include_test_in_rag", False):
        print(f"[RAG]   exclude {excluded_test.relative_to(root)}")
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
            f"score={chunk.score:.4f}, preview={preview}"
        )
    return "\n".join(lines)


def assert_no_test_leakage(chunks: List[RetrievedChunk]) -> None:
    for chunk in chunks:
        if chunk.metadata.get("source_file") == "test.jsonl":
            raise RuntimeError("Evaluation leakage detected: test.jsonl was retrieved.")


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
