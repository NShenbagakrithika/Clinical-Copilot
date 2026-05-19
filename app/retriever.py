# app/retriever.py
import os, glob, re
from typing import List, Dict, Any

_KNOWLEDGE_DIR = os.getenv(
    "KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(__file__), "knowledge")
)

_DOCS: List[Dict[str, Any]] = []

def index_memory(chunks: List[Dict[str, Any]]) -> None:
    global _DOCS
    _DOCS = chunks[:]

def _load_files() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for fp in glob.glob(os.path.join(_KNOWLEDGE_DIR, "*.txt")):
        try:
            with open(fp, "r") as f:
                txt = (f.read() or "").strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        for i in range(0, len(txt), 900):
            docs.append({"source": os.path.basename(fp), "text": txt[i:i+900]})
    return docs

if not _DOCS:
    _DOCS = _load_files()

def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores); hi = max(scores)
    if hi - lo == 0:
        return scores[:]  # already flat
    return [(s - lo) / (hi - lo) for s in scores]

def search(query: str, k: int = 6) -> List[Dict[str, Any]]:
    if not _DOCS or not query:
        return []
    q = _tok(query)
    raw_scores: List[float] = []
    for d in _DOCS:
        t = _tok(d["text"])
        inter = len(q & t)
        denom = (len(q) + len(t) - inter) or 1
        jaccard = inter / denom
        raw_scores.append(jaccard)

    norm_scores = _minmax_normalize(raw_scores)
    ranked = sorted(zip(norm_scores, _DOCS), key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    for s, d in ranked[:k]:
        out.append({"text": d["text"], "source": d["source"], "score": float(s)})
    return out