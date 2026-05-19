# app/model_infer.py
import json, os
from typing import Any, Dict, List

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = None

from .rules_fallback import fallback_options

MODEL_DIR = os.getenv("TINY_MODEL_DIR", "checkpoints/tiny-json-gpt")

_tok = None
_model = None

def _can_use_model() -> bool:
    return torch is not None and AutoTokenizer is not None and bool(MODEL_DIR)

def _device():
    # keep it simple & safe on Mac/CPU
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model():
    """Lazy-load tokenizer/model; never raise to the API layer."""
    global _tok, _model
    if not _can_use_model():
        return
    try:
        if _tok is None:
            _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            if getattr(_tok, "pad_token", None) is None:
                _tok.pad_token = _tok.eos_token
        if _model is None:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                torch_dtype=getattr(torch, "float32", None),
                low_cpu_mem_usage=True
            ).to(_device())
            _model.eval()
    except Exception as e:
        # make sure we fall back gracefully
        _tok = None
        _model = None

def _build_prompt(question: str, facts, evidence_docs):
    fact_lines = "\n".join(f"- {f.get('text','')}" for f in (facts or [])) or "- (none)"
    uniq_sources, doc_lines = [], []
    if evidence_docs:
        seen = set()
        for d in evidence_docs:
            src = (d.get("source") or "").strip()
            if src and src not in seen:
                uniq_sources.append(src); seen.add(src)
            txt = (d.get("text","") or "").replace("\n"," ")[:300]
            doc_lines.append(f"- ({src}) {txt}")
    src_lines = "\n".join(f"- {s}" for s in uniq_sources) or "- (none)"
    docs_block = "\n".join(doc_lines) or "- (none)"

    return f"""You are a clinical reasoning assistant. Output STRICT JSON ONLY.

Schema:
{{
  "options": [
    {{
      "title": str,
      "rationale": str,
      "steps": [str],
      "risks": [str],
      "contraindications": [str],
      "monitoring": [str],
      "citations": [str]   // filenames ONLY from the list below
    }},
    {{...}}, {{...}}
  ]
}}

Rules:
- Cite ONLY from these allowed filenames:
{src_lines}
- If none fit, use [].
- Keep lists concise (≤6). No text outside JSON.

<Question>
{question.strip()}

<Patient_Facts>
{fact_lines}

<Evidence_Snippets>
{docs_block}

<JSON>
"""

def generate_options(question: str, facts: List[Dict[str, Any]], evidence_docs: List[Dict[str, Any]], max_new_tokens=500) -> Dict[str, Any]:
    # Try model; on any failure return rules fallback (never 500)
    try:
        load_model()
        if _tok is None or _model is None:
            raise RuntimeError("tiny-model-unavailable")
        prompt = _build_prompt(question, facts, evidence_docs)
        ids = _tok(prompt, return_tensors="pt").to(_device())
        with torch.no_grad():
            out = _model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2
            )
        text = _tok.decode(out[0], skip_special_tokens=True)
        jstart, jend = text.find("{"), text.rfind("}")
        if jstart == -1 or jend == -1:
            raise ValueError("model_output_not_json")
        obj = json.loads(text[jstart:jend+1])
        obj["options"] = list(obj.get("options", []))[:3]
        return obj
    except Exception:
        return fallback_options(question, facts, evidence_docs)