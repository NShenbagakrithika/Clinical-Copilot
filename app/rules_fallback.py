# app/rules_fallback.py
from typing import Any, Dict, List

def fallback_options(
    question: str,
    facts: List[Dict[str, Any]],
    hits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Lightweight rule engine used when the tiny model is unavailable
    or returns unusable output.

    - Uses simple keyword checks on the question.
    - Pulls up to two unique source filenames from retriever hits for citations.
    - Returns up to three options.
    """
    q = (question or "").lower()
    has_fever = "fever" in q
    has_cough = "cough" in q
    has_chest = "chest" in q or "chest pain" in q or "pleuritic" in q

    # Collect up to 2 unique source filenames from retriever hits
    cite: List[str] = []
    if hits:
        seen = set()
        for h in hits:
            src = (h.get("source") or "local").strip()
            if src and src not in seen:
                cite.append(src)
                seen.add(src)
            if len(cite) == 2:
                break

    opts: List[Dict[str, Any]] = []

    # CAP path
    if has_fever and has_cough:
        opts.append({
            "title": "Consider community-acquired pneumonia",
            "rationale": "Fever + cough pattern; confirm with chest radiograph.",
            "steps": ["Chest X-ray", "Pulse oximetry", "Empiric antibiotics per local protocol"],
            "risks": ["Antibiotic side effects", "Resistance"],
            "contraindications": [],
            "monitoring": ["Reassess O2 sat & symptoms in 24–48 h"],
            "citations": cite
        })

    # Chest pain / ischemia rule-out
    if has_chest:
        opts.append({
            "title": "Rule out cardiac ischemia",
            "rationale": "Chest symptoms warrant ECG and high-sensitivity troponin to exclude ACS.",
            "steps": ["ECG", "High-sensitivity troponin", "Aspirin if no contraindication"],
            "risks": ["Bleeding with antiplatelet"],
            "contraindications": ["Active bleeding", "ASA allergy"],
            "monitoring": ["Observation until ruled out"],
            "citations": cite
        })

    # Generic third option to ensure 2–3 actionable paths
    if len(opts) < 3:
        opts.append({
            "title": "Obtain more data and risk stratify",
            "rationale": "Evidence limited or mixed; gather diagnostics to narrow differential.",
            "steps": [
                "Detailed history (onset, severity, modifiers)",
                "Vitals, CBC/CMP",
                "Imaging/tests guided by exam"
            ],
            "risks": [],
            "contraindications": [],
            "monitoring": ["Close follow-up; escalate on red flags"],
            "citations": cite
        })

    return {"options": opts[:3]}