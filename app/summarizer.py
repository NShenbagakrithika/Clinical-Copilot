# app/summarizer.py
from typing import Any, Dict, List

def _txts(facts: List[Dict[str, Any]]) -> List[str]:
    return [str(f.get("text","")).strip() for f in (facts or []) if str(f.get("text","")).strip()]

def _has(blob: str, kws: List[str]) -> bool:
    b = (blob or "").lower()
    return any(k in b for k in kws)

def _blob(question: str, facts_txt: List[str]) -> str:
    return f"{question or ''}\n" + "\n".join(facts_txt)

def _uniq(seq: List[str], n: int) -> List[str]:
    out, seen = [], set()
    for s in seq:
        if s and s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= n: break
    return out

def make_summary(
    question: str,
    patient_facts: List[Dict[str, Any]],
    options: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Returns:
      {
        "problem": str,
        "factors": {"relevant":[str], "ignored":[str]},
        "differential": [str],
        "diagnostics": [str],
        "treatment": [str],
        "disposition": [str],
        "counseling": [str],
        "red_flags": [str],
        "notes": [str],
        "citations": [str]
      }
    """
    facts_txt = _txts(patient_facts)
    all_text = _blob(question, facts_txt).lower()

    # ----- Problem -----
    if _has(all_text, ["fall","fell","trauma"]):
        side = " right" if _has(all_text, [" right"]) else (" left" if _has(all_text, [" left"]) else "")
        limb = " leg" if _has(all_text, ["leg","hip","femur","tibia"]) else (" arm" if _has(all_text, ["arm","humerus","radius","ulna"]) else "")
        problem = f"Acute {('post-traumatic ' if limb else '')}pain after fall{(' in'+limb) if limb else ''}{side}."
    elif _has(all_text, ["chest pain","chest tightness"]):
        problem = "Acute chest pain."
    elif _has(all_text, ["fever","cough","pleuritic"]):
        problem = "Fever with cough; possible CAP."
    else:
        problem = (question or "Current concern").rstrip(".?") + "."

    # ----- Factors -----
    relevant, ignored = [], []
    if _has(all_text, ["rod","implant","metal"]):
        relevant.append("Metal implant present → MRI contraindicated; prefer X-ray/CT.")
    if _has(all_text, ["diabetes","hba1c"]):
        relevant.append("Type 2 diabetes → delayed bone healing & higher infection risk.")
    if _has(all_text, ["egfr","ckd","chronic kidney"]):
        relevant.append("Chronic kidney disease → check eGFR for med/contrast choices.")
    if _has(all_text, ["asthma"]) and not _has(all_text, ["wheeze","bronchospasm","sob","shortness of breath"]):
        ignored.append("Asthma (controlled) — not relevant to this presentation.")
    if _has(all_text, ["cancer"]) and not _has(all_text, ["recurr","metast","chemo","radiation"]):
        ignored.append("Past cancer (no active disease) — not currently relevant.")
    if not relevant:
        relevant.append("No comorbid factors obviously altering immediate management.")

    # ----- Differential (prioritized) -----
    differential: List[str] = []
    if _has(all_text, ["fall","fell","trauma"]):
        if _has(all_text, ["leg","hip","femur","tibia"]):
            differential += [
                "Fracture (hip/femur/tibia/fibula) ± occult fracture",
                "Ligament/meniscal injury or severe contusion",
                "Compartment syndrome (if escalating pain/swelling)"
            ]
        if _has(all_text, ["arm","humerus","radius","ulna","wrist"]):
            differential += [
                "Fracture (humerus/radius/ulna) ± dislocation",
                "Neurovascular injury (check pulses/sensation)"
            ]
    if _has(all_text, ["chest pain","chest tightness"]):
        differential += ["Acute coronary syndrome", "Aortic pathology", "Pulmonary embolism", "Musculoskeletal"]
    if _has(all_text, ["fever","cough","pleuritic"]):
        differential += ["Community-acquired pneumonia", "Viral bronchitis", "PE (if pleuritic/hypoxia)"]

    differential = _uniq(differential, 6)

    # ----- Pull useful steps from options if present -----
    pulled_steps: List[str] = []
    for o in options or []:
        for s in (o.get("steps") or []):
            pulled_steps.append(str(s))
    pulled_steps = _uniq(pulled_steps, 8)

    # ----- Diagnostics / Treatment / Disposition -----
    diagnostics: List[str] = []
    treatment: List[str] = []
    disposition: List[str] = []
    counseling: List[str] = []
    red_flags: List[str] = []

    if _has(all_text, ["fall","fell","trauma"]) and _has(all_text, ["leg","hip","femur","tibia","arm","humerus","radius","ulna"]):
        diagnostics += [
            "X-ray of affected limb (AP/lateral).",
            "CT if X-ray non-diagnostic or intra-articular/complex injury suspected.",
            "Neurovascular exam; document pulses, capillary refill, sensation, motor.",
        ]
        if _has(all_text, ["metal","rod","implant"]):
            diagnostics.append("Avoid MRI due to metal hardware.")
        treatment += [
            "Immobilize/splint; RICE (rest, ice, compression, elevation).",
            "Analgesia (acetaminophen ± short opioid if severe; avoid NSAIDs if fracture + CKD).",
        ]
        disposition += [
            "Limit weight-bearing until fracture excluded/managed.",
            "Orthopedics referral within 24–72 h (earlier if displaced/open fracture)."
        ]
        counseling += [
            "With diabetes, expect slower healing; monitor skin integrity & glucose closely.",
            "Return immediately for numbness, escalating pain, worsening swelling, fever."
        ]
        red_flags += ["Pain out of proportion (compartment syndrome)", "Neurovascular deficit", "Open fracture"]

    if _has(all_text, ["chest pain","chest tightness"]):
        diagnostics += [
            "12-lead ECG now; repeat with symptoms.",
            "High-sensitivity troponin at 0/1–3 h per protocol.",
        ]
        treatment += ["Aspirin 160–325 mg chewed if no contraindication.", "Monitor on telemetry; IV access."]
        disposition += ["Risk-stratify for ACS; admit if elevated risk or abnormal troponin/ECG."]
        red_flags += ["New ST-changes, rising troponin, hemodynamic instability"]

    if _has(all_text, ["fever","cough","pleuritic"]):
        diagnostics += ["Chest radiograph", "Pulse oximetry and vitals trend"]
        treatment += ["Start empiric CAP antibiotics per local resistance when clinical suspicion high."]
        red_flags += ["O₂ sat < 90–92% RA", "Respiratory distress"]

    # if options provided steps, prepend them (so guidelines still surface)
    diagnostics = _uniq(pulled_steps + diagnostics, 8) if pulled_steps else diagnostics
    treatment = _uniq(treatment, 8)
    disposition = _uniq(disposition, 6)
    counseling = _uniq(counseling, 6)
    red_flags = _uniq(red_flags, 6)

    # Notes & citations
    notes = ["Filtering: large records are trimmed to comorbidities/constraints that change decisions (e.g., diabetes, CKD, metal hardware)."]
    citations: List[str] = []
    for o in options or []:
        for c in (o.get("citations") or []):
            if c not in citations:
                citations.append(c)
    citations = citations[:6]

    # If nothing useful in diagnostics/treatment came through, keep a safe default
    if not diagnostics:
        diagnostics = ["Focused history & exam", "Targeted labs/imaging based on differential"]
    if not treatment:
        treatment = ["Symptom control", "Safety-net and close follow-up"]
    if not disposition:
        disposition = ["Outpatient vs. ED observation depending on vitals, pain control, and red flags"]

    return {
        "problem": problem,
        "factors": {"relevant": relevant, "ignored": ignored},
        "differential": differential,
        "diagnostics": diagnostics,
        "treatment": treatment,
        "disposition": disposition,
        "counseling": counseling,
        "red_flags": red_flags,
        "notes": notes,
        "citations": citations
    }