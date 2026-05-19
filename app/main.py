# app/main.py
import os, glob, json, time
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Local rules engine (from your backend) ---
from .insight_model import infer

# --- ML service pieces (from your MLService) ---
from .retriever import index_memory, search
from .rules_fallback import fallback_options
from .model_infer import generate_options
from .summarizer import make_summary

# ---------------- App & CORS ----------------
app = FastAPI(title="Clinical Copilot (Unified)", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Knowledge indexing -------------
KNOW_DIR = os.getenv(
    "KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(__file__), "knowledge")
)

def _load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for fp in glob.glob(os.path.join(KNOW_DIR, "*.txt")):
        try:
            with open(fp, "r") as f:
                txt = (f.read() or "").strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        # naive chunking ~900 chars
        for i in range(0, len(txt), 900):
            chunks.append({"source": os.path.basename(fp), "text": txt[i:i+900]})
    return chunks

@app.on_event("startup")
def _startup_index():
    chunks = _load_chunks()
    if not chunks:
        print(f"[unified] WARNING: no files under {KNOW_DIR}. Add knowledge/*.txt")
    else:
        index_memory(chunks)
        print(f"[unified] Indexed {len(chunks)} chunks from {KNOW_DIR}")

# ---------------- Data models ----------------
class IngestPayload(BaseModel):
    patient_id: str
    bundle: Dict[str, Any]

# ---------------- Status routes ---------------
@app.get("/status")
def status():
    return {"ok": True, "ts": time.time()}

# --------- Simple FHIR ingestion (optional demo) ---------
# You can leave these out if you’re not demoing ingestion today.
FHIR_DB: Dict[str, Dict[str, Any]] = {}
DB_FILE = Path("fhir_db.json")

def save_db():
    DB_FILE.write_text(json.dumps(FHIR_DB))

def load_db():
    global FHIR_DB
    if DB_FILE.exists():
        FHIR_DB = json.loads(DB_FILE.read_text())

load_db()

def parse_fhir_bundle(bundle: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    notes, labs, meds, problems = [], [], [], []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Condition":
            problems.append({
                "resource": f"{rtype}/{res.get('id','')}",
                "code": res.get("code", {}).get("text") or "Unknown condition",
                "onset": res.get("onsetDateTime") or res.get("recordedDate"),
                "clinicalStatus": (res.get("clinicalStatus", {}) or {}).get("text",""),
            })

        elif rtype == "Observation":
            labs.append({
                "resource": f"{rtype}/{res.get('id','')}",
                "name": (res.get("code", {}).get("text") or "").strip(),
                "value": (res.get("valueQuantity",{}) or {}).get("value"),
                "unit":  (res.get("valueQuantity",{}) or {}).get("unit"),
                "effectiveDateTime": res.get("effectiveDateTime") or res.get("issued"),
            })

        elif rtype in ("MedicationStatement","MedicationRequest"):
            med_txt = (res.get("medicationCodeableConcept") or {}).get("text","")
            dose = ""
            if res.get("dosageInstruction"):
                dose = res["dosageInstruction"][0].get("text","")
            meds.append({
                "resource": f"{rtype}/{res.get('id','')}",
                "name": med_txt,
                "dose": dose,
                "when": res.get("authoredOn") or (res.get("effectivePeriod",{}) or {}).get("start","")
            })

        elif rtype in ("DocumentReference","Composition"):
            text = ""
            if rtype == "Composition":
                text = (res.get("title","") + " " + (res.get("text",{}) or {}).get("div","")).strip()
            else:
                text = res.get("description","") or ""
            if text:
                clean = (text.replace("<div>"," ")
                             .replace("</div>"," ")
                             .replace("<p>"," ")
                             .replace("</p>"," ")
                             .strip())
                notes.append({
                    "resource": f"{rtype}/{res.get('id','')}",
                    "text": clean,
                    "date": res.get("date") or res.get("created") or ""
                })

    return {"notes":notes, "labs":labs, "meds":meds, "problems":problems}

@app.post("/ingest/fhir")
def ingest(payload: IngestPayload):
    parsed = parse_fhir_bundle(payload.bundle)
    FHIR_DB[payload.patient_id] = parsed
    save_db()
    return {"ingested": True, "patient_id": payload.patient_id,
            "counts": {k:len(v) for k,v in parsed.items()}}

# -------- Original rules engine route (optional) --------
@app.post("/ask_text")
def ask_text(p: dict = Body(...)):
    # p = {"question": "...", "facts": {...}}  # facts optional
    return infer(p["question"], p.get("facts"))

# ----------------- ML reasoning route -------------------
@app.post("/reason")
def reason_api(p: dict = Body(...)):
    """
    Input: {"question": "...", "patient_facts":[{"text":"..."}]}
    Returns: options + evidence + engine + backend-made summary.
    """
    facts = p.get("patient_facts", [])
    q = p["question"]

    # retrieve guideline chunks
    hits = search(q, k=6)

    engine = "model"
    try:
        out = generate_options(q, facts, hits)
        if not out.get("options"):
            raise RuntimeError("empty options")
    except Exception as e:
        print(f"[warn] tiny-model fallback: {e}")
        out = fallback_options(q, facts, hits)
        engine = "rules"

    options = out.get("options", [])[:3]
    summary = make_summary(q, facts, options)

    return {
        "question": q,
        "engine": engine,
        "options": options,
        "summary": summary,                    # <— new
        "evidence": {"docs": hits, "patient_snippets": facts}
    }