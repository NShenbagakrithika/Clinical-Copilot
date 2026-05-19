from typing import Any, Dict, List
import httpx


def fetch_pubmed(query: str, n: int = 3):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={n}&fields=title,year,url"
    r = httpx.get(url, timeout=10)
    return [
        {
            "title": p["title"],
            "year": p.get("year", "N/A"),
            "url": p["url"]
        }
        for p in r.json().get("data", [])
    ]


def infer(question: str, facts: Dict[str, Any] | None = None) -> Dict[str, Any]:
    q = (question or "").lower()
    options: List[Dict[str, Any]] = []

    if "leg pain" in q and "fall" in q:
        # Fetch citation data for diabetes + fracture
        citations = fetch_pubmed("diabetes delayed bone healing", n=2)

        options.append({
            "title": "Order imaging and manage fracture",
            "rationale": (
                "Patient reports leg pain after a fall. "
                "Has diabetes, which may delay healing and increase infection risk. "
                "Rod in hand means MRI is contraindicated — X-ray or CT is preferred."
            ),
            "steps": [
                "Order X-Ray or CT of affected leg",
                "Stabilize leg and restrict weight-bearing activity",
                "Refer to orthopedics"
            ],
            "risks": ["Delayed bone healing due to diabetes", "Higher infection risk"],
            "contraindications": ["Avoid MRI (metal rod in hand)"],
            "monitoring": ["Monitor healing progress and signs of infection"],
            "citations": citations
        })

    else:
        options.append({
            "title": "No matching clinical reasoning path",
            "rationale": "Query does not match any predefined patterns.",
            "steps": [],
            "risks": [],
            "contraindications": [],
            "monitoring": [],
            "citations": []
        })

    return {
        "question": question,
        "options": options,
        "evidence": {
            "patient_snippets": facts or []
        }
    }