import os
from datetime import date, timedelta
from typing import List, Optional, Dict, Any

import re
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------
# Config
# ---------------------------

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    # Fail fast if the key is missing so Render logs clearly show it
    raise RuntimeError("Please set the NEWSAPI_KEY environment variable")

NEWSAPI_AI_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"

# Domain vocab

DEFAULT_MATERIALS = ["cement", "concrete", "rebar", "steel", "aggregate", "aggregates"]

# Risk words focused on supply disruption / operations, not generic politics
RISK_KEYWORDS = [
    # Supply / volume
    "shortage", "shortages", "scarcity", "scarce",
    "disruption", "disruptions", "disrupted",
    "delay", "delays", "delayed",
    "bottleneck", "bottlenecks",
    "backlog", "backlogs",
    # Labor / operations
    "strike", "strikes", "walkout", "walkouts",
    "lockout", "lockouts",
    "shutdown", "shut down", "shutting down", "closure", "closures",
    "suspension", "suspended", "suspend production",
    "layoff", "layoffs",
    # Incidents / disasters
    "fire", "explosion", "blast", "accident", "collapse",
    "flood", "earthquake", "hurricane", "storm", "typhoon",
    # Logistics / ports
    "port", "ports", "terminal", "terminals",
    "congestion", "backed up",
    "shipping", "shipment", "shipments", "freight",
    "supply chain", "logistics",
    # Finance / commercial
    "bankrupt", "bankruptcy", "insolvency", "insolvent",
    "default", "restructuring",
    "sanction", "sanctions", "embargo", "export ban", "export bans",
    "tariff", "tariffs", "export controls",
    # Prices
    "price spike", "price spikes", "price surge", "price surges",
    "price increase", "price increases",
    "cost increase", "cost pressure", "inflation",
]

# Additional “construction context” words
CONSTRUCTION_KEYWORDS = [
    "construction", "infrastructure", "bridge", "bridges",
    "highway", "highways", "road", "roads",
    "tunnel", "tunnels", "dam", "dams",
    "cement plant", "cement factory", "cement works",
    "steel mill", "steel plant",
    "quarry", "quarries", "ready-mix", "ready mix",
    "batch plant", "asphalt plant", "concrete plant",
]


# ---------------------------
# Models
# ---------------------------

class NewsQuery(BaseModel):
    query: str = Field(
        ...,
        description="Main risk phrase, e.g. 'concrete disruption' or 'steel shortage'",
    )
    region: Optional[str] = Field(
        None,
        description="City / region / country to focus on, e.g. 'Montreal' or 'United States'",
    )
    materials: Optional[List[str]] = Field(
        None,
        description="List of material keywords, e.g. ['concrete','cement','rebar']",
    )
    days_back: int = Field(
        5,
        ge=1,
        le=365,
        description="How many days back to search from today (capped to 30 for free NewsAPI.ai accounts)",
    )
    risk_filter: bool = Field(
        True,
        description="If true, apply heuristic risk filter to drop obviously irrelevant articles.",
    )
    max_docs: int = Field(
        15,
        ge=1,
        le=100,
        description="Maximum number of documents to return after filtering & ranking.",
    )
    min_risk_score: Optional[int] = Field(
        None,
        description="Optional minimum risk_score. If not provided, backend uses a sensible default.",
    )


app = FastAPI(
    title="Procurement News Backend",
    version="0.1.0",
    description="Backend service that calls NewsAPI.ai (Event Registry) and returns normalized news documents for supply-chain risk monitoring.",
)


# ---------------------------
# Helpers
# ---------------------------

def build_material_risk_hint(q: NewsQuery, effective_days_back: int) -> str:
    parts: List[str] = []
    if q.query:
        parts.append(f"query={q.query!r}")
    if q.materials:
        parts.append("materials=" + ", ".join(q.materials))
    if q.region:
        parts.append(f"region={q.region}")
    parts.append(f"days_back={effective_days_back}")
    return "; ".join(parts)


def region_to_location_uri(region: str) -> str:
    slug = region.strip().replace(" ", "_")
    return f"http://en.wikipedia.org/wiki/{slug}"


def normalize_materials(user_materials: Optional[List[str]]) -> List[str]:
    if user_materials:
        base = [m.strip().lower() for m in user_materials if m.strip()]
    else:
        base = []
    # Always include our defaults to keep recall high
    base.extend(DEFAULT_MATERIALS)
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for m in base:
        if m and m not in seen:
            seen.add(m)
            result.append(m)
    return result


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Simple sentence splitter; good enough for heuristics
    return re.split(r"[.!?]\s+", text)


def count_occurrences(text: str, terms: List[str]) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for term in terms:
        # Use word boundary when possible
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        hits = len(re.findall(pattern, text_lower))
        count += hits
    return count


def compute_risk_features(
    title: str,
    body: str,
    material_terms: List[str],
) -> Dict[str, Any]:
    title = title or ""
    body = body or ""
    full_text = f"{title}\n{body}"
    text_lower = full_text.lower()
    material_hits = count_occurrences(text_lower, material_terms)
    risk_keyword_hits = count_occurrences(text_lower, RISK_KEYWORDS)
    construction_hits = count_occurrences(text_lower, CONSTRUCTION_KEYWORDS)

    # Sentence-level co-occurrence
    sentences = split_sentences(full_text)
    joint_sentences = 0
    for s in sentences:
        s_lower = s.lower()
        if any(m in s_lower for m in material_terms) and any(
            rk in s_lower for rk in RISK_KEYWORDS
        ):
            joint_sentences += 1

    title_lower = title.lower()
    title_has_material = any(m in title_lower for m in material_terms)
    title_has_risk = any(rk in title_lower for rk in RISK_KEYWORDS)
    title_joint = bool(title_has_material and title_has_risk)

    # Simple heuristic risk score
    risk_score = (
        3 * int(title_joint)
        + 2 * min(joint_sentences, 3)
        + 1 * min(material_hits, 5)
        + 1 * min(risk_keyword_hits, 5)
        + 1 * min(construction_hits, 3)
    )

    return {
        "material_hits": material_hits,
        "risk_keyword_hits": risk_keyword_hits,
        "construction_hits": construction_hits,
        "joint_sentences": joint_sentences,
        "title_joint": title_joint,
        "risk_score": risk_score,
    }


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/news/search")
async def news_search(payload: NewsQuery):
    effective_days_back = min(payload.days_back, 30)

    today = date.today()
    date_start = today - timedelta(days=effective_days_back)

    # --------------------------------------------------
    # Build keyword(s) for NewsAPI.ai
    # --------------------------------------------------
    raw_q = (payload.query or "").strip()

    keywords: List[str] | str
    keyword_oper = "or"

    # Handle "A OR B OR C" style queries
    or_split = re.split(r"\bOR\b", raw_q, flags=re.IGNORECASE)
    parts = [p.strip() for p in or_split if p.strip()]

    if len(parts) > 1:
        keywords = parts
        keyword_oper = "or"
    else:
        keywords = raw_q

    materials = normalize_materials(payload.materials)

    body = {
        "apiKey": NEWSAPI_KEY,
        "keyword": keywords,
        "keywordOper": keyword_oper,
        "lang": ["eng"],
        "dateStart": date_start.strftime("%Y-%m-%d"),
        "dateEnd": today.strftime("%Y-%m-%d"),
        "resultType": "articles",
        "articlesSortBy": "date",
        "articlesCount": 50,
    }

    if payload.region:
        body["sourceLocationUri"] = region_to_location_uri(payload.region)

    # Call NewsAPI.ai
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                NEWSAPI_AI_ENDPOINT,
                json=body,
                headers={"Content-Type": "application/json"},
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling NewsAPI.ai: {exc}",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"NewsAPI.ai returned HTTP {resp.status_code}: {resp.text}",
        )

    data = resp.json()

    if "error" in data:
        raise HTTPException(
            status_code=502,
            detail=f"NewsAPI.ai error: {data['error']}",
        )

    # ======================================================
    # 🔍 DEBUG BLOCK
    # ======================================================
    print("=== NewsAPI.ai DEBUG ===", flush=True)
    print("Request body:", body, flush=True)
    print("HTTP status:", resp.status_code, flush=True)
    print("Raw top-level keys:", list(data.keys()), flush=True)

    articles_block = data.get("articles", {})
    print("articles_block type:", type(articles_block), flush=True)

    results_preview = [
        a.get("title") for a in articles_block.get("results", [])[:5]
    ]
    print("First article titles:", results_preview, flush=True)
    print("totalResults:", articles_block.get("totalResults"), flush=True)
    print("======================================================", flush=True)
    # ======================================================

    # Parse article block
    articles = articles_block.get("results", []) or []
    total_raw = articles_block.get("totalResults", len(articles))

    documents: List[Dict[str, Any]] = []
    risk_hint = build_material_risk_hint(payload, effective_days_back)

    for a in articles:
        title = a.get("title") or ""
        content = a.get("body") or ""

        risk_features = compute_risk_features(title, content, materials)
        source = a.get("source", {})
        source_title = source.get("title") if isinstance(source, dict) else source

        doc = {
            "id": a.get("uri") or a.get("url"),
            "title": title,
            "content": content,
            "url": a.get("url"),
            "source": source_title,
            "publishedAt": (
                a.get("dateTimePub") or a.get("dateTime") or a.get("date")
            ),
            "material_risk_hint": risk_hint,
            "material_hits": risk_features["material_hits"],
            "risk_keyword_hits": risk_features["risk_keyword_hits"],
            "construction_hits": risk_features["construction_hits"],
            "joint_sentences": risk_features["joint_sentences"],
            "title_joint": risk_features["title_joint"],
            "risk_score": risk_features["risk_score"],
            "risk_features": risk_features,  # full structure for debugging/agent use
        }
        documents.append(doc)

    matched_before_filter = len(documents)

    # ---------------------------
    # Risk filtering logic
    # ---------------------------
    if payload.risk_filter:
        min_score = payload.min_risk_score if payload.min_risk_score is not None else 4

        filtered_docs: List[Dict[str, Any]] = []
        for d in documents:
            mf = d["risk_features"]

            # Must have at least one material and one risk word somewhere
            if mf["material_hits"] == 0 or mf["risk_keyword_hits"] == 0:
                continue

            # Require some fairly strong signal:
            # - either same-sentence co-occurrence or strong construction context
            if mf["joint_sentences"] == 0 and mf["construction_hits"] == 0 and not mf["title_joint"]:
                continue

            if mf["risk_score"] < min_score:
                continue

            filtered_docs.append(d)

        documents = filtered_docs

    # Rank by risk_score and cap the number of docs returned
    documents.sort(key=lambda d: d["risk_score"], reverse=True)
    documents = documents[: payload.max_docs]

    return {
        "status": "ok",
        "effective_days_back": effective_days_back,
        "total_results_raw": int(total_raw),
        "matched_docs_before_filter": matched_before_filter,
        "returned_docs": len(documents),
        "risk_filter": payload.risk_filter,
        "min_risk_score": payload.min_risk_score,
        "documents": documents,
    }
