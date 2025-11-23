import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

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
# You can also use "https://newsapi.ai/api/v1/article/getArticles" –
# both domains point to the same backend. If one ever fails, just swap.


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
    # Intelligence layer knobs
    risk_filter: bool = Field(
        True,
        description="If true, only return articles that look like material-related risks."
    )
    max_docs: int = Field(
        200,
        ge=1,
        le=1000,
        description="Maximum number of articles to return after filtering & ranking."
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
    """
    Short human-readable description of the filter that produced this result.
    This is just for debugging / transparency for the WatsonX/OpenAI agent.
    """
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
    """
    Map a free-text region to a Wikipedia URI for Event Registry's sourceLocationUri.
    This is a heuristic but works well for most cities/countries.
    """
    slug = region.strip().replace(" ", "_")
    return f"http://en.wikipedia.org/wiki/{slug}"


# ---------------------------
# Simple risk + material heuristics
# ---------------------------

RISK_KEYWORDS: List[str] = [
    "shortage", "shortages",
    "strike", "walkout", "protest",
    "disruption", "disruptions", "bottleneck",
    "delay", "delays",
    "shutdown", "shut down", "closure", "closed",
    "outage",
    "congestion", "port congestion",
    "backlog", "backlogs",
    "price spike", "price spikes", "price surge", "price surges",
    "price increase", "price increases", "price hike", "price hikes",
    "tariff", "tariffs",
    "sanction", "sanctions",
    "export ban", "export bans", "embargo",
    "restriction", "restrictions", "quota", "quotas",
    "regulation", "regulations", "new law", "new laws",
    "environmental rule", "environmental rules",
]

MATERIAL_SYNONYMS = {
    "cement": ["cement", "clinker", "ready-mix", "ready mix"],
    "concrete": ["concrete", "ready-mix", "ready mix"],
    "rebar": ["rebar", "reinforcing steel", "steel rebar"],
    "steel": ["steel", "steel mill", "steelmaker", "steel producer"],
    "aggregates": ["aggregates", "gravel", "crushed stone", "sand", "construction aggregates"],
    "asphalt": ["asphalt", "bitumen"],
}


def _flatten_material_terms(materials: Optional[List[str]]) -> List[str]:
    """
    From payload.materials build a list of concrete terms we care about.
    If nothing provided, fall back to a global construction-material list.
    """
    if not materials:
        materials = ["cement", "concrete", "rebar", "steel", "aggregates", "asphalt"]

    terms: List[str] = []
    for m in materials:
        m_lower = m.lower()
        if m_lower in MATERIAL_SYNONYMS:
            terms.extend(MATERIAL_SYNONYMS[m_lower])
        else:
            terms.append(m_lower)
    # dedupe & normalize to lowercase
    return sorted({t.lower() for t in terms})


def score_article_for_risk(text: str, material_terms: List[str]) -> Tuple[int, int, int]:
    """
    Return (material_hits, risk_hits, total_score) for a given piece of text.
    Very simple: count substring matches.
    """
    t = text.lower()

    mat_hits = sum(1 for m in material_terms if m in t)
    risk_hits = sum(1 for k in RISK_KEYWORDS if k in t)

    # Simple combined score; we can tune weights later.
    total_score = mat_hits * 2 + risk_hits  # materials slightly higher weight
    return mat_hits, risk_hits, total_score


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

    keywords: list[str] | str
    keyword_oper = "or"

    # If user typed something like "A OR B OR C",
    # split on OR and send a keyword list
    or_split = re.split(r"\bOR\b", raw_q, flags=re.IGNORECASE)
    parts = [p.strip() for p in or_split if p.strip()]

    if len(parts) > 1:
        # Multiple tokens -> use list + OR
        keywords = parts
        keyword_oper = "or"
    else:
        # Single phrase -> send as-is
        keywords = raw_q

    body = {
        "apiKey": NEWSAPI_KEY,
        "keyword": keywords,          # string or list
        "keywordOper": keyword_oper,  # ignored for single string, fine for list
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
    # 🔍 DEBUG BLOCK — Print exactly what we sent and got
    # ======================================================
    print("=== NewsAPI.ai DEBUG ===", flush=True)
    print("Request body:", body, flush=True)
    print("HTTP status:", resp.status_code, flush=True)
    print("Raw top-level keys:", list(data.keys()), flush=True)

    articles_block = data.get("articles", {})
    print("articles_block:", type(articles_block), articles_block, flush=True)

    results_preview = [
        a.get("title") for a in articles_block.get("results", [])[:5]
    ]
    print("First article titles:", results_preview, flush=True)

    print("totalResults:", articles_block.get("totalResults"), flush=True)
    print("======================================================", flush=True)
    # ======================================================

    # 3) Parse article block
    articles = articles_block.get("results", []) or []
    total_raw = articles_block.get("totalResults", len(articles))

    documents: List[dict] = []
    risk_hint = build_material_risk_hint(payload, effective_days_back)

    # pre-compute material terms for scoring
    material_terms = _flatten_material_terms(payload.materials)

    for a in articles:
        source = a.get("source", {})
        source_title = source.get("title") if isinstance(source, dict) else source

        title = a.get("title") or ""
        body_text = a.get("body") or ""
        full_text = f"{title}\n{body_text}"

        # Heuristic risk scoring
        mat_hits, risk_hits, score = score_article_for_risk(full_text, material_terms)

        # If risk_filter is ON:
        #   - Require at least one material hit AND one risk hit.
        if payload.risk_filter and (mat_hits == 0 or risk_hits == 0):
            continue

        documents.append(
            {
                "id": a.get("uri") or a.get("url"),
                "title": title,
                "content": body_text,
                "url": a.get("url"),
                "source": source_title,
                "publishedAt": (
                    a.get("dateTimePub")
                    or a.get("dateTime")
                    or a.get("date")
                ),
                "material_risk_hint": risk_hint,
                # Expose heuristics to the agent / future LLM layer
                "material_hits": mat_hits,
                "risk_keyword_hits": risk_hits,
                "risk_score": score,
            }
        )

    # Rank by risk_score and cap the number of docs returned
    documents.sort(key=lambda d: d["risk_score"], reverse=True)
    documents = documents[: payload.max_docs]

    return {
        "status": "ok",
        "effective_days_back": effective_days_back,
        "total_results_raw": int(total_raw),
        "returned_docs": len(documents),
        "documents": documents,
    }
