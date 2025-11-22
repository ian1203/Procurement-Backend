from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import httpx
import os
from typing import List, Optional

# Read API key from environment (Render -> Environment -> NEWSAPI_KEY)
NEWS_API_KEY = os.environ.get("NEWSAPI_KEY")

# Risk-related keywords to bias NewsAPI and our own scoring
RISK_TERMS = [
    "shortage", "shortages",
    "disruption", "disruptions",
    "strike", "strikes",
    "shutdown", "shutdowns",
    "outage", "outages",
    "delay", "delays",
    "bottleneck", "bottlenecks",
    "logistics", "supply chain", "port congestion",
    "price spike", "price spikes", "price surge", "price surges",
    "shipping", "freight", "transport disruption",
    "labor dispute", "labour dispute"
]


class NewsQuery(BaseModel):
    """
    Query schema for the news search endpoint.

    - query: main risk phrase (e.g., "concrete disruption", "oil price")
    - region: geographic focus (e.g., "Montreal", "North America")
    - materials: list of material keywords (e.g., ["cement", "steel"])
    - days_back: how many days back to search (will be clamped to <= 30 due to NewsAPI limits)
    """
    query: str
    region: Optional[str] = None
    materials: Optional[List[str]] = None
    days_back: int = 5


app = FastAPI(
    title="Procurement News Backend",
    version="0.2.0",
    description=(
        "Backend service that calls NewsAPI.org and returns normalized, "
        "supply-chain-risk-oriented news documents for the Construction Supply Chain Risk Monitor agent."
    ),
)


def _score_article(
    art: dict,
    query: Optional[str],
    materials: Optional[List[str]],
    region: Optional[str],
    risk_terms: List[str],
) -> int:
    """
    Simple heuristic relevance scorer.

    Higher score = more likely to be a true supply-chain risk article for the requested materials/region.
    """
    text = " ".join([
        art.get("title") or "",
        art.get("description") or "",
        art.get("content") or "",
    ]).lower()

    score = 0

    # Tokens from main query
    if query:
        for token in query.lower().split():
            token = token.strip()
            if token and token in text:
                score += 1

    # Materials are strong signals
    if materials:
        for m in materials:
            m_norm = m.lower().strip()
            if m_norm and m_norm in text:
                score += 2

    # Region as a soft signal
    if region:
        if region.lower() in text:
            score += 1

    # Risk / disruption vocabulary
    for r in risk_terms:
        if r in text:
            score += 2

    return score


@app.post("/news/search")
async def news_search(q: NewsQuery):
    """
    Search normalized news articles from NewsAPI.org with a supply-chain risk focus.

    Returns:
    - status: "ok" on success
    - effective_days_back: the actual days used (clamped to <=30)
    - total_results_raw: NewsAPI's totalResults
    - documents: list of normalized articles with relevance scores
    """
    if not NEWS_API_KEY:
        # If this happens in Render, check Environment -> NEWSAPI_KEY
        return {"error": "NEWSAPI_KEY not configured"}

    # Clamp days_back to NewsAPI's ~30-day limit
    days_back = max(1, min(q.days_back, 30))
    from_date = (datetime.utcnow() - timedelta(days=days_back)).date().isoformat()

    # --- Build smarter query string for NewsAPI ---
    terms: List[str] = []

    # Main phrase in quotes to promote phrase matches
    if q.query:
        terms.append(f"\"{q.query}\"")

    # Materials block (OR)
    if q.materials:
        cleaned_mats = [m.strip() for m in q.materials if m.strip()]
        if cleaned_mats:
            mat_or = " OR ".join(cleaned_mats)
            terms.append(f"({mat_or})")

    # Region as a soft filter
    if q.region:
        terms.append(f"({q.region})")

    # Risk vocabulary block (always included)
    risk_or = " OR ".join(RISK_TERMS)
    terms.append(f"({risk_or})")

    final_query = " AND ".join(terms)

    params = {
        "q": final_query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",            # more relevant than just latest
        "pageSize": 30,                   # a bit more than default to have material to score
        "searchIn": "title,description",  # avoid noisy matches in full content blobs
        "apiKey": NEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://newsapi.org/v2/everything", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        # Bubble up a clean error for the agent
        raise HTTPException(status_code=502, detail=f"Error calling NewsAPI: {e}") from e

    articles = data.get("articles", [])

    # Score and normalize
    scored_docs = []
    for art in articles:
        score = _score_article(art, q.query, q.materials, q.region, RISK_TERMS)
        scored_docs.append((score, art))

    # Sort highest relevance first
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    documents = []
    for score, art in scored_docs:
        documents.append({
            "id": art.get("url"),
            "title": art.get("title"),
            "content": art.get("content") or art.get("description") or "",
            "url": art.get("url"),
            "source": (art.get("source") or {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "material_risk_hint": final_query,
            "relevance_score": score,
            # threshold is arbitrary; tweak if you want fewer/ more "high risk" hits
            "high_relevance": score >= 4,
        })

    return {
        "status": "ok",
        "effective_days_back": days_back,
        "total_results_raw": data.get("totalResults", 0),
        "documents": documents,
    }
