from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from datetime import datetime, timedelta

NEWS_API_KEY = os.environ.get("NEWSAPI_KEY")

app = FastAPI()


class NewsQuery(BaseModel):
    query: str
    region: str | None = None          # e.g. "US OR Canada OR Mexico"
    materials: list[str] | None = None # e.g. ["cement", "steel"]
    days_back: int = 5                 # requested days back (will be clamped to 30)


def build_newsapi_query(q: NewsQuery) -> tuple[str, int]:
    """
    Build a NewsAPI-compatible query string and compute an effective days_back
    (capped at 30 days, due to NewsAPI limits).
    """
    max_days = 30
    # Clamp days_back between 1 and 30
    effective_days = max(1, min(q.days_back, max_days))

    core = (q.query or "").strip()
    if not core:
        raise ValueError("query must not be empty")

    # --- Materials block: "cement" OR "concrete" OR "steel"
    materials_block = ""
    if q.materials:
        mats = [m.strip() for m in q.materials if m and m.strip()]
        if mats:
            materials_block = " OR ".join(f'"{m}"' for m in mats)

    # --- Region block: split on OR so "US OR Canada" is turned into "US" OR "Canada"
    region_block = ""
    if q.region:
        parts = [p.strip() for p in q.region.split("OR")]
        parts = [p for p in parts if p]
        if parts:
            region_block = " OR ".join(f'"{p}"' for p in parts)

    # --- Final boolean expression:
    # (core) AND (materials_block OR region_block)  [if any optional blocks exist]
    final_query = f"({core})"
    optional_terms = []

    if materials_block:
        optional_terms.append(f"({materials_block})")
    if region_block:
        optional_terms.append(f"({region_block})")

    if optional_terms:
        final_query += " AND (" + " OR ".join(optional_terms) + ")"

    return final_query, effective_days


@app.post("/news/search")
async def news_search(q: NewsQuery):
    if not NEWS_API_KEY:
        # Using HTTPException so FastAPI returns a proper error code to callers (like Orchestrate)
        raise HTTPException(status_code=500, detail="NEWSAPI_KEY not configured")

    try:
        final_query, effective_days = build_newsapi_query(q)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # NewsAPI free tier allows max 30 days, we already clamped effective_days
    from_date = (datetime.utcnow() - timedelta(days=effective_days)).date().isoformat()

    params = {
        "q": final_query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "searchIn": "title,description",  # focus on title & description to cut noise
        "apiKey": NEWS_API_KEY,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://newsapi.org/v2/everything", params=params)

    if resp.status_code != 200:
        # Try to surface the NewsAPI error message so we can debug easily
        try:
            payload = resp.json()
            message = payload.get("message") or payload
        except Exception:
            message = resp.text
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"NewsAPI error: {message}",
        )

    data = resp.json()
    articles = data.get("articles", [])
    total_results_raw = data.get("totalResults", len(articles))

    docs = []
    for art in articles:
        docs.append({
            "id": art.get("url"),
            "title": art.get("title"),
            "content": art.get("content") or art.get("description") or "",
            "url": art.get("url"),
            "source": (art.get("source") or {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "material_risk_hint": final_query,
        })

    return {
        "status": "ok",
        "effective_days_back": effective_days,
        "total_results_raw": total_results_raw,
        "documents": docs,
    }
