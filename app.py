from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from datetime import datetime, timedelta

NEWS_API_KEY = os.environ.get("NEWSAPI_KEY")

app = FastAPI()

class NewsQuery(BaseModel):
    query: str
    region: str | None = None        # e.g. "Montreal", "Quebec", "Canada"
    materials: list[str] | None = None  # e.g. ["concrete", "rebar"]
    days_back: int = 5               # how far back to search

MAX_DAYS_BACK = 30  # NewsAPI free tier limitation

@app.post("/news/search")
async def news_search(q: NewsQuery):
    # 1) Check key
    if not NEWS_API_KEY:
        return {"status": "error", "message": "NEWSAPI_KEY not configured"}

    # 2) Clamp the date window so we never ask NewsAPI for more than 30 days
    effective_days = min(max(q.days_back, 1), MAX_DAYS_BACK)

    # 3) Build a smart query string
    terms = [q.query]
    if q.materials:
        terms.extend(q.materials)
    if q.region:
        terms.append(q.region)
    final_query = " AND ".join(terms)

    from_date = (datetime.utcnow() - timedelta(days=effective_days)).date().isoformat()

    params = {
        "q": final_query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://newsapi.org/v2/everything", params=params)
    except httpx.RequestError as exc:
        # Network / connection issue
        return {
            "status": "error",
            "message": f"Network error while calling NewsAPI: {exc}"
        }

    # If NewsAPI returns non-200, don’t crash – return a clean error payload
    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        return {
            "status": "error",
            "message": f"NewsAPI returned HTTP {resp.status_code}",
            "newsapi_body": body,
        }

    data = resp.json()

    # 4) Normalize to "documents" for Orchestrate
    docs = []
    for art in data.get("articles", []):
        docs.append({
            "id": art.get("url"),
            "title": art.get("title"),
            "content": art.get("content") or art.get("description") or "",
            "url": art.get("url"),
            "source": art.get("source", {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "material_risk_hint": final_query,
        })

    return {
        "status": "ok",
        "effective_days_back": effective_days,
        "documents": docs,
        "total_results_raw": data.get("totalResults"),
    }
