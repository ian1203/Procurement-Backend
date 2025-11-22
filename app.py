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

@app.post("/news/search")
async def news_search(q: NewsQuery):
    if not NEWS_API_KEY:
        return {"error": "NEWSAPI_KEY not configured"}

    # Build a smart query string
    terms = [q.query]
    if q.materials:
        terms.extend(q.materials)
    if q.region:
        terms.append(q.region)
    final_query = " AND ".join(terms)

    from_date = (datetime.utcnow() - timedelta(days=q.days_back)).date().isoformat()

    params = {
        "q": final_query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://newsapi.org/v2/everything", params=params)
        resp.raise_for_status()
        data = resp.json()

    # Normalize to what Orchestrate expects as "documents"
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

    return {"status": "ok", "documents": docs}
