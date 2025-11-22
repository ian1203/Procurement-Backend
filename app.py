from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from datetime import datetime, timedelta

app = FastAPI()

class NewsQuery(BaseModel):
    query: str
    region: str | None = None
    materials: list[str] | None = None
    days_back: int = 5

def get_news_api_key() -> str:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="NEWSAPI_KEY not configured")
    return key

@app.post("/news/search")
async def news_search(q: NewsQuery):
    NEWS_API_KEY = get_news_api_key()   # <-- read per request

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
