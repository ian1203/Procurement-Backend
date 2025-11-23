import os
from datetime import date, timedelta
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------
# Config
# ---------------------------

NEWSAPI_AI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_AI_KEY:
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
    This is just for debugging / transparency for the WatsonX agent.
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
# Routes
# ---------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/news/search")
async def news_search(payload: NewsQuery):
    # NewsAPI.ai free / normal plans are typically limited to the last 30 days,
    # so we cap there to avoid confusing “older than 30 days” failures.
    effective_days_back = min(payload.days_back, 30)

    today = date.today()
    date_start = today - timedelta(days=effective_days_back)

    # Build keyword list: main risk phrase + materials
    keywords: List[str] = [payload.query]
    if payload.materials:
        keywords.extend(m for m in payload.materials if m)

    # If there is more than one keyword we use AND; otherwise it doesn't matter
    keyword_oper = "and" if len(keywords) > 1 else "or"

    # Build request body for NewsAPI.ai / Event Registry
    body = {
        # authentication
        "apiKey": NEWSAPI_AI_KEY,
        # filters
        "keyword": keywords if len(keywords) > 1 else keywords[0],
        "keywordOper": keyword_oper,
        "lang": ["eng"],
        "dateStart": date_start.strftime("%Y-%m-%d"),
        "dateEnd": today.strftime("%Y-%m-%d"),
        # result options
        "resultType": "articles",
        "articlesSortBy": "date",     # newest first
        "articlesCount": 50,          # up to 100 allowed per call
        # we *don’t* request heavy extra fields (concepts, categories) for now
    }

    # Optional region filter: use sourceLocationUri when user provides region
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
        # Event Registry style error
        raise HTTPException(
            status_code=502,
            detail=f"NewsAPI.ai error: {data['error']}",
        )

    articles_block = data.get("articles", {})
    articles = articles_block.get("results", []) or []
    total_raw = articles_block.get("totalResults", len(articles))

    # Normalize into the format your WatsonX tool expects
    documents = []
    risk_hint = build_material_risk_hint(payload, effective_days_back)

    for a in articles:
        source = a.get("source", {})
        if isinstance(source, dict):
            source_title = source.get("title")
        else:
            source_title = source

        documents.append(
            {
                "id": a.get("uri") or a.get("url"),
                "title": a.get("title"),
                "content": a.get("body"),
                "url": a.get("url"),
                "source": source_title,
                "publishedAt": (
                    a.get("dateTimePub")
                    or a.get("dateTime")
                    or a.get("date")
                ),
                "material_risk_hint": risk_hint,
            }
        )

    return {
        "status": "ok",
        "effective_days_back": effective_days_back,
        "total_results_raw": int(total_raw),
        "documents": documents,
    }
