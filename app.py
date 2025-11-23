import os
import re
import json
from datetime import date, timedelta
from typing import List, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# ---------------------------
# Config
# ---------------------------

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise RuntimeError("Please set the NEWSAPI_KEY environment variable")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

NEWSAPI_AI_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"

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
        description="If true, only return articles that look like material-related risks.",
    )
    max_docs: int = Field(
        20,
        ge=1,
        le=200,
        description="Maximum number of articles to return after filtering & ranking.",
    )
    min_risk_score: Optional[int] = Field(
        None,
        description="Optional minimum heuristic risk_score threshold when risk_filter=true.",
    )


class ArticleForClassification(BaseModel):
    id: Optional[str] = None
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    publishedAt: Optional[str] = None
    material_risk_hint: Optional[str] = None


class ClassificationRequest(BaseModel):
    materials: Optional[List[str]] = None
    region: Optional[str] = None
    documents: List[ArticleForClassification]


class RiskRequest(BaseModel):
    materials: Optional[Any] = None  # can be list or string; we normalize
    region: Optional[Any] = None
    days_back: int = 7
    max_docs: int = 10
    query: Optional[str] = None


# ---------------------------
# App
# ---------------------------

app = FastAPI(
    title="Procurement News Backend",
    version="0.2.0",
    description="Backend service that calls NewsAPI.ai for news ingestion and an LLM for supply-chain risk classification.",
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


def normalize_materials(raw) -> List[str]:
    """
    Normalize the 'materials' field into a clean list of lowercase material names.
    Handles:
    - None  -> default broad list
    - list[str] -> cleaned
    - string like '["steel", "cement"]' -> parsed as JSON
    - string like 'steel, cement' -> split on commas
    """
    default_materials = ["cement", "concrete", "rebar", "steel", "aggregates", "copper", "lumber"]

    if raw is None:
        return default_materials

    if isinstance(raw, list):
        cleaned = [str(m).strip().lower() for m in raw if str(m).strip()]
        return cleaned or default_materials

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return default_materials

        # Try JSON list first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                cleaned = [str(m).strip().lower() for m in parsed if str(m).strip()]
                if cleaned:
                    return cleaned
        except Exception:
            pass

        # Fallback: comma-separated list
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        return parts or default_materials

    return default_materials


def normalize_region(raw: Optional[Any]) -> Optional[str]:
    """
    Convert things like 'null' / '' into proper None.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.lower() in {"null", "none", "global", "world", "worldwide"}:
            return None
        return s
    return str(raw)


# Risk scoring keyword lists
RISK_KEYWORDS = [
    "shortage",
    "shortages",
    "disruption",
    "disruptions",
    "strike",
    "strikes",
    "shutdown",
    "shutdowns",
    "closure",
    "closures",
    "fire",
    "explosion",
    "flood",
    "earthquake",
    "hurricane",
    "typhoon",
    "sanction",
    "sanctions",
    "ban",
    "export ban",
    "blockade",
    "delay",
    "delays",
    "bottleneck",
    "congestion",
]
CONSTRUCTION_KEYWORDS = [
    "construction",
    "infrastructure",
    "project",
    "projects",
    "site",
    "bridge",
    "tunnel",
    "road",
    "highway",
    "building",
    "plant",
    "factory",
]


def compute_risk_features(text: str, title: str, materials: List[str]) -> dict:
    text_l = (text or "").lower()
    title_l = (title or "").lower()

    material_hits = 0
    for m in materials:
        if not m:
            continue
        pattern = re.escape(m.lower())
        material_hits += len(re.findall(pattern, text_l))

    risk_hits = 0
    for k in RISK_KEYWORDS:
        pattern = r"\b" + re.escape(k.lower()) + r"\b"
        risk_hits += len(re.findall(pattern, text_l))

    constr_hits = 0
    for k in CONSTRUCTION_KEYWORDS:
        pattern = r"\b" + re.escape(k.lower()) + r"\b"
        constr_hits += len(re.findall(pattern, text_l))

    sentences = re.split(r"[.!?]\s+", text_l)
    joint_sentences = 0
    for s in sentences:
        if not s:
            continue
        if any(m in s for m in materials) and any(k in s for k in RISK_KEYWORDS):
            joint_sentences += 1

    title_joint = any(m in title_l for m in materials) and any(
        k in title_l for k in RISK_KEYWORDS
    )

    risk_score = (
        material_hits
        + 2 * risk_hits
        + constr_hits
        + 2 * joint_sentences
        + (2 if title_joint else 0)
    )
    # Clamp to a simple 0–10 scale
    risk_score = max(0, min(int(risk_score), 10))

    return {
        "material_hits": material_hits,
        "risk_keyword_hits": risk_hits,
        "construction_hits": constr_hits,
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

    keywords: Any
    keyword_oper = "or"

    or_split = re.split(r"\bOR\b", raw_q, flags=re.IGNORECASE)
    parts = [p.strip() for p in or_split if p.strip()]

    if len(parts) > 1:
        keywords = parts
        keyword_oper = "or"
    else:
        keywords = raw_q

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
        async with httpx.AsyncClient(timeout=20.0) as client_http:
            resp = await client_http.post(
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

    # Debug
    print("=== NewsAPI.ai DEBUG ===", flush=True)
    print("Request body:", body, flush=True)
    print("HTTP status:", resp.status_code, flush=True)
    print("Raw top-level keys:", list(data.keys()), flush=True)

    articles_block = data.get("articles", {})
    print("articles_block_type:", type(articles_block), flush=True)
    print("totalResults:", articles_block.get("totalResults"), flush=True)
    print("======================================================", flush=True)

    # Parse article block
    articles = articles_block.get("results", []) or []
    total_raw = articles_block.get("totalResults", len(articles))

    materials = normalize_materials(payload.materials)
    risk_hint = build_material_risk_hint(payload, effective_days_back)

    documents = []
    for a in articles:
        source = a.get("source", {})
        source_title = source.get("title") if isinstance(source, dict) else source

        title = a.get("title") or ""
        content = a.get("body") or ""
        features = compute_risk_features(
            text=f"{title}\n{content}",
            title=title,
            materials=materials,
        )

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
            "material_hits": features["material_hits"],
            "risk_keyword_hits": features["risk_keyword_hits"],
            "construction_hits": features["construction_hits"],
            "joint_sentences": features["joint_sentences"],
            "title_joint": features["title_joint"],
            "risk_score": features["risk_score"],
        }
        documents.append(doc)

    matched_docs_before_filter = len(documents)
    docs_filtered = documents

    min_score_used: Optional[int] = None

    if payload.risk_filter:
        min_score_used = (
            payload.min_risk_score
            if payload.min_risk_score is not None
            else 3
        )
        docs_filtered = [
            d for d in documents if d["risk_score"] >= min_score_used
        ]

    # Rank by risk_score and cap the number of docs returned
    docs_filtered.sort(key=lambda d: d["risk_score"], reverse=True)
    docs_filtered = docs_filtered[: payload.max_docs]

    return {
        "status": "ok",
        "effective_days_back": effective_days_back,
        "total_results_raw": int(total_raw),
        "matched_docs_before_filter": matched_docs_before_filter,
        "returned_docs": len(docs_filtered),
        "risk_filter": payload.risk_filter,
        "min_risk_score": min_score_used,
        "documents": docs_filtered,
    }


@app.post("/news/classify")
async def news_classify(payload: ClassificationRequest):
    """
    Classify a set of articles as external risks or not, using OpenAI.
    """
    if not payload.documents:
        return {
            "status": "ok",
            "total_docs": 0,
            "risk_docs": 0,
            "classified": [],
            "risks_only": [],
        }

    materials_str = ", ".join(payload.materials or [])
    region_str = payload.region or "Global"

    articles_parts = []
    for idx, doc in enumerate(payload.documents, start=1):
        articles_parts.append(
            f"Article {idx}:\n"
            f"id: {doc.id}\n"
            f"title: {doc.title}\n"
            f"content: {doc.content or ''}\n"
            f"source: {doc.source or ''}\n"
            f"publishedAt: {doc.publishedAt or ''}\n"
            f"material_risk_hint: {doc.material_risk_hint or ''}\n"
        )
    articles_block = "\n\n".join(articles_parts)

    system_prompt = (
        "You are a supply chain risk classifier for construction materials. "
        "You decide whether each news article describes a REAL external event "
        "that could disrupt the supply, availability, or delivery of the materials."
    )

    user_prompt = f"""
Materials of interest: {materials_str or "various construction materials"}
Region of interest: {region_str}

Only mark is_risk=true if:
- there is an actual external event (plant fire, strike, port closure, sanctions, export ban,
  extreme weather, major regulatory change, shipping disruption, etc.), AND
- it plausibly affects supply or delivery timelines of construction materials.

If the article is only about price forecasts, generic market commentary, company earnings,
or unrelated politics, set is_risk=false.

Return a JSON object with a single key "classified" whose value is a list of objects,
one per article in the SAME ORDER. Each object MUST have:

- id (string or null)
- title (string)
- url (string or null)
- source (string or null)
- publishedAt (string or null)
- material_risk_hint (string or null)
- is_risk (boolean)
- risk_level (integer 1-5; 1 = very low, 5 = very high)
- affected_materials (list of strings)
- affected_regions (list of strings)
- time_horizon_days (integer or null)
- reason (short explanation)

Articles:
{articles_block}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {exc}",
        ) from exc

    raw_content = completion.choices[0].message.content
    try:
        parsed = json.loads(raw_content)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse OpenAI JSON: {exc}; raw={raw_content}",
        ) from exc

    classified = parsed.get("classified", [])
    risk_only = [c for c in classified if c.get("is_risk")]

    return {
        "status": "ok",
        "total_docs": len(payload.documents),
        "risk_docs": len(risk_only),
        "classified": classified,
        "risks_only": risk_only,
    }


@app.post("/news/risk")
async def news_risk(payload: RiskRequest):
    """
    High-level endpoint:
    - builds a broad news query for the given materials/region
    - fetches candidate articles from NewsAPI.ai
    - runs LLM classification
    - returns only external risk events
    """

    # Normalize inputs (tolerate WatsonX sending strings)
    materials = normalize_materials(payload.materials)
    region = normalize_region(payload.region)

    days_back: int = payload.days_back
    if isinstance(days_back, str):
        try:
            days_back = int(days_back)
        except ValueError:
            days_back = 7
    days_back = max(1, min(days_back, 30))

    max_docs: int = payload.max_docs
    if isinstance(max_docs, str):
        try:
            max_docs = int(max_docs)
        except ValueError:
            max_docs = 10
    max_docs = max(1, min(max_docs, 50))

    # Decide query string
    if payload.query:
        query_str = payload.query
    else:
        query_str = " OR ".join(materials)

    # Call internal news_search (risk_filter disabled here)
    news_query = NewsQuery(
        query=query_str,
        region=region,
        materials=materials,
        days_back=days_back,
        risk_filter=False,      # keep broad recall; LLM will filter
        max_docs=max_docs,
        min_risk_score=None,
    )

    search_result = await news_search(news_query)
    candidate_docs = search_result["documents"]

    if not candidate_docs:
        return {
            "status": "ok",
            "effective_days_back": search_result["effective_days_back"],
            "total_results_raw": search_result["total_results_raw"],
            "candidate_docs": 0,
            "risk_docs": 0,
            "risks": [],
        }

    # Build classification request
    class_docs = []
    for d in candidate_docs:
        class_docs.append(
            ArticleForClassification(
                id=d.get("id"),
                title=d.get("title") or "",
                content=d.get("content") or "",
                url=d.get("url"),
                source=d.get("source"),
                publishedAt=d.get("publishedAt"),
                material_risk_hint=d.get("material_risk_hint"),
            )
        )

    class_req = ClassificationRequest(
        materials=materials,
        region=region,
        documents=class_docs,
    )

    class_result = await news_classify(class_req)

    return {
        "status": "ok",
        "effective_days_back": search_result["effective_days_back"],
        "total_results_raw": search_result["total_results_raw"],
        "candidate_docs": len(candidate_docs),
        "risk_docs": class_result["risk_docs"],
        "risks": class_result["risks_only"],
    }
