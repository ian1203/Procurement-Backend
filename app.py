import os
import re
import json
from datetime import date, timedelta
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------
# Config
# ---------------------------

# News API
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise RuntimeError("Please set the NEWSAPI_KEY environment variable")

NEWSAPI_AI_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"

# LLM (for classification). Optional: only required for /news/classify
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Domain vocab
DEFAULT_MATERIALS = ["cement", "concrete", "rebar", "steel", "aggregate", "aggregates"]

RISK_KEYWORDS = [
    "shortage", "shortages", "scarcity", "scarce",
    "disruption", "disruptions", "disrupted",
    "delay", "delays", "delayed",
    "bottleneck", "bottlenecks",
    "backlog", "backlogs",
    "strike", "strikes", "walkout", "walkouts",
    "lockout", "lockouts",
    "shutdown", "shut down", "closure", "closures",
    "suspension", "suspended",
    "port", "ports", "terminal", "terminals",
    "congestion", "backed up",
    "shipping", "shipment", "shipments", "freight",
    "supply chain", "logistics",
    "sanction", "sanctions", "embargo", "export ban", "export bans",
    "tariff", "tariffs", "export controls",
    "price spike", "price spikes", "price surge", "price surges",
    "price increase", "price increases",
    "cost increase", "cost pressure",
]

CONSTRUCTION_KEYWORDS = [
    "construction", "infrastructure",
    "bridge", "bridges",
    "highway", "highways", "road", "roads",
    "tunnel", "tunnels", "dam", "dams",
    "cement plant", "cement factory", "cement works",
    "steel mill", "steel plant",
    "quarry", "quarries", "ready-mix", "ready mix",
    "batch plant", "asphalt plant", "concrete plant",
]


# ---------------------------
# Pydantic models
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
        description="If true, apply a simple heuristic risk filter on the raw articles.",
    )
    max_docs: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of documents to return after filtering & ranking.",
    )
    min_risk_score: Optional[int] = Field(
        None,
        description="Optional minimum heuristic risk_score; if omitted a default is used.",
    )


class ArticleForClassification(BaseModel):
    id: Optional[str] = None
    title: str
    content: Optional[str] = ""
    url: Optional[str] = None
    source: Optional[str] = None
    publishedAt: Optional[str] = None
    material_risk_hint: Optional[str] = None


class ClassificationRequest(BaseModel):
    materials: Optional[List[str]] = None
    region: Optional[str] = None
    documents: List[ArticleForClassification]

class RiskRequest(BaseModel):
    """
    High-level request for pre-filtered, LLM-classified external risks.

    The backend will:
    - build a reasonable query from the materials if query is not provided
    - call /news/search with risk_filter=false
    - call /news/classify on the returned documents
    - return only risk events
    """
    materials: Optional[List[str]] = Field(
        None,
        description="Materials of interest. If omitted, defaults to cement/concrete/rebar/steel/aggregates.",
    )
    region: Optional[str] = Field(
        None,
        description="Region of interest (country / area), e.g. 'United States', 'Mexico', 'EU'.",
    )
    days_back: int = Field(
        7,
        ge=1,
        le=365,
        description="How many days back to search from today (capped to 30 in backend).",
    )
    max_docs: int = Field(
        10,
        ge=1,
        le=50,
        description="Max candidate docs fetched from NewsAPI before classification.",
    )
    query: Optional[str] = Field(
        None,
        description="Optional override for raw keyword query. "
                    "If omitted, backend builds 'material1 OR material2 ...'.",
    )


class ClassifiedArticle(BaseModel):
    id: Optional[str]
    title: str
    url: Optional[str]
    source: Optional[str]
    publishedAt: Optional[str]
    material_risk_hint: Optional[str]
    is_risk: bool
    risk_level: int
    affected_materials: List[str]
    affected_regions: List[str]
    time_horizon_days: Optional[int]
    reason: str


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="Procurement News Backend",
    version="0.2.0",
    description="Backend service that calls NewsAPI.ai for news ingestion and an LLM for risk classification.",
)


# ---------------------------
# Helper functions
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
    base.extend(DEFAULT_MATERIALS)
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
    return re.split(r"[.!?]\s+", text)


def count_occurrences(text: str, terms: List[str]) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for term in terms:
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

    # treat "concrete" carefully: if no construction context nearby, it's often metaphorical
    material_terms_effective = []
    for m in material_terms:
        if m == "concrete":
            if any(ck in text_lower for ck in CONSTRUCTION_KEYWORDS):
                material_terms_effective.append(m)
        else:
            material_terms_effective.append(m)

    material_hits = count_occurrences(text_lower, material_terms_effective)
    risk_keyword_hits = count_occurrences(text_lower, RISK_KEYWORDS)
    construction_hits = count_occurrences(text_lower, CONSTRUCTION_KEYWORDS)

    sentences = split_sentences(full_text)
    joint_sentences = 0
    for s in sentences:
        s_lower = s.lower()
        if any(m in s_lower for m in material_terms_effective) and any(
            rk in s_lower for rk in RISK_KEYWORDS
        ):
            joint_sentences += 1

    title_lower = title.lower()
    title_has_material = any(m in title_lower for m in material_terms_effective)
    title_has_risk = any(rk in title_lower for rk in RISK_KEYWORDS)
    title_joint = bool(title_has_material and title_has_risk)

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


async def call_openai_chat(prompt: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set; classification endpoint is not configured.",
        )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies news articles about supply "
                    "chain risks for construction materials."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {resp.status_code} {resp.text}",
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response format from OpenAI.",
        )

    # Expect content to be a JSON object; if not, try to extract JSON part
    content = content.strip()
    try:
        if content.startswith("```"):
            # handle ```json ... ```
            content = re.sub(r"^```[a-zA-Z]*", "", content)
            content = re.sub(r"```$", "", content).strip()
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse JSON from OpenAI response: {content}",
        )


def build_classification_prompt(
    doc: ArticleForClassification,
    materials: List[str],
    region: Optional[str],
) -> str:
    materials_str = ", ".join(materials) if materials else "cement, concrete, rebar, steel and other construction materials"
    region_str = region or "any region relevant to the article"

    return (
        "You are classifying whether this news article describes an EXTERNAL RISK that "
        "could affect the SUPPLY or DELIVERY of construction materials.\n\n"
        f"Materials of interest: {materials_str}\n"
        f"Region of interest: {region_str}\n\n"
        "For the given article, answer with a single JSON object with this schema:\n"
        "{\n"
        '  "is_risk": true/false,\n'
        '  "risk_level": integer 1-5, 1 = very low, 5 = very high,\n'
        '  "affected_materials": [list of material names],\n'
        '  "affected_regions": [list of regions/countries/ports],\n'
        '  "time_horizon_days": integer or null,\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "Classify as is_risk = true ONLY if the article clearly describes a disruption, shortage, "
        "shutdown, strike, port congestion, export ban, price spike, accident or similar event "
        "that could realistically impact supply or delivery of the materials.\n\n"
        "Article title:\n"
        f"{doc.title}\n\n"
        "Article content (may be truncated):\n"
        f"{(doc.content or '')[:2000]}\n"
    )


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

    # Build keyword(s) for NewsAPI.ai
    raw_q = (payload.query or "").strip()
    keywords: List[str] | str
    keyword_oper = "or"

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

    # Debug
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

    # Parse article block
    articles = articles_block.get("results", []) or []
    total_raw = articles_block.get("totalResults", len(articles))

    documents: List[Dict[str, Any]] = []
    risk_hint = build_material_risk_hint(payload, effective_days_back)

    for a in articles:
        title = a.get("title") or ""
        content = a.get("body") or ""
        source = a.get("source", {})
        source_title = source.get("title") if isinstance(source, dict) else source

        risk_features = compute_risk_features(title, content, materials)

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
        }
        documents.append(doc)

    matched_before_filter = len(documents)

    # Simple heuristic filter (still recall-first; classification does the heavy lifting)
    if payload.risk_filter:
        min_score = payload.min_risk_score if payload.min_risk_score is not None else 2
        filtered_docs: List[Dict[str, Any]] = []
        for d in documents:
            if (
                d["material_hits"] == 0
                or d["risk_keyword_hits"] == 0
                or d["risk_score"] < min_score
            ):
                continue
            filtered_docs.append(d)
        documents = filtered_docs

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


@app.post("/news/classify")
async def news_classify(payload: ClassificationRequest):
    """
    Takes a batch of documents (e.g. from /news/search) and uses an LLM
    to classify which ones are true external supply risks.
    """
    materials = normalize_materials(payload.materials)
    region = payload.region

    classified_results: List[ClassifiedArticle] = []

    for doc in payload.documents:
        prompt = build_classification_prompt(doc, materials, region)
        result = await call_openai_chat(prompt)

        classified = ClassifiedArticle(
            id=doc.id,
            title=doc.title,
            url=doc.url,
            source=doc.source,
            publishedAt=doc.publishedAt,
            material_risk_hint=doc.material_risk_hint,
            is_risk=bool(result.get("is_risk", False)),
            risk_level=int(result.get("risk_level", 1)),
            affected_materials=[str(m) for m in result.get("affected_materials", [])],
            affected_regions=[str(r) for r in result.get("affected_regions", [])],
            time_horizon_days=(
                int(result["time_horizon_days"])
                if result.get("time_horizon_days") is not None
                else None
            ),
            reason=str(result.get("reason", "")),
        )
        classified_results.append(classified)

    # Optionally filter to only risk == true
    risks_only = [c for c in classified_results if c.is_risk]

    return {
        "status": "ok",
        "total_docs": len(classified_results),
        "risk_docs": len(risks_only),
        "classified": [c.dict() for c in classified_results],
        "risks_only": [c.dict() for c in risks_only],
    }

@app.post("/news/risk")
async def news_risk(payload: RiskRequest):
    """
    High-level endpoint:
    - builds a broad news query for the given materials/region
    - fetches candidate articles from NewsAPI.ai
    - runs LLM classification
    - returns only external risk events

    This is the endpoint your agent should usually call.
    """

    # 1) Decide query string
    # If user provided a query, use it; otherwise build "mat1 OR mat2 ..."
    materials = normalize_materials(payload.materials)
    if payload.query:
        query_str = payload.query
    else:
        # Simple OR query over materials, e.g. "cement OR concrete OR rebar ..."
        query_str = " OR ".join(materials)

    # 2) Call the internal /news/search logic with risk_filter disabled
    news_query = NewsQuery(
        query=query_str,
        region=payload.region,
        materials=materials,
        days_back=payload.days_back,
        risk_filter=False,          # keep broad recall; LLM will filter
        max_docs=payload.max_docs,
        min_risk_score=None,
    )

    search_result = await news_search(news_query)

    candidate_docs = search_result.get("documents", []) or []

    # If no candidates, short-circuit
    if not candidate_docs:
        return {
            "status": "ok",
            "effective_days_back": search_result.get("effective_days_back"),
            "total_results_raw": search_result.get("total_results_raw"),
            "candidate_docs": 0,
            "risk_docs": 0,
            "risks": [],
        }

    # 3) Build ClassificationRequest from search docs
    classification_docs: List[ArticleForClassification] = []
    for d in candidate_docs:
        classification_docs.append(
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

    classification_request = ClassificationRequest(
        materials=materials,
        region=payload.region,
        documents=classification_docs,
    )

    classification_result = await news_classify(classification_request)

    return {
        "status": "ok",
        "effective_days_back": search_result.get("effective_days_back"),
        "total_results_raw": search_result.get("total_results_raw"),
        "candidate_docs": search_result.get("returned_docs"),
        "risk_docs": classification_result.get("risk_docs"),
        "risks": classification_result.get("risks_only"),
    }
